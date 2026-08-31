#!/usr/bin/env python3
"""Fine-tune a pretrained WeatherNext 2 checkpoint.

This follows the official WeatherNext 2 demo's loss/gradient path and adds an
Optax update loop plus checkpoint serialization.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Iterator

import haiku as hk
import jax
import numpy as np
import optax
import pandas as pd
import xarray
import xarray_jax
from google.cloud import storage

from weathernext.utils import checkpoint
from weathernext.utils import data_utils
from weathernext.utils import fiddle_config_io
from weathernext.utils import xarray_tree
from weathernext.weathernext2 import fgn


BUCKET_NAME = "dm_graphcast"
OUTPUT_CHECKPOINT = Path("weather-me-fine_tune_weight.npz")
MODEL_CONFIGS = {
    "WeatherNext2": "WeatherNext2",
    "WeatherNextCyclones": "WeatherNextCyclones",
    "WeatherNextCyclones_Mini": "WeatherNextCyclones_Mini",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Google DeepMind WeatherNext 2 checkpoint."
    )
    parser.add_argument(
        "--data",
        help=(
            "Training NetCDF/Zarr path. Supports a local path or gs:// URI. "
            "If omitted, the official 1-degree sample batch is used."
        ),
    )
    parser.add_argument(
        "--model-name",
        choices=tuple(MODEL_CONFIGS),
        default="WeatherNextCyclones_Mini",
        help="Use Mini for a first run; WeatherNext2 is the full 0.25-degree model.",
    )
    parser.add_argument(
        "--split",
        choices=("2023", "2024", "2025"),
        default="2024",
        help="Checkpoint training cutoff. WeatherNext2 only supports 2025.",
    )
    parser.add_argument(
        "--model-member",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="WeatherNext2/WeatherNextCyclones ensemble checkpoint member.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-gradient", type=float, default=1.0)
    parser.add_argument("--target-lead-time", default="6h")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CHECKPOINT,
        help="Output checkpoint (default: weather-me-fine_tune_weight.npz).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.model_name == "WeatherNext2" and args.split != "2025":
        raise ValueError("WeatherNext2 has public pretrained weights only for split 2025")
    if args.model_name == "WeatherNextCyclones_Mini" and args.split not in {
        "2023",
        "2024",
    }:
        raise ValueError("WeatherNextCyclones_Mini supports splits 2023 and 2024")


def gcs_client() -> storage.Client:
    return storage.Client.create_anonymous_client()


def checkpoint_blob_path(args: argparse.Namespace) -> str:
    if args.model_name == "WeatherNextCyclones_Mini":
        return f"weathernext2/params/{args.model_name}_<{args.split}.npz"
    return (
        f"weathernext2/params/{args.model_name}_<{args.split}_"
        f"model{args.model_member}.npz"
    )


def sample_blob_path() -> str:
    return (
        "weathernext2/dataset/"
        "source-hres_forecast_init-2024-10-07 00:00:00_"
        "res-1.0_levels-13_steps-04.nc"
    )


def load_pretrained_checkpoint(args: argparse.Namespace) -> fgn.CheckPoint:
    bucket = gcs_client().bucket(BUCKET_NAME)
    path = checkpoint_blob_path(args)
    print(f"Loading pretrained checkpoint: gs://{BUCKET_NAME}/{path}")
    with bucket.blob(path).open("rb") as source:
        return checkpoint.load(source, fgn.CheckPoint)


def _open_gcs_dataset(uri: str) -> xarray.Dataset:
    without_scheme = uri.removeprefix("gs://")
    bucket_name, _, blob_path = without_scheme.partition("/")
    if not bucket_name or not blob_path:
        raise ValueError(f"Invalid GCS URI: {uri}")
    blob = gcs_client().bucket(bucket_name).blob(blob_path)
    if blob_path.endswith(".zarr"):
        raise ValueError(
            "Remote Zarr requires a configured fsspec/gcsfs store; mount or "
            "download it locally and pass its path."
        )
    with blob.open("rb") as source:
        return xarray.load_dataset(source)


def load_training_dataset(path: str | None) -> xarray.Dataset:
    if path is None:
        bucket = gcs_client().bucket(BUCKET_NAME)
        blob_path = sample_blob_path()
        print(f"Using official sample data: gs://{BUCKET_NAME}/{blob_path}")
        with bucket.blob(blob_path).open("rb") as source:
            return xarray.load_dataset(source)

    if path.startswith("gs://"):
        return _open_gcs_dataset(path)

    local_path = Path(path).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    if local_path.suffix == ".zarr" or local_path.is_dir():
        return xarray.open_zarr(local_path)
    return xarray.open_dataset(local_path)


def _time_step(time_values: np.ndarray) -> pd.Timedelta:
    if len(time_values) < 2:
        raise ValueError("Training data needs at least two time steps")
    step = pd.Timedelta(time_values[1] - time_values[0])
    if step <= pd.Timedelta(0):
        raise ValueError("Training times must be strictly increasing")
    return step


def _window_size(task_config: Any, target_lead_time: str, step: pd.Timedelta) -> int:
    total = pd.Timedelta(task_config.input_duration) + pd.Timedelta(target_lead_time)
    ratio = total / step
    if not float(ratio).is_integer():
        raise ValueError(
            f"input duration + target lead ({total}) is not divisible by data step ({step})"
        )
    return int(ratio) + 1


def _prepare_relative_time(window: xarray.Dataset) -> xarray.Dataset:
    time_values = window.coords["time"].values
    if np.issubdtype(time_values.dtype, np.datetime64):
        datetimes = time_values
        relative = datetimes - datetimes[-1]
        window = window.assign_coords(
            time=("time", relative),
            datetime=("time", datetimes),
        )
    elif "datetime" not in window.coords:
        raise ValueError(
            "Data with timedelta 'time' must also contain an absolute 'datetime' coordinate"
        )
    if "batch" not in window.dims:
        window = window.expand_dims(batch=[0])
    return window


def training_windows(
    dataset: xarray.Dataset,
    task_config: Any,
    target_lead_time: str,
    steps: int,
) -> Iterator[xarray.Dataset]:
    """Yield deterministic rolling windows, cycling when steps exceed samples."""
    time_values = dataset.coords["time"].values
    step = _time_step(time_values)
    size = _window_size(task_config, target_lead_time, step)
    if dataset.sizes["time"] < size:
        raise ValueError(
            f"Dataset has {dataset.sizes['time']} time steps; at least {size} are required"
        )

    starts = list(range(dataset.sizes["time"] - size + 1))
    for index in range(steps):
        start = starts[index % len(starts)]
        window = dataset.isel(time=slice(start, start + size)).load()
        yield _prepare_relative_time(window)


def configure_accelerator(config: Any) -> None:
    backend = jax.default_backend()
    transformer_kwargs = config.predictor_kwargs["noisy_function_kwargs"][
        "mesh_model_ctor"
    ].keywords["transformer_kwargs"]
    if backend == "gpu":
        transformer_kwargs["attention_type"] = "triblockdiag_mha"
    elif backend == "tpu":
        transformer_kwargs.update(
            {
                "block_q": 128,
                "block_kv": 128,
                "block_kv_compute": 128,
                "block_q_dkv": 128,
                "block_kv_dkv": 128,
                "block_kv_dkv_compute": 128,
            }
        )


def build_loss(config: Any):
    @hk.transform
    def loss_fn(inputs, targets, forcings):
        predictor = fgn.construct_predictor(config)
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda value: xarray_jax.unwrap_data(
                value.mean(), require_jax=True
            ),
            (loss, diagnostics),
        )

    return loss_fn


def extract_example(window: xarray.Dataset, task_config: Any, lead_time: str):
    return data_utils.extract_inputs_targets_forcings(
        window,
        target_lead_times=slice(lead_time, lead_time),
        **dataclasses.asdict(task_config),
    )


def save_checkpoint(params: Any, source: fgn.CheckPoint, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    value = fgn.CheckPoint(
        params=jax.device_get(params),
        description=(
            "WeatherNext 2 fine-tuned checkpoint. Base checkpoint: "
            f"{source.description}"
        ),
        license=source.license,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        checkpoint.dump(destination, value)
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    validate_args(args)
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.local_devices()}")

    config_name = f"weathernext2/configs/{MODEL_CONFIGS[args.model_name]}"
    config = fiddle_config_io.get_fiddle_config_by_name(config_name)
    configure_accelerator(config)

    pretrained = load_pretrained_checkpoint(args)
    dataset = load_training_dataset(args.data)
    windows = training_windows(
        dataset, config.task, args.target_lead_time, args.steps
    )

    loss_fn = build_loss(config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.clip_gradient),
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
    )
    params = pretrained.params
    optimizer_state = optimizer.init(params)

    def update(params, optimizer_state, rng, inputs, targets, forcings):
        def objective(current_params):
            return loss_fn.apply(
                current_params, rng, inputs, targets, forcings
            )

        (loss, diagnostics), grads = jax.value_and_grad(
            objective, has_aux=True
        )(params)
        updates, optimizer_state = optimizer.update(
            grads, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        return params, optimizer_state, loss, diagnostics, grad_norm

    update_jitted = jax.jit(update)
    rng = jax.random.PRNGKey(args.seed)
    history: list[dict[str, float | int]] = []

    for step_index, window in enumerate(windows, start=1):
        inputs, targets, forcings = extract_example(
            window, config.task, args.target_lead_time
        )
        rng, step_rng = jax.random.split(rng)
        params, optimizer_state, loss, _, grad_norm = update_jitted(
            params,
            optimizer_state,
            step_rng,
            inputs,
            targets,
            forcings,
        )
        loss_value = float(jax.device_get(loss))
        grad_value = float(jax.device_get(grad_norm))
        history.append(
            {"step": step_index, "loss": loss_value, "grad_norm": grad_value}
        )
        print(
            f"step={step_index:04d}/{args.steps:04d} "
            f"loss={loss_value:.6f} grad_norm={grad_value:.6f}"
        )

    save_checkpoint(params, pretrained, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint_format": "weathernext.weathernext2.fgn.CheckPoint",
                "checkpoint_kind": "fine_tuned",
                "weathernext_release": "v0.3.0",
                "model_name": args.model_name,
                "split": args.split,
                "model_member": args.model_member,
                "fine_tune_steps": args.steps,
                "target_lead_time": args.target_lead_time,
                "inference_ready": True,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved fine-tuned weights: {args.output}")
    print(f"Saved training metrics: {metrics_path}")
    print(f"Saved checkpoint metadata: {metadata_path}")


if __name__ == "__main__":
    main()
