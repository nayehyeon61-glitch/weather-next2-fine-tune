# WeatherNext 2 Fine-tuning

Google DeepMind의 공개 WeatherNext 2 체크포인트를 HRES/ERA5 형식 데이터로
파인튜닝하는 실행 프로젝트입니다. 공식 `weathernext` v0.3.0의
`predictor.loss(...)`와 체크포인트 직렬화 형식을 그대로 사용합니다.

학습이 끝나면 기본적으로 다음 두 파일을 만듭니다.

- `weather-me-fine_tune_weight.npz`: 파인튜닝된 모델 가중치
- `weather-me-fine_tune_weight.metrics.json`: step별 loss와 gradient norm

가중치 파일은 크기가 크므로 Git에는 올라가지 않도록 `.gitignore`에 포함되어
있습니다.

## 1. 설치

Python 3.10 또는 3.11 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TPU를 사용할 때는 실행 환경에 맞는 JAX TPU 패키지를 추가로 설치하십시오.
GPU에서는 스크립트가 공식 데모와 동일하게 attention 구현을
`triblockdiag_mha`로 변경합니다.

## 2. 먼저 Mini 모델로 실행 확인

아래 명령은 공개된 1° 샘플 데이터와 Mini 체크포인트를 자동으로 내려받아
한 번 업데이트합니다.

```bash
python fine_tune.py --steps 1
```

> 첫 실행은 JAX 컴파일 때문에 오래 걸립니다. 공식 문서상 Mini 추론은 P100
> 수준에서도 가능하지만 gradient 계산은 더 많은 메모리를 요구합니다.

## 3. WeatherNext 2 전체 모델 파인튜닝

0.25° WeatherNext 2는 공개 체크포인트 네 개로 구성됩니다. 한 실행에서 한
member를 파인튜닝하며, member별 output 경로를 별도로 지정할 수 있습니다.

```bash
python fine_tune.py \
  --model-name WeatherNext2 \
  --split 2025 \
  --model-member 1 \
  --data /path/to/hres_or_era5_training.zarr \
  --steps 1000 \
  --learning-rate 1e-6 \
  --output weather-me-fine_tune_weight.npz
```

공식 안내상 0.25° 모델은 추론에도 H100급 메모리가 필요하므로 full-gradient
파인튜닝은 더 큰 accelerator 메모리가 필요할 수 있습니다. 메모리가 부족하면
Mini 모델로 파이프라인을 먼저 검증하고, `--steps 1`로 전체 모델의 메모리 사용량을
확인하십시오.

## 4. 학습 데이터 규격

`--data`는 로컬 NetCDF(`.nc`), 로컬 Zarr(`.zarr`), 또는 `gs://.../*.nc`를
받습니다. WeatherBench2의 ERA5 또는 IFS HRES t=0 analysis처럼 WeatherNext 2
task config가 요구하는 변수, pressure level, 위도·경도 격자를 포함해야 합니다.

필수 시간 구조는 다음 중 하나입니다.

- `time`이 절대 `datetime64`인 연속 시계열
- `time`이 `timedelta64`이고 별도의 절대 `datetime` 좌표가 있는 데이터

스크립트는 config의 `input_duration`과 `--target-lead-time`을 이용해 rolling
window를 만들며, 기본값은 6시간 one-step autoregressive 학습입니다.

## 5. 출력 체크포인트 다시 불러오기

저장 형식은 공식 `fgn.CheckPoint` 형식입니다.

```python
from weathernext.utils import checkpoint
from weathernext.weathernext2 import fgn

with open("weather-me-fine_tune_weight.npz", "rb") as f:
    fine_tuned = checkpoint.load(f, fgn.CheckPoint)

params = fine_tuned.params
```

## 주요 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--model-name` | `WeatherNextCyclones_Mini` | Mini, Cyclones, WeatherNext2 선택 |
| `--split` | `2024` | 공개 체크포인트 cutoff |
| `--model-member` | `1` | 전체 모델 ensemble member |
| `--steps` | `10` | optimizer update 횟수 |
| `--learning-rate` | `1e-6` | AdamW learning rate |
| `--target-lead-time` | `6h` | one-step target lead time |
| `--output` | `weather-me-fine_tune_weight.npz` | 저장할 가중치 파일 |

## 참고

- [Google DeepMind WeatherNext 저장소](https://github.com/google-deepmind/weathernext)
- [WeatherNext 2 공식 안내](https://developers.google.com/weathernext)
- [WeatherBench2 데이터 가이드](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)

WeatherNext는 실험적 연구 모델이며 공식 기상 특보를 대체하지 않습니다. 데이터와
가중치의 개별 라이선스 및 이용 조건도 함께 확인하십시오.
