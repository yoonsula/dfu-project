# DFU Foot Analysis Pipeline

당뇨발(DFU) 이미지 분석 파이프라인입니다. **발/궤양 세그멘테이션**과 **DFU 3-class 분류**를 하나의 프로젝트에서 실행할 수 있습니다.

> **재학습(train.py)** 은 별도 학습 데이터가 필요합니다.

## Quick Start

```bash
# 1) 환경 (Python 3.10+, CUDA 권장)
conda create -n dfu-venv python=3.11 -y
conda activate dfu-venv
pip install -r requirements.txt

# 2) 에셋 확인
python verify_setup.py

# 3) 통합 inference (발 탐지 → 궤양 → DFU 분류)
python infer.py \
  --checkpoint checkpoints/best.pt \
  --image /path/to/image.jpg \
  --image-size 384 \
  --device cuda
```

브라우저 UI:

```bash
python app_gradio.py --checkpoint checkpoints/best.pt --image-size 384 --device cuda
# http://127.0.0.1:7861
```

## Pipeline

```
Input Image
    │
    ▼
[1] Foot + Ulcer Segmentation (MultiTaskSegModel)
    │   DINOv3 ViT-S/16 + FastInst-style heads
    │
    ├── foot 미탐지 → DFU 분류 스킵
    │
    └── foot 탐지됨
            │
            ├── [2] Ulcer: foot 중앙 정렬까지 만족 시 활성화
            │
            └── [3] DFU Classification (DinoV3LinearClassifier)
                    ├── TS6_normal skin
                    ├── diabetic ulcer
                    └── other_injury
```

| 단계 | 모델 | 게이트 조건 |
|------|------|-------------|
| Foot mask | `checkpoints/best.pt` | foot area ratio ∈ [0.08, 0.5] |
| Ulcer mask | 동일 checkpoint | foot 탐지 + 화면 중앙 ±0.25 |
| DFU 분류 | `checkpoints/dinov3_linear_best_0.001.pt` | **foot 탐지 시** |

## Bundled Assets (~570 MB)

| 경로 | 용도 | 크기(약) |
|------|------|----------|
| `checkpoints/best.pt` | 발/궤양 세그멘테이션 | 120 MB |
| `checkpoints/dinov3_linear_best_0.001.pt` | DFU 3-class 분류 head | 83 MB |
| `assets/dinov3/` | Meta DINOv3 repo + ViT-S backbone (세그멘테이션) | 85 MB |
| `assets/dinov3-hf/` | HuggingFace DINOv3 (분류 backbone) | 165 MB |
| `checkpoints/last.pt` | 학습 재개용 (inference 불필요) | 120 MB |

외부 경로(`../dinov3`, `../dfu-classification` 등) 없이 **프로젝트 내부 경로만** 사용합니다.  
경로 변경이 필요하면 `.env.example` → `.env` 복사 후 수정하거나 `paths.py`의 환경 변수를 사용하세요.

## Project Structure

```
dfu-project/
├── assets/
│   ├── dinov3/                         # Meta DINOv3 (segmentation backbone)
│   │   ├── dinov3/
│   │   ├── hubconf.py
│   │   └── checkpoint/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
│   └── dinov3-hf/
│       └── dinov3-vits16-pretrain-lvd1689m/   # HF DINOv3 (classification)
├── checkpoints/
│   ├── best.pt                         # segmentation (inference)
│   ├── dinov3_linear_best_0.001.pt     # classification (inference)
│   └── last.pt                         # training resume only
├── models/
│   ├── backbone.py                     # DINOv3 feature extractor
│   ├── dfu_classifier.py               # DINOv3 + linear head
│   ├── fastinst_head.py                 # shared segmentation head block
│   ├── foot_head.py / ulcer_head.py
│   └── multitask_model.py
├── datasets/
│   ├── catalog.py                       # training data source list
│   ├── source_loaders.py                # COCO / FUSeg / Wound loaders
│   ├── samples.py                       # SegmentationSample
│   └── diabetic_foot_dataset.py
├── data/loaders.py                      # DataLoader factory
├── cli/dataset_args.py                  # shared dataset CLI flags
├── inference/pipeline.py                # staged foot → ulcer inference
├── utils/                               # runtime / image helpers
├── infer.py                            # 통합 inference (seg + classification)
├── infer_classification.py             # 분류 단독 inference
├── app_gradio.py                       # Gradio + FastRTC UI
├── verify_setup.py                     # 에셋 존재 확인
├── train.py                            # 재학습 (데이터 별도 필요)
├── paths.py                            # 경로 기본값
└── requirements.txt
```

## Inference

### 통합 (권장)

```bash
python infer.py \
  --checkpoint checkpoints/best.pt \
  --image /path/to/image_or_dir \
  --image-size 384 \
  --device cuda
```

출력 (`output/inference/`):

- `{name}_foot_mask.png`, `{name}_ulcer_mask.png`, `{name}_overlay.png`
- `{name}.json` — 타이밍, gate 상태, **분류 결과** 포함

분류 비활성화:

```bash
python infer.py ... --no-classification
```

### 분류만

```bash
python infer_classification.py --image /path/to/image.jpg
```

### Shared backbone + separate head checkpoints

Backbone을 freeze하고 task별 환경에서 head만 따로 학습한 경우에는 checkpoint 3개를 함께 지정합니다. 이 경로는 inference에서 DINOv3 backbone을 한 번만 실행한 뒤 같은 feature map을 foot, wound/ulcer, DFU classification head에 공유합니다.

```bash
python infer.py \
  --foot-head-checkpoint output/train/foot_head_v1/best.pt \
  --ulcer-head-checkpoint output/train/wound_head_v1/best.pt \
  --shared-classification-checkpoint output/train/dfu_head_v1/best.pt \
  --image /path/to/image_or_dir \
  --image-size 384 \
  --device cuda
```

## Gradio + FastRTC (Realtime UI)

웹캠은 **FastRTC** (`WebRTC` 컴포넌트)로 스트리밍합니다.

```bash
pip install -r requirements.txt

python app_gradio.py \
  --checkpoint checkpoints/best.pt \
  --display-max-size 512 \
  --device cuda \
  --amp
```

| 탭 | 왼쪽 | 오른쪽 |
|----|------|--------|
| **이미지** | Input + Overlay + Run | 결과 JSON |
| **실시간** | WebRTC 오버레이 | JSON (0.5s 갱신) |

> WebRTC 실시간 파이프라인에 DFU 분류 연동은 별도 작업 예정입니다. 현재 UI는 세그멘테이션(발/궤양) 중심입니다.

## Training (Optional)

재학습 시 학습 데이터를 별도로 준비해야 합니다. 기본 데이터 경로는 `../../03_데이터/`이며 `.env` 또는 환경 변수로 변경 가능합니다.

```bash
python train.py \
  --epochs 20 \
  --image-size 384 \
  --batch-size 64 \
  --amp \
  --foot-augment
```

| 데이터 | 기본 경로 |
|--------|-----------|
| Foot (Roboflow) | `../../03_데이터/roboflow-foot` |
| Foot (DFU SAM3) | `../../03_데이터/dfu-foot-sam3-filtered/train` |
| Body hard negatives | `../../03_데이터/roboflow-body` |
| Human body hard negatives | `../../03_데이터/roboflow-humanbody` |
| Ulcer (FUSeg) | `../../03_데이터/wound-segmentation/data/Foot Ulcer Segmentation Challenge` |
| Wound Image Dataset | `../../03_데이터/Wound Image Dataset` |

### Dataset Catalog

학습 데이터는 `datasets/catalog.py`에서 한눈에 볼 수 있게 관리합니다.

- **Foot detection**: `roboflow-foot`, `dfu-foot-sam3-filtered/train`, `roboflow-body`, `roboflow-humanbody`
- **Ulcer detection**: `Foot Ulcer Segmentation Challenge`, `Wound Image Dataset`

새 COCO foot dataset을 기본 학습에 추가하려면 `datasets/catalog.py`의 foot source 목록에 한 줄을 추가하면 됩니다. 실험용으로만 추가할 때는 `--foot-root`를 반복해서 넘길 수 있습니다.

```bash
python train.py \
  --foot-root ../../03_데이터/roboflow-foot \
  --foot-root ../../03_데이터/dfu-foot-sam3-filtered/train \
  --image-size 384 --batch-size 64 --amp
```

## Head-Only Training With Shared Frozen Backbone

세 task의 데이터셋과 학습 환경이 서로 다르면, backbone은 고정하고 각 head를 따로 학습하는 방식을 권장합니다. 전체 흐름은 다음과 같습니다.

```text
각 task dataset image
  -> 같은 frozen DINOv3Backbone
  -> feature cache 저장
  -> task별 head만 학습
  -> foot_head.pt / wound_head.pt / dfu_head.pt
```

중요한 조건은 task별 환경이 달라도 아래 설정은 같아야 한다는 점입니다.

- 같은 DINOv3 모델: `dinov3_vits16`
- 같은 backbone checkpoint: `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- 같은 `--image-size`
- 같은 preprocessing/normalization
- 같은 feature 형태: `[B, 384, H/16, W/16]`

### 1. DFU classification dataset format

DFU classification 데이터는 ImageFolder 형식을 권장합니다.

```text
dfu_root/
  train/
    TS6_normal skin/
    diabetic ulcer/
    other_injury/
  val/
    TS6_normal skin/
    diabetic ulcer/
    other_injury/
```

`train/val` 폴더가 없으면 class 폴더만 두고 `--val-ratio`로 자동 분할할 수 있습니다.

```text
dfu_root/
  TS6_normal skin/
  diabetic ulcer/
  other_injury/
```

CSV도 사용할 수 있습니다. CSV에는 `image_path`/`path`/`image` 중 하나와 `label`/`class`/`class_name` 중 하나가 필요합니다. `split` 컬럼이 있으면 `train`, `val`을 그대로 사용합니다.

### 2. Feature cache 생성

각 환경에서 자기 task 데이터만 cache하면 됩니다. 단 `--image-size`, DINOv3 repo/checkpoint는 inference에서 쓸 설정과 동일하게 유지합니다.

```bash
# Foot segmentation features
python cache_features.py \
  --task foot \
  --split both \
  --output-dir output/feature_cache/shared_v1 \
  --image-size 384 \
  --batch-size 16 \
  --device cuda

# Wound/ulcer segmentation features
python cache_features.py \
  --task ulcer \
  --split both \
  --output-dir output/feature_cache/shared_v1 \
  --image-size 384 \
  --batch-size 16 \
  --device cuda

# DFU classification features
python cache_features.py \
  --task dfu \
  --split both \
  --dfu-root /path/to/dfu_root \
  --dfu-class "TS6_normal skin" \
  --dfu-class "diabetic ulcer" \
  --dfu-class "other_injury" \
  --output-dir output/feature_cache/shared_v1 \
  --image-size 384 \
  --batch-size 16 \
  --device cuda
```

생성되는 cache 구조:

```text
output/feature_cache/shared_v1/
  foot/train/
  foot/val/
  ulcer/train/
  ulcer/val/
  dfu/train/
  dfu/val/
```

### 3. Head-only 학습

`train_feature_head.py`는 저장된 feature만 읽어서 해당 task head만 학습합니다. Backbone forward와 gradient 계산이 없어서 빠르고, task별 환경에서 독립적으로 실행할 수 있습니다.

```bash
python train_feature_head.py \
  --task foot \
  --cache-dir output/feature_cache/shared_v1 \
  --output-dir output/train \
  --run-name foot_head_v1 \
  --epochs 30 \
  --batch-size 64 \
  --device cuda

python train_feature_head.py \
  --task ulcer \
  --cache-dir output/feature_cache/shared_v1 \
  --output-dir output/train \
  --run-name wound_head_v1 \
  --epochs 30 \
  --batch-size 64 \
  --device cuda

python train_feature_head.py \
  --task dfu \
  --cache-dir output/feature_cache/shared_v1 \
  --output-dir output/train \
  --run-name dfu_head_v1 \
  --epochs 30 \
  --batch-size 64 \
  --device cuda
```

각 run은 `best.pt`와 `last.pt`를 저장합니다.

```text
output/train/foot_head_v1/best.pt
output/train/wound_head_v1/best.pt
output/train/dfu_head_v1/best.pt
```

이 세 checkpoint는 `infer.py`에서 `--foot-head-checkpoint`, `--ulcer-head-checkpoint`, `--shared-classification-checkpoint`로 함께 로드합니다.

## Dataset Acknowledgments

재학습에 사용하는 외부 데이터셋의 저작자를 표기합니다.

### Lower Limb and Feet Wound Image Dataset (Mendeley)

궤양 세그멘테이션 학습에 `Wound Image Dataset` (`wound_main` / `wound_mask` / `Nomal`)을 사용합니다.

- **제목:** Lower Limb and Feet Wound Image Dataset for Medical Analysis
- **저자:** Md Masudul Islam
- **라이선스:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **DOI:** [10.17632/hsj38fwnvr.3](https://doi.org/10.17632/hsj38fwnvr.3)
- **URL:** [https://data.mendeley.com/datasets/hsj38fwnvr/3](https://data.mendeley.com/datasets/hsj38fwnvr/3)

### Roboflow Universe

```bibtex
@misc{ human-parts-bru4g_dataset,
  title = { Human parts Dataset },
  type = { Open Source Dataset },
  author = { Personal },
  howpublished = { \url{ https://universe.roboflow.com/personal-ekd6m/human-parts-bru4g } },
  url = { https://universe.roboflow.com/personal-ekd6m/human-parts-bru4g },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2025 },
  month = { feb },
  note = { visited on 2026-06-16 },
}

@misc{ body-idcrc_dataset,
  title = { body Dataset },
  type = { Open Source Dataset },
  author = { hell },
  howpublished = { \url{ https://universe.roboflow.com/hell-khakz/body-idcrc } },
  url = { https://universe.roboflow.com/hell-khakz/body-idcrc },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2025 },
  month = { apr },
  note = { visited on 2026-06-16 },
}

@misc{ foot-segmentation-dd6qi_dataset,
  title = { Foot Segmentation Dataset },
  type = { Open Source Dataset },
  author = { cv },
  howpublished = { \url{ https://universe.roboflow.com/cv-v6vyj/foot-segmentation-dd6qi } },
  url = { https://universe.roboflow.com/cv-v6vyj/foot-segmentation-dd6qi },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2025 },
  month = { jul },
  note = { visited on 2026-06-16 },
}
```

| 프로젝트 경로 | Roboflow 데이터셋 |
|---------------|-------------------|
| `roboflow-foot` | [Foot Segmentation](https://universe.roboflow.com/cv-v6vyj/foot-segmentation-dd6qi) |
| `roboflow-body` | [body](https://universe.roboflow.com/hell-khakz/body-idcrc) |
| `roboflow-humanbody` | [Human parts](https://universe.roboflow.com/personal-ekd6m/human-parts-bru4g) |

## Architecture

### Segmentation

```
Input Image → DINOv3Backbone (ViT-S/16, 384-dim)
    ├── FastInstFootHead  (num_queries=8)  → foot mask
    └── FastInstUlcerHead (num_queries=16) → ulcer mask
```

### Classification

```
Input Image → DINOv3 ViT-S/16 (HF, frozen) → CLS token → Linear(384→3)
```

클래스: `TS6_normal skin`, `diabetic ulcer`, `other_injury`  
분류 val accuracy: **95.1%** (checkpoint epoch 10)

## Inference Gate Logic

**Ulcer** (기존):

1. Foot area ratio ∈ [`min_foot_ratio`, `max_foot_ratio`] (기본 0.08 ~ 0.5)
2. Foot center가 화면 중앙 ± `center_tolerance` (기본 0.25) 이내

**DFU Classification** (신규):

- `foot_detected=True`일 때만 실행 (중앙 정렬 조건 없음)

## Environment

| 항목 | 요구사항 |
|------|----------|
| Python | 3.10+ |
| GPU | CUDA 권장 (CPU도 가능, 느림) |
| Conda env | `dfu-venv` 등 |

```bash
pip install -r requirements.txt
# torch, torchvision, transformers, gradio, fastrtc, pillow, numpy
```

## Troubleshooting

| 증상 | 확인 |
|------|------|
| `DINOv3 repo not found` | `python verify_setup.py` 실행, `assets/dinov3/` 존재 확인 |
| `Classification checkpoint not found` | `checkpoints/dinov3_linear_best_0.001.pt` 확인 |
| OOM (WSL) | `--num-workers 0`, `--image-size 384` 유지 |
| inference 속도 | 학습 해상도와 `--image-size` 일치 (384) |
