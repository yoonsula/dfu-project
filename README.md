# DFU Foot Analysis Pipeline

당뇨발(DFU) 이미지 분석 파이프라인입니다. **발/궤양 세그멘테이션**과 **DFU binary 분류**를 하나의 프로젝트에서 실행할 수 있습니다.

> **재학습(train.py)** 은 별도 학습 데이터가 필요합니다.

## Quick Start

```bash
# 1) 환경 (Python 3.10+, CUDA 권장)
conda create -n dfu-venv python=3.11 -y
conda activate dfu-venv
pip install -r requirements.txt

# 2) 에셋 확인
python scripts/verify_setup.py

# 3) 통합 inference (발 탐지 → 궤양 → DFU 분류)
python infer.py \
  --foot-head-checkpoint checkpoints/foot_head_v1/best.pt \
  --wound-head-checkpoint checkpoints/wound_head_v1/best.pt \
  --dfu-head-checkpoint checkpoints/dfu_head_v1/best.pt \
  --image /path/to/image.jpg \
  --image-size 384 \
  --device cuda
```

브라우저 UI:

```bash
python app_gradio.py \
  --foot-head-checkpoint checkpoints/foot_head_v1/best.pt \
  --wound-head-checkpoint checkpoints/wound_head_v1/best.pt \
  --dfu-head-checkpoint checkpoints/dfu_head_v1/best.pt \
  --image-size 384 \
  --device cuda
# http://127.0.0.1:7861
```

## Pipeline

```
Input Image
    │
    ▼
[1] Foot + Wound Segmentation (DFUPipelineModel)
    │   DINOv3 ViT-S/16 + FastInst-style heads
    │
    ├── foot 미탐지 → DFU 분류 스킵
    │
    └── foot 탐지됨
            │
            ├── [2] Wound: foot 중앙 정렬까지 만족 시 활성화
            │
            └── [3] DFU Classification (DFUFeatureClassifierHead)
                    ├── dfu
                    └── other
```

| 단계 | 모델 | 게이트 조건 |
|------|------|-------------|
| Foot mask | foot head checkpoint | foot area ratio ∈ [0.08, 0.5] |
| Wound mask | wound head checkpoint | foot 탐지 + 화면 중앙 ±0.25 |
| DFU 분류 | `--dfu-head-checkpoint` (shared backbone feature) | **foot 탐지 시** |

## Bundled Assets

| 경로 | 용도 | 크기(약) |
|------|------|----------|
| `assets/dinov3/` | Meta DINOv3 repo + ViT-S backbone (세그멘테이션 + 분류 feature) | 85 MB |
| `checkpoints/{run_name}/` | 학습된 head checkpoint (`best.pt`, `last.pt`) | task별 상이 |

외부 경로(`../dinov3`, `../dfu-classification` 등) 없이 **프로젝트 내부 경로만** 사용합니다.  
경로 변경이 필요하면 `.env.example` → `.env` 복사 후 수정하거나 `paths.py`의 환경 변수를 사용하세요.

## Project Structure

```
dfu-project/
├── assets/
│   └── dinov3/                         # Meta DINOv3 (shared backbone)
│       ├── dinov3/
│       ├── hubconf.py
│       └── checkpoint/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
├── checkpoints/
│   └── {run_name}/                     # train.py default output (best.pt, last.pt, train_log.json)
├── models/
│   ├── backbone.py                     # DINOv3 feature extractor
│   ├── dfu_feature_head.py             # shared-feature DFU classifier head
│   ├── fastinst_head.py                # shared segmentation head block
│   ├── foot_head.py / wound_head.py
│   └── pipeline_model.py               # inference assembly (backbone + heads)
├── datasets/
│   ├── catalog.py                      # training data source list
│   ├── source_loaders.py               # COCO / FUSeg / Wound loaders
│   ├── samples.py                      # SegmentationSample
│   └── diabetic_foot_dataset.py
├── data/loaders.py                     # DataLoader factory
├── cli/dataset_args.py                 # shared dataset CLI flags
├── inference/
│   ├── checkpoints.py                  # pipeline / head checkpoint loading
│   ├── classification.py               # DFU head classification
│   └── pipeline.py                     # staged foot → wound inference
├── trainers/
│   ├── common.py                       # shared CLI + training utilities
│   ├── losses.py                       # segmentation loss / metrics
│   ├── training_log.py                 # run logging helpers
│   ├── segmentation.py                 # foot/wound segmentation loop
│   ├── foot_trainer.py
│   ├── wound_trainer.py
│   └── dfu_trainer.py
├── scripts/
│   ├── verify_setup.py                 # 에셋 존재 확인
│   ├── prepare_dfu_classification_data.py
│   └── export_augmented_foot_coco.py
├── utils/                              # runtime / image helpers
├── infer.py                            # 통합 inference CLI
├── app_gradio.py                       # Gradio + FastRTC UI
├── train.py                            # `--task` dispatcher
├── paths.py                            # 경로 기본값
└── requirements.txt
```

## Inference

### 통합 (권장)

```bash
python infer.py \
  --foot-head-checkpoint checkpoints/foot_head_v1/best.pt \
  --wound-head-checkpoint checkpoints/wound_head_v1/best.pt \
  --dfu-head-checkpoint checkpoints/dfu_head_v1/best.pt \
  --image /path/to/image_or_dir \
  --image-size 384 \
  --device cuda
```

출력 (`output/inference/`):

- `{name}_foot_mask.png`, `{name}_wound_mask.png`, `{name}_overlay.png`
- `{name}.json` — 타이밍, gate 상태, **분류 결과** 포함

Wound head는 기본적으로 탐지된 foot mask의 bbox에 `--wound-crop-margin 0.1` 만큼 여유를 둔 feature crop만 사용합니다. Overlay에는 이 crop 영역이 노란 bbox로 표시되고, JSON에는 원본 이미지 좌표의 `wound_crop_bbox: [xmin, ymin, xmax, ymax]`가 기록됩니다. 전체 feature map에서 wound를 돌리고 싶으면 `--no-wound-feature-crop`을 추가하세요.

촬영 거리/중앙 정렬 guide 없이 모든 후속 단계를 계속 실행하려면 `--no-guide`를 추가하세요. 이 옵션은 guide 문구와 guide gate만 끄며, foot mask가 존재하면 wound head, feature crop bbox 시각화, DFU classification은 계속 실행됩니다.

분류 비활성화:

```bash
python infer.py ... --no-classification
```

### Shared backbone + separate head checkpoints

Backbone을 freeze하고 task별 환경에서 head만 따로 학습한 경우에는 foot/wound head checkpoint를 함께 지정합니다. 이 경로는 inference에서 DINOv3 backbone을 한 번만 실행한 뒤 같은 feature map을 foot, wound, DFU classification head에 공유합니다.

```bash
python infer.py \
  --foot-head-checkpoint checkpoints/foot_head_v1/best.pt \
  --wound-head-checkpoint checkpoints/wound_head_v1/best.pt \
  --dfu-head-checkpoint checkpoints/dfu_head_v1/best.pt \
  --image /path/to/image_or_dir \
  --image-size 384 \
  --device cuda
```

## Gradio + FastRTC (Realtime UI)

웹캠은 **FastRTC** (`WebRTC` 컴포넌트)로 스트리밍합니다.

```bash
pip install -r requirements.txt

python app_gradio.py \
  --foot-head-checkpoint checkpoints/foot_head_v1/best.pt \
  --wound-head-checkpoint checkpoints/wound_head_v1/best.pt \
  --dfu-head-checkpoint checkpoints/dfu_head_v1/best.pt \
  --display-max-size 512 \
  --device cuda \
  --amp
```

| 탭 | 왼쪽 | 오른쪽 |
|----|------|--------|
| **이미지** | Input + Overlay + Run | 결과 JSON (segmentation + classification) |
| **실시간** | WebRTC 오버레이 | JSON (0.5s 갱신, classification 포함) |

`--dfu-head-checkpoint`를 지정하면 segmentation backbone feature를 공유해 `dfu` / `other` 분류 결과가 JSON에 함께 기록됩니다. `--no-classification`으로 끌 수 있습니다.

## Training (Optional)

재학습 시 학습 데이터를 별도로 준비해야 합니다. 기본 데이터 경로는 `../../03_데이터/`이며 `.env` 또는 환경 변수로 변경 가능합니다.

```bash
python train.py \
  --task foot \
  --epochs 20 \
  --image-size 384 \
  --batch-size 64 \
  --amp \
  --foot-augment
```

`train.py`는 `--task`에 따라 `trainers/` 아래 trainer로 위임합니다. 학습은 task마다 **frozen backbone + head 하나**만 사용하고, 추론에서 `DFUPipelineModel`이 backbone과 head checkpoint를 조합합니다.

```bash
python train.py --task foot ...
python train.py --task wound ...
python train.py --task dfu ...

# 또는 trainer 직접 실행
python -m trainers.foot_trainer ...
python -m trainers.wound_trainer ...
python -m trainers.dfu_trainer ...
```

`train.py`는 cache 없이 이미지에서 frozen DINOv3 backbone을 실행한 뒤 선택한 task head만 학습합니다. `--task foot` 또는 `--task wound`를 각각 실행해 독립 head checkpoint를 만들 수 있습니다.

| 데이터 | 기본 경로 |
|--------|-----------|
| Foot (Roboflow) | `../../03_데이터/roboflow-foot` |
| Foot (DFU SAM3) | `../../03_데이터/dfu-foot-sam3-filtered/train` |
| Body hard negatives | `../../03_데이터/roboflow-body` |
| Human body hard negatives | `../../03_데이터/roboflow-humanbody` |
| Wound (FUSeg) | `../../03_데이터/wound-segmentation/data/Foot Ulcer Segmentation Challenge` |
| Wound Image Dataset | `../../03_데이터/Wound Image Dataset` |

### Dataset Catalog

학습 데이터는 `datasets/catalog.py`에서 한눈에 볼 수 있게 관리합니다.

- **Foot detection**: `roboflow-foot`, `dfu-foot-sam3-filtered/train`, `roboflow-body`, `roboflow-humanbody`
- **Wound detection**: `Foot Ulcer Segmentation Challenge`, `Wound Image Dataset`

새 COCO foot dataset을 기본 학습에 추가하려면 `datasets/catalog.py`의 foot source 목록에 한 줄을 추가하면 됩니다. 실험용으로만 추가할 때는 `--foot-root`를 반복해서 넘길 수 있습니다.

```bash
python train.py \
  --task foot \
  --foot-root ../../03_데이터/roboflow-foot \
  --foot-root ../../03_데이터/dfu-foot-sam3-filtered/train \
  --image-size 384 --batch-size 64 --amp
```

## Head-Only Training With Shared Frozen Backbone

세 task의 데이터셋과 학습 환경이 서로 다르면, backbone은 고정하고 각 head를 따로 학습하는 방식을 권장합니다. `train.py`는 cache 없이 이미지에서 feature를 즉시 추출해 선택한 segmentation head만 학습합니다.

```text
각 task dataset image
  -> 같은 frozen DINOv3Backbone
  -> task별 head만 학습
  -> foot_head.pt / wound_head.pt / dfu_head.pt
```

```bash
# Foot head
python train.py --task foot --image-size 384 --batch-size 64 --amp --device cuda

# Wound head
python train.py --task wound --image-size 384 --batch-size 64 --amp --device cuda
```

중요한 조건은 task별 환경이 달라도 아래 설정은 같아야 한다는 점입니다.

- 같은 DINOv3 모델: `dinov3_vits16`
- 같은 backbone checkpoint: `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- 같은 `--image-size`
- 같은 preprocessing/normalization
- 같은 feature 형태: `[B, 384, H/16, W/16]`

### 1. DFU classification dataset format

DFU classification 데이터는 `dfu` vs `other`의 binary ImageFolder 형식을 권장합니다. 원본 `DFU Dataset`과 `dfu_partA_20260617`은 아래 스크립트로 하나의 학습 폴더로 재생성합니다.

```bash
python scripts/prepare_dfu_classification_data.py --overwrite
```

기본 출력:

```text
../../03_데이터/dfu_classification_data/
  train/
    dfu/
    other/
  val/
    dfu/
    other/
  test/
    dfu/
    other/
```

Label mapping:

```text
DFU Dataset/*/Diabetic Foot Ulcer -> dfu
DFU Dataset/*/Healthy             -> other
DFU Dataset/*/Wound               -> other
dfu_partA_20260617/dfu            -> dfu
dfu_partA_20260617/others         -> other
```

`dfu_partA_20260617`은 파일명 앞의 patient/group id 기준으로 train/val/test에 나눕니다. 생성된 `manifest.csv`에는 원본 경로, target 경로, split, label, group id가 기록됩니다. 디스크 사용량을 줄이고 싶으면 `--mode hardlink`를 사용할 수 있습니다.

### 2. Segmentation head 학습

`train.py`는 저장된 cache 없이 이미지를 읽고 frozen backbone을 통과시킨 뒤 선택한 segmentation head만 학습합니다.

```bash
python train.py \
  --task foot \
  --output-dir checkpoints \
  --run-name foot_head_v1 \
  --epochs 30 \
  --batch-size 64 \
  --device cuda \
  --amp

python train.py \
  --task wound \
  --output-dir checkpoints \
  --run-name wound_head_v1 \
  --epochs 30 \
  --batch-size 64 \
  --device cuda \
  --amp
```

각 run은 `best.pt`와 `last.pt`를 저장합니다.

```text
checkpoints/foot_head_v1/best.pt
checkpoints/wound_head_v1/best.pt
```

이 두 checkpoint는 `infer.py`에서 `--foot-head-checkpoint`, `--wound-head-checkpoint`로 함께 로드합니다.

### 3. DFU classification head 학습

`train.py --task dfu`는 `dfu_classification_data/`를 읽고, foot/wound와 **같은 frozen DINOv3 backbone feature map** 위에서 DFU classification head만 학습합니다. inference에서는 backbone을 한 번만 실행한 뒤 foot, wound, dfu head가 같은 feature를 공유합니다.

노트북(`01_classification.ipynb`)에서 성능이 좋았던 설정을 참고해 기본값을 맞춰 두었습니다.

- `--dfu-head-type linear`: notebook과 같이 단순 linear head
- `--dfu-lr 5e-3`, `--dfu-batch-size 32`, `--epochs 10`
- `--class-weight none`
- `--warmup-ratio 0.1` + cosine scheduler
- `--dfu-best-metric f1`
- `--image-size 384`: foot/wound checkpoint와 동일하게 유지

```bash
python train.py \
  --task dfu \
  --dfu-root ../../03_데이터/dfu_classification_data \
  --output-dir checkpoints \
  --run-name dfu_head_v1 \
  --image-size 384 \
  --epochs 10 \
  --device cuda \
  --amp
```

기본 클래스는 `dfu`, `other`입니다. MLP head나 class weight를 쓰고 싶으면 `--dfu-head-type mlp --class-weight balanced`를 추가하세요.

학습 결과:

```text
checkpoints/dfu_head_v1/best.pt
checkpoints/dfu_head_v1/last.pt
checkpoints/dfu_head_v1/train_log.json
```

저장된 `best.pt`는 `infer.py`의 `--dfu-head-checkpoint`에 바로 사용할 수 있습니다.

```bash
python infer.py \
  --foot-head-checkpoint checkpoints/foot_head_v1/best.pt \
  --wound-head-checkpoint checkpoints/wound_head_v1/best.pt \
  --dfu-head-checkpoint checkpoints/dfu_head_v1/best.pt \
  --image /path/to/image_or_dir \
  --image-size 384 \
  --device cuda
```

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
    └── FastInstWoundHead (num_queries=16) → wound mask
```

### Classification

```
Input Image → DINOv3 ViT-S/16 (HF, frozen) → CLS token → Linear(384→3)
```

클래스: `TS6_normal skin`, `diabetic ulcer`, `other_injury`  
분류 val accuracy: **95.1%** (checkpoint epoch 10)

## Inference Gate Logic

**Wound** (기존):

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
| `DINOv3 repo not found` | `python scripts/verify_setup.py` 실행, `assets/dinov3/` 존재 확인 |
| `DFU head checkpoint not found` | `--dfu-head-checkpoint` 경로 확인 또는 `--no-classification` 사용 |
| OOM (WSL) | `--num-workers 0`, `--image-size 384` 유지 |
| inference 속도 | 학습 해상도와 `--image-size` 일치 (384) |
