# DFU Foot Analysis Pipeline

당뇨발(DFU) 이미지 분석 파이프라인입니다. **발/궤양 세그멘테이션**과 **DFU binary 분류**(`dfu` vs `other`)를 하나의 shared DINOv3 backbone 위에서 실행합니다.

> **재학습(`train.py`)** 은 별도 학습 데이터가 필요합니다.

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
  --device cuda
```

`--image-size`를 생략하면 foot head checkpoint의 `args.image_size`를 읽고, 없으면 **384**를 사용합니다.

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
[1] DINOv3 ViT-S/16 (frozen, 1회 실행)
    │   feature map [B, 384, H/16, W/16]
    │
    ├── FastInstFootHead  → foot mask
    ├── FastInstWoundHead → wound mask (foot crop feature, gate 통과 시)
    └── DFUFeatureClassifierHead → dfu / other (foot 탐지 시)
```

| 단계 | 모델 | 게이트 조건 |
|------|------|-------------|
| Foot mask | `--foot-head-checkpoint` | foot area ratio ∈ [0.08, 0.5] |
| Wound mask | `--wound-head-checkpoint` | foot 탐지 + (guide 켜짐 시) 화면 중앙 ±0.25 |
| DFU 분류 | `--dfu-head-checkpoint` | **foot 탐지 시** (중앙 정렬 조건 없음) |

`--no-guide`를 쓰면 촬영 가이드 문구와 중앙 정렬 gate만 꺼지고, foot이 탐지되면 wound·분류는 계속 실행됩니다.

## Architecture

학습과 추론은 **backbone 1개 + task별 head checkpoint** 구조입니다.

- **학습**: `train.py --task {foot,wound,dfu}` — frozen `DINOv3Backbone` 위에서 해당 head만 학습
- **추론**: `DFUPipelineModel` — backbone 1회 실행 후 foot / wound / dfu head가 같은 feature map 공유

### Segmentation heads

```
Input [3, H, W]
  → DINOv3Backbone (ViT-S/16, 384-dim)
  → FastInstFootHead  (num_queries=8)  → foot mask
  → FastInstWoundHead (num_queries=16) → wound mask
```

### DFU classification head

```
feature map [B, 384, H/16, W/16]
  → spatial mean pooling
  → Linear(384→2) 또는 MLP
  → dfu / other
```

`DFUFeatureClassifierHead`는 `--head-type linear|mlp`로 학습하며, checkpoint의 `head_state_dict`를 `infer.py --dfu-head-checkpoint`로 로드합니다.

참고 run (`dfu_head_v1`, linear, image_size=384): val accuracy **98.8%**, val F1 **97.8%** (epoch 16).

## Configuration

경로 기본값은 `paths.py`에 있고, `.env.example` → `.env` 복사 후 환경 변수로 덮어쓸 수 있습니다.

| 환경 변수 | 기본값 | 용도 |
|-----------|--------|------|
| `DINOV3_MODEL_PATH` | `assets/dinov3-hf` | 로컬 HF DINOv3 ViT-S/16 스냅샷 (`config.json` + weights) |
| `DFU_CHECKPOINT_DIR` | `checkpoints/` | 학습 checkpoint 루트 |
| `DFU_TRAIN_OUTPUT_DIR` | `checkpoints/` | `train.py` 기본 `--output-dir` |
| `DFU_INFERENCE_OUTPUT_DIR` | `output/inference/` | `infer.py` 출력 |
| `DFU_DATA_ROOT` | `../../03_데이터/` | 학습 데이터 루트 |

## Bundled Assets

| 경로 | 용도 |
|------|------|
| `assets/dinov3-hf/` | 로컬 DINOv3 backbone HF 스냅샷 (`DINOV3_MODEL_PATH`) |
| `checkpoints/{run_name}/` | 학습된 head (`best.pt`, `last.pt`, `train_log.json`) |
| `output/inference/` | inference 마스크·overlay·JSON |

## Project Structure

```
dfu-project/
├── assets/dinov3-hf/                   # local HF backbone snapshot (DINOV3_MODEL_PATH)
├── checkpoints/{run_name}/             # train output (default)
├── output/inference/                   # infer.py output
├── models/
│   ├── backbone.py
│   ├── dfu_feature_head.py
│   ├── foot_head.py / wound_head.py
│   ├── fastinst_head.py
│   └── pipeline_model.py               # DFUPipelineModel
├── datasets/
│   ├── catalog.py
│   ├── classification_dataset.py
│   ├── source_loaders.py
│   ├── samples.py
│   └── diabetic_foot_dataset.py
├── data/loaders.py
├── cli/dataset_args.py
├── inference/
│   ├── checkpoints.py                  # head / pipeline loading
│   ├── classification.py               # DFU head inference
│   └── pipeline.py                     # gated foot → wound
├── trainers/
│   ├── common.py / losses.py / training_log.py
│   ├── segmentation.py                 # foot & wound loop
│   ├── foot_trainer.py / wound_trainer.py / dfu_trainer.py
├── scripts/
│   ├── verify_setup.py
│   ├── prepare_dfu_classification_data.py
│   ├── export_augmented_foot_coco.py
│   └── evaluate.py                     # test set evaluation
├── eval/
│   ├── sample_loaders.py
│   └── runners.py
├── example/backbone_features.ipynb
├── infer.py                            # CLI inference
├── app_gradio.py                       # Gradio + FastRTC UI
├── train.py                            # --task dispatcher
├── paths.py
└── requirements.txt
```

## Inference

```bash
python infer.py \
  --foot-head-checkpoint checkpoints/foot_head_v1/best.pt \
  --wound-head-checkpoint checkpoints/wound_head_v1/best.pt \
  --dfu-head-checkpoint checkpoints/dfu_head_v1/best.pt \
  --image /path/to/image_or_dir \
  --device cuda
```

**출력** (`output/inference/`):

- `{name}_foot_mask.png`, `{name}_wound_mask.png`, `{name}_overlay.png`
- `{name}.json` — 타이밍, gate 상태, 분류 결과

**주요 옵션**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--wound-crop-margin` | `0.15` | foot bbox 주변 feature crop 여유 |
| `--no-wound-feature-crop` | off | wound head를 전체 feature map에 적용 |
| `--no-guide` | off | 촬영 가이드·중앙 정렬 gate 비활성화 |
| `--no-classification` | off | DFU 분류 스킵 |
| `--image-size` | checkpoint에서 추론 | foot/wound/dfu 학습 해상도와 일치 권장 (384) |

Wound head는 기본적으로 foot mask bbox를 `--wound-crop-margin`만큼 확장한 feature crop만 사용합니다. overlay에는 노란 bbox가 표시되고 JSON에 `wound_crop_bbox: [xmin, ymin, xmax, ymax]`가 기록됩니다.

## Gradio + FastRTC (Realtime UI)

```bash
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
| **이미지** | Input + Overlay + Run | 결과 JSON |
| **실시간** | WebRTC 오버레이 | JSON (0.5s 갱신) |

`app_gradio.py`의 `--wound-crop-margin` 기본값은 **0.1**입니다 (`infer.py`는 0.15).

## Training

재학습 시 데이터는 `../../03_데이터/` 기본 경로 또는 `.env`로 지정합니다.

`train.py`는 `--task`에 따라 trainer로 위임합니다. 각 task는 **frozen backbone + head 1개**만 학습하고, 추론 시 `DFUPipelineModel`이 checkpoint를 조합합니다.

```bash
python train.py --task foot ...
python train.py --task wound ...
python train.py --task dfu ...

# 또는 trainer 직접 실행
python -m trainers.foot_trainer ...
python -m trainers.wound_trainer ...
python -m trainers.dfu_trainer ...
```

**공통 기본값** (`trainers/common.py`): foot/wound는 `--image-size 384`, `--batch-size 32`, `--lr 5e-4`.  
`--run-name`을 생략하면 `checkpoints/{timestamp}/`에 저장됩니다.

> foot / wound / dfu head는 inference에서 같은 backbone feature를 공유하므로, **세 task 모두 동일한 `--image-size`(권장 384)와 `DINOV3_MODEL_PATH`**를 사용하세요.

### 학습 데이터 경로

| 데이터 | 기본 경로 |
|--------|-----------|
| Foot (Roboflow) | `../../03_데이터/roboflow-foot` |
| Foot (DFU SAM3) | `../../03_데이터/dfu-foot-sam3-filtered/train` |
| Body hard negatives | `../../03_데이터/roboflow-body` |
| Human body hard negatives | `../../03_데이터/roboflow-humanbody` |
| Wound (FUSeg) | `../../03_데이터/wound-segmentation/data/Foot Ulcer Segmentation Challenge` |
| Wound Image Dataset | `../../03_데이터/Wound Image Dataset` |
| DFU classification | `../../03_데이터/dfu_classification_data` |

`datasets/catalog.py`에서 foot / wound source 목록을 관리합니다. 실험용 COCO root만 추가할 때는 `--foot-root`를 반복 지정할 수 있습니다.

### Foot / Wound segmentation

```bash
python train.py \
  --task foot \
  --run-name foot_head_v1 \
  --image-size 384 \
  --epochs 30 \
  --batch-size 64 \
  --amp \
  --device cuda

python train.py \
  --task wound \
  --run-name wound_head_v1 \
  --image-size 384 \
  --epochs 30 \
  --batch-size 64 \
  --amp \
  --device cuda
```

출력 예:

```text
checkpoints/foot_head_v1/best.pt
checkpoints/wound_head_v1/best.pt
```

checkpoint payload: `head_state_dict`, `args`, `metrics`, `epoch`.

### DFU classification 데이터 준비

`dfu` vs `other` ImageFolder 형식입니다. 원본 `DFU Dataset`과 `dfu_partA_20260617`은 아래 스크립트로 합칩니다.

```bash
python scripts/prepare_dfu_classification_data.py --overwrite
```

```text
../../03_데이터/dfu_classification_data/
  train/{dfu,other}/
  val/{dfu,other}/
  test/{dfu,other}/
```

Label mapping:

```text
DFU Dataset/*/Diabetic Foot Ulcer -> dfu
DFU Dataset/*/Healthy             -> other
DFU Dataset/*/Wound               -> other
dfu_partA_20260617/dfu            -> dfu
dfu_partA_20260617/others         -> other
```

`dfu_partA_20260617`은 patient/group id 기준 train/val/test split. `--mode hardlink`로 디스크 절약 가능.

### DFU classification head 학습

```bash
python train.py \
  --task dfu \
  --dfu-root ../../03_데이터/dfu_classification_data \
  --run-name dfu_head_v1 \
  --image-size 384 \
  --epochs 30 \
  --batch-size 32 \
  --lr 5e-4 \
  --device cuda \
  --amp
```

| 옵션 | dfu 기본값 | 설명 |
|------|------------|------|
| `--head-type` | `linear` | `linear` 또는 `mlp` |
| `--lr` | `5e-3` | head optimizer LR |
| `--batch-size` | `32` | classification batch size |
| `--class-weight` | `none` | `balanced`로 inverse-frequency 가중치 |
| `--best-metric` | `f1` | `best.pt` 선택 기준 (`accuracy` 가능) |
| `--warmup-ratio` | `0.1` | cosine scheduler warmup |

출력:

```text
checkpoints/dfu_head_v1/best.pt
checkpoints/dfu_head_v1/train_log.json
```

`best.pt`는 `infer.py --dfu-head-checkpoint`에 바로 사용합니다.

## Evaluation (Test Set)

`scripts/evaluate.py`로 학습된 head checkpoint의 hold-out 성능을 확인합니다.

| Task | 입력 | 평가 지표 |
|------|------|-----------|
| foot | `--data-root` (COCO) | dice, iou, accuracy |
| wound | `--image-dir` + `--mask-dir` | dice, iou, accuracy |
| dfu | `--data-root` (`dfu/`, `other/` 하위 폴더) | accuracy, precision, recall, f1 |

```bash
# Foot (COCO)
python scripts/evaluate.py \
  --task foot \
  --checkpoint checkpoints/foot_head_v1/best.pt \
  --data-root /path/to/coco-foot-test \
  --device cuda

# Wound (image + mask folders)
python scripts/evaluate.py \
  --task wound \
  --checkpoint checkpoints/wound_head_v1/best.pt \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --device cuda

# DFU classification (ImageFolder root — e.g. test split folder)
python scripts/evaluate.py \
  --task dfu \
  --checkpoint checkpoints/dfu_head_v1/best.pt \
  --data-root ../../03_데이터/dfu_classification_data/test \
  --device cuda
```

`--output-json report.json`으로 전체 리포트를 저장할 수 있습니다.

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

## Inference Gate Logic

**Foot**

- area ratio ∈ [`min_foot_ratio`, `max_foot_ratio`] (기본 0.08 ~ 0.5)

**Wound** (`--no-guide`가 아닐 때)

1. foot 탐지
2. foot center가 화면 중앙 ± `center_tolerance` (기본 0.25) 이내

**DFU Classification**

- `foot_detected=True`일 때만 실행

## Environment

| 항목 | 요구사항 |
|------|----------|
| Python | 3.10+ |
| GPU | CUDA 권장 (CPU 가능) |
| 패키지 | `torch`, `torchvision`, `transformers`, `gradio`, `fastrtc`, `pillow`, `numpy` |

```bash
pip install -r requirements.txt
```

## Troubleshooting

| 증상 | 확인 |
|------|------|
| DINOv3 backbone load 실패 | `python scripts/verify_setup.py`, `assets/dinov3-hf/`에 HF 스냅샷 존재, `transformers>=4.56` |
| `DFU head checkpoint not found` | `--dfu-head-checkpoint` 경로 또는 `--no-classification` |
| head key mismatch warning | 구 checkpoint prefix (`ulcer_head.` 등) — non-strict 로드, 동작 확인 |
| OOM (WSL) | `--num-workers 0`, `--image-size 384` |
| inference 속도 | 학습 해상도와 `--image-size` 일치 (384) |
