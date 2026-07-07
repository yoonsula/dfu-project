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

`--image-size`를 생략하면 foot head checkpoint의 `args.image_size`를 읽고, 없으면 **512**를 사용합니다.

브라우저 UI:

```bash
python app_gradio.py \
  --foot-head-checkpoint checkpoints/foot_head_v1/best.pt \
  --wound-head-checkpoint checkpoints/wound_head_v1/best.pt \
  --dfu-head-checkpoint checkpoints/dfu_head_v1/best.pt \
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

상세 실험 기록은 [DFU Classification](#dfu-classification) 섹션을 참고하세요.

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
| `--image-size` | checkpoint에서 추론 | foot/wound/dfu 학습 해상도와 일치 권장 (512) |

Wound head는 기본적으로 foot mask bbox를 `--wound-crop-margin`만큼 확장한 feature crop만 사용합니다. overlay에는 노란 bbox가 표시되고 JSON에 `wound_crop_bbox: [xmin, ymin, xmax, ymax]`가 기록됩니다.

## Gradio + FastRTC (Realtime UI)

```bash
python app_gradio.py \
  --foot-head-checkpoint checkpoints/foot_head_v1/best.pt \
  --wound-head-checkpoint checkpoints/wound_head_v1/best.pt \
  --dfu-head-checkpoint checkpoints/dfu_head_v1/best.pt \
  --display-max-size 520 \
  --device cuda \
  --amp
```

| 탭 | 왼쪽 | 오른쪽 |
|----|------|--------|
| **이미지** | Input + Overlay + Run | 결과 JSON |
| **실시간** | WebRTC 오버레이 | JSON (0.5s 갱신) |

`app_gradio.py`의 `--wound-crop-margin` 기본값은 **0.1**입니다 (`infer.py`는 0.15).

## Training

재학습 데이터는 `../../03_데이터/` (또는 `.env`의 `DFU_DATA_ROOT`) 아래에 둡니다.

`train.py --task {foot,wound,dfu}`는 frozen DINOv3 backbone 위에서 **해당 head만** 학습합니다. 추론 시 `DFUPipelineModel`이 checkpoint를 조합합니다.

```bash
python train.py --task foot ...
python train.py --task wound ...
python train.py --task dfu ...

# 또는 trainer 직접 실행
python -m trainers.foot_trainer ...
python -m trainers.wound_trainer ...
python -m trainers.dfu_trainer ...
```

**공통 기본값** (`trainers/common.py`): `--image-size 512`, `--batch-size 32`, `--lr 5e-4`, `--epochs 30`.

> foot / wound / dfu head는 inference에서 **같은 backbone feature**를 공유합니다. 세 task 모두 **동일한 `--image-size`(512)와 `DINOV3_MODEL_PATH`**를 사용하세요.

| 공통 옵션 | 기본값 | 설명 |
|-----------|--------|------|
| `--run-name` | timestamp | `checkpoints/{run_name}/`에 저장 |
| `--dinov3-model` | `assets/dinov3-hf` | frozen backbone |
| `--amp` | off | CUDA mixed precision |
| `--early-stopping-patience` | 7 | val score 개선 없으면 조기 종료 |

checkpoint payload: `head_state_dict`, `args`, `metrics`, `epoch`.

---

### Foot Segmentation

발 영역 binary mask를 예측합니다. `FastInstFootHead` (num_queries=8).

#### 데이터셋

| 소스 | 기본 경로 | 역할 | 형식 |
|------|-----------|------|------|
| Roboflow Foot | `../../03_데이터/roboflow-foot` | positive (primary) | COCO (`_annotations.coco.json`) |
| DFU SAM3 Foot | `../../03_데이터/dfu-foot-sam3-filtered/train` | positive (자동 merge) | COCO |
| Roboflow Body | `../../03_데이터/roboflow-body` | hard negative (category_id=1) | COCO |
| Roboflow Human body | `../../03_데이터/roboflow-humanbody` | hard negative (category_id=5,10) | COCO |

- val split: primary positive `--val-ratio` (기본 10%), negative `--val-negative-ratio` (기본 25%)
- train negative oversample: `--negative-oversample 4`, empty mask loss 가중: `--neg-loss-weight 3.0`
- 추가 COCO root: `--foot-root /path/to/coco` 반복 지정

Foot 학습에는 wound 데이터를 사용하지 않습니다. (`datasets/catalog.py`의 `foot_primary_sources`, `foot_extra_coco_sources`)

#### 학습

```bash
python train.py \
  --task foot \
  --run-name foot_head_v2 \
  --image-size 512 \
  --epochs 30 \
  --batch-size 32 \
  --amp \
  --device cuda
```

#### 평가

```bash
python scripts/evaluate.py \
  --task foot \
  --checkpoint checkpoints/foot_head_v2/best.pt \
  --data-root /path/to/coco-foot-test \
  --device cuda
```

지표: dice, iou, accuracy. `--output-json report.json`으로 리포트 저장 가능.

#### 참고 run

| run | image_size | best val dice | best val iou | checkpoint |
|-----|------------|---------------|--------------|------------|
| `foot_head_v1` | 512 | 0.954 | 0.933 | `checkpoints/foot_head_v1/best.pt` |
| `foot_head_v2` | 512 | 0.959 | 0.935 | `checkpoints/foot_head_v2/best.pt` |

---

### Wound Segmentation

궤양 영역 binary mask를 예측합니다. `FastInstWoundHead` (num_queries=16).  
Inference 시 foot bbox 기준 feature crop 사용 (`--wound-crop-margin`).

#### 데이터셋

| 소스 | 기본 경로 | 역할 | 형식 |
|------|-----------|------|------|
| FUSeg | `../../03_데이터/wound-segmentation/data/Foot Ulcer Segmentation Challenge` | positive | `train\|validation\|test/images` + `labels/` |
| Wound Image Dataset | `../../03_데이터/Wound Image Dataset` | positive + negative | `wound_main/` + `wound_mask/`, `Nomal/` (negative) |

- FUSeg: split별 고정 (train/validation/test)
- Wound Image Dataset: `--val-ratio` (기본 10%)로 train/val 랜덤 split
- Mendeley 데이터 제외: `--no-wound-image`

Wound 학습에는 foot COCO 데이터를 사용하지 않습니다. (`datasets/catalog.py`의 `wound_sources`)

#### 학습

```bash
python train.py \
  --task wound \
  --run-name wound_head_v1 \
  --image-size 512 \
  --epochs 30 \
  --batch-size 64 \
  --amp \
  --device cuda
```

#### 평가

```bash
python scripts/evaluate.py \
  --task wound \
  --checkpoint checkpoints/wound_head_v1/best.pt \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --device cuda
```

지표: dice, iou, accuracy.

#### 참고 run

| run | image_size | best val dice | best val iou | checkpoint |
|-----|------------|---------------|--------------|------------|
| `wound_head_v1` | 512 | 0.814 | 0.738 | `checkpoints/wound_head_v1/best.pt` |

---

### DFU Classification

발 이미지를 **dfu vs other** binary로 분류합니다. `DFUFeatureClassifierHead` (spatial mean pooling → Linear/MLP).

#### 데이터셋

학습에 사용한 원본 3종과 로컬 경로입니다.

| # | 소스 | 로컬 경로 | 전처리 | binary 라벨 |
|---|------|-----------|--------|-------------|
| 1 | **AI Hub** (욕창 데이터, dataSetSn=509) | AI Hub 반출 데이터 | **원본 그대로** 사용 | dfu / other |
| 2 | **Kaggle — DFU Dataset** | `../../03_데이터/DFU Dataset` | **foot segmentation → crop** 후 사용 | dfu / other |
| 3 | **PART A** (2026-06-17 CRC 라벨링) | `../../03_데이터/dfu_partA_20260617` | **foot segmentation → crop** 후 사용 | dfu / other |

> Kaggle `DFU Dataset`과 PART A(`dfu_partA_20260617`)는 `scripts/crop_foot_classification_dataset.py`로 foot head를 돌려 발 영역을 crop한 뒤 분류 학습에 넣었습니다. AI Hub 데이터는 crop 없이 사용합니다.

**출처 상세**

| 소스 | 설명 | URL |
|------|------|-----|
| AI Hub | 욕창 관련 AI Hub 공개 데이터 | [aihub.or.kr (dataSetSn=509)](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%9A%95%EC%B0%BD&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=509) |
| Kaggle DFU Dataset | Srivatsan Mk, MIT License | [kaggle.com/datasets/srivatsanmk2004/dfu-dataset](https://www.kaggle.com/datasets/srivatsanmk2004/dfu-dataset) |
| PART A | 2026-06-17 기준 CRC 라벨링 데이터 | _(내부 수집)_ |

**학습용 ImageFolder 경로**

| 경로 | 설명 |
|------|------|
| `../../03_데이터/dfu_classification_data` | Kaggle + PART A merge 출력 (`prepare_dfu_classification_data.py`) |
| `../../03_데이터/dfu_classification_cropped` | foot crop 적용 후 최종 학습·평가용 (Kaggle + PART A + AI Hub 통합) |

**Label mapping** (Kaggle 3-class → binary)

| 원본 | → | binary |
|------|---|--------|
| `DFU Dataset/*/Diabetic Foot Ulcer` | → | `dfu` |
| `DFU Dataset/*/Healthy` | → | `other` |
| `DFU Dataset/*/Wound` | → | `other` |
| `dfu_partA_20260617/dfu` | → | `dfu` |
| `dfu_partA_20260617/others` | → | `other` |

#### 데이터 준비

`dfu` vs `other` ImageFolder 형식입니다.

**1) Kaggle + PART A merge**

```bash
python scripts/prepare_dfu_classification_data.py --overwrite
# 디스크 절약: --mode hardlink
```

**2) Kaggle + PART A foot crop** (foot head checkpoint 필요)

```bash
python scripts/crop_foot_classification_dataset.py \
  --input-root ../../03_데이터/dfu_classification_data \
  --output-root ../../03_데이터/dfu_classification_cropped \
  --foot-head-checkpoint checkpoints/foot_head_v2/best.pt \
  --device cuda
```

AI Hub 반출 데이터는 crop 없이 `dfu_classification_cropped` ImageFolder에 통합합니다.  
`dfu_partA_20260617`은 patient/group id 기준 train/val/test split.

```text
../../03_데이터/dfu_classification_cropped/
  train/{dfu,other}/
  val/{dfu,other}/
  test/{dfu,other}/
```

#### 학습

```bash
python train.py \
  --task dfu \
  --dfu-root ../../03_데이터/dfu_classification_cropped \
  --run-name dfu_head_v1 \
  --epochs 30 \
  --batch-size 32 \
  --head-type mlp \
  --lr 1e-3 \
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

#### 평가

```bash
python scripts/evaluate.py \
  --task dfu \
  --checkpoint checkpoints/dfu_head_mlp_1e3_aug/best.pt \
  --data-root ../../03_데이터/dfu_classification_cropped/test \
  --device cuda \
  --output-json report.json
```

지표: accuracy, precision, recall, f1. 일괄 비교: `scripts/evaluate_dfu_heads_on_test.py`, 오분류 리포트: `scripts/report_dfu_test_errors.py`.

#### 참고 run

| run | head | lr | aug | val acc | val F1 | test acc | test F1 | checkpoint |
|-----|------|----|-----|---------|--------|----------|---------|------------|
| `dfu_head_v1` | linear | 5e-4 | — | 95.1% | 90.9% | _TBD_ | _TBD_ | `checkpoints/dfu_head_v1/best.pt` |
| `dfu_head_mlp_1e3_aug` | mlp | 1e-3 | ✓ | 97.8% | 94.4% | 96.2% | 93.1% | `checkpoints/dfu_head_mlp_1e3_aug/best.pt` |
| `dfu_head_mlp_1e3` | mlp | 1e-3 | — | 98.0% | 93.4% | 96.1% | 92.9% | `checkpoints/dfu_head_mlp_1e3/best.pt` |
| _(새 run)_ | | | | | | | | |

**실험 메모** _(자유 기록)_

- 원본: AI Hub (원본) + Kaggle DFU Dataset (foot crop) + PART A (foot crop)
- crop checkpoint: `foot_head_v2`
- class imbalance 처리:
- 상세 비교: `analysis/head_checkpoint_comparison/`

---

## Evaluation (Test Set)

`scripts/evaluate.py`로 hold-out 성능을 확인합니다. task별 데이터 형식과 명령은 위 [Foot](#foot-segmentation), [Wound](#wound-segmentation), [DFU](#dfu-classification) 섹션을 참고하세요.

| Task | 입력 | 평가 지표 |
|------|------|-----------|
| foot | `--data-root` (COCO) | dice, iou, accuracy |
| wound | `--image-dir` + `--mask-dir` | dice, iou, accuracy |
| dfu | `--data-root` (`dfu/`, `other/` 하위 폴더) | accuracy, precision, recall, f1 |

## Dataset Acknowledgments

재학습에 사용하는 외부 데이터셋의 저작자를 표기합니다.

### Foot (Segmentation)

| 프로젝트 경로 | 출처 |
|---------------|------|
| `roboflow-foot` | [Foot Segmentation (cv-v6vyj)](https://universe.roboflow.com/cv-v6vyj/foot-segmentation-dd6qi) |
| `roboflow-body` | [body (hell-khakz)](https://universe.roboflow.com/hell-khakz/body-idcrc) |
| `roboflow-humanbody` | [Human parts (personal-ekd6m)](https://universe.roboflow.com/personal-ekd6m/human-parts-bru4g) |

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

@misc{ foot-dydgp-wkb8b_dataset,
  title = { foot Dataset },
  type = { Open Source Dataset },
  author = { ysla },
  howpublished = { \url{ https://universe.roboflow.com/ysla/foot-dydgp-wkb8b } },
  url = { https://universe.roboflow.com/ysla/foot-dydgp-wkb8b },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2026 },
  month = { jun },
  note = { visited on 2026-07-07 },
}
```

### Wound (Segmentation)

**Lower Limb and Feet Wound Image Dataset (Mendeley)**

- **제목:** Lower Limb and Feet Wound Image Dataset for Medical Analysis
- **저자:** Md Masudul Islam
- **라이선스:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **DOI:** [10.17632/hsj38fwnvr.3](https://doi.org/10.17632/hsj38fwnvr.3)
- **URL:** [https://data.mendeley.com/datasets/hsj38fwnvr/3](https://data.mendeley.com/datasets/hsj38fwnvr/3)
- **로컬 경로:** `Wound Image Dataset` (`wound_main/` + `wound_mask/`, `Nomal/`)

**FUSeg — Foot Ulcer Segmentation Challenge**

- **저자:** uwm-bigdata
- **URL:** [https://github.com/uwm-bigdata/wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation)
- **로컬 경로:** `wound-segmentation/data/Foot Ulcer Segmentation Challenge`

### DFU Classification

**AI Hub — 욕창 데이터 (dataSetSn=509)**

- **URL:** [https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=509](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%9A%95%EC%B0%BD&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=509)
- **전처리:** 원본 이미지 그대로 사용 (foot crop 없음)

**Kaggle — DFU Dataset**

- **제목:** DFU Dataset
- **제작자:** Srivatsan Mk (Kaggle)
- **라이선스:** MIT License
- **URL:** [https://www.kaggle.com/datasets/srivatsanmk2004/dfu-dataset](https://www.kaggle.com/datasets/srivatsanmk2004/dfu-dataset)
- **로컬 경로:** `../../03_데이터/DFU Dataset`
- **전처리:** foot segmentation head로 발 영역 crop 후 사용

**PART A — CRC 라벨링 데이터**

- **설명:** 2026-06-17 기준 CRC 라벨링 데이터
- **로컬 경로:** `../../03_데이터/dfu_partA_20260617`
- **전처리:** foot segmentation head로 발 영역 crop 후 사용

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
| OOM (WSL) | `--num-workers 0`, `--image-size 512` |
| inference 속도 | 학습 해상도와 `--image-size` 일치 (512) |
