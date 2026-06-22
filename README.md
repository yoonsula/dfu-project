# AI Hub DFU Classification

`dfu` vs `other` 이진 분류 head만 학습하는 프로젝트입니다.

## 구조

```
aihub-dfu/
├── assets/dinov3-hf/          # 로컬 HF backbone (config.json + model.safetensors)
├── train.py
├── paths.py
├── models/
├── datasets/
├── trainers/
├── scripts/
└── checkpoints/               # 학습 출력 (gitignore)
```

`assets/dinov3-hf/`에는 Hugging Face 스냅샷을 넣습니다.  
`dfu-project/assets/dinov3-hf/` 내용을 복사하거나 `DINOV3_MODEL_PATH`로 경로를 지정하세요.

## 데이터 형식

```
dfu_classification_data/
├── train/
│   ├── dfu/
│   └── other/
└── val/
    ├── dfu/
    └── other/
```

## 설치

```bash
pip install -r requirements.txt
python scripts/verify_setup.py
```

## 학습

```bash
python train.py \
  --data-root ../../03_데이터/dfu_classification_data \
  --run-name dfu_head_v1 \
  --image-size 384 \
  --batch-size 32 \
  --lr 5e-4 \
  --device cuda --amp
```

## 평가

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/dfu_head_v1/best.pt \
  --data-root ../../03_데이터/dfu_classification_data/test
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DINOV3_MODEL_PATH` | `assets/dinov3-hf` | 로컬 HF 스냅샷 |
| `DFU_CLASSIFICATION_DATA_ROOT` | `../../03_데이터/dfu_classification_data` | 학습 데이터 |
| `DFU_CHECKPOINT_DIR` | `checkpoints/` | checkpoint 저장 위치 |

Backbone은 `local_files_only=True`로 로드합니다 (네트워크 불필요).
