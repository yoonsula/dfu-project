# DFU Foot Analysis Pipeline — Portfolio

당뇨발(DFU) 이미지 분석 파이프라인 포트폴리오입니다.  
원본 프로젝트: **[yoonsula/dfu-project](https://github.com/yoonsula/dfu-project)**

## 구성

| 파일 | 설명 |
|------|------|
| `export/DFU_Portfolio.pptx` | 16:9 PowerPoint (제출용) |
| `export/DFU_Portfolio.pdf` | 16:9 PDF |
| `index.html` | 브라우저 슬라이드 덱 (동일 내용) |
| `generate_pptx.py` | PPTX 재생성 스크립트 |
| `scripts/export_pdf.sh` | HTML → PDF (Chrome headless) |

## 슬라이드 구성

1. **표지** — 프로젝트명, 기간, 역할
2. **프로젝트 설명** — 작업기간 / 개요 / 기여도 / 역할
3. **전체 아키텍처** — Shared DINOv3 + 3 Heads
4. **스킬 정리**
5. **기술 스택**
6–7. **주요 알고리즘 · 코드 스니펫**
8. **타임라인**
9–11. **프로젝트 나열** (문제 → 실행 → 성과 → 인사이트)
12. **기능 · 성능 테스트 결과**
13. **GitHub 링크 · 산출물**
14. **한줄 정리**

## 핵심 성과

| Task | Metric | Value |
|------|--------|-------|
| Foot Segmentation | Val Dice / IoU | **0.959 / 0.935** |
| Wound Segmentation | Val Dice / IoU | **0.814 / 0.738** |
| DFU Classification | Test Acc / F1 | **96.2% / 93.1%** |

## 디자인

- Apple / Notion 스타일 밝은 미니멀
- 화이트 중심 + 블루(`#0071E3`) 포인트
- 다이어그램 · 표 · 배지 · 타임라인 · 카드 중심

## 재생성

```bash
pip install python-pptx
python generate_pptx.py
./scripts/export_pdf.sh
```

## Author

윤수 라 · yoonsu@rexsw.com
