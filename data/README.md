# 데이터

이 저장소에는 **데이터를 포함하지 않는다.** LG Aimers 5기 본선(Data Intelligence) 제공 데이터로, 재배포할 수 없다.

## 필요한 파일

| 파일 | 내용 | 출처 |
|---|---|---|
| `data/train.csv` | 제품 40,506개 × 468열 (정답 `target`: Normal / AbNormal) | LG Aimers 5기 본선 |
| `data/test.csv` | 제품 17,361개 × 469열 (정답 없음, `Set ID`) | LG Aimers 5기 본선 |
| `data/Sub Assembly Line 공정_feature 종류.hwp` | 공정 · 변수 설명 문서 | LG Aimers 5기 본선 |
| `hand_data.xlsx` (선택) | 현장 수기 측정 21일치 (UV 에너지 · 두께 · 온도 · 습도) | 17단계에서만 사용, 경로를 인자로 지정 |

## 만들어지는 중간 파일

`src/s02_preprocess.py`, `src/s03_features.py`를 실행하면 `data/clean.parquet`, `data/features.parquet`가 생긴다. 이후 단계는 이 파일을 읽는다.
