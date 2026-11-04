# 06. DL 모델링 (표 데이터 MLP)

> 사용 스킬: `pytorch-lightning`(LightningModule, Trainer, EarlyStopping, seed_everything·deterministic으로 재현성 확보)
> 코드: `src/s06_dl.py` · 결과: `reports/06_dl_results.csv`, `data/oof.parquet`(mlp 열)

## 1. 구조와 학습 설정

| 항목 | 설정 |
|---|---|
| 입력 | 5단계 `full` 피처 166개(수치) + Model.Suffix **임베딩(8차원)** |
| 전처리 | QuantileTransformer(정규분포 변환, **학습 폴드로만 fit**). 결측은 학습 폴드 중앙값으로 채움 |
| 네트워크 | 174 → 256 → 128 → 1. BatchNorm + SiLU + Dropout 0.3 |
| 손실 | BCEWithLogits, pos_weight = √(음성/양성) ≈ 4.0 (불균형 완화) |
| 최적화 | AdamW(lr 1e-3, wd 1e-4) + Cosine. 최대 30 epoch, val loss 기준 EarlyStopping(patience 5) |
| 검증 | 5단계와 **같은 5-fold 분할과 시드** → OOF로 직접 비교 |
| 환경 | CPU 8코어, RAM 3GB (GPU 없음) |

## 2. 결과

| 모델 | ROC-AUC | PR-AUC | 최적 F1 |
|---|---|---|---|
| **MLP** | 0.686 | 0.236 | 0.237 |
| LightGBM | 0.696 | **0.284** | 0.281 |
| GBDT 앙상블 (LGB+XGB+Cat 순위 평균) | **0.701** | **0.286** | 0.281 |
| GBDT 앙상블 + MLP | 0.702 | 0.272 | 0.267 |

- 폴드별 MLP AUC는 0.686 ~ 0.698, 조기 종료는 9 ~ 19 epoch에서 일어났다.
- MLP와 LightGBM 예측 순위의 상관은 0.74다. 다른 패턴을 일부 보지만, 그 부분은 대부분 잡음이었다.

## 3. 해석

1. **MLP는 GBDT보다 PR-AUC가 약 0.05 낮다.** 이 데이터는 **이산 설정값**(이동속도 4000/5000/9000…, 압력 단계)과 **임계형 규칙**(리드타임 > 2시간 → 불량)이 많아, 축에 맞춰 자르는 트리 모델이 구조적으로 유리하다.
2. **앙상블에 MLP를 넣으면 오히려 나빠진다**(PR-AUC 0.286 → 0.272). 상위 위험 구간의 정밀도를 MLP가 희석하기 때문이다.
3. **GBDT 3종 앙상블의 이득도 작다**(PR-AUC +0.002). 단일 LightGBM이 성능·속도·해석성의 균형이 가장 좋다.

## 4. 결론 — 최종 모델 선택

| 용도 | 선택 | 이유 |
|---|---|---|
| **운영 예측** | LightGBM 단일(또는 GBDT 3종 앙상블) + 하드 규칙 인터락 | 성능 동급, 학습 38초, SHAP 해석이 정확함 |
| **원인 해석** | LightGBM `process_only` | 시간·ID 효과를 빼고 공정 조건만 해석 |
| DL | 채택하지 않음 | 현재 데이터 크기(4만 건)와 피처 성격에서는 이득 없음 |

> DL이 이득을 보려면 **원시 시계열**(도포 궤적, 압력 곡선)이나 **검사 이미지**처럼 트리가 다루기 어려운 비정형 데이터가 필요하다. 이건 8단계 최종 제안에 넣는다.
