# HyperbolicLR 출판 계획: ExpHyperbolicLR 중심 v13

작성일: 2026-08-18. 원 논문: [arXiv:2407.15200](https://arxiv.org/abs/2407.15200), "HyperbolicLR: Epoch Insensitive Learning Rate Scheduler". 저널 목표는 Machine Learning (Springer)이며, v1 LaTeX 원본은 `manuscript/v1_arxiv/`에 회수되어 있다.

## 1. 결론

논문은 실패 특성화 논문이 아니라 **ExpHyperbolicLR 방법 논문**으로 유지한다. 문제 설정은 cross-budget HPO transfer이고, 실증 대상은 작은 budget에서 선택한 설정을 더 긴 training horizon에 그대로 배포할 때의 성능이다.

> **핵심 주장.** ExpHyperbolicLR는 작은 budget에서 선택한 peak learning rate와 초기 학습 궤적을 더 긴 horizon에서도 보존하면서, 추가 budget을 실제 loss 감소에 사용하도록 설계된 log-space hyperbolic schedule이다.

이 문장은 현재 DeepONet 한 설정에서는 강하게 지지되지만, 아직 저널 수준의 일반성은 없다. 독립 HPO 반복, 다른 operator architecture, 다른 task와 optimizer에서 확인되기 전에는 최종 주장으로 확정하지 않는다.

Cross-budget collapse는 방법이 필요한 이유를 보여주는 현상이다. 논문의 정체성은 collapse 원인 규명이 아니라 low-budget tuning fidelity와 long-horizon utility를 함께 제공하는 EH의 설계와 검증이다.

## 2. 논문이 해결하는 실무 문제

실무에서는 짧은 proxy run으로 learning-rate 설정을 선택한 뒤 더 긴 run에 배포한다. Horizon에 맞춰 재매개화되는 cosine, linear, polynomial, WSD 계열은 deployment horizon이 바뀌면 실제 LR trajectory도 바뀐다. 반대로 horizon-independent exponential decay는 trajectory를 보존하지만 긴 budget을 충분히 활용하지 못할 수 있다.

좋은 cross-budget schedule은 두 요구를 동시에 만족해야 한다.

1. **Fidelity:** 작은 tuning budget에서 선택한 설정이 긴 deployment horizon에서도 유효해야 한다.
2. **Utility:** 긴 horizon이 제공하는 추가 optimization opportunity를 사용해 최종 성능을 개선해야 한다.

Stability 또는 collapse는 별도 보조축이다. Fidelity와 utility를 직접 측정하지 않고 collapse count만으로 방법의 우위를 주장하지 않는다.

## 3. H와 EH의 역할

H와 EH를 같은 강도로 밀지 않는다.

- **H:** shared-asymptote hyperbolic family의 선형 LR-space 원형이다. Family geometry와 smooth low-sensitivity 배치를 보여주는 ablation이다.
- **EH:** 같은 geometry를 log LR에 적용한 권장 방법이다. Multiplicative LR scale에 맞는 anchoring을 제공하고, 현재 DeepONet 결과에서 H의 tuning drift와 long-horizon loss gap을 줄였다.
- **E:** horizon-independent trajectory는 보존하지만 long-horizon utility가 약한 대조군이다.
- **POW와 WSQD:** horizon-free base phase를 가진 직접 경쟁 방법이다. EH가 유일하다는 주장을 막는 필수 baseline이다.

POW와 WSQD는 base phase의 fixed-time prefix를 EH보다 더 정확히 보존한다. EH의 차이는 exact prefix invariance가 아니라 phase boundary가 없는 하나의 smooth curve, bounded upper horizon $U$, log-space closed form, 그리고 fidelity와 endpoint progress 사이의 연속 tradeoff다. 이것은 설계 차이이며 성능 우월성은 아니다. Main experiment는 이 차이가 low-budget HPO 선택과 long-horizon utility에 실제 이득을 주는지 판정한다.

논문에서 사용할 정확한 관계는 다음과 같다.

> Hyperbolic geometry provides a smooth low-sensitivity family, while its exponentially anchored form, ExpHyperbolicLR, closes the remaining cross-budget utility gap in the current DeepONet campaign.

마지막 절은 새 실험이 통과해야만 여러 task에 일반화한다.

## 4. 현재 증거

### 4.1 DeepONet exploratory campaign

`collider:~/HyperbolicLR/paper/results_extrap/integral-DeepONet/`의 252개 parquet은 14 scheduler, tuning budget 2개, deployment horizon 3개, training seed 3개를 포함한다. 아래 표는 B=10에서 튜닝한 설정을 N=200에 배포한 결과다.

| Scheduler | 관측 collapse | N=200 median final loss | B=10 LR에서 B=50 LR로의 변화 |
|---|---:|---:|---:|
| **EH** | **0/3** | **1.2434816e-6** | `6.9232772e-4 -> 7.2479880e-4`, 4.7% |
| H | 0/3 | 3.4706446e-6 | `5.9089757e-4 -> 2.0225296e-4`, 2.92배 감소 |
| POW | 0/3 | 3.5988181e-6 | `7.0193726e-4 -> 3.0955665e-4`, 2.27배 감소 |
| WSQD | 1/3 | 3.6061474e-6 | `1.0465244e-3 -> 4.0554727e-4`, 2.58배 감소 |
| E | 0/3 | 2.9466078e-5 | 거의 동일 |

이 설정에서 EH는 다음 세 조건을 동시에 만족한 유일한 schedule이다.

1. B=10 tuning으로 N=200까지 배포했을 때 관측 collapse가 없다.
2. B=10과 B=50 HPO가 거의 같은 peak LR를 선택한다.
3. B=10 tuning으로 얻은 N=200 median final loss가 가장 낮다.

이 결과는 EH의 저널 가설을 세우기에 충분하지만, HPO 결정은 scheduler-budget당 한 번뿐이다. Deployment seed 3개는 HPO 선택의 독립 반복이 아니다. 현재 수치는 exploratory evidence로만 취급한다.

### 4.2 중요한 반례와 제한

- B=50에서 다시 튜닝하면 WSQD의 N=200 median final loss가 `1.2638153e-6`으로 가장 낮고 EH는 `3.4664359e-6`이다. EH의 사용 조건은 full-budget retuning이 어렵거나 deployment horizon이 바뀌는 경우다.
- DeepONetLN 정규화 캠페인 81/81 run은 완료되었다. B=10 tuning의 N=200 median final loss는 L `1.132e-5`, P `1.956e-5`, EH `1.972e-5` 순이었고 EH는 단독 최선이 아니다. EH, H, E, C, L, N, WSD의 선택 LR가 `1.01e-4`에서 `1.12e-4`에 몰려 search lower bound `1e-4`와 거의 같다. 이 결과는 normalization이 regime을 바꾼다는 증거이자 search-space truncation 경고다. 수정된 LR 범위 없이 EH 일반성 판정에 사용하지 않는다.
- CIFAR Phase 2의 8pp 격차는 scheduler별 LR를 튜닝하지 않은 fixed-lr0 비교였다. EH의 anchoring 가격으로 해석하지 않는다.
- CIFAR validation loss anytime 결과는 과적합 때문에 accuracy 순위와 반대로 움직였다. Classification에서는 validation accuracy와 held-out test accuracy만 쓴다.
- F24와 F26은 `dwell -> Adam edge-of-stability crossing -> collapse` 인과 사슬을 반증했다. Dwell과 sensitivity는 schedule geometry descriptor로만 사용하고 물리적 collapse mechanism으로 쓰지 않는다.

## 5. Novelty ledger

### 5.1 논문이 주장할 신규성

1. **ExpHyperbolicLR 방법:** 2024년 arXiv v1에서 공개한 log-space hyperbolic scheduler 자체가 원 방법 기여다.
2. **Cross-budget HPO transfer 해석:** EH를 단순 epoch-insensitive curve가 아니라 low-budget tuning fidelity와 long-horizon utility를 함께 겨냥한 방법으로 정식화한다.
3. **Log-space schedule coordinates:** prefix deformation, endpoint anchoring, horizon sensitivity를 multiplicative LR scale에서 폐형식으로 계산하고 H와 EH의 차이를 분리한다.
4. **직접 transfer benchmark:** 동일 candidate set 또는 동일 HPO protocol로 B=10 선택을 N=10, 50, 100, 200에 배포하고, within-family transfer regret와 utility를 함께 측정한다.

### 5.2 이미 선점되어 인용할 주장

- Horizon이 바뀌면 최적 peak LR가 움직일 수 있다는 관찰은 Power Scheduler와 WSqD를 포함한 선행연구에 있다.
- Geometric 또는 inverse-time decay 자체는 고전적이며 Schaul et al.과 Ge et al.의 선행이 있다.
- WSD, continued training, anytime evaluation, averaging으로 horizon 문제를 완화하는 아이디어는 선행연구가 있다.
- 누적 LR, edge-of-stability, preconditioned sharpness는 우리 신규성이 아니다.

### 5.3 금지할 주장

- EH가 모든 task와 optimizer에서 최선이라는 주장.
- EH만 horizon-free라는 주장.
- POW 또는 WSQD보다 이론적으로 우월하다는 주장.
- Dwell threshold가 collapse 원인이라는 주장.
- AdamW의 `38/eta` crossing이 transfer failure를 설명한다는 주장.
- `D + A >= H` 또는 sensitivity conservation identity를 논문의 headline theorem으로 포장하는 서술.
- 한 HPO seed와 세 deployment seed를 독립 표본 여섯 개처럼 취급하는 통계.

## 6. 이론부의 새 역할

본문 이론은 EH의 설계와 실험 변수를 설명하는 데 필요한 부분만 남긴다.

1. Cross-budget schedule family `eta_N(t)`와 deployment protocol을 정의한다.
2. LR-space와 log LR-space의 prefix deviation, endpoint anchoring, tail headroom을 정의한다.
3. EH의 log-space hyperbolic arc, shared asymptote, monotone decay, closed-form deviation과 anchoring을 제시한다.
4. H와 EH를 같은 geometry의 linear-space와 log-space 버전으로 비교한다.
5. `D + A >= H`와 sensitivity mass identity는 coordinate bookkeeping으로 제시하고 증명은 appendix로 보낸다.
6. Stop-time minimax game과 spectral-tension theorem은 cross-budget HPO 주장을 직접 지지하지 않으므로 appendix 또는 별도 이론 논문 후보로 분리한다.
7. Edge-of-stability와 dwell mechanism은 이론부에서 제거한다. Negative result는 limitations에 한 문단으로 보고한다.

## 7. Confirmatory experiment protocol

### 7.1 실험 단위

HPO 반복이 추론 단위다. 하나의 HPO가 선택한 설정을 여러 training seed로 재학습한 결과는 한 cluster 안의 반복 측정이다.

- Confirmatory methods EH, H, POW, WSQD의 독립 low-budget selection 반복 수: `R=12`.
- Descriptive baselines E, C, D2Z, WSD의 반복 수: `R=3`.
- 각 반복은 optimizer별 공통 30-point log LR grid를 전부 평가한다.
- HPO budget: 정확히 `10 epochs`.
- Deployment seed 수: 선택 설정당 `K=3`. 각 replicate는 고유한 세 seed를 쓰고, 같은 replicate의 scheduler끼리만 그 seed 세트를 공유한다. HPO training seed와도 겹치지 않는다.
- Deployment horizons: `N in {10, 50, 100, 200}`.
- 모든 scheduler는 같은 split, 같은 LR candidate set, 대응되는 HPO seed와 deployment seed block을 사용한다.
- Adaptive sampler와 pruner는 primary protocol에서 쓰지 않는다. 유한 candidate grid를 전부 평가해 search-path와 resume-state 차이를 제거한다.
- HPO training seed, deployment seed, split seed를 별도 필드로 저장한다.

Seed allocation은 `master_seed=20260818`에서 문자열 key `hyperboliclr-v13|role|setting|replicate|slot`의 SHA-256 앞 64 bit를 취해 `1..2^31-2`로 사상한다. Role은 dataset, split, HPO training, deployment로 분리한다. 생성된 모든 seed는 manifest에 값으로 저장하고 역할 사이 충돌을 검사한다. 라이브러리 RNG의 spawn 규칙에 seed 목록을 맡기지 않는다.

### 7.2 Core scheduler set

Main table은 다음 8개로 고정한다.

| 역할 | Scheduler |
|---|---|
| 제안 방법 | EH |
| Family ablation | H |
| Horizon-independent but low-utility control | E |
| 직접 경쟁자 | POW, WSQD |
| 표준 horizon-rescaled baseline | C, D2Z |
| 널리 쓰이는 cooldown baseline | WSD |

Polynomial, linear-with-floor, warmup variants, Schedule-Free는 appendix 확장으로 유지한다. Core 결과를 본 뒤 baseline을 빼거나 추가하지 않는다.

### 7.3 Setting matrix

| ID | Task | Architecture | Optimizer | 역할 |
|---|---|---|---|---|
| S1 | GRF integral operator | DeepONet | AdamW | 현재 결과의 독립 재현 |
| S2 | 같은 operator data | TFONet | AdamW | architecture transfer |
| S3 | CIFAR-10 | ResNet-18 for CIFAR | SGD with momentum | task와 optimizer transfer |
| S4 | GRF integral operator | DeepONetLN | AdamW | normalization interaction, corrected LR range |
| S5 | CIFAR-10 | SimpleViT | AdamW | S2 또는 S3가 혼합일 때의 추가 판정 |

S1부터 S3까지가 제출 필수다. S4는 현재 lower-bound HPO 결함을 바로잡는 ablation이다. S5는 결과가 한 architecture family에만 남을 때 실행한다.

S1과 S2는 raw operator data, batch size 100, AdamW `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.01`을 쓴다. S1은 `DeepONet(norm=False)`, S2는 현재 registry의 TFONet 구조를 고정한다. S3는 batch size 128, SGD `momentum=0.9`, `weight_decay=5e-4`, standard random crop과 horizontal flip을 train에만 적용한다. 이 값은 scheduler별로 바꾸지 않는다.

### 7.4 데이터 분할

- Operator data는 split별 generator seed를 고정해 raw `train=8,000`, `validation=1,000`, `test=1,000` sample을 독립 생성한다. 기존 generator의 GRF amplitude scaling은 train split에서만 fit하고 같은 계수를 validation과 test에 적용한다. 추가적인 model-input standardization은 하지 않는다. S4의 normalization은 데이터 전처리가 아니라 architecture LayerNorm ablation이다. 현재 train/val 두 파일만 있는 campaign은 exploratory로 남긴다.
- CIFAR-10은 split seed를 고정해 train 45,000, validation 5,000을 stratified하게 나누고 official test 10,000을 그대로 둔다. HPO 중 official test set을 열거나 transform하지 않는다.
- Split manifest에는 sample identity, split별 seed와 count, source checksum, transform spec을 저장한다. 모든 run artifact에 manifest hash를 기록한다.
- HPO objective는 validation metric만 사용한다. Deployment는 validation과 test metric을 모두 기록하지만 설정 선택에는 어느 것도 다시 사용하지 않는다. Final claim은 held-out test metric으로만 판정한다.

### 7.5 Search space와 공정성

두 arm을 분리한다.

**Arm A, fixed-shape transfer landscape:** S1에서 peak LR만 바꾼다. AdamW `1e-5`에서 `5e-2`까지 30개 log-spaced point, deployment seed 3개, `N in {10,50,100,200}`을 쓴다. 인접 LR 비율은 약 1.341이다. Shape는 EH/H `U=400, r=eta_min/eta_0=1e-3`, E `gamma=0.963`, C `r=1e-3`, D2Z exact zero, WSD `cooldown_frac=0.2, r=1e-3, warmup=0.1N`, POW `b=-0.51, decay_frac=0.1, absolute warmup=5`, WSQD `alpha=0.2, T0=10, absolute warmup=5`로 고정한다. 이 arm이 candidate-set fidelity의 primary controlled measurement다. S2와 S3에서는 전체 grid를 반복하지 않고 Arm B 결과가 혼합일 때 selected LR 주변 5-point local grid를 진단용으로 실행한다.

**Arm B, replicated low-budget selection:** Arm A와 같은 fixed shape를 사용하고 peak LR만 optimizer별 30-point common grid에서 선택한다. 각 replicate 안에서는 모든 candidate와 scheduler가 같은 HPO training seed를 공유한다. 선택된 peak LR와 shape 전체를 deployment horizon에서 고정한다. Base optimizer의 weight decay, momentum, batch size, augmentation은 setting별로 사전 고정하고 scheduler에 따라 바꾸지 않는다.

AdamW grid는 `1e-5`에서 `5e-2`, SGD with momentum grid는 `1e-3`에서 `1`이다. 2026-08-18 exploratory DB audit에서 기존 AdamW core best LR는 새 lower bound보다 최소 10.2배 높았고, DeepONetLN best LR 군은 30-point grid의 index 8 부근에 놓였다. G1에서 어느 scheduler든 best candidate가 양쪽 끝 두 point 안에 들면 그 optimizer의 모든 scheduler에 대해 해당 방향 경계를 10배 확장하고 G1 전체를 다시 실행한다. 결과를 보고 범위를 좁히지는 않는다.

Shape search를 primary HPO에 섞지 않는다. EH/H의 `U`와 floor, WSD cooldown처럼 방법마다 차원이 다른 search를 같은 30 trial 안에 넣으면 scheduler quality와 search-space burden을 분리할 수 없다. Shape sensitivity는 primary 결과를 고정한 뒤 selected LR를 그대로 두고 별도 paired ablation으로 측정한다.

Arm A의 LR 선택과 평가는 seed leave-one-out cross-fitting을 쓴다. 각 held-out seed에 대해 나머지 두 seed의 validation score로 B=10 LR와 horizon-specific oracle LR를 고르고, held-out seed의 test score로 transfer regret와 utility를 계산한다. 같은 seed로 oracle을 선택하고 평가하는 winner bias를 허용하지 않는다.

### 7.6 Primary metrics

회귀와 operator task의 한 run score는 마지막 `w=max(3, ceil(0.1N))` epoch metric의 median이다. Classification HPO score는 이 창의 validation accuracy median이다. Deployment에서는 HPO가 읽지 않은 test set을 같은 창에만 평가하고 test accuracy median을 최종 표에 쓴다.

**Within-family transfer regret:** Arm A에서 B=10 최적 LR를 `hat_eta_s,10`, horizon N의 within-family 최적 LR를 `eta*_s,N`이라 두면 회귀 task는

$$R_s(N)=\log_{10}\frac{L_s(\hat\eta_{s,10},N)}{L_s(\eta^*_{s,N},N)}.$$

Classification은 $R_s(N)=A_s(\eta^*_{s,N},N)-A_s(\hat\eta_{s,10},N)$을 percentage point로 보고한다. `R=0`이 완전 transfer다.

**Long-horizon utility:** 회귀 task는

$$U_s(N)=\log_{10}\frac{L_s(\hat\eta_{s,10},10)}{L_s(\hat\eta_{s,10},N)}.$$

Classification은 accuracy gain을 percentage point로 보고한다. `U>0`이 긴 budget을 실제로 사용했다는 뜻이다.

**Replicated-selection endpoint:** Arm B의 각 반복에서 B=10 validation metric으로 선택한 설정을 N=200에 배포한 held-out test metric이다. EH와 H, POW, WSQD의 paired difference가 primary comparison이다.
**Diagnostics:** selected-LR drift, full-curve rank correlation, top-k overlap, continuous rebound ratio를 함께 보고한다. Regression collapse는 `non-finite` 또는 final-window loss가 best-window loss의 2배를 넘는 경우로 사전 등록하고 1.1, 2, 10 threshold sensitivity를 공개한다. Classification은 과적합과 instability를 섞지 않기 위해 non-finite run만 collapse로 세고, best-to-final accuracy drop은 별도 연속량으로 보고한다. Collapse count를 primary endpoint로 쓰지 않는다. Primary scheduler의 selected deployment에 non-finite outcome이 있으면 이를 결측으로 버리지 않고 viability failure로 보고하며, EH에 발생한 setting은 방법 성공 gate를 통과할 수 없다.

### 7.7 통계

- HPO replicate별 deployment seed median을 먼저 계산한다. Confirmatory scheduler 비교는 12개의 paired cluster score에 대해 수행한다. Replicate 사이 deployment seed는 중복하지 않아 cluster 독립성을 보존한다.
- 각 setting마다 EH 대 H superiority와 EH 대 POW, WSQD non-inferiority를 하나의 confirmatory family로 둔다. Paired difference의 부호를 모두 열거하는 one-sided exact sign-flip permutation test를 쓰고 세 검정에 Holm correction을 적용한다. 이는 대칭 오차 null 아래의 permutation test이며, scheduler label을 실제 무작위 배정한 causal randomization test라고 부르지 않는다. Setting 사이의 p-value는 합치지 않는다.
- Regression은 `d=log10(L_comp)-log10(L_EH)`, classification은 `d=A_EH-A_comp`로 방향을 통일한다. H 비교는 `d>0` superiority, POW와 WSQD 비교는 regression `d>-0.1`, classification `d>-0.5 percentage point` non-inferiority를 검정한다.
- Effect size는 paired log-loss difference 또는 accuracy-point difference와 95% hierarchical bootstrap interval로 보고한다. Bootstrap은 HPO cluster를 먼저 resample하고 그 안에서 deployment seed를 resample한다.
- Scheduler 전체 순위와 나머지 baseline 비교는 descriptive다. 여러 task의 p-value를 하나로 합쳐 일반성을 만들지 않는다. Exact sign test와 leave-one-replicate-out effect를 sensitivity analysis로 함께 공개한다.

### 7.8 Workload와 실행 순서

Arm A S1은 `8 schedulers x 30 LR x 3 seeds x 4 horizons = 2,880` runs다. Arm B는 setting당 HPO 1,800 trials와 deployment 720 runs다. S1부터 S3까지의 Arm B는 HPO 5,400 trials와 deployment 2,160 runs다.

기존 252-run artifact의 first-to-last timestamp span은 13.77시간이지만 두 GPU의 병렬 실행, scheduler별 HPO, idle 구간이 섞여 있어 새 matrix의 GPU-hour 환산 기준으로 쓰지 않는다. G1에서 task별 `seconds per 10-epoch HPO trial`과 `seconds per deployment epoch`을 분리 측정한 뒤 총 비용을 계산한다. 측정 전 완료일을 약속하지 않는다.

G1은 두 단계다. G1a는 별도 dry artifact root에서 tiny data와 축소 epoch를 쓰되 8 scheduler와 30 candidate 전체를 통과시킨다. G1b는 S1 full data에서 replicate 0의 10-epoch selection과 `N in {10,200}`, `K=3` deployment를 실행한다. G1b가 search-boundary gate를 통과하면 replicate 0으로 그대로 보존하고, 범위를 확장해야 하면 모든 G1b artifact를 exploratory로 표기한 뒤 같은 replicate seed로 공통 grid 전체를 다시 실행한다.

실행은 S1 G1, S1 Arm A, S1 Arm B 순이다. S1이 성공 기준 1을 통과해야 S2와 S3로 간다. S2와 S3는 독립적으로 실행하며 한 setting 결과를 보고 다른 setting의 search range나 margin을 바꾸지 않는다. S4와 S5는 primary 결과가 확정된 뒤 실행한다.

### 7.9 실패와 결측 처리

- Non-finite loss 또는 accuracy가 발생한 training run은 방법의 outcome이다. 같은 seed를 교체하거나 rerun해 지우지 않는다.
- HPO trial의 non-finite metric은 회귀에서 `+infinity`, classification에서 `-infinity`로 기록하고 trial status를 별도 저장한다.
- SSH 단절, host reboot, 파일 쓰기 실패처럼 model outcome 전에 난 infrastructure failure만 같은 `run_id`로 재개한다. 새 seed로 바꾸지 않는다.
- OOM이 한 scheduler에만 발생하면 그 scheduler를 위한 batch 축소를 하지 않는다. Setting 전체에 같은 batch와 accumulation 조합을 적용해 다시 시작하거나 해당 setting을 infeasible로 판정한다.
- 각 gate는 expected run key set과 artifact key set의 차집합이 0인지 확인한다. Missing run을 제외한 채 통계를 계산하지 않는다.

## 8. 필수 ablation

1. **H 대 EH:** exponentiation과 log anchoring이 실제 utility gap을 닫는지 판정한다.
2. **U sensitivity:** `U in {250, 400, 800}`을 S1과 S2에서 비교한다. 최대 deployment horizon 200에서 구현이 요구하는 `U>N`을 모두 만족하며, `U/N_max in {1.25,2,4}`를 이룬다.
3. **Relative floor:** `eta_min/eta_0 in {1e-4, 1e-3, 1e-2}`로 endpoint dependence를 확인한다.
4. **Warmup matching:** native warmup arm과 모든 방법에 같은 absolute warmup을 적용한 arm을 S1에서 비교한다.
5. **Normalization:** S4를 corrected LR range로 다시 튜닝한다. 기존 lower-bound 결과를 방법 비교로 재사용하지 않는다.
6. **Full-budget reference:** 기존 B=50 HPO는 descriptive reference로만 둔다. 새 confirmatory HPO는 프로젝트 규칙에 따라 10 epochs로 고정한다.

## 9. 성공과 중단 기준

방법 중심 저널 논문은 다음 조건을 모두 만족해야 한다.

1. S1에서 EH의 Arm A transfer regret가 `0.1 log10` 이내이고 utility가 양수다.
2. S2 또는 S3 중 적어도 하나에서 같은 방향이 재현된다.
3. 적어도 2개 setting에서 Arm B의 EH 대 H superiority가 Holm-corrected test를 통과하고, EH 대 POW와 EH 대 WSQD가 각각 사전 non-inferiority margin 안에 있다.
4. EH의 이점이 한 HPO replicate 또는 한 deployment seed에만 의존하지 않는다.
5. U ablation에서 결론이 `U=400` 한 점에만 의존하지 않는다.

판정 경로는 다음과 같다.

- **S1, S2, S3 통과:** 일반 cross-budget method paper로 진행한다.
- **Operator setting 두 개만 통과:** 제목과 주장을 neural operator training으로 좁힌다.
- **S1만 통과:** Machine Learning 제출을 중단한다. DeepONet case study 또는 failure-characterization 논문으로 다시 판단한다.
- **EH가 POW 또는 WSQD에 두 setting 이상 명확히 열위:** EH 단독 권장 주장을 버리고 comparative study로 전환한다.

## 10. Figure plan

1. **Problem figure:** B=10에서 같은 LR 후보를 선택한 뒤 N=200에 배포하는 protocol을 한 장으로 보여준다.
2. **Main Pareto figure:** x축 Arm A transfer regret, y축 long-horizon utility, marker outline은 collapse sensitivity다. 점 하나가 아니라 HPO replicate 또는 setting uncertainty를 표시한다.
3. **Response-surface figure:** scheduler별 peak LR 대 horizon 성능 surface와 optimum ridge를 보여준다. EH의 ridge가 수직에 가까운지가 직접 증거다.
4. **H 대 EH ablation:** 같은 hyperbolic geometry에서 linear LR space와 log LR space의 endpoint 및 performance 차이를 보여준다.
5. **Cross-setting forest plot:** EH 대 H, POW, WSQD의 paired effect와 confidence interval을 setting별로 보여준다.
6. **Limitations figure 또는 appendix:** normalization interaction과 EoS negative result를 짧게 정리한다.

기존 dwell threshold figure는 main figure에서 제외한다.

## 11. 원고 구조

1. **Introduction:** proxy-budget HPO, deployment horizon 변화, fidelity와 utility의 두 요구, EH의 해결 방식, 기여 목록.
2. **Problem and Evaluation Protocol:** tuning budget, deployment horizon, transfer regret, utility, HPO cluster 단위.
3. **ExpHyperbolicLR:** H geometry, log-space exponentiation, closed-form coordinates, H와 EH의 역할 분리.
4. **Related Schedules:** cosine, D2Z, WSD, Power, WSqD, exponential을 같은 transfer 좌표로 비교.
5. **Experimental Protocol:** fixed-shape landscape, replicated HPO, split and seed discipline, statistics.
6. **Results:** DeepONet 재현, TFONet, CIFAR ResNet, ablations, direct-prior comparison.
7. **What Schedule Geometry Explains and Does Not Explain:** curve deformation은 설명하지만 optimizer instability mechanism은 설명하지 않는다고 선을 긋는다.
8. **Limitations and Use Conditions:** full-budget retuning이 가능할 때 WSQD가 더 좋을 수 있음, architecture dependence, bounded `U`, scale scope.
9. **Appendix:** coordinate identities, proofs, stop-time minimax material, full baseline tables, negative mechanism audit.

## 12. 제목

실험 완료 전 작업 제목은 다음으로 고정한다.

**Hyperbolic Learning-Rate Schedules for Cross-Budget Hyperparameter Transfer**

EH가 S1부터 S3까지 통과하면 더 직접적인 제목으로 바꿀 수 있다.

**ExpHyperbolicLR: Preserving Low-Budget Tuning Across Training Horizons**

Operator setting에만 남으면 다음으로 좁힌다.

**ExpHyperbolicLR for Cross-Budget Hyperparameter Transfer in Neural Operators**

## 13. 구현 선행조건

Confirmatory run 전에 다음 코드 계약을 만족해야 한다.

1. Primary HPO는 optimizer별 30-point LR candidate manifest를 한 번 생성하고 모든 scheduler와 replicate가 같은 값을 쓰게 한다. Candidate manifest hash가 run metadata와 일치하지 않으면 실행을 거부한다.
2. Output path와 parquet schema에 `run_id`, `selection_id`, HPO replicate, candidate index, HPO train seed, deployment seed, split hash, candidate-grid hash, optimizer, full optimizer와 scheduler parameters, code commit, dirty flag, environment lock hash, device, run status를 넣는다. HPO trial artifact와 deployment artifact는 `selection_id`로 연결한다. 현재 filename은 `sched_extra`가 달라도 충돌할 수 있다.
3. HPO objective를 단일 마지막 epoch loss에서 `w=max(3,ceil(0.1N))` final-window validation metric으로 바꾼다.
4. CIFAR runner에 ResNetCIFAR와 SimpleViT 선택, proper train/val/test split, SGD momentum을 추가한다.
5. Operator generator에 고정 seed, train-only GRF scaling, 독립 train/validation/test split과 manifest를 추가한다.
6. LR floor는 absolute `infimum_lr` 독립 탐색이 아니라 peak LR 대비 ratio로 고정한다.
7. Arm A analysis는 seed leave-one-out cross-fitting으로 transfer regret와 utility를 계산하고, rank correlation과 top-k overlap은 원자료에서 재계산한다.
8. Dry run은 scheduler별 output naming, deterministic replay, resume, split isolation, seed isolation, test-set non-access during HPO, expected-key completeness를 확인해야 한다.

G1a acceptance는 다음을 모두 요구한다: expected artifact key set 일치, 같은 command replay의 byte-stable manifest와 동일 metric hash, 중간 중단 후 resume 시 duplicate나 누락 0개, HPO 중 test-loader access 0회, role 간 seed collision 0개, scheduler parameter drift 0개, candidate-grid hash 불일치 시 fail-closed. G1b는 여기에 full-data run count, search-boundary gate, task별 runtime 측정을 추가한다. Infrastructure exception, malformed artifact, provenance mismatch는 campaign을 중단한다. Non-finite metric은 재실행하거나 결측 처리하지 않고 method outcome으로 저장한다. Production run은 clean Git commit과 environment lock hash가 없으면 시작하지 않는다.

### 13.1 동결 구현과 실행 표면

- `paper/protocol/confirmatory_v13.json`이 protocol hash `efd5c3298b6490982731b2ff959d5825bb93b911a1b3808bc6aedefa6e968e22`의 단일 source of truth다. 후보 grid, scheduler shape, seed block, split role, metric direction, output schema를 runner가 재정의하지 않는다.
- `paper/protocol/operator_v13.json`과 `paper/protocol/cifar10_v13.json`은 각각 독립 operator split과 CIFAR-10 45k/5k/10k split을 동결한다. Runner는 새 run마다 manifest hash와 실제 source file checksum을 확인한다.
- `paper/confirmatory_runner.py`는 한 run을 원자적으로 기록하고 `paper/confirmatory_orchestrator.py`는 phase별 expected-key set, deterministic lane sharding, resume, selection, search-boundary gate, artifact checksum, clean-commit provenance를 관리한다.
- `paper/confirmatory_analysis.py`는 Arm A seed leave-one-out cross-fitting, transfer regret, utility, candidate rank correlation, paired cluster effect, exact sign-flip test, Holm correction, hierarchical BCa interval, sign test, leave-one-replicate-out sensitivity를 원자료에서 재계산한다.
- Remote 실행 전 `_provision_confirmatory_v13.sh`로 환경과 data checksum을 검증하고, collider lane은 `_run_confirmatory.sh <phase> <setting> <pilot|full> <horizons> <device> <lane/count>`를 `setsid`로 분리 실행한다. Collider에서는 shared pueue state를 건드리지 않는다.

G1a는 scheduler 8개, candidate 30개, HPO 240개와 deployment 48개를 합친 288-curve dry campaign으로 통과했다. 7개 job 뒤 중단한 run을 재개했을 때 `executed=281`, `skipped=7`, duplicate와 missing이 0이었고, 독립 replay의 manifest byte와 metric hash가 일치했다. 양쪽 audit hash는 `227c999442fd57753c3ec5381c4d29f860c4f86ccb882685a93bd159067ffd83`다.

G1b는 clean commit `71b4de0`에서 통과했다. S1 replicate 0의 HPO 240개와 `N in {10,200}`, `K=3` deployment 48개가 모두 검증되어 expected key `288/288`, boundary selection 0개, non-finite outcome 0개였다. Audit hash는 `f52140a7f819a119a6e057ad9f19700053592aec6d4224f4efa6068c4732dd20`, median runtime은 HPO trial당 `10.0685 s`, deployment epoch당 `1.0107 s`다. 한 HPO lane이 224개 artifact 시점에 종료되었으나 동일 run ID로 resume한 뒤 duplicate와 missing 없이 audit를 통과했다.

G1b의 단일 HPO replicate는 방법 우열을 판정하지 않는다. N=200 held-out median loss는 EH `4.1004e-6`, H `1.8924e-6`, D2Z `2.4851e-6`였고 EH utility는 `2.0521 log10`으로 양수였다. 이 결과는 full-data 범위와 파이프라인의 sanity check이며, EH 대 H, POW, WSQD 결론은 G2와 G3의 사전 정의된 반복 단위로만 낸다.

## 14. 타겟과 일정

1순위는 **Machine Learning (Springer)**다. SCIE와 2025 JIF 4.9가 확인되어 있고, 방법과 실증을 함께 싣기에 맞다. 2순위는 Neurocomputing, 색인 제약이 사라질 때의 대안은 TMLR이다. 이중 투고는 하지 않는다.

기존 2026년 10월 초 목표는 confirmatory matrix를 생략해야만 가능한 날짜라 폐기한다. 새 일정은 날짜보다 gate로 관리한다.

| Gate | 완료 조건 |
|---|---|
| G0 protocol freeze | metric, split, seed, search space, output schema, success rule 고정 |
| G1 pipeline proof | G1a에서 scheduler 8개와 30 candidate의 deterministic dry replay, resume, split/test isolation, expected-key gate 통과. G1b에서 S1 replicate 0, N=10/200 full-data pilot와 search-boundary gate 통과 |
| G2 landscape | S1 Arm A의 30-point LR grid, 3 seeds, 4 horizons 완료 |
| G3 replicated selection | S1부터 S3까지 Arm B 완료. EH/H/POW/WSQD는 R=12, 나머지는 R=3, 모두 K=3이고 replicate별 seed block은 중복 없음 |
| G4 ablations | H/EH, `U in {250,400,800}`, floor, warmup, normalization 판정 |
| G5 manuscript | 본문과 appendix 완성, 원자료에서 모든 숫자 재생성 |
| G6 review | 적대적 리뷰와 citation audit 통과 후 제출 |

## 15. 현재 상태

- DeepONet 252-run exploratory campaign은 완결되었고 EH 가설을 지지한다.
- Sharpness resolution-1과 measured-dwell audit은 완결되었고 EoS mechanism을 반증했다.
- DeepONetLN independent HPO campaign 81/81 run은 완료되었다. EH가 L과 P보다 근소하게 나빴고, 다수 scheduler의 HPO가 LR search lower bound에 붙었다. S4는 corrected range로 재실행한다.
- HyperbolicLR GPU 작업은 collider에서만 실행한다. darkmatter의 다른 프로젝트 작업은 건드리지 않는다.
- G0, G1a, G1b는 완료되었다. G2 S1 Arm A 2,880-run landscape를 clean commit `71b4de0`에서 collider L40 두 lane으로 실행 중이며, 완료 후 expected-key audit와 seed leave-one-out cross-fitted transfer regret 및 utility를 계산한다.
