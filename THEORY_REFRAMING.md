# 재프레이밍: 스케줄러 설계의 보존 법칙과 트레이드오프 (v2)

작성: 2026-06-11 (v1), 같은 날 v2로 전면 개정. 목적: HyperbolicLR 논문을 "새 스케줄 함수" 논문에서 "horizon 불확실성 하의 스케줄 설계에 내재한 정량적 트레이드오프와 보존 법칙을 정식화하고, 그 위에서 HyperbolicLR의 좌표를 폐형식으로 제시하는" 논문으로 재구성한다. 모든 명제는 sympy 또는 numpy로 검증 완료 (검증 항목마다 표기).

## 0. 한 문장 주장

> 총 budget이 불확실한 상황에서 annealing 스케줄러는 세 가지를 동시에 가질 수 없다: 재현 가능한 초기 궤적 (작은 prefix deviation $D$), 목표 하한 도달 (작은 anchoring error $A$), 연장된 budget의 실질 활용 (큰 tail headroom $H$). 이들은 가정 없는 부등식 $D + A \ge H$로 묶이며, budget 민감도의 총량은 보존 법칙 $\int_0^N \partial_N \eta_N\,dt = B'(N) - \eta_N(N)$을 따른다. 모든 기존 스케줄러는 이 보존된 민감도 질량의 (총량, 배치) 좌표로 분류되고, HyperbolicLR은 가장 작은 총량을 가장 늦게 배치하는 매끄러운 구성이며 그 좌표 전부가 폐형식이다.

## 1. 형식 체계

**설정.** 스케줄러 패밀리는 budget $N \in \mathcal{N} \subset \mathbb{R}_{>0}$마다 LR 곡선 $\eta_N : [0, N] \to [\eta_{\min}, \eta_0]$를 주는 사상. $\Delta\eta = \eta_0 - \eta_{\min}$, $\eta_N(0) = \eta_0$.

**정의 1 (Prefix deviation).** $N_1 < N_2$, 창 $T \le N_1$에 대해 $D_T(N_1, N_2) = \sup_{t \le T} |\eta_{N_1}(t) - \eta_{N_2}(t)|$, $L^1$ 버전 $\bar{D}_T = \int_0^T |\eta_{N_1} - \eta_{N_2}|\,dt$.

**정의 2 (Anchoring error).** $A(N) = \eta_N(N) - \eta_{\min} \ge 0$.

**정의 3 (Tail headroom).** $H(N_1, N_2) = \eta_{N_2}(N_1) - \eta_{\min}$. 긴 budget 런이 시점 $N_1$ 이후의 연장 구간 $[N_1, N_2]$를 시작할 때 남아 있는 LR 여유. $H \approx 0$이면 연장 구간 전체를 바닥 LR로만 학습한다 (연장 budget의 낭비).

**정의 4 (Cumulative budget과 민감도 질량).** $B(N) = \int_0^N \eta_N(t)\,dt$ (총 LR 예산). 민감도 질량 $M(N) = \int_0^N \partial_N \eta_N(t)\,dt$ (budget을 한 단위 늘릴 때 LR 곡선이 창 전체에서 움직이는 총량).

**정의 5 (Budget elasticity).** $\mathcal{E}(N) = \partial \log |\dot\eta_N(0)| / \partial \log N$. 초기 감쇠율의 budget 민감도.

## 2. 다리 명제: 스케줄 편차가 학습 곡선 편차를 지배한다

**명제 1 (Bridge).** $f$가 $L$-smooth, $\|\nabla f\| \le G$라 하자. 동일 초기값과 공유 randomness (동일 mini-batch 순서)로 LR 수열 $\{\eta^1_k\}, \{\eta^2_k\}$를 쓰는 SGD 궤적 $\theta^1, \theta^2$는

$$\|\theta^1_T - \theta^2_T\| \;\le\; G \exp\!\Big(L \textstyle\sum_{k<T} \eta^1_k\Big) \sum_{k<T} |\eta^1_k - \eta^2_k|$$

*증명.* $\delta_k = \theta^1_k - \theta^2_k$, $g_k(\theta) = \nabla f_{i_k}(\theta)$ (공유 인덱스 $i_k$). $\delta_{k+1} = \delta_k - \eta^1_k(g_k(\theta^1_k) - g_k(\theta^2_k)) - (\eta^1_k - \eta^2_k) g_k(\theta^2_k)$이므로 $\|\delta_{k+1}\| \le (1 + \eta^1_k L)\|\delta_k\| + |\eta^1_k - \eta^2_k|\,G$. $\prod_k (1 + \eta^1_k L) \le \exp(L \sum \eta^1_k)$로 이산 Grönwall 전개하면 끝. $\square$

따름정리: 두 budget의 스케줄이 창 $[0,T]$에서 일치하면 궤적이 (공유 randomness 하에) 정확히 일치하고, 학습 곡선 decoupling은 $\bar{D}_T$에 비례하는 양으로 상한된다.

**실측 metric과의 정합.** `paper/analyze`의 SLCD 구현은 Savitzky-Golay 평활 후 공통 창에서 $\frac{1}{T}\sum_{t \le T} |L^1_t - L^2_t| / (L^1_t + L^2_t)$를 계산한다. 즉 실측 SLCD는 학습 곡선의 정규화 $L^1$ prefix 거리이고, 명제 1이 그 비정규화 버전을 $\bar{D}_T$로 상한한다 (2026-06-11 구현 확인). 명제 1의 지수 인자는 일반 smooth 함수에서는 피할 수 없지만, quadratic에서는 다음과 같이 제거된다.

**명제 1' (Quadratic에서의 타이트 bridge).** noiseless quadratic $f = \tfrac12\sum\lambda_i\theta_i^2$에서 모드 $i$의 기대 손실 기여 $m_{t,i} = \theta_{0,i}^2\prod_{k<t}(1-\eta_k\lambda_i)^2$는

$$\left|\log m^1_{t,i} - \log m^2_{t,i}\right| \;\le\; \frac{2\lambda_i}{1 - \eta_{\max}\lambda_i}\,\sum_{k<t}|\eta^1_k - \eta^2_k| \;=\; \frac{2\lambda_i}{1-\eta_{\max}\lambda_i}\,\bar{S}_t$$

즉 학습 곡선의 log-decoupling은 $L^1$ prefix deviation에 **선형**으로 (지수 인자 없이) 지배되며, $\eta\lambda \ll 1$ 모드에서는 사실상 등호다. *증명.* $|\log(1-a) - \log(1-b)| \le |a-b|/(1-\max(a,b))$를 각 step에 적용. $\square$ [수치 검증 2026-06-11: budgets 100 vs 200, hyperbolic/cosine, $\lambda \in \{0.005, 0.05, 0.5, 1\}$에서 bound 성립 및 tightness 비 0.982~1.000]

## 3. 정량적 불가능성: 삼각 부등식

**정리 1 (No free lunch).** 임의의 패밀리와 $N_1 < N_2$에 대해, 아무 가정 없이

$$D_{N_1}(N_1, N_2) + A(N_1) \;\ge\; H(N_1, N_2)$$

*증명.* $H = \eta_{N_2}(N_1) - \eta_{\min} = [\eta_{N_2}(N_1) - \eta_{N_1}(N_1)] + [\eta_{N_1}(N_1) - \eta_{\min}] \le D_{N_1} + A(N_1)$. $\square$

**따름정리 1.1 (질적 불가능성, v1의 정리 1).** full-window prefix invariance ($D \equiv 0$) + exact anchoring ($A \equiv 0$) + 강단조 annealing은 동시 불가. ($D = A = 0 \Rightarrow H = 0 \Rightarrow \eta_{N_2}(N_1) = \eta_{\min}$인데 강단조면 $\eta_{N_2}$는 $N_1 < N_2$에서 $\eta_{\min}$ 위에 있어야 하므로 모순.)

**따름정리 1.2 (퇴화 모서리).** $D = A = 0$을 고집하면 $H = 0$: 모든 budget이 최소 budget 안에 감쇠를 끝내고 바닥에 눕는 drop-then-floor 패밀리만 남는다. 연장 budget은 전부 $\eta_{\min}$에서 소비된다.

이 부등식이 논문의 중심 결과다. 세 양은 모두 실측 가능하고, 모든 스케줄러는 이 부등식을 어디서 얼마나 (그리고 등호로 포화하는지) 채우는지로 특성화된다.

## 4. 보존 법칙: 민감도 질량은 사라지지 않고 배치될 뿐이다

**정리 2 (민감도 질량 보존).** $N$에 대해 매끄러운 패밀리에서

$$M(N) = \int_0^N \partial_N \eta_N(t)\,dt = B'(N) - \eta_N(N)$$

*증명.* Leibniz 규칙: $B'(N) = \eta_N(N) + \int_0^N \partial_N \eta_N\,dt$. $\square$ [sympy 검증: cosine 양변 $\Delta\eta/2$; hyperbolic 수치 검증 상대오차 $\le 10^{-7}$]

**해석.** $B'(N)$은 budget 한 단위 추가가 사 오는 LR 예산 (연장 구간의 평균 LR 수준), $\eta_N(N)$은 끝단 LR. 그 차이만큼의 민감도 질량이 반드시 창 $[0, N]$ 어딘가에 분포해야 한다. 따라서 **스케줄러 설계는 2단계 선택이다: 질량 총량 $M(N)$ (얼마나 budget에 적응할 것인가)과 배치 프로파일 $\partial_N \eta_N(t)$ (그 적응을 창의 어디에 둘 것인가).** prefix invariance란 질량을 관측 창 뒤로 미는 것에 다름 아니다.

**따름정리 2.1 (Rescaling).** $\eta_N(t) = \eta_{\min} + \Delta\eta\,\tilde\varphi(t/N)$이면 $B(N) = N(\eta_{\min} + c_\varphi \Delta\eta)$, $c_\varphi = \int_0^1 \tilde\varphi$이므로 $M(N) = c_\varphi \Delta\eta$로 budget 무관 상수 (cosine, linear 모두 $\Delta\eta/2$). 배치도 자기닮음이라 조기 창 점유율이 고정된다.

**명제 2 (WSD의 좌표).** cooldown 비율 $c$, cooldown 프로파일 $\psi$ ($\psi(0)=1, \psi(1)=0$, $c_\psi = \int_0^1 \psi$)의 WSD: $\eta_N(t) = \eta_0$ for $t \le (1-c)N$, 이후 $\eta_{\min} + \Delta\eta\,\psi\!\big(\frac{t-(1-c)N}{cN}\big)$. $N_2 \ge N_1/(1-c)$이면

(a) $D_T = 0$ for $T \le (1-c)N_1$ (stable 구간 완전 invariance), 그러나 $D_{N_1} = \Delta\eta$ (가능한 최대값, cooldown 창에 전부 집중),
(b) $A \equiv 0$, $H(N_1, N_2) = \Delta\eta$, 삼각 부등식 등호 포화,
(c) $M(N) = \Delta\eta\,(1 - c(1 - c_\psi))$: $c$가 작을수록 $\Delta\eta$에 접근하는 **최대급 질량**을 마지막 $c$-구간에 몰아넣는 bang-bang 배치.

*증명.* (a) 직접 대입. (b) $\eta_{N_2}(N_1) = \eta_0$ (아직 stable). (c) $B(N) = (1-c)N\eta_0 + cN(\eta_{\min} + c_\psi\Delta\eta)$에서 $B'$ 계산, 정리 2 적용. $\square$

**분류 표 (트레이드오프 좌표, 모두 폐형식).**

| 패밀리 | $A(N)$ | $D_{N_1}$ (full) | 조기 창 $D$ | $M(N)/\Delta\eta$ | 질량 배치 | $\mathcal{E}(N)$ |
|---|---|---|---|---|---|---|
| rescaling (cos/lin/poly) | $0$ | $\ge \Delta\eta\tilde\varphi(N_1/N_2)$, 비소멸 | $\Theta(\cdot)$, 자기닮음 고정 | $c_\varphi$ (상수, cos/lin은 $\tfrac12$) | 전 구간 분산 | $-1$ |
| truncation (const, exp $\gamma$ 고정) | 통제 불가 | $0$ | $0$ | $0$ | 없음 | $0$ |
| WSD | $0$ | $\Delta\eta$ (최대) | $0$ (stable 구간) | $1 - c(1-c_\psi)$ (최대급) | 끝 $c$-구간 집중 (bang-bang) | $0$ (stable) |
| drop-then-floor | $0$ | $0$ | $0$ | $\approx 0$ | 없음 ($H=0$ 퇴화) | $0$ |
| **HyperbolicLR** | $\Delta\eta(1-\sqrt{1-(1-\rho)^2})$ | 폐형식 (정리 3d) | 최소 (아래 수치) | $0.29\sim0.36$ (최소급 비퇴화) | 매끄러운 후방 배치 (밀도 증가) | $-\frac{U-N}{2U-N}$ |

($\rho = N/U$. truncation과 drop-then-floor는 비퇴화 조건을 만족하지 못하는 경계 사례.)

## 5. HyperbolicLR의 폐형식 좌표

표기: $a = U - N$, $\rho = N/U$, $\eta_N(t) = \eta_0 + \Delta\eta(g_N(t) - g_N(0))$, $g_N(t) = \sqrt{(N-t)(2U-N-t)}/U$.

**보조정리 1 (Master 쌍곡선).** $g_N(t)^2 = \big((U-t)^2 - a^2\big)/U^2$. [sympy] 패밀리는 공통 점근선 $y = (U-t)/U$를 공유하는 쌍곡선 호들이고 $a$가 점근선까지의 거리다. budget 강건성은 rescaling이 아니라 이 점근 구조에서 나온다.

**보조정리 2 (편차 단조성).** $N_1 < N_2 \le U$이면 $t \mapsto \eta_{N_2}(t) - \eta_{N_1}(t)$는 $[0, N_1]$에서 $0$에서 출발해 단조 증가한다. 따라서 비음이고 $D_T$의 sup은 $t = T$에서 달성된다.

*증명.* $g_N'(t) = -(U-t)/(U^2 g_N(t))$. 보조정리 1에서 $a_1 > a_2 \Rightarrow g_{N_1}(t) < g_{N_2}(t)$이므로 $\frac{d}{dt}[\eta_{N_2} - \eta_{N_1}] = \Delta\eta\,\frac{U-t}{U^2}\Big(\frac{1}{g_{N_1}(t)} - \frac{1}{g_{N_2}(t)}\Big) > 0$ for $t < N_1$. $t = 0$에서 차이는 $0$. $\square$ [numpy 검증: 5개 budget 쌍 전수]

**정리 3 (좌표 일람).** $N \le U$에서:

(a) 강단조 감소. [sympy]
(b) $\dot\eta_N(0) = -\Delta\eta/\sqrt{N(2U-N)}$, $\mathcal{E}(N) = -\frac{U-N}{2U-N} \in (-\tfrac12, 0]$. [sympy]
(c) $A(N) = \Delta\eta\big(1 - \sqrt{1-(1-\rho)^2}\big)$, $\rho \to 1$에서 $0$. [sympy]
(d) $D_T(N_1, N_2) = \Delta\eta\big[(g_{N_1}(0) - g_{N_1}(T)) - (g_{N_2}(0) - g_{N_2}(T))\big]$ (보조정리 2로 sup이 창 끝에서 달성되므로 등호). [numpy 검증: 폐형식 = 수치 sup, 5쌍 × 3창]
(e) **삼각 부등식 등호 포화**: $D_{N_1}(N_1,N_2) + A(N_1) = H(N_1,N_2)$. [sympy 검증: 항등식] 즉 full-window 기준 낭비 없는 Pareto-효율 (linear, WSD도 포화; cosine은 mid-window sup 때문에 strict 부등). 효율적 스케줄러 간의 차이는 부등식이 아니라 **배치**에서 갈린다.
(f) 민감도 밀도 $\partial_N \eta_N(t) = \Delta\eta\,\frac{a}{U}\Big[\frac{1}{\sqrt{(U-t)^2-a^2}} - \frac{1}{\sqrt{U^2-a^2}}\Big] \ge 0$이고 $t$에 대해 단조 증가 (질량의 후방 배치). 총량은 $M(N) = \Delta\eta\,\frac{a}{U}\Big[\ln\frac{U+\sqrt{U^2-a^2}}{a} - \frac{N}{\sqrt{U^2-a^2}}\Big]$이고 $N \to U$에서 $0$. [sympy 부호; 폐형식은 quad 대비 상대오차 $<10^{-5}$로 5개 budget 전수 검증]
(g) $U$의 보간 역할: $U = N_{\max}$이면 anchoring 우선, $U \to \infty$이면 truncation으로 퇴화. 분류 표의 행들을 단일 knob으로 연속 보간.
(h) ExpHyperbolicLR은 $\log\eta$ 공간에서 동일 구조 ($\Delta\eta \to \log(\eta_0/\eta_{\min})$). 모든 결과 이식. [수치 검증]

**일반 창 하한 (rescaling과의 분리).** rescaling 패밀리는 $t = \beta N_1$에서 $D_{\beta N_1} \ge \Delta\eta\big[\tilde\varphi(\beta N_1/N_2) - \tilde\varphi(\beta)\big] > 0$이고 이 값은 budget 비에만 의존한다 (절대 규모를 키워도 비소멸). hyperbolic의 동일 창 deviation은 (d)의 폐형식으로 $(1-\rho)$ 차수로 통제된다.

## 6. 수치 검증 요약 ($\eta_0 = 10^{-2}$, $\eta_{\min} = 10^{-6}$, $U = 250$, 2026-06-11)

조기 창 $t \in [0, 0.2N_1]$ sup deviation ($\Delta\eta$ 단위): hyperbolic이 cosine 대비 3.1~4.2배 우위 (50→100: 0.0187 vs 0.0710; 50→200: 0.0285 vs 0.0893; 100→200: 0.0208 vs 0.0710; 150→200: 0.0099 vs 0.0410).

전체 창 sup deviation: hyperbolic이 대부분 우위이나 150→200에서 cosine 근소 우위 (0.2831 vs 0.2564). 주장은 조기/비례 창으로 한정한다.

민감도 질량 $M(N)/\Delta\eta$: hyperbolic 0.288 (N=50), 0.359 (100), 0.365 (150), 0.295 (200), 0.118 (240) vs rescaling 고정 0.5.

질량의 조기 창 (첫 20%) 점유율: hyperbolic 1.0~1.2%, cosine 2.5%, linear 4.0%. **총량이 작고 배치도 늦다**: 두 효과가 곱으로 조기 deviation을 줄인다.

anchoring error: $A(50) = 0.40\Delta\eta$, $A(100) = 0.20\Delta\eta$, $A(150) = 0.084\Delta\eta$, $A(200) = 0.020\Delta\eta$. $N \ll U$의 실질 가격이며 정리 1이 회피 불가능성을 보증한다.

## 7. 정확히 풀리는 동역학: loss 영역에서의 직접 검증 (2026-06-11 추가)

LR 곡선의 유사성은 설계의 결과라 그 자체로는 증거가 아니다. epoch insensitivity의 직접 증거는 **loss 궤적이 실제로 비슷하게 가는 것**이고, 명제 1은 그 연결의 정성적 보증일 뿐이다. 실제 딥러닝 실험 (Phase 2) 이전의 중간 층위로, 기대 loss가 정확히 계산되는 landscape에서 이를 보인다.

**설정.** $f(\theta) = \tfrac12\sum_i \lambda_i \theta_i^2$, power-law 스펙트럼 $\lambda_i = 1/i$ ($i = 1..200$), 모드별 gradient noise $\sigma_i^2 = c\,\lambda_i$. SGD의 기대 2차 모멘트는 모드별 재귀 $\mathbb{E}[\theta_{t+1,i}^2] = (1-\eta_t\lambda_i)^2\,\mathbb{E}[\theta_{t,i}^2] + \eta_t^2\sigma_i^2$로 닫혀서 **샘플링 없이 정확한 학습 곡선**을 준다. $\eta_0 = 0.8$, $\eta_{\min} = 8\times10^{-4}$, $U = 250$, WSD cooldown 20%, budget $\{50, 100, 150, 200\}$. 비교군에 **Constant ($\eta_0$ 고정)**를 포함한다: truncation 모서리의 살아 있는 대표. 구현: `paper/scripts/07_exact_dynamics.py`, 그림: `figs/exact_dynamics_loss_curves.png`, `figs/exact_dynamics_regime_map.png`.

**핵심 발견: 스케줄링 질문은 noise regime의 질문이다.** noise 계수 $c$를 쓸면 최종 loss의 승자가 세 regime으로 갈린다 (N=200 기준):

| noise $c$ | 승자 | 이유 |
|---|---|---|
| $\lesssim 0.02$ (약함) | **Constant** | noise floor가 무시 가능하면 감쇠는 손해다. 높은 LR이 bias를 가장 빨리 죽이고, prefix invariance도 자명하게 완벽 ($D = 0$) |
| $0.02 \sim 0.2$ (중간) | **WSD** | 높은 LR로 bias를 줄이고 마지막 cooldown으로 noise를 죽이는 분업이 최적 |
| $\gtrsim 0.2$ (강함) | **HyperbolicLR** (극단에선 linear/cosine) | WSD는 stable 구간 내내 최고 LR로 분산을 쌓는데 짧은 cooldown으로 다 못 턴다. 매끄러운 후방 감쇠가 분산을 점진 소산시키며 역전 |

noise가 약하면 constant가 모든 것을 이긴다는 사실 (2026-06-11 논의에서 지적됨)이 보정의 출발점이다: **annealing 스케줄러라는 물건 자체가 noise-dominated regime의 도구**이고, toy는 그 regime에서 평가해야 한다. 실제 minibatch SGD는 본질적으로 noisy하므로 딥러닝은 뒤쪽 regime에 산다고 보는 것이 자연스럽다 (Phase 2에서 확인할 가설).

**스케줄링이 유의미한 지점 ($c = 0.3$)에서의 결과.**

SLCD (정확 loss 곡선, 낮을수록 epoch-insensitive):

| Scheduler | 50-100 | 50-200 | 100-200 | 150-200 | 평균 |
|---|---|---|---|---|---|
| Constant | 0 | 0 | 0 | 0 | **0** (자명) |
| Cosine | 0.0150 | 0.0170 | 0.0255 | 0.0211 | 0.0196 |
| Linear | 0.0077 | 0.0076 | 0.0122 | 0.0100 | 0.0094 |
| WSD | 0.0132 | 0.0132 | 0.0257 | 0.0360 | 0.0220 |
| HyperbolicLR | 0.0067 | 0.0095 | 0.0130 | 0.0095 | **0.0097** |

최종 loss (N=200): **HyperbolicLR 0.2275** < WSD 0.2279 < Linear 0.2429 < Cosine 0.2496 < Constant 0.4483.

**해석.**
1. noise가 커지면 loss 수준이 순간 LR의 noise floor를 따라가므로, **WSD의 loss 영역 insensitivity가 붕괴한다** ($c=0.05$에서 SLCD 0.004 → $c=0.3$에서 0.022, cosine보다 나빠짐). 그림에서 WSD의 budget별 곡선은 각자의 cooldown 시점에 뿔뿔이 갈라진다. LR 영역에서 prefix-invariant라는 사실이 loss 영역 insensitivity를 보장하지 않는 regime이 존재한다는 것: 이것이 LR 곡선이 아니라 loss 곡선을 봐야 한다는 원칙의 정확한 실례다.
2. 같은 지점에서 hyperbolic은 SLCD 최저 (자명한 Constant 제외)이면서 최종 loss도 최저다: **noisy regime에서는 insensitivity와 최종 성능이 트레이드오프가 아니라 동시에 달성된다.** 매끄러운 후방 질량 배치가 분산 소산과 prefix 보존을 한꺼번에 해 주기 때문.
3. Constant는 분류 표의 truncation 모서리를 시각화한다: $D = 0$이지만 noise floor 아래로 내려갈 수 없다 (anchoring 실패가 loss로 직접 나타남). 정리 1의 세 욕구 중 둘을 완벽히 갖고 하나를 완전히 잃는 모서리해.
4. 발표 지점 $c = 0.3$은 선택이지만 cherry-picking이 아니다: regime map 그림이 $c \in [10^{-4}, 2]$ 전체를 공개하며, 각 regime의 승자와 crossover가 그대로 보인다. 논문에도 map 전체를 싣는다.
5. 한계: quadratic + 모드 독립 noise는 deep learning이 아니며, WSD cooldown 비율 (20%)을 키우면 강noise regime에서 WSD가 개선될 수 있다 ($c$-sweep과 cooldown-sweep의 2차원 map은 후속 작업). 이 층위는 명제 1과 Phase 2 사이의 다리이고, 최종 검증은 Phase 2다.

**Phase 2에 주는 검증 가능한 시그니처.** (i) 실제 task의 noise regime 추정: constant LR이 경쟁력 있으면 약noise, WSD가 압도하면 중간, hyperbolic이 따라잡거나 이기면 강noise. (ii) WSD 학습 곡선이 cooldown 위치에서 budget별로 갈라지는 패턴 (창 위치별 SLCD 분해로 검출). (iii) hyperbolic의 SLCD 우위가 noise 수준과 무관하게 유지되는지.

### 7.1 공정 튜닝 후의 모습 (hardening, 2026-06-11)

위 3-regime 서사는 **untuned** 비교다 (WSD cooldown 20% 고정, constant는 $\eta_0$ 고정). 리뷰어가 가장 먼저 칠 곳이므로 모든 스케줄러에 하이퍼파라미터 튜닝을 허용하고 재실행했다 (`scripts/08_hardened_toy.py`: $\eta_0$ 전원 grid 10점, WSD cooldown $\in \{0.1..0.7\}$, hyperbolic $U \in \{210..600\}$, (noise, budget)별 독립 튜닝). 결과 (`figs/tuned_regime_map.png`):

1. **튜닝된 WSD가 quadratic 세계의 대부분을 지배한다.** $N=200$ 기준 $c \in [5\times10^{-4}, 0.4]$ 전 구간 1위, 중간 noise에서 마진 12~57%. untuned에서 보였던 hyperbolic의 강noise 우승은 $c \gtrsim 0.9$의 마진 1% 미만 동률로 축소된다. constant의 영토는 $c \lesssim 10^{-3}$의 틈새로 줄어든다.
2. **스펙트럼 robustness** ($\lambda_i = i^{-\alpha}$, $\alpha \in \{0.5, 1, 2\}$): 승자는 $\alpha$에 따라 WSD/Cosine/HyperbolicLR로 흔들리고 마진은 대부분 수 % 이내. quadratic 최종 성능에서 어떤 스케줄러의 일관 우위도 주장할 수 없다.
3. **naive HPO transfer는 toy에서 변별력이 없다.** $N=50$에서 튜닝해 $N=200$에 재인스턴스 배포하는 프로토콜에서 cosine의 regret이 전 noise에서 0.00%다 (최적 $\eta_0$가 budget에 불변). hyperbolic은 0~5.8%로 오히려 열세. **이유**: 이 프로토콜은 "완주한 정확한 최종 loss"로 튜닝하므로, prefix invariance가 도움이 되는 실제 메커니즘 (부분 학습 곡선을 보고 내리는 튜닝 결정: Optuna pruning, early stopping, 저epoch HPO trial)이 toy에 존재하지 않는다.

**이 hardening이 논문에 주는 것.** (a) toy의 역할을 정직하게 한정한다: toy가 입증하는 것은 **insensitivity 메커니즘** (명제 1', SLCD 순서, WSD의 loss 영역 insensitivity 붕괴)이지 최종 성능 우위가 아니다. 최종 성능 우위 주장은 논문에서 하지 않는다 (parity가 목표). (b) Phase 2의 HPO transfer 프로토콜을 정밀화한다: 변별 메커니즘이 "부분 곡선 기반 튜닝 결정"이므로, **저epoch HPO (10 epochs/trial) + Optuna median pruning을 켠 상태에서 튜닝 → 전체 epoch 배포**가 올바른 프로토콜이다 (`run_all.py`의 기존 구조와 일치, pruning 활성화 확인 필요). cosine은 부분 곡선이 budget별로 다르게 보이므로 pruning이 좋은 구성을 잘못 죽이는 비율이 높아진다는 것이 검증 가설. (c) WSD 대비 차별점은 최종 성능이 아니라 insensitivity + anytime성이라는 포지셔닝이 확정된다.

### 7.2 Anytime 프로토콜: 두 번째 기둥의 정량화 (2026-06-11)

**게임 정의.** 중단 시점 $\tau \in [50, 200]$를 모르는 채로 각 스케줄러가 단 하나의 구성 (자기 하이퍼파라미터 + horizon 추정 $N_{plan}$, $t \ge N_{plan}$ 이후 마지막 LR 유지)을 사전에 commit하고 한 번 달린다. 평가는 anytime regret $R(\tau) = L(\tau)/L_{oracle}(\tau) - 1$, oracle은 $\tau$를 알고 전 패밀리를 튜닝한 최적. 각 contender는 자기 grid에서 **minimax regret**을 최소화하는 구성을 고른다 (전원 공정 튜닝). WSD는 stop (그냥 중단)과 branch (중단 요청 시 checkpoint에서 $0.2\tau$ cooldown 추가 실행) 두 변형. 구현: `scripts/09_anytime_protocol.py`, 그림: `figs/anytime_regret.png`.

**결과 ($c = 0.3$).**

| Contender | minimax regret | mean regret | 비고 |
|---|---|---|---|
| Constant | 75.1% | 67.2% | |
| Cosine | 35.5% | 22.5% | $N_{plan}=175$ commit |
| Linear | 31.6% | 23.6% | $N_{plan}=175$ |
| WSD-stop | 49.9% | 32.8% | cooldown 밖 중단에 취약 |
| **HyperbolicLR** | **29.9%** | **17.7%** | $N_{plan}=100$, $U=1.2N_{plan}$ |
| WSD-branch | 9.1% | 7.5% | **+20% 컴퓨트, checkpoint 인프라 필요** |

robustness: $c=0.05$에서 minimax 순서 Hyperbolic 39% < Linear 43% < Cosine 50% < WSD-stop 78%; $c=1.0$에서 Hyperbolic 15% = Linear 15% < Cosine 20% < WSD-stop 24%. 전 noise에서 hyperbolic이 단일 런 1위 또는 동률이고, 우위는 minimax보다 mean regret에서 뚜렷하다 (regret 골짜기가 넓고 깊음: 그림 참조).

**해석.** (i) **anchoring error가 anytime 게임에서는 자산으로 반전된다**: hyperbolic의 committed 구성은 중간 horizon에 $N_{plan}$을 두고 높은 tail LR을 유지하므로, $\tau > N_{plan}$에서도 계속 전진한다. cosine은 $N_{plan}$에서 $\eta_{\min}$으로 떨어진 뒤 사실상 정지한다. (ii) WSD-stop의 V자 regret (cooldown 창 밖 중단에 취약)이 명제 2(a)의 anytime 버전이다. (iii) WSD-branch가 전체 1위라는 것도 그대로 공개한다: 단 +20% 컴퓨트와 checkpoint 분기 인프라가 전제이고, 이 비용을 받아들일 수 있는 환경 (대규모 LLM 사전학습)과 아닌 환경 (자동화된 HPO 루프, 소규모 반복 실험)을 구분하는 것이 올바른 서술이다. (iv) 주의: hyperbolic은 knob이 2개라 탐색 자유도가 크다 (336 구성 vs cosine 84). 이는 방법의 내재적 특성이지만 본문에 명시한다.

### 7.3 Free-form anytime 최적해: 기하 감쇠와 ExpHyperbolicLR의 승격 (2026-06-11)

§7.2의 게임에서 패밀리 가정을 버리고 **LR 수열 200개 자체를 최적화**했다 (`scripts/10_freeform_anytime.py`, `11_dense_anytime.py`, `12_exp_family_anytime.py`; torch 미분 가능 정확 동역학, 초기화 3~4종 수렴 확인). 핵심 그림: `figs/eh_anytime.png` (`scripts/13`): log 스케일에서 free-form 최적해와 EH가 나란한 직선 (기하 감쇠)이고 H는 꺾여서 평탄해지며 cosine은 끝에서 추락한다. regret 곡선에서 EH의 골짜기가 H/cosine보다 넓고 낮다.

**결과 사다리 (dense 게임: $\tau$ = $[50,200]$의 모든 step, $c=0.3$, 전원 minimax 공정 튜닝).**

| Schedule | minimax | mean | 비고 |
|---|---|---|---|
| Free-form 최적 | **9.3%** | 8.6% | 게임 값의 추정치. WSD-branch (9.1%, +20% 컴퓨트)와 추가 컴퓨트 없이 동급 |
| Exponential (plateau+기하) | 21.1% | 11.3% | |
| **ExpHyperbolicLR** | 21.5% | **10.7%** | 명명 패밀리 중 mean 1위 |
| Sawtooth (envelope+주기 dip) | 27.8% | 18.2% | |
| HyperbolicLR | 29.9% | 17.0% | |
| Linear / Cosine | 31.6 / 35.6% | 23.4 / 21.9% | |
| WSD-stop | 49.9% | 32.3% | |

**최적해의 구조.** (i) sparse $\tau$ grid (10 step 간격)에서는 최적해가 **톱니** (각 중단 후보 직전 mini-cooldown 후 재상승, minimax 7.3%)다. dense grid에서는 톱니가 사라지므로 이는 grid 구조의 반영이지 아티팩트가 아니다: **중단이 checkpoint 시점에만 가능한 환경에서는 checkpoint 동기화된 cyclic schedule (SGDR/CyclicLR exp_range와 동형)이 최적**이라는 독립적 발견이며, warm restart 계열의 새로운 이론적 정당화다. (ii) dense grid에서는 최적해가 **plateau → $\tau_{\min}$ 직전 notch → log 스케일 직선 (기하 감쇠)**이고, regret 곡선이 전 $\tau$에서 8~9%로 평탄하다 (minimax equalizer 성질). 직관: 모든 중단 시점이 같은 상대적 annealing 상태를 보려면 schedule이 시간 이동에 자기닮음, 즉 기하적이어야 한다.

**Wide-window 검증 (2026-06-11, `scripts/14_wide_window.py`).** "기하 감쇠" 판정이 좁은 창의 아티팩트가 아닌지 $\tau \in [20, 640]$ (동적 범위 32배)에서 재검: 지수 fit residual 0.034 vs power-law ($t^{-1.18}$) fit residual 0.101로 **지수가 3배 우세**, log-log에서 뚜렷한 아래 굽음 (`figs/wide_window_shape.png`). 명제 3이 단일 모드의 답을 $1/(\lambda t)$로 확정했으므로, 다중 모드 (스펙트럼)에서 기하로 바뀌는 전이는 실재하는 현상이다 (왜 기하인지의 해석적 해명은 open). 부수 관찰: (i) 창 끝 직전 급강하는 경계 효과 (더 늦은 $\tau$가 없으므로 $\tau_{\max}$에 정면 annealing), (ii) $\tau$ grid 간격 5의 잔물결이 남아 sparse-grid 톱니와 dense 매끄러움 사이를 보간, (iii) 게임 값은 창 비율과 함께 증가 (4배 창 9.3% → 32배 창 16.7%). **정직성 경고**: family 기반 oracle이 true 최적 제어 oracle보다 $\tau=40/160/640$에서 1.8/3.3/14.3% 느슨함을 확인. 정성 결론 (순서, 모양)은 불변이나 논문의 최종 regret 수치는 true oracle로 재산출해야 한다.

**함의.**
1. **ExpHyperbolicLR의 승격**: 기하 감쇠의 anchored 폐형식 버전이 EH다 ($\log\eta$가 hyperbolic 곡선, $N_{plan} \to U$에서 정확히 기하). 지금까지 H의 변형으로 취급된 EH가 anytime 이론의 정답 형태에 가장 가까운 우리 방법이며, cosine 대비 anytime regret을 minimax 40%/mean 51% 줄인다. 논문에서 EH를 "horizon 불확실 환경의 권장 기본값"으로 전면 배치한다.
2. **정직한 gap**: 명명 패밀리 최선 (21%)과 게임 값 (9.3%) 사이에 2배 격차가 남는다. free-form이 더 잘하는 구조는 $\tau_{\min}$ notch와 가변 log-기울기다. 이 gap의 폐형식 해는 open problem으로 명시하고, 새 스케줄러 설계 공간으로 제안한다.
3. Phase 2의 anytime 실측에서 EH를 주 대상으로 승격하고, checkpoint 환경용 cyclic 변형 (CY는 코드에 이미 있음)을 부 대상으로 추가한다.

### 7.4 경로 A: anytime 게임의 해석적 구조와 문헌 포지셔닝 (2026-06-11)

**명제 3 (단일 모드: anytime 긴장 부재).** 단일 곡률 $\lambda$, noise $\sigma^2$의 모드 동역학 $m_{t+1} = (1-\eta\lambda)^2 m_t + \eta^2\sigma^2$에서:
(a) horizon $\tau$가 알려진 최적 제어는 greedy다 ($m_{t+1}$이 $m_t$에 단조이므로 DP가 단계별 최소화로 붕괴).
(b) greedy 스케줄은 $\eta^*_t = \dfrac{\lambda m_t}{\sigma^2 + \lambda^2 m_t}$이고, $u_t = \lambda^2 m_t/\sigma^2$로 치환하면 $u_{t+1} = u_t/(1+u_t)$, 즉 $u_t = \dfrac{u_0}{1 + t\,u_0}$ **정확히**. 따라서 $\eta^*_t \to 1/(\lambda t)$ (고전적 Robbins-Monro/강凸 SGD rate).
(c) $\eta^*$가 $\tau$에 의존하지 않으므로 **minimax anytime regret = 0**: 단일 곡률에서는 horizon 불확실성이 비용을 만들지 않는다.
[수치 검증: 폐형식 상대오차 $1.7\times10^{-15}$; greedy vs free-form 최적 제어 regret $\sim 10^{-7}$ at $\tau \in \{50,100,200\}$]

**정리 4 (스펙트럼 긴장; 2026-06-11 증명 완성).** 두 모드 $\lambda_1 > \lambda_2 > 0$, $\sigma_1^2 > 0$, $\sigma_2^2 = 0$, $m_2(0) > 0$, 스케줄 클래스 $\eta_t \in [0, \bar\eta]$, $\bar\eta\lambda_1 < 1$. 충분조건 (C): $\bar\eta\sigma_1^2 > \lambda_1\bar m_1 + (\lambda_2^2/\lambda_1)m_2(0)$, 여기서 $\bar m_1 = \max\big(m_1(0),\, \bar\eta\sigma_1^2/(\lambda_1(2-\bar\eta\lambda_1))\big)$ ($m_1$의 불변 상계; $m_1(0)$이 noise floor 이하면 (C)는 $\bar\eta\sigma_1^2\frac{1-\bar\eta\lambda_1}{2-\bar\eta\lambda_1} > \frac{\lambda_2^2}{\lambda_1}m_2(0)$로 단순화). 그러면 임의의 $1 \le \tau_1 < \tau_2$에 대해 **어떤 단일 스케줄도 $\tau_1$과 $\tau_2$에서 동시에 최적일 수 없고, 따라서 minimax anytime regret $V(\tau_1, \tau_2) > 0$** (스케줄 공간이 compact하고 loss가 연속이므로 $V$는 달성되는 min이고, 0이려면 동시 최적 스케줄이 존재해야 함).

*증명.* $J_\tau = \sum_i w_i m_i(\tau)$, $w_i = \lambda_i/2$에 대해 연쇄법칙으로

$$\frac{\partial J_\tau}{\partial \eta_t} = 2\sum_i w_i\,\Phi_i^{(\tau)}(t)\,B_i(t), \qquad \Phi_i^{(\tau)}(t) = \prod_{s=t+1}^{\tau-1}(1-\eta_s\lambda_i)^2, \quad B_i(t) = \eta_t\sigma_i^2 - \lambda_i(1-\eta_t\lambda_i)m_i(t).$$

동시 최적 스케줄 $\eta$가 존재한다고 가정하자. $t = \tau_1 - 1$을 보자.

(1) *내부성*: $\eta_t = 0$이면 $\partial J_{\tau_1}/\partial\eta_t = -2\sum w_i\lambda_i m_i(t) < 0$ ($m_2(t) = m_2(0)\prod(1-\eta_s\lambda_2)^2 > 0$이 $\bar\eta\lambda_2 < 1$에서 보장)이므로 $\tau_1$-최적성 모순. $\eta_t = \bar\eta$이면 $\partial J_{\tau_1}/\partial\eta_t \ge 2w_1[\bar\eta\sigma_1^2 - \lambda_1\bar m_1 - (\lambda_2^2/\lambda_1)m_2(0)] > 0$ (조건 (C); $\bar m_1$의 불변성은 $\eta\sigma_1^2 \le \bar m_1\lambda_1(2-\eta\lambda_1)$ 한 줄 귀납)이므로 역시 모순. 따라서 $\eta_t$는 내부점이고 두 문제 모두에서 stationarity 등식이 성립한다.

(2) *두 stationarity의 충돌*: $\tau_1$ 문제에서 $\Phi_i^{(\tau_1)}(t) = 1$이므로 $w_1B_1 + w_2B_2 = 0$. $\tau_2$ 문제에서 $\Phi_i^{(\tau_2)}(t) = F_i := \prod_{s=\tau_1}^{\tau_2-1}(1-\eta_s\lambda_i)^2$이므로 $w_1F_1B_1 + w_2F_2B_2 = 0$. 두 식에서 $w_1B_1(F_1 - F_2) = 0$. 그런데 $\sigma_2^2 = 0$이므로 $B_2 = -\lambda_2(1-\eta_t\lambda_2)m_2(t) < 0$ 엄격, 첫 식에서 $w_1B_1 = -w_2B_2 > 0$ 엄격. 따라서 $F_1 = F_2$.

(3) *연장 구간의 퇴화*: $\eta_s \in (0, \bar\eta]$이면 $0 < 1-\eta_s\lambda_1 < 1-\eta_s\lambda_2 < 1$이므로 $F_1 \le F_2$이고 등호는 $\eta_s = 0\ \forall s \in [\tau_1, \tau_2)$일 때뿐. 그 경우 $L(\tau_2) = L(\tau_1) = L^*(\tau_1)$인데, oracle은 엄격히 개선된다: $\tau_1$-최적 상태에서 $\eta = \varepsilon$ 한 step의 1차 효과가 $-2\sum w_i\lambda_i m_i(\tau_1) < 0$이므로 $L^*(\tau_2) \le L^*(\tau_1 + 1) < L^*(\tau_1)$. 따라서 $\tau_2$-최적성 모순. $\square$

**따름정리 4.1.** $V$는 $\sigma_2^2$에 연속이므로 (compact 스케줄 공간 위 균등 연속), $\sigma_2^2 = 0$에서 $V > 0$이면 충분히 작은 $\sigma_2^2 > 0$에서도 $V > 0$.

[수치 검증 2026-06-11: $\lambda = (0.5, 0.05)$, $\sigma_1^2 = 0.05$, $\tau = (60, 240)$에서 $V = 12.93\%$; $\tau_1$-최적 스케줄에서 $\eta_{\tau_1-1} = 0.056$ 내부, stationarity 잔차 $2\times10^{-12}$, $B_2 = -5.2\times10^{-4} < 0$, $w_1B_1 = -w_2B_2$ 정확히 균형, $F_1 = 5.9\times10^{-3} < F_2 = 6.0\times10^{-1}$ 엄격. (C)는 충분조건일 뿐이며 수치상 $V > 0$은 (C) 밖에서도 성립. 일반 스펙트럼 비 의존성: 비 $1, 4, 16, 64$에서 regret $0.00, 2.33, 5.67, 7.37\%$ (비 1의 0%는 명제 3의 대조군)]

**귀결: §7.2~7.3에서 관찰된 모든 anytime 비용의 근원은 noise가 아니라 곡률 스펙트럼의 다양성이며, 이것이 이제 정리다.** 단일 곡률이면 비용이 정확히 0 (명제 3), 곡률이 둘 이상 갈라지는 순간 양수 (정리 4). Codex 리뷰가 "논문의 운명을 가르는 한 가지"로 지목한 항목이 해결되었다.

**문헌 포지셔닝 (2026-06-11 조사).** anytime 학습 영역은 2025~26년에 급속히 점유되었다:
- Schaipp, Hägele, Taylor, Şimşekli, Bach (ICML 2025, arXiv:2501.18965): 비매끄러운 convex bound로 WSD/cosine 거동, cooldown 최적 길이, 연장 학습 LR 전이를 재현. "convex 이론으로 스케줄 설명" 프레임 선점.
- Anytime Pretraining (arXiv:2602.03702, 2026-02): constant/$1/\sqrt{t}$ + **weight averaging**으로 horizon-free하게 cosine 동급 달성 (150M~300M LM). 선형 회귀 minimax rate 이론.
- ScheduleFree+ (arXiv:2605.19095, 2026-05): optimizer 쪽 anytime, 장기 학습에서 WSD 대비 큰 우위 주장.

**우리 기여의 생존 영역과 차별점**: (i) 이들은 점근 rate 이론이고, 우리는 **유한 창 minimax regret 게임** (oracle 대비 상수 수준의 정량화, equalizer 구조, 게임 값 계산)이다. (ii) averaging 계열이 강한 이유가 우리 프레임워크로 설명된다: 정리 1은 LR 궤적의 제약인데 averaging은 학습 LR과 평가 지점을 분리해 게임을 우회한다 (사실상 컴퓨트 무료 WSD-branch; WSD-branch regret 9.1% ≈ pure-schedule 게임 값 9.3%의 일치가 방증). (iii) 명제 3~4의 특성화 (단일 곡률 무긴장 / 스펙트럼 긴장)는 어느 문헌에도 없는 형태다. (iv) 방법 주장은 "averaging이 부적합한 환경 (drop-in 스케줄러, 메모리 제약, fine-tuning 파이프라인)에서의 pure-schedule 처방 = EH"로 한정한다.

**Codex (GPT-5.5) 적대적 리뷰 반영 (2026-06-11).** 추가로 확인된 선행연구와 판정:
- **Schaul et al. 2013** ("No more pesky learning rates"): noisy separable quadratic에서 greedy 최적 LR을 이미 유도. 명제 3의 greedy 부분은 재발견이고, 신규성은 "유한 horizon DP와의 동치 + zero anytime regret" 표현뿐. 명제 3은 정리가 아니라 **보조정리로 강등**하고 Schaul 인용 필수.
- **Ge et al. 2019** (least squares에서 geometric step decay의 near-optimality, anytime final-iterate의 미묘함): §7.3의 "기하 감쇠 최적" 관찰과 직접 겹침. 우리 신규성은 "stop-time minimax 게임에서의" 기하 출현으로 한정해야 하며 Ge 인용 필수.
- **Eigencurve** (Hessian 스펙트럼에서 스케줄 설계): "스펙트럼 인지 스케줄"의 선행. 명제 4의 신규성은 "스펙트럼 다양성 = anytime 비용의 원천"이라는 인과 특성화로 좁혀짐. **수치만으로는 부족, 정식 증명이 논문 운명을 가름** (Codex 평가: "the one thing that can make the paper more than empirical schedule archaeology").
- **Wen et al. 2024 (arXiv:2410.05192)**: WSD의 branchable anytime성이 이미 명시적. 관찰 2/명제 2의 포지셔닝에 인용 필수.
- **REX (arXiv:2107.04197)**: budget 미지 LR 스케줄링 문제 설정 선행. Phase 2 baseline에 추가.
- 게임 정식화 (commit + 적대적 stop time + oracle regret + 유한 창 상수)는 Codex도 선행을 모름: **생존하는 핵심 신규성**. 단 $D+A\ge H$와 보존 법칙은 "측정 언어"로 강등 (tautological 공격 대응).
- 제목 제안: "Committed Learning-Rate Schedules Under Horizon Uncertainty". EH는 "universal optimum"이 아니라 "closed-form drop-in approximation"으로 서술.

## 8. 변분 관점: 솔직한 결론

"주어진 anchoring 예산에서 조기 deviation을 최소화하는 스케줄은 무엇인가"의 답은 hyperbolic이 아니다. **고정 master prefix + budget 의존 cooldown** (WSD와 그 일반화: prefix를 상수 대신 임의의 고정 감쇠 곡선으로) 구성이 조기 창 deviation을 정확히 $0$으로 만들면서 $A = 0$을 달성한다. 기하만 보면 조각적 구성이 변분 문제의 bang-bang 최적해다.

따라서 hyperbolic의 가치 주장은 "기하적 유일 최적"이 아니라 다음 조합이다: (i) 강단조 annealing을 유지하는 매끄러운 ($C^1$) 해 중에서 질량 총량과 조기 배치가 최소급, (ii) cooldown 트리거라는 이산 결정이 불필요 (WSD는 cooldown 시작 시점에서 사실상 budget을 알아야 한다: horizon 문제의 해소가 아니라 지연), (iii) 모든 좌표가 폐형식이라 $U$ 선택 시점에 비용을 사전 계산 가능, (iv) 어느 쪽이 실제 학습 성능 (HPO transfer, 최종 loss)에서 이기는지는 이론이 결정 못 하는 실증 질문이고 우리는 정확히 그것을 측정한다. 이 서술이 "왜 그냥 WSD를 쓰지 않는가"라는 리뷰어 공격에 대한 정면 답변이다.

미해결 질문 (논문에 명시): $C^1$ 강단조 패밀리로 제한했을 때, 주어진 $A$ 예산 하에서 조기 질량을 최소화하는 변분 문제의 해가 hyperbolic 형태인가. (점근선 공유 구조가 답일 것이라는 추측까지만.)

## 9. 이론이 예측하고 실험이 검증할 것

| 이론 | 예측 | 실험 |
|---|---|---|
| 명제 1, 1' + §6 | 조기 창 SLCD 순서: H/EH < WC/C/P/L | Phase 2 SLCD (5 seeds) |
| §7.1(3) | 부분 곡선 기반 튜닝에서만 prefix invariance가 HPO transfer 이득으로 전환 | **HPO transfer (주 실험, 프로토콜 정밀화)**: 저epoch HPO + median pruning 활성 상태에서 튜닝 → 전체 epoch 배포. pruning 오살률 (좋은 구성이 잘못 잘리는 비율)을 스케줄러별로 측정 |
| 정리 3(c) | $A(N)$ 큰 설정에서 정밀 수렴 손해 | 최종 성능 parity 검정, $A(N)$과 격차의 상관 |
| 명제 2(a) | WSD의 SLCD: stable 구간 0, cooldown 창에서 점프 | 창 위치별 SLCD 분해 측정 (신규 분석, 추가 실험 불필요: 기존 학습 곡선 재분석) |
| 정리 3(g) | $U$ 민감도는 $(1-\rho)$로 사전 예측 가능 | $U \in \{200,250,300,400\}$ ablation |
| 정리 2 | 민감도 질량 배치가 학습 곡선 decoupling 위치를 결정 | 학습 곡선 차이의 시간 분포 vs $\partial_N\eta$ 프로파일 대조 |

## 10. 논문 구조 (개정)

1. **Introduction**: horizon 불확실성 → $D, A, H$ 세 욕구 → 정리 1 한 줄 → 기여: (i) 트레이드오프 부등식과 보존 법칙, (ii) 기존 스케줄러의 (질량, 배치) 분류, (iii) 폐형식 좌표를 갖는 매끄러운 해 HyperbolicLR, (iv) HPO transfer 실증.
2. **The Tradeoff and the Conservation Law**: 정의, 명제 1, 정리 1, 정리 2, 따름정리들, 분류 표.
3. **Hyperbolic Schedules as Smooth Mass Placement**: 보조정리 1~2, 정리 3, 명제 2와의 대조, §8의 솔직한 변분 논의.
4. **Exact Dynamics**: §7의 quadratic 정확 동역학 (Figure 2 후보). 명제 1' 검증, tuned regime map (§7.1), anytime regret 프로토콜 (§7.2)까지 포함: toy 층위에서 두 기둥 (insensitivity, anytime)을 모두 정량화.
5. **Experiments**: SLCD와 질량 배치 검증, pruning 기반 HPO transfer (주 결과 1), anytime regret 실측 (주 결과 2), 최종 성능 parity (WSD/SF/WC 포함), $U$ ablation.
6. **Related Work**: 분류 표의 좌표로 통일 서술 (rescaling 계열, WSD/cooldown 계열, schedule-free/LR-free 계열, REX).
7. **Limitations**: $A(N)$의 가격, 전체 창 비우위 사례, 변분 비최적성 (§8), toy에서의 WSD 우위 (§7), convex last-iterate 이론 (decay-to-zero 최적성)과의 긴장, 실험 스코프.

## 11. 남은 작업

이론 (실험 불필요):
1. ~~정리 1 정량화~~ (완료: 삼각 부등식), ~~보존 법칙~~ (완료), ~~보조정리 2 증명~~ (완료), ~~WSD 정식화~~ (완료), ~~SLCD 구현 정합 확인~~ (완료).
2. ~~$M(N)$ 폐형식 확인~~ (완료: log 표현으로 정리, 수치 전수 일치).
3. 명제 1의 momentum/AdamW 확장 또는 "SGD에 한정" 명시 결정.
4. ~~이론 그림~~ (완료: `paper/scripts/06_theory_figures.py` → `figs/theory_tradeoff_overview.png`. 4-panel: deviation 곡선, 민감도 밀도 배치, $A(N)$ 곡선, $M(N)$ 총량. 모든 곡선이 폐형식과 일치 확인: cosine/linear 0.5, WSD 0.8667, hyperbolic 0.3592).
5. 변분 문제의 정확한 진술: $\mathcal{F} = \{C^1$ 강단조, $\eta_N(0)=\eta_0$, $A(N) \le \alpha\}$에서 $\inf_{\{\eta_N\} \in \mathcal{F}} \sup_{N_1<N_2 \le N_{\max}} D_{\beta N_1}(N_1, N_2)$의 최소해가 점근선 공유 (쌍곡선형) 구조인가. 조각적 패밀리를 허용하면 하한 $0$이 달성되므로 (§8) $C^1$ 강단조 제약이 본질적이다.
6. ~~anytime성의 정식화~~ (완료: §7.2의 minimax anytime regret 프로토콜. toy에서 hyperbolic이 단일 런 1위 확인. Phase 2에서 동일 프로토콜을 실제 NN에 적용하는 것이 남은 작업: 중단 시점 grid에서 val loss 측정, WSD-branch는 cooldown 재실행 컴퓨트를 기록).
7. ~~Part B의 0.00% regret이 grid 아티팩트인지 확인~~ (완료: 60점 fine grid에서도 cosine 0~1.1%, hyperbolic 2.5~5.4%로 결론 유지. 단 cosine 최적 $\eta_0$가 $c \le 0.3$에서 grid 상한 (1.9, 안정성 한계 2 직전)에 닿으므로 boundary 효과 명시 필요).
8. (확인 완료, 작업 남음) `run_all.py`의 Optuna에 pruner가 **없음**을 확인 (2026-06-11 grep). 변별 메커니즘인 부분 곡선 기반 튜닝 결정이 현재 파이프라인에 부재. Phase 2 전에 MedianPruner 추가 + "pruning 오살률" 측정 코드 작성 (GPU 불필요한 코드 작업).
9. (경로 A 파생) 논문 최종 regret 수치를 family oracle 대신 **true 최적 제어 oracle**로 재산출 (§7.3 경고 참조; free-form per-$\tau$ 최적화, 코드는 `scripts/14`의 `true_oracle_at` 재사용).
10. ~~명제 4의 정식 증명~~ (**완료 2026-06-11**: 정리 4로 승격, Pontryagin식 1차 조건 충돌 논법, 전 단계 수치 검증. 남은 세부: (a) $\sigma_2^2 > 0$ 일반 케이스의 직접 증명 (현재는 연속성 따름정리), (b) 정량 하한 $V \ge f(\lambda_1/\lambda_2)$의 폐형식, (c) "스펙트럼 → 최적 모양" 사상은 open problem으로 명시).
11. (Codex 리뷰 파생) 인용/포지셔닝 필수 추가: Schaul et al. 2013 (명제 3 greedy 선행), Ge et al. 2019 (geometric decay 선행), Eigencurve, Wen et al. 2024 (WSD anytime), REX. Phase 2 baseline에 REX와 D2Z (linear decay-to-zero) 추가 검토.
12. (Codex 리뷰 파생) HPO 오살률의 정확한 정의 확정: "저budget에서 prune된 구성 중 full budget이면 top-k에 들었을 비율"을 스케줄러별로 측정. Hyperband/ASHA와의 차별점은 측정이지 게임 해석이 아님을 명시.

실험 의존 (Phase 2 이후): §8의 표 전체.
