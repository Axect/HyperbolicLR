# ExpHyperbolicLR의 cross-budget 설계 이론 (v3)

작성: 2026-06-11, v3 전면 재정립 2026-08-18. 목적은 HyperbolicLR과 ExpHyperbolicLR의 수학을 cross-budget HPO transfer라는 실무 문제에 맞춰 정리하는 것이다. 기존의 트레이드오프와 보존 법칙은 schedule coordinate를 제공하지만 논문의 주인공은 아니다. 본문 기여는 log-space hyperbolic 설계와 그 실증이며, stop-time game과 spectral tension은 appendix 또는 별도 이론 작업으로 분리한다.

## 0. 중심 주장과 이론의 한계

> 작은 budget에서 선택한 learning-rate 설정을 긴 horizon에 그대로 배포하려면 초기 trajectory의 fidelity와 추가 budget의 utility가 모두 필요하다. Hyperbolic geometry는 horizon 변화에 따른 curve deformation을 뒤쪽에 매끄럽게 배치하고, ExpHyperbolicLR는 이 geometry를 log LR에 적용해 multiplicative anchoring을 제공한다. 이 설계가 실제 optimization에서 H, Power Scheduler, WSqD보다 나은지는 정리가 아니라 replicated HPO experiment가 판정한다.

## 1. 형식 체계

**설정.** 스케줄러 패밀리는 budget $N \in \mathcal{N} \subset \mathbb{R}_{>0}$마다 LR 곡선 $\eta_N : [0, N] \to [\eta_{\min}, \eta_0]$를 주는 사상. $\Delta\eta = \eta_0 - \eta_{\min}$, $\eta_N(0) = \eta_0$.

**정의 1 (Prefix deviation).** $N_1 < N_2$, 창 $T \le N_1$에 대해 $D_T(N_1, N_2) = \sup_{t \le T} |\eta_{N_1}(t) - \eta_{N_2}(t)|$, $L^1$ 버전 $\bar{D}_T = \int_0^T |\eta_{N_1} - \eta_{N_2}|\,dt$.

**정의 2 (Anchoring error).** $A(N) = \eta_N(N) - \eta_{\min} \ge 0$.

**정의 3 (Tail headroom).** $H(N_1, N_2) = \eta_{N_2}(N_1) - \eta_{\min}$. 긴 budget 런이 시점 $N_1$ 이후의 연장 구간 $[N_1, N_2]$를 시작할 때 남아 있는 LR 여유. $H \approx 0$이면 연장 구간 전체를 바닥 LR로만 학습한다 (연장 budget의 낭비).

**정의 4 (Cumulative budget과 민감도 질량).** $B(N) = \int_0^N \eta_N(t)\,dt$ (총 LR 예산). 민감도 질량 $M(N) = \int_0^N \partial_N \eta_N(t)\,dt$ (budget을 한 단위 늘릴 때 LR 곡선이 창 전체에서 움직이는 총량).

**정의 5 (Budget elasticity).** $\mathcal{E}(N) = \partial \log |\dot\eta_N(0)| / \partial \log N$. 초기 감쇠율의 budget 민감도.

**정의 6 (Log-space coordinates).** $\ell_N(t)=\log\eta_N(t)$, $\ell_{\min}=\log\eta_{\min}$로 두고 $D_T^{\log}(N_1,N_2)=\sup_{t\le T}|\ell_{N_1}(t)-\ell_{N_2}(t)|$, $A^{\log}(N)=\ell_N(N)-\ell_{\min}$, $H^{\log}(N_1,N_2)=\ell_{N_2}(N_1)-\ell_{\min}$을 정의한다. Peak LR를 log scale로 탐색하는 HPO에서는 additive LR 차이보다 이 좌표가 직접적이다. $\eta_{\min}=r\eta_0$처럼 relative floor $r$를 고정하면 좌표가 peak LR의 절대 scale에 대해 불변이다.

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

**정리 1 (좌표 삼각관계, 원고에서는 보조식).** 임의의 패밀리와 $N_1 < N_2$에 대해, 아무 가정 없이

$$D_{N_1}(N_1, N_2) + A(N_1) \;\ge\; H(N_1, N_2)$$

*증명.* $H = \eta_{N_2}(N_1) - \eta_{\min} = [\eta_{N_2}(N_1) - \eta_{N_1}(N_1)] + [\eta_{N_1}(N_1) - \eta_{\min}] \le D_{N_1} + A(N_1)$. $\square$

**따름정리 1.1 (질적 불가능성, v1의 정리 1).** full-window prefix invariance ($D \equiv 0$) + exact anchoring ($A \equiv 0$) + 강단조 annealing은 동시 불가. ($D = A = 0 \Rightarrow H = 0 \Rightarrow \eta_{N_2}(N_1) = \eta_{\min}$인데 강단조면 $\eta_{N_2}$는 $N_1 < N_2$에서 $\eta_{\min}$ 위에 있어야 하므로 모순.)

**따름정리 1.2 (퇴화 모서리).** $D = A = 0$을 고집하면 $H = 0$: 모든 budget이 최소 budget 안에 감쇠를 끝내고 바닥에 눕는 drop-then-floor 패밀리만 남는다. 연장 budget은 전부 $\eta_{\min}$에서 소비된다.

이 부등식은 세 설계 욕구를 한 좌표계에 놓는 bookkeeping relation이다. 증명은 삼각 부등식 한 줄이므로 headline theorem으로 쓰지 않는다. 방법의 가치는 EH가 이 좌표에서 어디에 놓이는지와 그 위치가 실제 cross-budget transfer로 이어지는지를 함께 보일 때 생긴다.

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

**분류 표.** H 행은 LR-space 좌표, EH 행은 같은 식의 log-space 좌표다.

| 패밀리 | Anchoring | Full-prefix deviation | 조기 deviation | 정규화 sensitivity mass | 배치 | 초기 elasticity |
|---|---|---|---|---|---|---|
| rescaling (cos/lin/poly) | $0$ | budget 비에 따라 비소멸 | 자기닮음 고정 | cos/lin $1/2$ | 전 구간 분산 | $-1$ |
| truncation (const, exp $\gamma$ 고정) | 통제 불가 | $0$ | $0$ | $0$ | 없음 | $0$ |
| WSD | $0$ | cooldown에서 최대 | stable 구간 $0$ | 최대급 | 끝 구간 집중 | stable 구간 $0$ |
| drop-then-floor | $0$ | $0$ | $0$ | $\approx0$ | 없음, utility 퇴화 | $0$ |
| **HyperbolicLR** | $\Delta\eta[1-\sqrt{1-(1-\rho)^2}]$ | 정리 3(d) | 작은 후방 배치 | $0.29\sim0.36$ | 매끄러운 후방 증가 | $-\frac{U-N}{2U-N}$ |
| **ExpHyperbolicLR** | $\Delta\ell[1-\sqrt{1-(1-\rho)^2}]$ | H와 같은 정규화 식 | log-space 후방 배치 | H와 같은 정규화 식 | 매끄러운 후방 증가 | $-\frac{U-N}{2U-N}$ |

$\rho=N/U$. Truncation과 drop-then-floor는 utility 또는 anchoring을 포기하는 경계 사례다. 표는 성능 순위를 뜻하지 않는다.

## 5. HyperbolicLR과 ExpHyperbolicLR의 폐형식 좌표

표기: $a=U-N$, $\rho=N/U$, $g_N(t)=\sqrt{(N-t)(2U-N-t)}/U$. HyperbolicLR는 $\eta_N(t)=\eta_0+\Delta\eta(g_N(t)-g_N(0))$이고, ExpHyperbolicLR는 $\ell_N(t)=\ell_0+\Delta\ell(g_N(t)-g_N(0))$, $\Delta\ell=\log(\eta_0/\eta_{\min})$로 정의한다. 두 방법은 같은 hyperbolic arc를 공유하지만 H는 additive LR geometry, EH는 multiplicative log-LR geometry를 쓴다.

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
(h) ExpHyperbolicLR은 $\log\eta$ 공간에서 동일 구조를 가지므로 (a)부터 (g)까지가 $\Delta\eta\to\Delta\ell=\log(\eta_0/\eta_{\min})$로 이식된다. Relative floor를 고정하면 normalized log trajectory가 peak LR와 분리되며, 이것이 H와 EH를 가르는 핵심 설계 가설이다. [수치 검증]

**일반 창 하한 (rescaling과의 분리).** rescaling 패밀리는 $t = \beta N_1$에서 $D_{\beta N_1} \ge \Delta\eta\big[\tilde\varphi(\beta N_1/N_2) - \tilde\varphi(\beta)\big] > 0$이고 이 값은 budget 비에만 의존한다 (절대 규모를 키워도 비소멸). hyperbolic의 동일 창 deviation은 (d)의 폐형식으로 $(1-\rho)$ 차수로 통제된다.

## 6. 수치 검증 요약 ($\eta_0 = 10^{-2}$, $\eta_{\min} = 10^{-6}$, $U = 250$, 2026-06-11)

조기 창 $t \in [0, 0.2N_1]$ sup deviation ($\Delta\eta$ 단위): hyperbolic이 cosine 대비 3.1~4.2배 우위 (50→100: 0.0187 vs 0.0710; 50→200: 0.0285 vs 0.0893; 100→200: 0.0208 vs 0.0710; 150→200: 0.0099 vs 0.0410).

전체 창 sup deviation: hyperbolic이 대부분 우위이나 150→200에서 cosine 근소 우위 (0.2831 vs 0.2564). 주장은 조기/비례 창으로 한정한다.

민감도 질량 $M(N)/\Delta\eta$: hyperbolic 0.288 (N=50), 0.359 (100), 0.365 (150), 0.295 (200), 0.118 (240) vs rescaling 고정 0.5.

질량의 조기 창 (첫 20%) 점유율: hyperbolic 1.0~1.2%, cosine 2.5%, linear 4.0%. **총량이 작고 배치도 늦다**: 두 효과가 곱으로 조기 deviation을 줄인다.

anchoring error: $A(50) = 0.40\Delta\eta$, $A(100) = 0.20\Delta\eta$, $A(150) = 0.084\Delta\eta$, $A(200) = 0.020\Delta\eta$. $N \ll U$의 실질 가격이며 정리 1이 회피 불가능성을 보증한다.

## 7. Secondary exact dynamics와 stop-time theory (appendix 또는 별도 논문 후보)

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
1. Untuned toy에서 schedule별 loss curve deformation이 크게 달라진다. 이는 geometry가 observable trajectory를 바꿀 수 있다는 existence proof다.
2. 같은 toy를 schedule별로 공정 튜닝하면 WSD가 대부분 regime을 지배하고 HyperbolicLR의 우위는 사라진다.
3. 따라서 quadratic toy는 EH 또는 H의 neural-network 성능 우위를 지지하지 않는다.
4. Dwell, sensitivity mass, noise regime을 deep-learning collapse의 원인으로 옮기지 않는다.
5. 원고에서 이 결과를 쓴다면 coordinate illustration 또는 appendix negative control로만 쓴다.

Cross-budget 방법 주장은 toy가 아니라 동일 LR candidate grid와 독립 HPO 반복에서 직접 측정한다.

### 7.1 공정 튜닝 후 얻은 제한사항 (2026-06-11)

모든 schedule에 peak LR를 튜닝하고 WSD cooldown과 H의 $U$도 탐색한 결과 (`scripts/08_hardened_toy.py`, `figs/tuned_regime_map.png`):

1. N=200의 대부분 noise 구간에서 WSD가 1위였고 HyperbolicLR의 untuned 우위는 사라졌다.
2. 스펙트럼을 바꾸면 WSD, cosine, H 사이의 순위가 바뀌어 일관된 최종성능 우위가 없다.
3. 완주한 final loss로 N=50에서 고른 peak LR를 N=200에 옮기는 단순 toy에서는 cosine regret도 0이 될 수 있다.

이 결과는 pruning을 켜야만 EH가 유리하다는 결론을 허용하지 않는다. Confirmatory protocol은 pruning을 끄고 B=10 final-window metric으로 설정을 고른 뒤 긴 horizon에 직접 배포한다. Pruning failure는 별도 연구 질문이다.

### 7.2 Anytime 프로토콜: cross-budget 본문과 분리할 보조 결과 (2026-06-11)

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

**현재 논문에서의 해석.**
1. ExpHyperbolicLR가 dense stop-time toy에서 명명 패밀리 중 mean regret 1위라는 결과는 log-space 설계의 보조 동기다. 실제 neural-network cross-budget superiority를 대신하지 않는다.
2. 명명 패밀리 최선과 free-form game value 사이에는 2배 gap이 남는다. 이 gap과 checkpoint-synchronized sawtooth는 별도 schedule-design 문제다.
3. Classification의 anytime 순위는 metric에 따라 뒤집혔으므로 Phase 2 anytime 표를 EH의 실증 근거로 쓰지 않는다.

### 7.4 Stop-time spectral result와 선행연구 경계

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

**범위.** 이 정리는 정의된 quadratic stop-time game 안에서 단일 곡률과 다중 곡률을 구분한다. Neural-network cross-budget transfer 또는 EH의 우위를 증명하지 않으며, 현재 방법 논문에서는 appendix 결과다.

**문헌 포지셔닝.** Schedule transfer와 horizon-free training은 이미 점유된 문제다.

- Schaul et al. 2013은 noisy quadratic의 greedy optimal LR를 유도했다.
- Ge et al. 2019는 least squares에서 geometric step decay의 near-optimality를 다뤘다.
- REX는 training horizon을 알지 못하는 LR scheduling을 명시적으로 다뤘다.
- Wen et al. 2024는 WSD의 branchable anytime 성질을 설명했다.
- Power Scheduler와 2026년 WSqD는 training horizon이 바뀌어도 base phase를 재사용하는 직접 경쟁 방법이다.
- Anytime Pretraining과 ScheduleFree+는 averaging 또는 optimizer state를 이용해 다른 방식으로 horizon 문제를 푼다.

따라서 EH의 신규성은 horizon-free 문제의 발견, geometric decay의 발견, 또는 유일성에 있지 않다. 주장 가능한 차이는 bounded shared-asymptote hyperbolic arc를 log LR에 적용한 구체적 방법, 그 closed-form coordinate, 그리고 동일 HPO protocol에서 측정한 fidelity와 utility다.

Stop-time minimax game과 spectral-tension theorem은 이 방법 주장과 논리적으로 독립이다. 현재 원고에서는 appendix 또는 별도 이론 논문으로 취급하고, 이론 novelty로 EH 실험 부족을 보상하지 않는다.

## 8. 변분 관점과 EH 주장의 정확한 범위

주어진 anchoring budget에서 prefix deviation만 최소화하면 hyperbolic이 답이 아니다. 고정 master prefix와 budget-dependent cooldown을 허용하면 조기 deviation 0과 exact anchoring을 함께 얻는 piecewise construction이 가능하다. 따라서 EH를 geometric optimum이나 유일한 horizon-free schedule로 부르지 않는다.

EH의 방법 가치는 다음 조합에 있다.

1. 강단조이고 매끄러운 hyperbolic arc를 log LR에 적용한다.
2. 하나의 bounded-horizon parameter $U$로 prefix deformation과 endpoint anchoring을 폐형식으로 계산할 수 있다.
3. H와 달리 multiplicative scale에서 endpoint를 제어한다.
4. E처럼 긴 run을 일찍 floor에 묶지 않으면서, rescaling schedule보다 작은 cross-budget deformation을 겨냥한다.
5. 현재 DeepONet exploratory campaign에서 low-budget tuning fidelity, long-horizon utility, zero observed collapse를 함께 보였다.

POW와 WSQD도 horizon-free base phase를 제공하므로 EH가 유일하다는 주장은 틀리다. 이론은 세 방법의 성능 순서를 정하지 못하며, 동일 candidate set과 replicated HPO가 판정해야 한다.

미해결 질문은 $C^1$ 강단조 패밀리에서 주어진 log-space anchoring budget 아래 조기 log-deviation을 최소화하는 해가 shared-asymptote 구조인지다. 이 문제는 현재 원고의 필요조건이 아니며 추측으로만 남긴다.

## 9. 이론 가설과 판정 실험

| 설계 요소 | 사전 가설 | 판정 실험 |
|---|---|---|
| Log-space hyperbolic geometry | EH의 B=10 최적 peak LR가 N 증가에 따라 덜 이동한다 | Fixed-shape LR response surface와 optimum-ridge drift |
| Fidelity와 utility의 결합 | E는 fidelity만, rescaling 계열은 utility 또는 collapse, EH는 두 축의 Pareto corner에 놓인다 | Within-family transfer regret와 long-horizon utility |
| Exponentiation | 같은 geometry에서 EH가 H보다 낮은 N=200 transfer loss를 보인다 | H 대 EH paired ablation |
| Bounded horizon $U$ | EH 결론이 정확한 $U=N_{\max}$ 한 점에만 의존하지 않는다 | $U\in\{200,400,800\}$ |
| Relative floor | Peak LR scale와 endpoint ratio를 분리하면 search boundary가 줄어든다 | $\eta_{\min}/\eta_0\in\{10^{-4},10^{-3},10^{-2}\}$ |
| Architecture dependence | DeepONet 결과가 TFONet 또는 CIFAR ResNet에서 재현된다 | 독립 setting matrix와 HPO replicate |

다음은 더 이상 사전 가설이 아니다.

- Dwell time 또는 sensitivity mass가 AdamW instability의 물리적 cause라는 주장.
- $\eta\lambda_{\max}^{P}/38$ crossing이 collapse를 예측한다는 주장.
- Normalization이 EH의 이점을 보존한다는 주장.
- Anytime toy 순위가 classification 성능 순위를 예측한다는 주장.

F24와 F26이 앞의 두 mechanism claim을 반증했다. 이 음성 결과는 limitations에 보고하고 새 proxy로 구조하지 않는다.

## 10. 원고 배치

1. **Problem and protocol:** tuning budget, deployment horizon, fidelity, utility, transfer regret.
2. **ExpHyperbolicLR:** H와 EH의 공통 hyperbolic arc, log-space exponentiation, relative floor, $U$.
3. **Closed-form schedule coordinates:** $D^{\log}$, $A^{\log}$, sensitivity profile. 삼각관계와 보존 identity는 짧게 쓰고 증명은 appendix로 보낸다.
4. **Direct comparators:** E, cosine, D2Z, WSD, Power Scheduler, WSqD.
5. **Experiments:** fixed-shape landscape, replicated HPO, DeepONet, TFONet, CIFAR ResNet, H/EH와 $U$ ablation.
6. **Scope:** geometry가 curve deformation은 설명하지만 optimizer instability mechanism은 설명하지 않는다고 명시한다.
7. **Appendix:** 기존 exact dynamics, stop-time minimax game, spectral-tension theorem, 전체 baseline과 negative mechanism audit.

## 11. 남은 이론 작업

원고 필수:

1. EH의 log-space 정의와 relative-floor parameterization을 논문 notation으로 고정한다.
2. $D_T^{\log}$, $A^{\log}$, $H^{\log}$의 closed form을 H 좌표에서 정확히 옮기고 symbolic check를 다시 실행한다.
3. $U$가 deployment horizon의 oracle 정보가 아니라 사전 upper bound라는 조건을 정리한다.
4. Momentum과 AdamW에서 bridge proposition을 주장하지 않는다. 명제 1은 SGD 한정임을 본문에 명시한다.
5. Power Scheduler와 WSqD를 같은 coordinate table에 넣되 성능 순서는 이론으로 주장하지 않는다.

Appendix 또는 후속 작업:

1. Stop-time regret를 true optimal-control oracle로 다시 계산한다.
2. Spectral-tension theorem의 $\sigma_2^2>0$ 일반화와 정량 하한을 연구한다.
3. Shared-asymptote variational problem을 정식화한다.

실험 프로토콜과 성공 기준은 `PUBLICATION_PLAN.md` v13을 단일 기준으로 쓴다.
