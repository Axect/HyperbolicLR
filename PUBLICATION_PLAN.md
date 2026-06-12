# HyperbolicLR 출판 계획 (arXiv → TMLR)

작성일: 2026-06-11. 대상 논문: [arXiv:2407.15200](https://arxiv.org/abs/2407.15200) "HyperbolicLR: Epoch Insensitive Learning Rate Scheduler" (Tae-Geun Kim, 단독 저자).

## 목표와 타겟 venue

1순위 타겟은 **TMLR (Transactions on Machine Learning Research)**. 게재 기준이 "claims supported by evidence"라서 스케줄러 실증 연구와 잘 맞고, rolling submission이라 실험 보강이 끝나는 대로 제출 가능하다. 선례로 WSD cooldown 분석 논문이 2025년 TMLR에 게재됐다 (arXiv:2508.01483).

| 순위 | Venue | 비고 |
|---|---|---|
| 1 | TMLR | rolling, OpenReview double-blind, arXiv 선공개 허용, 리뷰 약 2~3개월 |
| 2 | Neural Networks / Machine Learning (Springer) | 전통 저널, 리뷰 6개월~1년 |
| 3 | Neurocomputing | 위 두 곳에서 막힐 경우 |
| 도전 | ICLR / NeurIPS / AAAI 본 트랙 | Phase 3 (LM pretraining) 결과가 충분히 강할 때만 |

제출 목표일: **2026년 8월 말** (TMLR).

## 현황 진단

보유 자산:

- 코드 인프라 완비: `paper/scheduler_config.py` (스케줄러 13종 중앙 레지스트리), `paper/run_all.py` (Optimize → Optimize-Compare 자동화, multi-seed 지원), W&B 로깅, Rust 분석 도구 (`analyze/`, `metric/`, `learning_curves/`).
- Baseline: N/P/C/E/L/S (전통), OC/CY (cyclic), WC/WH/WEH (warmup), Prodigy/DAdapt (LR-free).
- 분석 스크립트: decay rate (`scripts/03`), cumulative LR budget (`scripts/04`).
- Task 3종: CIFAR-10/100 (SimpleCNN/ResNetCIFAR/SimpleViT), OSC LSTM Seq2Seq, Integral operator learning (DeepONet/TFONet).

리뷰어가 지적할 갭 (우선순위순):

1. **Schedule-Free AdamW (Defazio et al., NeurIPS 2024)와 WSD (Warmup-Stable-Decay) 비교 부재.** 둘 다 "training horizon을 몰라도 되는 학습"이라는 동일 문제를 공략하는 2024년 이후 주류 방법이라 정면 비교가 필수.
2. **스케일 증거 부재.** CIFAR + operator learning만으로는 "epoch insensitivity가 규모에서도 성립하는가"에 답할 수 없음. 소규모 LM pretraining 트랙 1개 필요.
3. **통계 처리.** 단일 seed 결과로는 부족. seed 5개 이상 mean ± std와 유의성 검정 필요.
4. **epoch insensitivity의 정량화.** 현재는 학습 곡선 그림 중심. SLCD metric을 주장의 중심 증거로 승격해야 함.
5. **Related work가 2023년 이전에 멈춰 있음.** Schedule-Free, WSD, REX, Prodigy/DAdapt를 포함해 재작성 필요.

## Phase 1: Baseline 보강 (코드 작업)

목표: WSD 스케줄러와 Schedule-Free AdamW를 실험 파이프라인에 추가한다.

### 1.1 WSD 스케줄러 추가

- 파일: `paper/scheduler_config.py`
- 코드명 `WSD`. `LambdaLR` 기반 factory `_create_wsd(optimizer, lr, infimum_lr, epochs, cooldown_frac)` 구현. 구조: linear warmup (`epochs // 10`) → constant (lr 유지) → cooldown (마지막 `cooldown_frac` 구간, 1-sqrt decay, Hägele et al. 2024 방식, infimum_lr까지).
- 레지스트리 (`get_schedulers`), 기본 파라미터 (`get_scheduler_params`), Optuna search space (`get_optuna_search_space`: `infimum_lr` 1e-7~1e-4 log, `cooldown_frac` 0.1~0.4), `build_scheduler_from_trial`, `get_optimize_compare_params`에 모두 등록.
- 주의: WSD는 epoch 단위 step (per-batch 아님), `STEP_PER_BATCH_SCHEDULERS`에 넣지 않는다.

### 1.2 Schedule-Free AdamW 추가

- 의존성: `schedulefree` 패키지 (`paper/requirements.txt`에 추가).
- 파일: `paper/scheduler_config.py`의 `get_lr_free_optimizers()`에 `"ScheduleFree": ScheduleFreeAdamW` 추가 (import 실패 시 graceful skip 유지).
- Schedule-Free는 Prodigy/DAdapt와 달리 실제 lr이 필요하므로 `get_lr_free_optuna_search_space`에 `lr` (1e-4~1e-2 log)과 `warmup_steps` 탐색 추가. lr 기본값 처리를 위해 LR-free 권장 lr을 optimizer별로 분기 (`Prodigy`/`DAdapt`는 1.0, `ScheduleFree`는 탐색값).
- **핵심 주의사항**: Schedule-Free는 train/eval 모드 전환이 필수. `paper/util.py`의 `Trainer`와 `OperatorTrainer`에서 `hasattr(self.optimizer, "train")`이고 `optimizer.train`이 callable이면 `train_epoch` 시작 시 `optimizer.train()`, `evaluate` 시작 시 `optimizer.eval()` 호출. 이 호출이 빠지면 평가 성능이 조용히 틀어지므로 (z-iterate vs x-iterate) 반드시 검증할 것.
- `paper/run_all.py`의 LR-free 경로 (`optimize_lr_free_one` 등)가 lr=1.0 하드코딩이라면 ScheduleFree 분기 추가.

### 1.3 검증 (Phase 1 완료 기준)

- LR 곡선 단위 검증: 임시 스크립트로 epochs=100, WSD의 lr 시퀀스를 뽑아 warmup/constant/cooldown 3구간 형태와 경계값 (init_lr, infimum_lr 도달) 확인.
- Dry run 1: `uv run python run_all.py --task cifar10 --model SimpleCNN --epochs 2 --n-trials 2 --n-seeds 1 --wandb-mode disabled --schedulers WSD --skip-lr-free` 정상 종료.
- Dry run 2: LR-free 경로 포함 dry run에서 ScheduleFree가 에러 없이 돌고 `optimizer.train()/eval()` 전환이 실제로 호출되는지 확인.
- `paper/README.md` 스케줄러 표와 LR-free 표 갱신.

## Phase 2: 본 실험 재실행 (기존 3개 task)

목표: 신규 baseline을 포함한 전체 매트릭스를 multi-seed로 재실행한다.

- 실행: `run_all.py`로 task별 전체 파이프라인. HPO는 epochs=50 (낮은 budget에서 최적화한 뒤 큰 budget으로 전이하는 것이 epoch-insensitivity 시험의 핵심), comparison은 `--epoch-budgets 50,100,150,200 --compare-seeds 5`.
- 매트릭스: cifar10 × {ResNetCIFAR, SimpleViT}, osc × LSTM_Seq2Seq, integral × {DeepONet, TFONet}. SimpleCNN(subset)은 dry run 용도로만.
- 스케줄러: 전체 14종 (WSD 포함) + LR-free 3종 (Prodigy, DAdapt, ScheduleFree).
- 인프라: 로컬 GPU + pueue (`pueue group add HyperbolicLR`), task당 1개 큐 항목. 다른 프로젝트 GPU 작업은 절대 건드리지 않는다.
- 완료 기준: 5개 모델 조합 모두 `results_{project}.json` 생성, W&B에 학습 곡선 적재, seed 5개 분산 확인.

## Phase 3: LM pretraining 스케일업 트랙 (신규)

목표: "epoch insensitivity"를 token-budget insensitivity로 일반화해 transformer LM pretraining에서 검증한다. TMLR 안정권 확보와 학회 도전 가능성을 가르는 phase.

- 디렉토리: `paper/nanogpt/` 신설. karpathy nanoGPT 구조를 차용하되 스케줄러를 `scheduler_config.py`에서 주입.
- 인프라 결정 (2026-06-11): 기존 `paper/` 파이프라인은 pytorch_template 기반이 아니므로 `/pytorch-migrate` 적용 대상이 아니며, 검증 완료된 현재 코드를 그대로 유지한다. 대신 신규 트랙인 이 phase는 최신 pytorch_template로 스캐폴딩하는 것을 검토 (preflight, checkpoint/resume이 vast.ai 장시간 학습에 유리).
- 모델: GPT-2 small 축소판 (~50M params, n_layer=8, n_embd=512 수준). 데이터: FineWeb-Edu sample 또는 OpenWebText 일부.
- 실험 설계: token budget {1x, 2x, 4x} (예: 1B/2B/4B tokens)에서 C/WC/WSD/H/EH/WH/WEH + ScheduleFree 비교. 핵심 질문: "budget을 늘렸을 때 초기 학습 곡선이 유지되는가 (decoupling 없음)"와 "최종 val loss 경쟁력".
- 인프라: vast.ai GPU 임대 (RTX 4090/5090 또는 A100 1~2장, 예산 $50~150), pueue로 관리.
- 완료 기준: budget 3개 × 스케줄러 8종 × seed 2 이상의 val loss 곡선 확보, SLCD metric 산출.
- 리스크 대응: HyperbolicLR이 ScheduleFree/WSD에 최종 loss에서 밀리면 narrative를 "동급 성능을 유지하면서 epoch-insensitive한 유일한 annealing 계열 스케줄러"로 조정. epoch insensitivity 자체가 차별점이므로 동급이면 충분.

## Phase 4: 이론 핵심부와 통계 분석

목표: 논문의 중심 기여를 "prefix invariance vs anchoring 트레이드오프"의 정식화로 승격한다. 상세 프레임워크와 검증된 명제들은 `THEORY_REFRAMING.md` 참조 (2026-06-11 sympy/numpy 검증 완료).

- 이론 본문 작성: 정의 (prefix deviation, anchoring error, elasticity), 명제 1 (Grönwall bridge, SLCD 정당화), 정리 1 (3성질 동시 불가), 정리 2 (rescaling 패밀리의 decoupling 하한), 정리 3 (HyperbolicLR의 폐형식 좌표), 스케줄러 분류 표.
- SLCD metric을 $L^1$ prefix deviation의 실측 대응물로 재정의 (`paper/metric/` Rust 구현과 정합 확인).
- **HPO transfer를 주 실험으로 승격**: 저budget 튜닝 → 고budget 배포 시 성능 보존율. `run_all.py`의 Optimize → Optimize-Compare 구조가 이미 이 프로토콜.
- 유의성 검정: task × budget 조합에 대해 Wilcoxon signed-rank test, mean ± std (5 seeds).
- $U$ 민감도 ablation ($U \in \{200, 250, 300, 400\}$), anchoring error $A(N)$과 최종 성능 격차의 상관 분석.
- 산출물: `paper/scripts/05_slcd_statistics.py`, 논문용 LaTeX 표, 남은 이론 작업은 `THEORY_REFRAMING.md` §9.

## Phase 5: 논문 개정

목표: TMLR 양식으로 원고를 재작성한다.

- LaTeX 소스 확보: 현재 repo에 tex 소스가 없으므로 위치 확인 (Overleaf 추정) 후 TMLR 템플릿 (OpenReview 제공)으로 이식.
- Related work 재작성: Schedule-Free (Defazio et al. 2024), WSD/cooldown (Hägele et al. 2024, Dremov et al. 2025), Prodigy/D-Adaptation, REX, OneCycle. 본 연구의 위치를 "horizon-robust training" 문헌 안에 명시.
- 구성 변경: `THEORY_REFRAMING.md` §8의 개편안을 따른다. (1) 트레이드오프 정식화를 §2 핵심 섹션으로, (2) HyperbolicLR을 "트레이드오프의 매끄러운 해"로 재서술, (3) HPO transfer를 주 결과로, (4) LM pretraining 결과 추가, (5) 통계 표 전면 교체, (6) Limitations에 anchoring error의 가격과 전체 창 비우위 사례를 명시.
- 그림 재생성: `result_plot.py`, `learning_curves/`, `metric_plot.py` 기반으로 신규 baseline 포함 버전.
- 작성 후 adversarial review (desk-reject 체크)를 돌리고 지적사항 반영.

## Phase 6: 제출과 대응

- TMLR 제출 (OpenReview). 동시에 arXiv v2 업데이트 (보강된 버전).
- 리뷰 대응 후 게재 확정 시: GitHub README에 TMLR 인용 정보 갱신, `hyperbolic_lr.py` PyPI 패키징 검토 (선택).
- TMLR reject 시: 리뷰 코멘트 반영 후 Neural Networks 또는 Machine Learning (Springer)으로 이동.

## 일정 (목표)

| Phase | 기간 | 비고 |
|---|---|---|
| 1. Baseline 보강 | ~6월 중순 | 코드 작업, 즉시 시작 |
| 2. 본 실험 재실행 | 6월 하순 ~ 7월 초 | 로컬 GPU, pueue |
| 3. LM pretraining | 7월 | vast.ai, Phase 2와 일부 병행 가능 |
| 4. Metric/통계 | 7월 하순 | Phase 2 결과 나오는 대로 착수 |
| 5. 논문 개정 | 7월 하순 ~ 8월 중순 | |
| 6. TMLR 제출 | 8월 말 | |

## 진행 기록

- 2026-06-11: 계획 수립. Phase 1 착수.
- 2026-06-11: Phase 1 완료. WSD 스케줄러 (`scheduler_config.py`의 `_create_wsd`, LambdaLR 기반 warmup → stable → 1-sqrt cooldown)와 Schedule-Free AdamW (`get_lr_free_optimizers`의 `ScheduleFree`, `util.py` Trainer/OperatorTrainer에 optimizer.train()/eval() 전환 추가) 구현 및 검증. LR 곡선 형태, Optuna 연동, dry run 모두 통과.
- 2026-06-11: CIFAR-10/100 확보 완료. cs.toronto.edu가 403 차단이라 PaddlePaddle bcebos 미러에서 다운로드, 공식 md5 일치 및 torchvision 무결성 검사 통과. `paper/data/`에 배치.
- 2026-06-11: 실제 CIFAR-10으로 전체 dry run 통과 (`run_all.py --epoch-budgets 2,4`, WSD + Prodigy/DAdapt/ScheduleFree). HPO와 비교 단계 모두 정상, results json 생성 확인. Phase 1 완전 종료, Phase 2 착수 가능 상태.
- 인프라 결정: `paper/` 코드는 pytorch_template 기반이 아니므로 `/pytorch-migrate` 비적용. 현재 파이프라인 유지, Phase 3 신규 트랙만 template 스캐폴딩 검토.
- 2026-06-11: 이론 재프레이밍 완료 (`THEORY_REFRAMING.md`). 핵심: prefix invariance + exact anchoring + strict annealing의 동시 불가능성 정리, 기존 스케줄러 분류 (rescaling/truncation/WSD/asymptotic), HyperbolicLR의 폐형식 트레이드오프 좌표 (elasticity $-(U-N)/(2U-N)$, anchoring error $\Delta\eta(1-\sqrt{1-(1-N/U)^2})$, sup deviation 폐형식). 모든 명제 sympy/numpy 검증. HPO transfer를 주 실험으로 승격.
- 2026-06-12 (v10, Phase 2 결정 실험 완료): vast.ai RTX 4070 Ti SUPER 17.8시간, **총 $3.37**. 산출물 206 parquet (`paper/results_phase2/`, 로컬 회수 및 무결성 검증 완료, 인스턴스 파기 완료). 분석 (`scripts/15_phase2_analysis.py`): (1) **SLCD: H가 양 task (CIFAR/CNN, OSC/LSTM) 모두 annealing 1위** — kill 기준 1 통과. (2) **pruning 오살률: WSD 100% (전 시점), EH 최저 (epoch 20+에서 0%)** — kill 기준 3 통과, toy 예측 메커니즘 그대로. (3) anytime: 회귀 (OSC log-loss)에서 H/EH 1~2위 통과; CIFAR accuracy에서는 {warmup + 깊은 tail} 그룹 (WC 77.7 / WSD 77.1 / SF 75.4)이 H-family (~69.8)에 8pp 우위. WH/WEH 보강 런으로 warmup 단독 교란 배제 → 격차의 원인은 **anchoring error의 실측 가격** (H-family의 높은 tail LR): 정리 1의 트레이드오프 ($D$ 최저 = SLCD/오살률 승리, $A$의 대가 = 최종 acc 8pp)가 실제 딥러닝에서 end-to-end로 확인됨. 후속: $U \to N$ ablation으로 가격 경감 검증. 부수 발견: lr 0.01 + 무warmup 전 스케줄 발산 (Limitations용). **종합 판정: GO** — 프레임워크/특성화 논문으로 Phase 4~5 (통계 정리, 논문 작성) 진행.
- 2026-06-11 (v9, 정리 4 증명 완성): 스펙트럼 긴장의 정식 증명 완료. 두 모드 ($\sigma_2^2=0$) 설정에서 "어떤 단일 스케줄도 두 horizon에서 동시 최적 불가 → minimax anytime regret > 0"을 Pontryagin식 1차 조건 충돌로 증명 (내부성 보조단계 + 두 stationarity의 가중비 충돌 + 연장 퇴화 배제). 모든 증명 단계 수치 검증 (V=12.93%, stationarity 잔차 1e-12, $F_1 < F_2$ 엄격). $\sigma_2^2>0$은 연속성 따름정리. Codex가 지목한 "논문 운명을 가르는 한 가지" 해결: 이제 논문은 정확해 (보조정리 3) + 정리 (정리 4) + 검증된 특성화를 갖춘 프레임워크 논문. 상세 `THEORY_REFRAMING.md` §7.4 정리 4.
- 2026-06-11 (v8, Codex 적대적 리뷰): GPT-5.5와 novelty 검증 세션. 판정: 생존하는 핵심 신규성은 (1) 유한 창 stop-time oracle-regret 게임 정식화, (2) 스펙트럼 다양성 = anytime 비용 원천의 특성화 (단, **정식 증명 필수**). 신규 선행연구 발견: Schaul 2013 (명제 3 greedy 선행 → 보조정리로 강등), Ge 2019 (geometric decay 선행 → 게임 한정으로 좁힘), Eigencurve, Wen 2024 (WSD anytime 선행), REX. $D+A \ge H$/보존 법칙은 측정 언어로 강등. 제목 후보 "Committed Learning-Rate Schedules Under Horizon Uncertainty". Phase 2 baseline에 REX/D2Z 추가, HPO 오살률 정의 확정. 상세 `THEORY_REFRAMING.md` §7.4 끝부분, §11 항목 10~12.
- 2026-06-11 (v7, 경로 A 완료): anytime 게임의 해석적 구조 확립. (1) **명제 3**: 단일 곡률 모드에서 greedy가 horizon-free 최적 ($u_{t+1}=u_t/(1+u_t)$ 정확 폐형식, $\eta^* \to 1/(\lambda t)$ = Robbins-Monro), minimax anytime regret 정확히 0. 기계 정밀도 검증. (2) **명제 4**: anytime 긴장의 근원은 스펙트럼 다양성 (두 모드 비 1/4/16/64에서 regret 0/2.3/5.7/7.4%). (3) wide-window [20,640]에서 기하 감쇠 판정 유지 (지수 fit이 power law 대비 residual 3배 우세). (4) 문헌 포지셔닝: anytime 방법 헤드라인은 averaging 계열 (Anytime Pretraining 2602.03702, ScheduleFree+ 2605.19095)에 선점됨; averaging이 정리 1을 우회하는 메커니즘을 프레임워크로 설명; 우리 기여는 유한 창 minimax 게임 정식화 + 특성화 (명제 3~4)로 확정. (5) family oracle이 true oracle 대비 최대 14% 느슨함 발견, 논문 수치는 true oracle로 재산출 필요. 논문 정체성: 방법 논문 → 프레임워크/특성화 논문으로 확정. 상세 `THEORY_REFRAMING.md` §7.4.
- 2026-06-11 (v6, free-form 최적해): anytime 게임의 free-form 최적 LR 수열을 직접 계산 (`scripts/10~12`). 발견 3건: (1) 중단이 checkpoint 시점에만 가능하면 최적해는 checkpoint 동기화 톱니 (warm restart의 이론적 정당화, minimax 7.3%), (2) 임의 시점 중단 (dense)에서는 plateau + 기하 감쇠 (minimax 9.3%, WSD-branch와 추가 컴퓨트 없이 동급, regret equalizer 성질 확인), (3) 기하 감쇠의 anchored 폐형식인 **ExpHyperbolicLR이 명명 패밀리 중 mean regret 1위 (10.7%)**로 cosine 대비 절반. EH를 논문의 전면으로 승격. 명명 패밀리 최선과 게임 값 사이 2배 gap은 open problem으로 명시. 상세 `THEORY_REFRAMING.md` §7.3.
- 2026-06-11 (v5, anytime): minimax anytime regret 프로토콜 설계 및 toy 구현 (`scripts/09_anytime_protocol.py`). 중단 시점을 모르는 단일 런 게임에서 hyperbolic이 전 noise 수준 1위 또는 동률 (c=0.3: minimax 29.9%, mean 17.7%; cosine 35.5%/22.5%, WSD-stop 49.9%/32.8%). anchoring error가 anytime 게임에서 자산으로 반전됨 (높은 tail LR이 N_plan 이후 전진 지속). WSD-branch (9.1%)가 전체 1위지만 +20% 컴퓨트와 checkpoint 인프라 필요. 논문의 두 기둥 (insensitivity + anytime)이 toy 층위에서 모두 정량화 완료. Phase 2에서 동일 프로토콜의 실측이 주 결과 2가 됨. 상세 `THEORY_REFRAMING.md` §7.2.
- 2026-06-11 (v4, hardening): 리뷰어 선제 방어 차원에서 toy를 공정 튜닝 체제로 재실행 (`scripts/08_hardened_toy.py`). 결과는 우리에게 불리한 방향으로 정직: (1) 튜닝된 WSD가 quadratic 최종 loss에서 광역 지배, hyperbolic의 강noise 우승은 1% 미만 동률로 축소, (2) naive HPO transfer (완주 loss 기반)는 cosine이 0% regret으로 전이되어 변별력 없음 (fine grid 확인), (3) 스펙트럼 바꾸면 승자 뒤바뀜. 대신 얻은 것: 명제 1' (quadratic 타이트 bridge, 지수 인자 없이 $L^1$ prefix deviation 선형 지배, tightness 0.98~1.00), 포지셔닝 확정 (최종 성능은 parity 주장, 차별점은 insensitivity + anytime), Phase 2 HPO transfer 프로토콜 정밀화 (저epoch HPO + median pruning 활성 + pruning 오살률 측정; 현재 run_all.py에 pruner 부재 확인, 추가 필요). 상세 `THEORY_REFRAMING.md` §2 명제 1', §7.1.
- 2026-06-11 (v3): 정확 동역학 toy를 noise regime map으로 확장 (`scripts/07_exact_dynamics.py`). 약noise에선 Constant가 전부 이긴다는 지적 (사용자)을 반영해 보정을 재설계: annealing 스케줄러는 noise-dominated regime의 도구임을 명시하고, 승자가 Constant → WSD → HyperbolicLR로 넘어가는 3-regime 구조를 발견. $c=0.3$에서 hyperbolic이 SLCD와 최종 loss 동시 최저, WSD의 loss 영역 insensitivity는 강noise에서 붕괴 (cooldown 위치에서 budget별 곡선 분리). 상세는 `THEORY_REFRAMING.md` §7.
- 2026-06-11 (v2): 실험 불요 이론 작업 일괄 완료. (1) 삼각 부등식 $D + A \ge H$ (무가정, 1줄 증명)와 hyperbolic/linear/WSD의 등호 포화, (2) 민감도 질량 보존 법칙 $M(N) = B'(N) - \eta_N(N)$과 (총량, 배치) 분해: rescaling 0.5 분산 / WSD 0.87 끝단 집중 / hyperbolic 0.29~0.36 후방 매끄러움, (3) 편차 단조성 보조정리와 $D_T$, $M(N)$ 폐형식 증명, (4) WSD 좌표 정식화 (명제 2), (5) SLCD Rust 구현 = 정규화 $L^1$ prefix 거리로 이론 정의와 정합 확인, (6) 변분 문제의 솔직한 결론 (조각적 구성이 bang-bang 최적; hyperbolic의 가치는 매끄러움/무트리거/폐형식 + 실증) 및 open problem 진술, (7) 이론 그림 `figs/theory_tradeoff_overview.png` 생성. 남은 이론 작업은 명제 1의 AdamW 확장 결정뿐.
