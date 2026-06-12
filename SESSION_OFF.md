# SESSION_OFF — 2026-06-12

## 현재 위치 (한 문장)

HyperbolicLR 출판 프로젝트는 이론 재프레이밍과 Phase 2 결정 실험을 모두 마치고 **GO 판정** 상태이며, 다음 작업은 논문 작성 (Phase 4~5)이다.

## 이번 세션 구간 (2026-06-11 ~ 06-12)에서 완료된 것

1. **Phase 1**: WSD 스케줄러와 Schedule-Free AdamW를 실험 파이프라인에 추가 (`paper/scheduler_config.py`, `paper/util.py`의 train/eval 전환), 실데이터 dry run 통과. CIFAR-10/100은 토론토 서버 403 차단을 우회해 PaddlePaddle 미러에서 확보 (공식 md5 일치). `hyperbolic_lr.py`의 max_iter 초과 시 크래시 버그 수정 (클램프로 final LR 유지).
2. **이론 재프레이밍** (`THEORY_REFRAMING.md`): 삼각 부등식 $D + A \ge H$, 민감도 질량 보존 법칙, 명제 1' (quadratic 타이트 bridge, tightness 0.98~1.00), 보조정리 3 (단일 모드 zero anytime regret = Robbins-Monro), **정리 4 (스펙트럼 긴장: 곡률 다양성이 anytime 비용의 근원, Pontryagin식 1차 조건 충돌로 완전 증명)**. free-form 최적해 계산: checkpoint 중단이면 톱니, 임의 중단이면 기하 감쇠 (32배 창에서 확인), EH가 명명 패밀리 중 최근접. Codex (GPT-5.5) 적대적 리뷰 반영 (Schaul 2013, Ge 2019, Eigencurve, Wen 2024, REX 포지셔닝; 제목 후보 "Committed Learning-Rate Schedules Under Horizon Uncertainty").
3. **Phase 2 결정 실험** (vast.ai RTX 4070 Ti SUPER, 17.8h, $3.37, 인스턴스 파기 완료): 206개 학습 곡선 (`paper/results_phase2/`, partA 162 + partB 44). 판정:
   - SLCD: H가 CIFAR/OSC 양 task에서 annealing 1위 → **통과**
   - pruning 오살률: WSD 100% 오살 vs EH 0% (epoch 20+) → **통과**, 가장 인용 가치 높은 단일 결과
   - anytime: 회귀 (OSC)에서 H/EH 1~2위 통과; CIFAR accuracy의 8pp 격차는 WH/WEH 보강 런으로 warmup 교란을 배제한 결과 **anchoring error의 실측 가격**으로 판명 = 정리 1 트레이드오프의 실제 딥러닝 확인
   - 부수 발견: lr 0.01 + 무warmup 전 스케줄 발산 (Limitations용), SF의 val loss/acc 괴리 (overconfidence)

## 핵심 파일

- `PUBLICATION_PLAN.md`: 전체 계획 + 진행 기록 v1~v10 (ground truth)
- `THEORY_REFRAMING.md`: 이론 전체 (정의, 정리, 증명, 수치 검증, 문헌 포지셔닝, 논문 구조 §10, 남은 작업 §11)
- `paper/scripts/06~15`: 이론 그림, toy 검증, free-form 최적화, Phase 2 분석
- `paper/results_phase2/`: Phase 2 원자료 (재실행 불필요)
- `paper/phase2_runner.py`, `phase2_orchestrate.py`: 실험 재현 코드 (resume-safe)
- `paper/figs/`: 논문 그림 후보 8장 (theory_tradeoff_overview, exact_dynamics_*, tuned_regime_map, anytime_regret, dense_anytime, eh_anytime, wide_window_shape)

## 다음 세션이 시작할 일 (우선순위순)

1. **논문 작성 착수** (Phase 5): `THEORY_REFRAMING.md` §10의 구조와 Codex 제안 제목으로 TMLR 양식 초안. 기존 arXiv tex 소스 위치 확인 필요 (Overleaf 추정, repo에 없음).
2. Phase 2 결과의 통계 마감: 시드별 분산 표기, Wilcoxon 검정, OSC anytime의 log-loss 정규화 표 정리 (`scripts/15` 확장).
3. $U \to N$ ablation 소규모 런 (anchoring 가격 경감 검증, CIFAR 18 runs 정도, 로컬 GPU 가능).
4. 남은 이론 세부 (`THEORY_REFRAMING.md` §11): 정리 4의 $\sigma_2^2 > 0$ 직접 증명 (선택), true oracle 재산출 (toy regret 수치 최종화), run_all.py pruner 추가 (논문 외 인프라).

## 주의사항

- vast.ai 계정의 다른 인스턴스들 (ms_*, vatd_* 라벨 7개)은 다른 프로젝트 소속, 절대 건드리지 말 것.
- `paper/results_phase2/`와 `paper/data/`는 gitignore 대상 점검 후 커밋 정리 필요 (현재 워킹트리에 Phase 1~2 코드 변경분 미커밋).
- 논문의 주장 한계: 최종 성능 우위 주장 금지 (parity + 트레이드오프 지도화), anytime 방법 헤드라인은 averaging 계열에 양보하고 측정/특성화로 차별화.
