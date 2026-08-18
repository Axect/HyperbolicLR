# SESSION_OFF 2026-08-18

## 현재 위치

HyperbolicLR 저널 프로젝트는 **ExpHyperbolicLR 방법 논문**으로 재정립되었다. 문제는 cross-budget HPO transfer다. 작은 budget에서 고른 설정을 긴 horizon에 그대로 배포할 때 tuning fidelity와 long-horizon utility를 함께 보존하는지가 핵심이다. G0 protocol freeze와 G1a deterministic dry campaign은 완료되었고, 다음 gate는 S1 full-data G1b다.

## 이번 구간에서 확정한 것

1. **현재 DeepONet 결과:** `collider:~/HyperbolicLR/paper/results_extrap/integral-DeepONet/` 252 run에서 B=10 tuning, N=200 deployment 기준 EH가 관측 collapse 0/3, median final loss `1.2434816e-6`, B=10 대 B=50 selected-LR drift 4.7%를 함께 보였다. H, POW, WSQD보다 낮은 loss지만 HPO 선택은 scheduler-budget당 한 번이라 exploratory evidence다.
2. **Full-budget 조건:** B=50에서 다시 튜닝하면 WSQD의 N=200 median final loss `1.2638153e-6`이 EH `3.4664359e-6`보다 낮다. EH의 사용 조건은 full-budget retuning이 어렵고 deployment horizon이 바뀌는 경우다.
3. **Normalization campaign:** `collider:~/HyperbolicLR/paper/results_extrap/integral-DeepONetLN/` 81/81 run 완료. B=10 tuning의 N=200 median final loss는 L `1.132e-5`, P `1.956e-5`, EH `1.972e-5`다. EH가 단독 최선이 아니다. 여러 방법의 selected LR가 search lower bound `1e-4` 근처에 붙어 있으므로 corrected LR range로 S4를 재실행해야 한다.
4. **Mechanism audit:** resolution-1 sharpness 재측정과 full 27-run pre-exposure audit에서 dwell 또는 AdamW `38/eta` crossing이 collapse를 예측하지 못했다. 정상 run이 crossing하고 collapse run이 crossing하지 않는 경우가 모두 있다. Dwell과 sensitivity mass는 schedule descriptor일 뿐 collapse mechanism으로 쓰지 않는다.
5. **CIFAR 정정 유지:** 기존 8pp 격차는 scheduler별 LR를 튜닝하지 않은 fixed-lr0 결과다. Anchoring 가격이 아니다. Validation-loss anytime 순위도 과적합 때문에 accuracy와 반대로 움직여 classification 근거로 쓰지 않는다.
6. **전략 문서 재작성:** `PUBLICATION_PLAN.md`를 v13으로 교체했다. EH novelty, 금지 주장, fixed-shape landscape, replicated HPO, setting matrix, metric, 통계, success gate, figure plan, 원고 구조를 한 문서에 고정했다.
7. **이론 문서 재작성:** `THEORY_REFRAMING.md`를 v3으로 바꿨다. EH의 log-space coordinate를 중심에 놓고, `D+A>=H`와 conservation identity는 bookkeeping으로 강등했다. Stop-time minimax와 spectral theorem은 appendix 또는 별도 이론 논문으로 분리했다. EoS와 dwell 인과 주장은 폐기했다.

## 논문의 현재 한 문장

> ExpHyperbolicLR applies a bounded shared-asymptote hyperbolic arc in log learning-rate space to preserve low-budget hyperparameter choices while continuing to exploit longer training horizons.

이 문장은 confirmatory experiment의 사전 가설이다. DeepONet, TFONet, CIFAR ResNet에서 독립 HPO 반복으로 통과하기 전에는 일반 사실로 쓰지 않는다.

## Confirmatory protocol

- Core methods: EH, H, E, POW, WSQD, C, D2Z, WSD.
- Required settings: S1 DeepONet plus AdamW, S2 TFONet plus AdamW, S3 CIFAR ResNet-18 plus SGD momentum.
- Confirmatory methods EH, H, POW, WSQD는 독립 HPO `R=12`, descriptive baselines E, C, D2Z, WSD는 `R=3`.
- HPO budget은 10 epochs, scheduler와 replicate당 30 trials.
- 선택 설정별 deployment seed는 `K=3`.
- Horizons: 10, 50, 100, 200.
- Pruner: `NopPruner()` 명시. Pruning kill-rate는 현재 논문 범위 밖이다.
- Primary metrics: within-family transfer regret, long-horizon utility, N=200 held-out metric.
- Inference unit: HPO replicate cluster. Deployment seed를 독립 HPO 표본으로 세지 않는다.
- Direct confirmatory comparisons: EH 대 H, POW, WSQD. Holm correction.

## 성공 기준

1. S1의 EH transfer regret가 `0.1 log10` 이내이고 utility가 양수다.
2. S2 또는 S3에서 같은 방향이 재현된다.
3. EH가 H보다 개선되고 POW와 WSQD에 대해 사전 margin 안에서 non-inferior한 setting이 2개 이상이다.
4. 결론이 한 HPO replicate, 한 deployment seed, 또는 `U=400` 한 점에 의존하지 않는다.

Operator setting 두 개에서만 통과하면 제목과 주장을 neural operator로 좁힌다. S1만 통과하면 Machine Learning 제출을 중단한다. EH가 POW 또는 WSQD에 두 setting 이상 명확히 열위면 EH 단독 권장 주장을 버린다.

## 다음 작업

1. Confirmatory 구현을 한 clean commit으로 고정하고 origin과 collider의 `~/HyperbolicLR`에 배포한다.
2. Collider에서 `_provision_confirmatory_v13.sh`를 실행해 environment lock, protocol hash, operator와 CIFAR source checksum, GPU 2개, clean working tree를 확인한다.
3. S1 replicate 0 G1b HPO 240개를 두 lane으로 실행한 뒤 selection이 30-point grid endpoint에 붙지 않았는지 audit한다.
4. Search-boundary gate를 통과하면 같은 replicate를 보존하고 `N in {10,200}`, `K=3` deployment 48개를 실행한다. Audit에서 expected-key set, provenance, test isolation, runtime을 확인한다.
5. G1b를 통과하면 S1 Arm A 2,880 runs를 실행하고 transfer regret와 utility gate를 판정한다. 그 뒤에만 S1 full Arm B와 S2, S3를 순서대로 연다.

## 핵심 파일

- `PUBLICATION_PLAN.md`: v13 단일 전략 문서와 confirmatory protocol.
- `THEORY_REFRAMING.md`: v3 EH 중심 이론과 appendix 경계.
- `manuscript/v1_arxiv/`: arXiv v1 LaTeX 원본.
- `paper/confirmatory_protocol.py`, `paper/protocol/confirmatory_v13.json`: frozen candidate grid, scheduler shape, seed blocks, metric, schema.
- `paper/confirmatory_data.py`, `paper/protocol/operator_v13.json`, `paper/protocol/cifar10_v13.json`: split manifests와 byte checksum gate.
- `paper/confirmatory_runner.py`: atomic single-run trainer와 provenance writer.
- `paper/confirmatory_orchestrator.py`: production matrix, deterministic lane sharding, resume, selection, completeness와 provenance audit.
- `paper/confirmatory_analysis.py`: cross-fitted Arm A metrics와 frozen confirmatory statistics.
- `paper/_provision_confirmatory_v13.sh`, `paper/_run_confirmatory.sh`: collider preflight와 pueue entry points.
- `collider:~/HyperbolicLR/paper/results_extrap/integral-DeepONet/`: 252-run 주 evidence.
- `collider:~/HyperbolicLR/paper/results_extrap/integral-DeepONetLN/`: 81-run normalization ablation.

## 금지 주장

- EH가 모든 task와 optimizer에서 최선이다.
- EH만 horizon-free다.
- Dwell 또는 EoS crossing이 collapse를 일으킨다.
- CIFAR 8pp가 anchoring 가격이다.
- `D+A>=H`가 강한 novelty theorem이다.
- 한 HPO seed에서 얻은 결과를 deployment seed 수만큼 복제해 유의성을 만든다.

## 호스트 정책

HyperbolicLR GPU 작업은 collider에서만 실행한다. Collider는 L40 2개이고 long-running worker는 pueue로 관리한다. Darkmatter와 collider는 `/home` NFS를 공유하지만 이 workstation의 local checkout과는 공유하지 않으므로 clean commit은 Git으로, ignored operator data는 rsync로 collider에 보낸다. Darkmatter의 Neural_Hamilton과 VaTD 작업은 건드리지 않는다.

현재 collider에 실행 중인 HyperbolicLR worker는 없다.

## Working tree

이번 변경은 exploratory campaign, recovered arXiv source, confirmatory freeze의 세 logical commit으로 분리했다. Production runner는 dirty tree를 거부하므로 G1b는 이 commit들을 origin과 collider에 동기화한 뒤 시작한다.
