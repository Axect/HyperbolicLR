# SESSION_OFF 2026-08-19

## 현재 위치

HyperbolicLR 저널 프로젝트는 **ExpHyperbolicLR 방법 논문**으로 재정립되었다. G0 protocol freeze, G1a deterministic dry campaign, S1 full-data G1b는 모두 통과했다. G2 S1 Arm A는 clean commit `71b4de0`에서 345/2,880 artifact pair까지 진행된 뒤 worker와 recovery service가 종료되어 일시 중단된 상태다. Arm A audit와 cross-fitted analysis는 아직 생성되지 않았다.

## 이번 구간에서 확정한 것

1. **현재 DeepONet 결과:** `collider:~/HyperbolicLR/paper/results_extrap/integral-DeepONet/` 252 run에서 B=10 tuning, N=200 deployment 기준 EH가 관측 collapse 0/3, median final loss `1.2434816e-6`, B=10 대 B=50 selected-LR drift 4.7%를 함께 보였다. H, POW, WSQD보다 낮은 loss지만 HPO 선택은 scheduler-budget당 한 번이라 exploratory evidence다.
2. **Full-budget 조건:** B=50에서 다시 튜닝하면 WSQD의 N=200 median final loss `1.2638153e-6`이 EH `3.4664359e-6`보다 낮다. EH의 사용 조건은 full-budget retuning이 어렵고 deployment horizon이 바뀌는 경우다.
3. **Normalization campaign:** `collider:~/HyperbolicLR/paper/results_extrap/integral-DeepONetLN/` 81/81 run 완료. B=10 tuning의 N=200 median final loss는 L `1.132e-5`, P `1.956e-5`, EH `1.972e-5`다. EH가 단독 최선이 아니다. 여러 방법의 selected LR가 search lower bound `1e-4` 근처에 붙어 있으므로 corrected LR range로 S4를 재실행해야 한다.
4. **Mechanism audit:** resolution-1 sharpness 재측정과 full 27-run pre-exposure audit에서 dwell 또는 AdamW `38/eta` crossing이 collapse를 예측하지 못했다. 정상 run이 crossing하고 collapse run이 crossing하지 않는 경우가 모두 있다. Dwell과 sensitivity mass는 schedule descriptor일 뿐 collapse mechanism으로 쓰지 않는다.
5. **CIFAR 정정 유지:** 기존 8pp 격차는 scheduler별 LR를 튜닝하지 않은 fixed-lr0 결과다. Anchoring 가격이 아니다. Validation-loss anytime 순위도 과적합 때문에 accuracy와 반대로 움직여 classification 근거로 쓰지 않는다.
6. **전략 문서 재작성:** `PUBLICATION_PLAN.md`를 v13으로 교체했다. EH novelty, 금지 주장, fixed-shape landscape, replicated HPO, setting matrix, metric, 통계, success gate, figure plan, 원고 구조를 한 문서에 고정했다.
7. **이론 문서 재작성:** `THEORY_REFRAMING.md`를 v3으로 바꿨다. EH의 log-space coordinate를 중심에 놓고, `D+A>=H`와 conservation identity는 bookkeeping으로 강등했다. Stop-time minimax와 spectral theorem은 appendix 또는 별도 이론 논문으로 분리했다. EoS와 dwell 인과 주장은 폐기했다.
8. **G1b full-data gate:** S1 replicate 0에서 HPO 240개와 `N in {10,200}`, `K=3` deployment 48개를 완료했다. Expected key `288/288`, boundary selection 0개, non-finite 0개, audit hash `f52140a7f819a119a6e057ad9f19700053592aec6d4224f4efa6068c4732dd20`이다. Median runtime은 HPO trial당 `10.0685 s`, deployment epoch당 `1.0107 s`다.
9. **G1b sanity result:** N=200 held-out median loss는 EH `4.1004e-6`, H `1.8924e-6`, D2Z `2.4851e-6`이고 EH utility는 `2.0521 log10`으로 양수다. 한 HPO replicate의 결과이므로 방법 우열로 해석하지 않는다. G2와 G3가 사전 정의된 추론 단위다.
10. **G2 최초 실행 중단:** S1 Arm A의 expected count는 2,880이다. Collider L40 두 lane을 `setsid`로 시작했지만 lane 0과 lane 1이 각각 121개와 152개 artifact pair를 기록한 뒤 Python traceback 없이 종료됐다. 마지막 기록은 정상 parquet와 summary pair였고 최초 중단 시점의 합계는 273개였다. Kernel journal은 권한 부족으로 확인하지 못해 종료 원인은 미확정이다.
11. **G2 recovery 중단:** `hlr-g2-recover.service`가 expected-key 기반 resume를 수행해 2026-08-19 15:26:54 KST 기준 345 pair까지 늘렸다. Scheduler별 count는 EH 341, H 4다. Recovery log의 마지막 heartbeat는 15:22:45이고 당시 lane 0과 lane 1은 각각 1,279개와 1,256개가 남아 있었다. 이후 transient service와 두 worker가 함께 사라졌고 failure marker와 traceback은 남지 않았다. 현재 collider GPU 두 개는 idle, service는 inactive, `g2_arm_a.done`, final audit, `arm_a_S1.json`은 모두 없다.

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

1. 새 worker를 바로 띄우지 말고 collider 장기 process가 종료되는 원인을 먼저 진단한다. `setsid` lane과 transient user service가 모두 로그 없이 사라졌으므로 user manager lifecycle, cgroup, host maintenance, signal 기록을 확인한다.
2. `~/HyperbolicLR/v13/paper/results_confirmatory/arm-a/S1/`의 parquet 345개와 summary 345개가 1:1로 대응하는지 다시 확인하고, scheduler별 count EH 341, H 4가 유지되는지 확인한다.
3. Clean production checkout `71b4de0`과 frozen operator byte checksum을 다시 preflight한 뒤 lane별 missing set 1,279와 1,256에서 resume한다. Collider에서는 shared pueue daemon을 시작하지 않는다.
4. Durable supervisor는 worker exit code와 signal을 보존하고 missing job만 재실행해야 한다. 2,880 pair 완료 후 full provenance audit와 `results_confirmatory/analysis/arm_a_S1.json` 생성을 같은 supervisor가 수행한다.
5. G2 audit가 통과하면 EH transfer regret `<=0.1 log10`과 positive utility gate를 판정한다. 이 gate를 통과한 뒤에만 S1 full Arm B를 실행하고, S2와 S3는 S1 성공 뒤 연다.

## 핵심 파일

- `PUBLICATION_PLAN.md`: v13 단일 전략 문서와 confirmatory protocol.
- `THEORY_REFRAMING.md`: v3 EH 중심 이론과 appendix 경계.
- `manuscript/v1_arxiv/`: arXiv v1 LaTeX 원본.
- `paper/confirmatory_protocol.py`, `paper/protocol/confirmatory_v13.json`: frozen candidate grid, scheduler shape, seed blocks, metric, schema.
- `paper/confirmatory_data.py`, `paper/protocol/operator_v13.json`, `paper/protocol/cifar10_v13.json`: split manifests와 byte checksum gate.
- `paper/confirmatory_runner.py`: atomic single-run trainer와 provenance writer.
- `paper/confirmatory_orchestrator.py`: production matrix, deterministic lane sharding, resume, selection, completeness와 provenance audit.
- `paper/confirmatory_analysis.py`: cross-fitted Arm A metrics와 frozen confirmatory statistics.
- `paper/_provision_confirmatory_v13.sh`, `paper/_run_confirmatory.sh`: collider preflight와 deterministic lane entry points.
- `collider:~/HyperbolicLR/logs/g2_arm_a_gpu{0,1}.log`: 최초 `setsid` lane 로그.
- `collider:~/HyperbolicLR/logs/g2_arm_a_recovery.log`, `g2_arm_a_gpu{0,1}_recovery.log`: 두 번째 resume와 마지막 heartbeat 기록.
- `collider:~/HyperbolicLR/logs/g2_recover.py`: untracked operational recovery watcher. Source protocol은 바꾸지 않지만 transient service 자체가 사라졌으므로 그대로 재실행하지 말고 liveness 원인부터 확인한다.
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

HyperbolicLR GPU 작업은 collider에서만 실행한다. Collider는 L40 2개다. Darkmatter와 collider는 `/home` NFS와 pueue state를 공유하므로 collider에서 pueue daemon을 시작하지 않는다. 이 workstation의 local checkout은 공유하지 않으므로 clean commit은 Git으로, ignored operator data는 exact byte archive로 collider에 보낸다. Darkmatter의 Neural_Hamilton과 VaTD 작업은 건드리지 않는다. `setsid`와 transient `systemd-run --user --collect`가 이번 장기 실행을 끝까지 유지하지 못했으므로 다음 세션에서는 둘 중 하나를 그대로 반복하지 않는다.

2026-08-19 15:26:54 KST snapshot에서 G2 Arm A는 345/2,880 pair, GPU 0과 1은 idle, `hlr-g2-recover.service`는 inactive다. 결과 population은 EH 341, H 4이며 나머지 scheduler는 아직 0이다. 완료 또는 실패 판정이 아니라 liveness 문제로 중단된 미완성 campaign이다.

## Working tree

Exploratory campaign `7374984`, recovered arXiv source `0775af5`, confirmatory freeze `8d03fba`, tracked paper scheduler fix `71b4de0`, G1b record `8719e9b`를 origin에 push했다. Collider production checkout은 `71b4de0` clean state를 유지하며, G2 raw artifacts와 operational recovery logs는 gitignored remote data다.
