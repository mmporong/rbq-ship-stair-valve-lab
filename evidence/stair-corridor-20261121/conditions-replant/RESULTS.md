# 재디딤 제어 17조건 결과

`rl-mid-replant-k3075-v1` 제어를 같은 고정 입력에 적용한 CPU MuJoCo 조건표다. 물리 14행은 바닥에서 작업 뒤 다음 착지까지의 `through-task` 범위이며, 전체 157착지 일정의 완료 판정과 구분한다.

이 범위에서는 성공 0건, 물리 실패 14건, 계획 실패 3건.

| 조건 | 실행 | 정격 조작(M1) | N1 | 최초 위반 |
|---|---:|---:|---:|---|
| [기준 조건](scenes/nominal.png) | physical | FAIL | FAIL | stance_slip_m 10.000208 mm (한계 ≤ 10.000000 mm) @ 127.831 s |
| [선박형 경사 60°](scenes/slope_60.png) | planning | N/A | N/A | unexpected_overlap_m 0.684302 mm (한계 0.100000 mm) @ 13.600 s |
| [단 높이 165 mm](scenes/rise_165.png) | planning | N/A | N/A | max_foot_ik_error_m 2.213606 mm (한계 2.000000 mm) @ 55.960 s |
| [단 높이 200 mm](scenes/rise_200.png) | planning | N/A | N/A | max_foot_ik_error_m 6.968600 mm (한계 2.000000 mm) @ 34.520 s |
| [발–계단 마찰 0.50](scenes/floor_mu_050.png) | physical | FAIL | FAIL | qp_attempt 1 (한계 0) @ 127.656 s |
| [발–계단 마찰 0.80](scenes/floor_mu_080.png) | physical | FAIL | FAIL | stance_slip_m 10.000560 mm (한계 ≤ 10.000000 mm) @ 127.830 s |
| [밸브 저항 5 N·m](scenes/resistance_5.png) | physical | FAIL | FAIL | max_base_shift_m 10.026888 mm (한계 ≤ 10.000000 mm) @ 100.620 s |
| [밸브 저항 15 N·m](scenes/resistance_15.png) | physical | FAIL | FAIL | max_base_shift_m 10.011330 mm (한계 ≤ 10.000000 mm) @ 102.579 s |
| [레버 0.10 m](scenes/sensitivity_lever_m_0_1.png) | physical | FAIL | FAIL | pad_contact_loss_s 0.021 (한계 ≤ 0.02) @ 100.558 s |
| [레버 0.14 m](scenes/sensitivity_lever_m_0_14.png) | physical | FAIL | FAIL | stance_slip_m 10.000001 mm (한계 ≤ 10.000000 mm) @ 125.939 s |
| [패드 마찰 0.6](scenes/sensitivity_pad_mu_0_6.png) | physical | FAIL | FAIL | stance_slip_m 10.000984 mm (한계 ≤ 10.000000 mm) @ 127.873 s |
| [패드 마찰 1.0](scenes/sensitivity_pad_mu_1.png) | physical | FAIL | FAIL | angle_deg 89.672° (한계 90.0 +/- 0.25) @ 111.651 s |
| [손가락 힘 한계 75 N](scenes/sensitivity_finger_force_limit_n_75.png) | physical | FAIL | FAIL | pad_contact_loss_s 0.021 (한계 ≤ 0.02) @ 100.460 s |
| [손가락 미끄럼 손실 15 N](scenes/sensitivity_finger_slide_frictionloss_n_15.png) | physical | FAIL | FAIL | stance_slip_m 10.000583 mm (한계 ≤ 10.000000 mm) @ 127.831 s |
| [팔 관절 한계 LIMS-EX](scenes/sensitivity_arm_limits_named_lims_ex_joint_limits.png) | physical | FAIL | FAIL | requested_torque_ratio 2.00682 (한계 ≤ 1) @ 100.401 s |
| [물리 간격 2.5 ms](scenes/sensitivity_timestep_s_0_0025.png) | physical | FAIL | FAIL | start_angle_deg 0.379° (한계 절댓값 ≤ 0.250°) @ 100.402 s |
| [접촉 임피던스 비 10](scenes/sensitivity_impratio_10.png) | physical | FAIL | FAIL | stance_slip_m 10.001475 mm (한계 ≤ 10.000000 mm) @ 4.197 s |

- 스레드 고정 물리행은 성공 0/14다. nominal 자체가 stance_slip_m 10.000208 mm (한계 ≤ 10.000000 mm) @ 127.831 s로 실패하고 M1도 실패해 성공 기준점이 없으므로 이 14행으로 pass/fail 강건성 브래킷을 만들 수 없다. 각 민감도 행은 최초 물리 위반 비교로만 사용한다.
- 발표 표의 물리 14행은 OPENBLAS/MKL/OMP=1, CUDA_VISIBLE_DEVICES=-1로 고정했다. 환경변수 미설정 14행은 `COMPARISON.md`에 분리했으며 공식 행과 섞지 않았다.
- geometry 3행은 물리 실행 전 참조 생성 실패다. M1/N1에 도달하지 않아 N/A이며 물리 14행의 분모에 포함하지 않는다.
- 60° 행의 13.600초 몸통–3단 겹침은 고정 계획의 결과다. [`별도 개선 계획`](../ship60/report.json)은 이 충돌을 피한 뒤 38.060초 RL swing IK 2.927358 mm에서 중단됐으므로 두 결과를 섞지 않았다.
- 임피던스비 10은 실제 시간순 최초 위반인 stance_slip_m 10.001475 mm (한계 ≤ 10.000000 mm) @ 4.197 s와 재디딤 판정 physical_replant 실패 (무하중·양의 들기·35 mm 이내 재착지 조건) @ 81.600 s를 함께 기록했다. 표 링크는 시간순 최초 위반 직전, 추가 링크는 [`재디딤 실패 직전`](scenes/sensitivity_impratio_10-replant.png)이다.
- 성공 물리행 그림은 마지막 실측 프레임, 실패 물리행 그림은 최초 위반 직전 프레임이다.
- 원래 일정 156착지에 재디딤 1착지를 추가해 전체 일정은 157착지다. 이 표의 nominal 단기 범위는 20착지를 평가한다.
- 재생 기록은 `.omx/mission-success-20261121/conditions-replant-one-thread`와 `.omx/mission-success-20261121/conditions-replant`에 있다. 보관한 항목은 각각 스레드 고정 발표 원시와 환경변수 미설정 대조 원시다. evidence에는 원시를 복사하지 않고 SHA-256·공식 재판정·장면만 넣었다.

동결 당시 소스를 다시 실행하려면 아래처럼 commit `2c91d83`의 9개 source/input 경로와 고정 바이너리 입력, 준비된 RBQ 자산을 한 트리에 모은다. [`replay.py`](replay.py)는 실행 전에 SHA-256을 대조한다.

```powershell
cd "$HOME/go2-wall-reach-lab"
$sourceRoot = ".local/source-2c91d83"
New-Item -ItemType Directory -Force "$sourceRoot" | Out-Null
$sourcePaths = @(
  "src"
  "tools"
  "configs"
  "pyproject.toml"
  "evidence/stair-mission-20261121/protocol.json"
  "evidence/t1-stair-ascent-20261120/target-round-trip-geometry.json"
  "evidence/t1-stair-ascent-20261120/target-round-trip-plan.json"
  "evidence/t1-stair-posture-20261120/front-pair.json"
  "evidence/t3-valve-20261120/fixed-body-candidates.json"
)
git archive --format=tar --output="$sourceRoot.tar" 2c91d83 -- $sourcePaths
tar -xf "$sourceRoot.tar" -C "$sourceRoot"
New-Item -ItemType Directory -Force "$sourceRoot/.assets" | Out-Null
Copy-Item -Recurse ".assets/rbq" "$sourceRoot/.assets/rbq"
New-Item -ItemType Directory -Force "$sourceRoot/evidence/stair-corridor-20261121/local-repair" | Out-Null
Copy-Item "evidence/stair-corridor-20261121/local-repair/walk50.npz" "$sourceRoot/evidence/stair-corridor-20261121/local-repair/walk50.npz"
New-Item -ItemType Directory -Force "$sourceRoot/evidence/stair-corridor-20261121/tracked/low-control" | Out-Null
Copy-Item "evidence/stair-corridor-20261121/tracked/low-control/loaded.json.gz" "$sourceRoot/evidence/stair-corridor-20261121/tracked/low-control/loaded.json.gz"
.\.venv\Scripts\python.exe evidence/stair-corridor-20261121/conditions-replant/replay.py all `
  --source-root "$sourceRoot" `
  --run-root ".local/t4-conditions-replant" `
  --jobs 3 --cpu-environment one-thread-cpu
```
