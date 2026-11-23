# 7판 내리막 진단과 8판 실험

임무 목표는 RBQ-10+RB1이 선박형 계단을 올라 10N·m 밸브를 90° 잠근 뒤 내려오는 것이다. 이번 실험은 T1의 내리막 정체를 계측하고, 목표 계단에서 연속 왕복을 판정할 수 있게 한다. T3의 동적 밸브 조작은 T1 왕복을 통과한 다음에 검증한다. [합격 기준](../PLAN.md)

## 7판에서 정체되는 시점

7판의 고정 CPU 번들과 5ms 물리·50Hz 제어를 사용했다. 5·6단계에서 닫힌·열린 내리막을 각각 6개씩 재생했다. 출발 평판에서 +X 방향으로 이어지는 발판에 발이 수직 하중 1N 이상을 싣는 첫 순간을 물리 스텝마다 찾았다. 재생 24개의 기존 거리·생존·높이 수치는 `sim2sim-4.0.0.json`의 같은 시작점과 일치했다. [접촉·속도·자세 시계열](../evidence/t1-v8-20261120/v7-descent-physics.json)

| 지형 | 아래 단 첫 접촉 | 마지막 2초의 전진 속도 중앙값 | 마지막 2초의 몸통 기울기 중앙값 |
|---|---:|---:|---:|
| 5단계 닫힌 내리막 | 5/6 | 0.344m/s | 2.75° |
| 5단계 열린 내리막 | 6/6 | 0.135m/s | 20.85° |
| 6단계 닫힌 내리막 | 0/6 | 0.00094m/s | 0.42° |
| 6단계 열린 내리막 | 0/6 | −0.00047m/s | 0.36° |

6단계의 12개 로봇은 0.6m/s 전진 명령을 유지했지만 모두 아래 단에 하중을 싣기 전에 멈췄다. 닫힌 계단에서는 앞발 중심이 평판 모서리를 최대 39mm 넘은 시행도 있었다. 발의 x좌표와 발판 접촉점을 함께 기록했다.

## 8판 학습 설정

7판 `model_8997.pt`에서 종료 벌점 가중치를 `−200`에서 `−50`으로 낮춰 PPO를 500회 더 학습했다. 7판과 같은 2,048개 환경, 10×10 지형, 시드 42, 24단계 수집 주기를 사용했다. 저장한 환경 설정을 대조한 결과 가중치 한 줄만 달랐다. 비교에 쓰는 체크포인트는 `9000`, `9100`, `9200`, `9300`, `9400`, `9496`이다. 반복 시간 중앙값은 4.37초였다. 재개 시 지형 단계는 0~2로 초기 배치되며, 종료 벌점 −200으로 500회를 더 학습한 대조군은 없다. 이번 비교는 두 정책의 동작 차이를 보여준다. [학습 기록과 체크포인트 해시](../evidence/t1-v8-20261120/training-run.json)

초기 20열 실행은 반복당 42~44초를 기록했다. 7판 설정을 대조한 뒤 그 실행의 로그·체크포인트·첫 장면을 보존하고 10열로 다시 시작했다. 10열 실행의 첫 두 반복은 4.52초와 3.76초였다. 20열 실행은 성능 비교 표본에서 제외했다.

## 같은 시작점에서의 CPU 하강 비교

보행 비교에 쓴 7판과 8판의 번들은 같은 장면 XML과 같은 시작 배열을 사용했다. 아래 표는 5ms 물리와 50Hz 제어에서 종료 없이 19.96초를 버티고, 월드 +X로 3m 넘게 이동하며, 몸통이 시작보다 0.10m 넘게 내려가고, 원래 계단의 아래 단에 하중을 실은 시행 수다. XY 거리로만 센 통과 수와 전체 높이 변화도 비교 자료에 남겼다. [시작점·거리·높이 비교](../evidence/t1-v8-20261120/matched-comparison.json)

| 정책 | 5단계 닫힌 | 5단계 열린 | 6단계 닫힌 | 6단계 열린 |
|---|---:|---:|---:|---:|
| 7판 `8997` | 3/6 | 4/6 | 0/6 | 0/6 |
| 8판 `9000` | 4/6 | 6/6 | 6/6 | 0/6 |
| 8판 `9100` | 4/6 | 2/6 | 5/6 | 1/6 |
| 8판 `9200` | 4/6 | 2/6 | 5/6 | 5/6 |
| 8판 `9300` | 4/6 | 2/6 | 4/6 | 1/6 |
| 8판 `9400` | 4/6 | 2/6 | 2/6 | 1/6 |
| 8판 `9496` | 3/6 | 3/6 | 2/6 | 0/6 |

`9200`의 하강 판정을 통과한 6단계 시행에서 몸통 하강 중앙값은 닫힌 계단 1.07m, 열린 계단 1.44m였다. 5단계 열린 내리막의 생존은 7판 6/6에서 `9200` 5/6으로 줄었다. 최종 `9496`은 6단계 열린 내리막 6개 모두 살아남았고, 3m 넘게 이동한 시행은 0개였다. [중간 하강 시계열](../evidence/t1-v8-20261120/v8-9200-descent-physics.json), [최종 하강 시계열](../evidence/t1-v8-20261120/v8-9496-descent-physics.json)

0~9단계 전체 300개에서 7판은 282개, `9496`은 277개가 살아남았다. 5단계 생존은 30/30에서 29/30으로 줄었고 생존 중 XY 3m 초과는 두 정책 모두 29/30이었다. `9496`은 900개 기준 상태에서 관측·정책 출력·제어 목표값 검사를 통과했다. 관절 위치 목표 최대 차이는 0.00003696rad였다. [최종 300개 CPU 평가](../evidence/t1-v8-20261120/v8-9496-native-4.0.0.json)

학습 중에는 체크포인트마다 WSL CPU에서 소수 표본을 장면으로 남겼다. 최종 장면의 열린 내리막은 두 로봇 모두 생존했고, 장면에 기록된 XY 거리 대표값은 0.65m였다. [체크포인트 장면 6개](../evidence/t1-v8-20261120/snapshots/)

## 목표 계단에서의 왕복 판정

우선 T0의 열린 계단 장면을 경사 50°, 단 높이 19cm, 발판 10개로 고정했다. 발판 깊이 127mm, 폭 610mm, 두께 30mm는 프로젝트 형상 가정이다. 발판 앞뒤 간격은 계산상 159mm이고 열린 틈은 약 32mm다. 동결한 CPU 평가 장면의 9단계 열린 오르막 첫 발판은 폭 2.13~2.26m다. [형상 가정](../configs/d1_parameters.json), [왕복 평가기](../train/eval_target_stairs.py)

출발 평면에서 상부 착지면까지 오른 뒤 180° 회전하고, 같은 계단을 내려와 출발 위치 근처에 정지하는 동안 한 시뮬레이션 상태를 유지한다. 5ms마다 몸통과 네 발의 계단 폭 및 착지면 이탈, 접촉면과 관절 구동기 토크를 확인한다. 오르막과 내리막 모두 아래·가운데·위 구간에서 각각 하중을 실은 발판 접촉을 확인한다. 출발점 15cm 안에서 네 발이 바닥을 지지하고, 기울기 10° 이내와 선속도 0.05m/s·각속도 0.15rad/s 이하가 0.5초 이어지면 복귀로 판정한다.

7판은 0.35·0.5·0.6m/s 모두 1단에 발이 닿았고, 몸통 충돌체가 3단에 닿아 오르막이 종료됐다. 첫 종료 접촉은 각각 3.48초·2.79초·2.37초에 기록됐다. [목표 계단 기준 결과](../evidence/t1-v8-20261120/v7-target-50deg-route.json)

`9496`도 같은 세 속도에서 모두 1단 지지 후 몸통이 3단과 충돌했다. 첫 종료 접촉은 각각 3.56초·4.225초·2.855초였다. 0.5m/s 시행에서는 3.28초에 앞왼발이 첫 발판을 지난 지점의 바닥을 짚었다. `9200`도 세 속도 모두 2단 이내에서 끝났다. 이 평가에서 목표 형상의 연속 왕복을 통과한 정책은 없었다. 별도의 발 접촉 계획과 토크 제어로 N1을 검증했다. [최종 목표 계단 정책 재생](../evidence/t1-v8-20261120/v8-9496-target-50deg-route.json), [N1 왕복](stair-transfer-50deg.md)

목표 계단의 시작 자세를 고정하고 정책과 명령 속도를 바꿔 33개 조합을 재생했다. 7판과 8판 6개 체크포인트의 동일 속도 비교 21회와 `9000`·`9200`·`9496`의 추가 속도 비교 12회를 합친 결과는 왕복 0/33이다. 어느 시행에서도 3단 이상의 발판에 하중을 실은 발 접촉은 없었다. [추가 속도 최종 정책](../evidence/t1-v8-20261120/v8-9496-target-speed-sweep.json), [몸통과 3단이 부딪힌 장면](../evidence/t1-train/failures/v8-9496-target-50deg-body-tread3.png)

## 중간 체크포인트의 관측 대조

`9200`의 관측 기준 900개 중 9단계 한 상태에서 높이 레이 한 개가 어긋났다. 원본 MuJoCo 3.5 장면과 내보낸 XML을 같은 3.5에서 재생해 원인을 확인했다. XML 직렬화가 발판 경계를 46μm 옮기면서 같은 레이가 188mm 높은 인접 발판에 닿았다. 그 두 발판의 원본 좌표를 돌려놓은 검증용 번들은 기존 허용 오차로 900개를 모두 통과했다. 정책 간 CPU 보행 비교는 공통 XML과 같은 시작 배열을 사용했다. [레이 재현](../evidence/t1-v8-20261120/v8-9200-ray-seam.json), [900개 재검증](../evidence/t1-v8-20261120/v8-9200-parity-corrected.json)

## 목표 계단의 진입 동작 비교

같은 50°·19cm 계단과 동결 정책으로 140개 진입을 CPU에서 재생했다. 시작 자세, 마찰 0.6, 5ms 물리 스텝, 발판·토크·왕복 판정은 기준 재생과 같다. 속도·보행 위상 표본 104개에는 기존 세 속도의 기준 재생 12개가 들어 있다. [실험 코드](../train/probe_stair_entry.py)

| 진입 조건 | 시행 수 | 상부 도달 | 몸통·팔 종료 접촉 | 전복 | 단계 시간 초과 |
|---|---:|---:|---:|---:|---:|
| 접근 속도·보행 위상 | 104 | 0 | 93 | 4 | 7 |
| 다리 관절 목표 보정 | 36 | 0 | 17 | 6 | 13 |

속도·보행 위상 비교는 체크포인트 4개에서 정책에 들어가는 위상을 제어 스텝 단위로 앞당기거나 오르막 속도를 바꿨다. 속도 시험은 몸통 X −0.65m부터 −0.35m 사이에서 0.5m/s를 0.22m/s 또는 0.75m/s로 바꾸고, −0.35m보다 앞에서는 새 속도를 유지했다. `9100`의 0.6m/s·위상 +25스텝 시행에서만 3단 하중 접촉이 나왔다. 2.245초에 발판 아래 바닥 지지, 2.935초에 앞왼발의 3단 수직 하중 161N, 4.30초에 전복이 기록됐다. 3단 접촉 시 몸통 X는 계단 시작보다 0.135m 뒤였다. [104개 실측](../evidence/t1-entry-cpu-20261120/timing.json), [접촉 시각과 하중](../evidence/t1-entry-cpu-20261120/tread3-touch.json), [3단 접촉 장면](../evidence/t1-train/failures/v8-9100-tread3-touch-before-fall.png)

다리 관절 목표 보정은 기본 자세의 발 자코비안으로 35mm 또는 70mm 아래쪽 발 목표에 해당하는 관절 변화를 구했다. 계단 진입에서 앞다리 또는 네 다리의 관절 목표에 이 변화를 더하고 관절 범위를 적용했다. 실제 몸통 높이와 발 하중은 물리 재생에서 측정했다. 36개 모두 2단 이하의 하중 접촉에 그쳤다. 가장 멀리 전진한 시행은 몸통 X가 3.50m에 도달했으나 상부에 올라서지 못했고 허용 경로 밖의 지지면을 밟았다. [36개 실측](../evidence/t1-entry-cpu-20261120/extension.json)

현재 시험한 범위에서는 진입 속도·보행 위상 조정과 진입 관절 보정 모두 N1 왕복을 충족하지 못했다. T3의 시작 조건은 목표 계단의 물리 왕복이다.

## 재현

`docs/stair-policy-sim2sim.md`의 CPU 번들이 있는 Windows 작업 디렉터리에서 실행한다. 결과 경로는 시행마다 새로 고른다.

```powershell
Set-Location "$HOME/go2-wall-reach-lab"
$py = '.venv/Scripts/python.exe'
$bundle = '.omx/q1-v7-cpu-bundle'
$run = 'evidence/t1-v8-repeat'
$env:OPENBLAS_NUM_THREADS = '1'
& $py train/diag_descent.py $bundle "$run/v7-descent-physics.json"
& $py train/eval_target_stairs.py $bundle "$run/v7-target-50deg-route.json"
```

8판의 학습 진입점은 고정 `unitree_rl_mjlab` 체크아웃의 Python에서 실행한다.

```bash
windows_home=$(powershell.exe -NoProfile -Command '$env:USERPROFILE' | tr -d '\r')
repo="$(wslpath "$windows_home")/go2-wall-reach-lab"
cd ~/go2-train/unitree_rl_mjlab
.venv/bin/python "$repo/train/rbq_rb1_stairs.py" train stairs \
  --env.scene.num-envs 2048 --env.scene.terrain.terrain-generator.num-cols 10 \
  --env.rewards.is-terminated.weight -50 \
  --agent.resume True --agent.load-run 2026-11-10_17-45-51_v7 \
  --agent.load-checkpoint model_8997.pt --agent.max-iterations 500 \
  --agent.save-interval 100 --agent.seed 42 --agent.logger tensorboard \
  --agent.run-name v8_term50_10col
```

8판 체크포인트를 가져올 때는 같은 WSL 환경에서 CPU 번들을 만든다. 주어진 실행 디렉터리는 이번 학습의 결과 경로다.

```bash
run=logs/rsl_rl/rbq10_rb1_stairs/2026-11-20_11-52-41_v8_term50_10col
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python "$repo/train/export_stairs_cpu.py" \
  "$run/model_9496.pt" "$repo/.omx/t1-v8-cpu-9496-repeat" --samples-per-column 3
```

Windows에서 동일한 평가를 실행한다.

```powershell
$bundle = '.omx/t1-v8-cpu-9496-repeat'
& $py train/eval_stairs_cpu.py $bundle "$run/v8-9496-native-4.0.0.json" --workers 4
& $py train/diag_descent.py $bundle "$run/v8-9496-descent-physics.json"
& $py train/eval_target_stairs.py $bundle "$run/v8-9496-target-50deg-route.json"
```

진입 비교를 재현하려면 `9000`·`9100`·`9200`·`9496`을 CPU 번들로 가져와 `.omx/t1-v8-cpu-<체크포인트>`에 둔다. 위 PowerShell 작업 디렉터리에서 새 출력 경로를 사용한다.

```powershell
& $py train/probe_stair_entry.py .omx 'evidence/t1-entry-repeat/timing.json' --mode timing --workers 6
& $py train/probe_stair_entry.py .omx 'evidence/t1-entry-repeat/extension.json' --mode extension --workers 4
```
