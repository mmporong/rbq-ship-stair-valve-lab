# 50° 계단 위 밸브 물리 검증

RBQ-10+RB1이 단 높이 19cm인 계단의 5단에 앞발 두 개, 3단에 뒷발 두 개를 딛고 밸브를 닫았다. 자세 탐색은 N1 연속 왕복의 발 배치와 몸통 자세를 기준으로 했다. 선택한 시행은 몸통 높이 목표를 그 자세보다 40mm 낮췄다. [N1 왕복](stair-transfer-50deg.md), [자세 및 팔 궤적 후보](../evidence/t3-valve-20261120/fixed-body-candidates.json)

밸브에는 0.12m 레버와 0~90° 회전 범위, 마찰 손실 10N·m인 수동 힌지를 뒀다. 0.12m는 T2에서 사용한 손잡이 길이 가정이다. 팔 끝과 레버 끝은 힘을 주고받는 이상화한 핀으로 연결하고, 연결된 상태에서 재생을 시작했다. 제어기는 로봇의 기존 관절 18개에 토크 명령을 보냈다. 힌지의 마찰 반력과 핀을 통해 힌지에 전달된 토크를 각 구속에서 계산했다. [밸브 용량과 입력 근거](valve-capacity.md), [장면과 힘 관측 코드](../src/go2_wall/stair_valve.py)

선택한 손잡이 위치의 첫 재생에서 실제 회전 저항은 최대 5.84N·m였다. 관절 마찰에 `solref`의 음수 형식 감쇠를 적용해 회전 중 측정 저항을 10.0N·m로 맞췄다. 이때 5초 유지 구간에서 밸브가 약 89.7°에 머물렀다. 마지막 정착 구간에서 3N·m에 해당하는 힘을 로봇의 팔·다리 힘 분배에 넣어 닫힘 위치를 유지했다. 이 값은 제어기의 예상 반력이며 힌지의 마찰 손실 설정은 10N·m다. [실패와 회복 비교](../evidence/t3-valve-20261120/replay-comparison.json), [MuJoCo 마찰 구속 모델](https://mujoco.readthedocs.io/en/stable/modeling.html#friction)

## N3 결과

MuJoCo 4.0 CPU에서 21.0초를 5ms 간격으로 재생하고 매 물리 스텝을 검사했다. 힌지는 실제로 90.016°까지 움직였다. 마지막 5초 동안 89.977~90.016°를 유지했고, 마지막 각도는 90.001°였다. 힌지 각속도가 0.02rad/s를 넘고 각도가 5~85°에 있는 1,647개 표본에서 반대 방향 마찰 토크가 모두 10.0N·m였다. 핀이 힌지에 전달한 토크의 최대 절댓값은 10.0015N·m였다. [각도·반력 시계열과 판정](../evidence/t3-valve-20261120/loaded-turn-replay.json), [마지막 자세](../evidence/t3-valve-20261120/loaded-turn.png)

| 조작 중 측정 | 결과 | M1 기준 |
|---|---:|---:|
| 닫힌 자세 유지 | 5.0초 | 5초 이상 |
| 몸통 이동 | 5.25mm | 10mm 이하 |
| 발 미끄러짐 | 4.44mm | 5mm 이하 |
| 마찰 사용률 | 0.4704 | 0.8 이하 |
| 관절 최대 토크 | 모델 한계의 45.4% | 100% 이하 |
| 네 지지 발판의 최소 하중 | 52.7N | 각각 1N 이상 |

발 중심 이동과 몸통 이동은 조작 시작 시점부터 마지막 유지 표본까지 계산했다. 전복, 토크 포화, 힘 분배 실패, 발 이외 접촉 없이 `valve.verified=true`를 확인했다. N1 왕복과 N3 조작은 각각 연속 재생으로 보존했다. 통합 임무에는 손잡이 접근, 파지와 해제, 바닥 복귀를 잇는 동작이 남아 있다. [물리 판정 및 로봇 상태](../evidence/t3-valve-20261120/loaded-turn-replay.json)

## 재현

```powershell
Set-Location "$HOME/go2-wall-reach-lab"
$py = '.venv/Scripts/python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$n1 = 'evidence/t1-stair-ascent-20261120'
$run = 'evidence/t3-valve-repeat'
& $py tools/replay_stair_valve.py find "$n1/target-round-trip-geometry.json" `
    "$n1/target-round-trip-plan.json" "$run/fixed-body-candidates.json"
& $py tools/replay_stair_valve.py replay "$run/fixed-body-candidates.json" `
    "$run/loaded-turn-replay.json" --candidate 4 --motion-seconds 8.75 `
    --settle-seconds 5.75 --gain 2.5 --hold-feedforward-nm 3 --timestep 0.005
```
