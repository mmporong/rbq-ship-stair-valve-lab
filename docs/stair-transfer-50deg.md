# 50°·19cm 계단의 발 이동과 지지

8판의 목표 계단 재생은 발판에 발을 디딘 뒤 몸통을 올리는 구간에서 끝났다. 이번에는 같은 형상에서 RBQ-10+RB1의 몸통 자세와 발 이동을 각각 계산하고, 연결된 일부 동작을 접촉 물리로 재생했다. [8판과 진입 비교](stair-v8-training.md)

## 자세 계획기 수정

Windows의 `spawn` 작업자에서 RBQ-10 선택이 사라지는 것을 재현했다. 요청한 RBQ-10 대신 기본 Go2+D1이 로드돼 모델 질량은 17.87kg, 다리 관절 한계는 23.7·23.7·45.43N·m로 들어갔다. 작업자 초기화에 기체 선택을 전달하고 그 기체의 토크 한계를 로드한다. 수정 후 같은 확인에서 RBQ-10+RB1의 48.95kg, 다리 관절 한계 100·100·150N·m, RB1 접기 자세가 확인됐다. [작업자 초기화](../src/go2_wall/stairs.py), [Windows 방식 회귀 검사](../tests/test_rbq.py)

## 정적 자세와 발 이동 판정

발판 깊이 127mm, 폭 610mm, 단 높이 190mm, 경사 50°인 열린 계단이다. 발판과 바닥 표면의 마찰은 0.6이며 정적 탐색은 그 70%를 사용했다. 발 이외 부위 30mm 여유와 체중 10% 수평 외란에서의 정적 평형을 확인했다. 발 이동은 가능한 중간 자세를 고른 뒤 몸통 이동과 발을 드는 구간을 20ms 간격으로 대조했다. [정적 탐색 코드](../tools/probe_stair_transfer.py)

| 시작 발 위치 | 정적 자세 수 | 몸통 원점이 첫 계단 경계를 지난 자세 |
|---|---:|---:|
| 네 발 바닥 | 4,171 | 0 |
| 앞발 1단, 뒷발 바닥 | 6,961 | 189 |
| 앞발 3단, 뒷발 1단 | 2,948 | 2,948 |

## 접촉 물리 재생

발 이동을 하나씩 재생할 때 같은 레인보우 모델, 5ms 물리 스텝, 10ms 제어·접촉 기록을 사용한다. 움직이는 발과 지지하는 발의 실제 하중을 발판별로 구한다. 각 시험은 표에 명시한 발 배치에서 시작한다. T1 판정은 출발면부터 상부 도달, 회전, 하강, 네 발 정지 복귀까지 연속으로 연결해야 성립한다. [접촉 재생 코드](../tools/replay_stair_transfer.py), [T1 기준](../PLAN.md)

| 이동 | 시작 지지 | 목표 지지 확인 | 실제 최대 토크 / 한계 | 포화 |
|---|---|---|---:|---:|
| 앞왼발 바닥에서 1단 | 네 발 바닥, 몸통 수평 | 네 발 하중 확인 | 0.301 | 0 |
| 앞왼발 1단에서 2단 | 앞발 1단, 뒷발 바닥 | 네 발 하중 확인 | 0.238 | 0 |
| 뒷왼발 1단에서 2단 | 앞발 3단, 뒷발 1단 | 네 발 하중 확인 | 0.418 | 0 |

각 재생은 4.5초이며 전복과 몸통·팔의 예상 밖 접촉이 기록되지 않았다. 끝 0.3초 동안 예정한 네 지지면에 각 발이 1N 이상의 수직 하중을 실었다. 바닥에서 첫 단으로 옮긴 앞왼발은 마지막 표본에서 38.5N을 실었고 최대 지지 미끄러짐 추정치는 1.14mm였다. 앞왼발을 1단에서 2단으로 옮긴 시행의 최대 지지 미끄러짐 추정치는 11.0mm로 기존 벽 왕복 검사 한계 10mm를 넘었다. [첫 단 물리 재생](../evidence/t1-stair-posture-20261120/floor-first.json), [앞왼발 1→2단](../evidence/t1-stair-posture-20261120/front-second.json), [뒷왼발 1→2단](../evidence/t1-stair-posture-20261120/rear-next.json)

첫 단의 최초 정적 연결 후보는 제어 목표를 만드는 과정에서 몸통 이동의 16% 지점에 끊겼다. 몸통 수평인 바닥 자세에서 시작하고 자세·하중 인계까지 샘플링할 수 있는 연결을 선택해 재생했다. 나머지 두 발 이동은 각자의 초기 자세에서 시행했다. 재생 결과의 `controller_metrics.passed`는 모든 발에 바닥 복귀를 요구하는 기존 검사값이다. 발판별 시작·종료 하중은 별도로 기록했다. [첫 탐색](../evidence/t1-stair-posture-20261120/poses.json), [연결된 자세 샘플](../evidence/t1-stair-posture-20261120/poses-continuous.json), [뒷왼발 2단 장면](../evidence/t1-stair-posture-20261120/rear-left-tread2-contacts.png)

## 두 앞발을 연속해서 첫 단에 올리기

첫 발판 시작보다 몸통 X가 0.405m 뒤인 수평 자세에서 시작했다. 앞왼발 이동의 끝 몸통 자세와 발 배치에서 앞오른발의 이동을 계획했다. 중간 지지 자세 3,695개 중 몸통 거리 기준으로 고른 36개를 연결해 제어 목표로 만들 수 있는 후보 6개를 찾았다. 첫 후보를 두 발 이동이 이어지는 8.6초의 한 재생으로 검증했다. [연속 재생 코드](../tools/replay_stair_entry.py), [세 시점의 발 하중 결과](../evidence/t1-stair-posture-20261120/front-pair.json)

네 발 바닥에서 시작해 앞왼발만 1단에 지지한 다음 두 앞발 모두 1단을 지지했다. 마지막 하중은 앞왼발 52.0N, 앞오른발 66.9N이었다. 마지막 몸통 원점은 계단 시작보다 0.442m 뒤에 있었다. 최대 토크는 한계의 30.7%, 지지 미끄러짐 추정치는 최대 2.23mm였다. 두 앞발의 순간 최대 마찰 사용률은 1.0이었다. 전복, 토크 포화, 예상 밖 접촉이 기록되지 않았다. [연속 재생 마지막 장면](../evidence/t1-stair-posture-20261120/front-pair-tread1.png)

계단 중간에서 뒷발 2단과 앞발 4단을 기준으로 네 다리를 모두 한 단씩 올리는 시험은 `RL, RR, FL, FR` 순서의 연결을 찾지 못했다. 시작 자세 2,869개와 스윙 자세 614·30·11·891개에서 후보 제한 60개와 분기 3개를 사용했다. [계단 중간 한 주기](../evidence/t1-stair-posture-20261120/cycle-gap2.json)

## 바닥에서 뒷발을 당겨 들어가는 연속 재생

두 앞발을 1단에 고정하고 뒷왼발과 뒷오른발을 바닥에서 각각 22cm 앞으로 옮겼다. 두 발의 최종 바닥 목표 X는 −0.38m였다. 바닥 출발부터 17.2초 동안 한 번의 물리 재생으로 이어졌다. 아래 표의 하중은 각 이동 끝 0.3초 동안 예정한 표면마다 1N 이상 유지한 발의 마지막 값이다. [17.2초 재생 경로](../evidence/t1-stair-entry-20261120/both-rear-deck-plan.json), [지지 하중과 관절 기록](../evidence/t1-stair-entry-20261120/both-rear-deck-replay.json), [마지막 물리 재생 자세](../evidence/t1-stair-entry-20261120/both-rear-deck.png)

| 경과 | 발 이동 | 네 발 하중 FL·FR·RL·RR | 몸통 X |
|---:|---|---|---:|
| 8.6초 | 두 앞발 1단 | 52.1 · 67.2 · 152.4 · 208.5N | −441.9mm |
| 12.7초 | 뒷왼발 바닥 전진 | 67.7 · 120.2 · 68.0 · 224.3N | −299.5mm |
| 17.2초 | 뒷오른발 바닥 전진 | 157.1 · 100.9 · 142.7 · 79.6N | −124.3mm |

제어 이득 1.5배와 1.2초 스윙에서 뒷왼발의 역기구학에 계획된 몸통 자세를 쓰면 착지 오차가 56.2mm였다. 측정한 몸통 자세를 쓰면 3.4mm였다. 이득과 스윙 시간을 달리한 네 쌍에서 같은 방향의 변화가 있었다. 본 재생은 측정된 몸통을 사용했고, 뒷오른발의 착지 오차는 1.2mm였다. 최대 토크는 모델 한계의 34.9%, 최대 지지 미끄러짐 추정치는 2.74mm였다. 토크 포화와 전복, 발 이외 접촉은 0건이었다. [스윙 제어 대조](../evidence/t1-stair-entry-20261120/rear-left-deck-control-comparison.json)

두 앞발을 1단 중앙에 둔 채 뒷왼발을 1단 중앙에 놓으려던 초기 탐색은 앞왼발과 목표가 겹쳤다. 해당 탐색은 다른 발 위치에서의 뒷발 진입 가능성을 평가하지 못했다. 연속 재생기는 다른 발과 구형 발의 지름보다 가까운 착지점을 입력에서 거부한다. 앞발을 먼저 2단으로 옮기는 경로에서 같은 쪽 1단 자리를 비웠다. [초기 탐색의 발 목표](../evidence/t1-stair-posture-20261120/poses-continuous.json), [연속 재생 코드](../tools/replay_stair_sequence.py)

## 발을 들어 올린 뒤 2단 모서리 넘기기

기존 스윙은 양 끝에서 발을 50mm씩 들어 그 사이를 대각선으로 잇는다. 1단에서 2단으로 넘어갈 때 발 중심 X가 2단 모서리와 같아지는 지점에서 계획된 발바닥은 2단보다 25.7mm 낮았다. 새 스윙은 두 표면 중 높은 곳을 기준으로 50mm 여유를 확보한 뒤 수평 이동을 시작한다. [궤적 코드](../src/go2_wall/stairs.py), [상·하행 모서리 검사](../tests/test_stair_route.py)

뒷발을 바닥에서 전진시킨 다음 앞왼발을 2단에 올리는 22.1초 동작에서 몸통 계획, 착지 목표, 측정 몸통을 쓰는 역기구학과 이득을 고정했다. 마지막 스윙 궤적만 바꿨다. [대각선 이동 경로](../evidence/t1-stair-entry-20261120/front-second-standard-plan.json), [높이 확보 후 수평 이동 경로](../evidence/t1-stair-entry-20261120/front-second-plan.json)

| 앞왼발 경로 | 2단 하중 | 착지 XY 오차 | 최대 토크 / 한계 | 토크 포화 | 전복 |
|---|---:|---:|---:|---:|---|
| 대각선 이동 | 0N | 388.0mm | 1.000 | 104스텝 | 발생 |
| 높이 확보 후 수평 이동 | 73.6N | 1.2mm | 0.349 | 0 | 없음 |

실패한 재생에서는 앞왼쪽 종아리, 앞오른쪽 넓적다리와 몸통이 환경에 닿았다. 마지막 표본에서 앞왼발은 목표 발판 바깥 바닥에 있었다. 높이 확보 후 수평 이동을 사용한 재생은 끝 0.3초 동안 네 발의 예정 지지면에 하중을 실었다. [실패 재생](../evidence/t1-stair-entry-20261120/front-second-standard.json), [실패 종료 자세](../evidence/t1-stair-entry-20261120/standard-landing-miss.png), [성공 재생](../evidence/t1-stair-entry-20261120/front-second-lift-first.json)

## 두 앞발 2단, 두 뒷발 1단

뒷발을 바닥에서 전진시킨 뒤 앞오른발과 앞왼발을 차례로 2단에 올렸다. 이어 뒷오른발과 뒷왼발을 1단에 올렸다. 같은 쪽 앞발과 뒷발을 먼저 올리는 두 순서에서는 다음 앞발을 2단에 올릴 연결을 찾지 못했다. [좌우 쌍 우선 탐색](../evidence/t1-stair-entry-20261120/paired-side-search.json), [네 발 재생 순서와 목표](../evidence/t1-stair-entry-20261120/four-feet-plan.json)

마지막 뒷왼발을 들기 전 몸통 이동에 1.5초를 쓰면 네 발이 목표에 닿았으나, 이동 중 몸통 접촉력이 1,265.8N에 이르렀다. 몸통 경로와 나머지 제어에는 같은 값을 쓰고 이 구간의 시간만 늘렸다. 2.5·3.5·4.5초 재생 모두 발 이외 접촉과 전복, 토크 포화가 기록되지 않았다. 가장 짧게 통과한 2.5초를 선택했다. [1.5초 경로](../evidence/t1-stair-entry-20261120/fast-body-shift-plan.json), [1.5초 재생 결과](../evidence/t1-stair-entry-20261120/fast-body-shift-replay.json), [몸통 이동 시간 대조](../evidence/t1-stair-entry-20261120/body-shift-comparison.json)

바닥에서 시작한 38.6초 재생 끝에 몸통 원점은 첫 계단 시작점에서 앞쪽 211.7mm였다. 끝 0.3초 동안 두 앞발은 2단, 두 뒷발은 1단에 하중을 실었다. 마지막 하중은 발 순서 FL·FR·RL·RR에서 130.1·202.2·45.9·102.1N이었다. 전체 최대 토크는 모델 한계의 45.9%, 지지 미끄러짐 추정치는 최대 5.12mm였다. [38.6초 재생과 하중](../evidence/t1-stair-entry-20261120/four-feet-replay.json), [네 발 지지 장면](../evidence/t1-stair-entry-20261120/four-feet.png)

`verified_sequence`는 저장한 부분 경로에서 각 착지의 0.3초 지지, 35mm 이내 착지 XY 오차, 폭과 지지 경로, 전복, 토크 한계와 포화, 10N 초과 발 이외 접촉을 검사한다. N1은 상부 도달과 회전, 하강 후 바닥 복귀까지 이어져야 한다. [재생 판정 코드](../tools/replay_stair_sequence.py), [N1 완료 기준](../PLAN.md)

## 1·2단에서 바닥까지 역순 복귀

두 앞발이 2단과 두 뒷발이 1단에 도달한 뒤 같은 몸통 목표와 지지 발을 역순으로 써서 내려왔다. 기존 대각선 궤적으로 1단에서 바닥으로 내딛을 때 앞오른발의 착지 오차가 260.8mm까지 커졌고, 전복과 토크 포화 10스텝이 기록됐다. 마지막 두 앞발의 스윙 궤적만 바꿔 높은 표면 위로 발을 먼저 들어 올린 뒤 바닥 쪽으로 옮겼다. [대각선 하강 경로](../evidence/t1-stair-entry-20261120/low-round-trip-standard-plan.json), [대각선 하강 결과](../evidence/t1-stair-entry-20261120/low-round-trip-standard-replay.json)

수정한 76.8초 재생은 바닥에서 출발해 1·2단을 밟고 바닥에 네 발로 돌아왔다. 끝 0.3초 동안 네 발에 바닥 하중이 유지됐다. 마지막 하중은 68.6·99.2·134.8·177.6N이었다. 최대 토크는 모델 한계의 48.3%이고, 최대 지지 미끄러짐 추정치는 7.09mm였다. 전복·토크 포화·발 이외 접촉은 기록되지 않았다. [낮은 구간 왕복 경로](../evidence/t1-stair-entry-20261120/low-round-trip-plan.json), [왕복 재생과 하중](../evidence/t1-stair-entry-20261120/low-round-trip-replay.json)

## 두 앞발 3단 지지

38.6초 진입 경로의 끝에 앞오른발과 앞왼발을 차례로 2단에서 3단에 올리는 동작을 붙였다. 48.4초 연속 재생 끝에 두 앞발 3단과 두 뒷발 1단의 하중은 117.7·81.8·158.3·122.4N이었다. 최대 토크는 모델 한계의 52.3%였으며 전복·토크 포화·발 이외 접촉은 없었다. [3단까지 이어진 경로](../evidence/t1-stair-entry-20261120/both-fronts-third-plan.json), [3단 재생과 하중](../evidence/t1-stair-entry-20261120/both-fronts-third-replay.json), [3단 마지막 장면](../evidence/t1-stair-entry-20261120/both-fronts-third.png)

## 한 단씩 반복해 상부 플랫폼에 앞발 올리기

앞오른발, 앞왼발, 뒷왼발, 뒷오른발 순으로 발판을 한 단씩 높였다. 첫 주기는 앞발 2단·뒷발 1단 자세에서 출발해 앞발 3단·뒷발 2단에 닿았다. 이후 각 주기의 발 목표와 몸통 목표를 앞쪽으로 159.43mm, 위로 190mm 옮겼다. 이전 주기의 마지막 몸통 자세에서 새 주기의 첫 스윙 자세까지는 2.5초에 걸쳐 옮겼다. [첫 주기 경로](../evidence/t1-stair-ascent-20261120/first-cycle-plan.json), [주기 반복 도구](../tools/repeat_stair_cycle.py)

| 종료 지지면 | 바닥 출발부터의 시간 | 실제 최대 토크 / 한계 |
|---|---:|---:|
| 앞발 3단·뒷발 2단 | 58.2초 | 0.568 |
| 앞발 4단·뒷발 3단 | 78.8초 | 0.568 |
| 앞발 10단·뒷발 9단 | 202.4초 | 0.568 |
| 앞발 상부 플랫폼·뒷발 10단 | 223.0초 | 0.568 |

10단까지의 40회 발 이동에서 각 착지 후 0.3초 동안 예정한 지지면의 네 발 하중을 확인했다. 마지막 하중은 185.0·178.2·57.3·59.7N이었다. 전복, 토크 포화, 발 이외 접촉은 기록되지 않았다. [첫 주기 재생](../evidence/t1-stair-ascent-20261120/first-cycle-replay.json), [두 번째 주기 재생](../evidence/t1-stair-ascent-20261120/second-cycle-replay.json), [10단 재생과 하중](../evidence/t1-stair-ascent-20261120/tread-ten-replay.json), [10단 마지막 자세](../evidence/t1-stair-ascent-20261120/tread-ten.png)

한 주기를 더 이어 두 앞발을 상부 플랫폼에 올렸다. 마지막 0.3초 동안 두 뒷발은 10단을 지지했다. 마지막 하중은 앞발 185.0·178.2N, 뒷발 57.3·59.7N이었다. 10ms마다 기록한 발 중심과 몸통 원점은 플랫폼 시작점을 넘은 뒤 플랫폼 경계 안에 있었다. 각 착지에서 예정한 지지면의 하중을 유지했고 전복·토크 포화·발 이외 접촉은 기록되지 않았다. [앞발 플랫폼 경로](../evidence/t1-stair-ascent-20261120/platform-front-plan.json), [223초 재생과 하중](../evidence/t1-stair-ascent-20261120/platform-front-replay.json), [플랫폼 진입 장면](../evidence/t1-stair-ascent-20261120/platform-front.png)

## 상부 플랫폼 네 발 지지와 회전

두 앞발을 플랫폼 안쪽으로 300mm씩 옮겨 10단에 있던 두 뒷발이 올라갈 자리를 마련했다. 바닥부터 244.2초 연속 재생 끝에 네 발의 플랫폼 하중은 121.7·163.5·81.2·113.8N이었다. 최대 토크는 모델 한계의 56.8%였으며 전복·토크 포화·발 이외 접촉은 없었다. [네 발 도착 경로](../evidence/t1-stair-ascent-20261120/platform-all-four-plan.json), [연속 재생과 하중](../evidence/t1-stair-ascent-20261120/platform-all-four-replay.json), [네 발 플랫폼 지지 장면](../evidence/t1-stair-ascent-20261120/platform-all-four.png)

플랫폼 중앙 네 발 자세에서 시작한 회전 시험은 발 위치를 30°씩 여섯 번 옮겼다. 98.8초에 발 이동 24회를 마쳐 실제 몸통 방향 179.96°를 기록했다. 각 착지에서 네 발이 플랫폼을 지지했고, 최대 토크는 한계의 35.2%였으며 전복·포화·발 이외 접촉은 없었다. 몸통 자세 계획에 월드 수직축 회전을 추가해 기존 좌표·피치 지정과 함께 썼다. [플랫폼 시작 회전 경로](../evidence/t1-stair-ascent-20261120/isolated-platform-turn-plan.json), [회전 실측](../evidence/t1-stair-ascent-20261120/isolated-platform-turn-replay.json)

244.2초 상승 경로에 플랫폼 중앙으로 옮기는 발 이동 6회와 회전 24회를 이었다. 바닥에서 시작한 367.2초 연속 재생은 발 이동 78회 모두에서 예정한 지지면의 하중을 유지했다. 마지막 몸통 방향은 179.95°였으며 네 발의 플랫폼 하중은 132.3·167.7·76.4·103.8N이었다. 최대 토크는 모델 한계의 56.8%였다. 전복·토크 포화·발 이외 접촉은 기록되지 않았다. [상승·회전 경로](../evidence/t1-stair-ascent-20261120/ascent-and-turn-plan.json), [367.2초 재생과 하중](../evidence/t1-stair-ascent-20261120/ascent-and-turn-replay.json), [회전 종료 자세](../evidence/t1-stair-ascent-20261120/ascent-and-turn.png)

두 앞발이 플랫폼과 두 뒷발이 10단인 상태에서 몸통 방향을 유지해 후진으로 내려온 시험은 445.6초에 바닥 네 발 지지까지 이어졌다. 발 이동 88회의 지지 검사를 통과했고 전복과 토크 포화는 없었다. 접은 팔목과 뒷오른쪽 넓적다리 사이에 30.3N의 접촉을 기록해 10N 한계를 넘겼다. [후진 복귀 경로](../evidence/t1-stair-ascent-20261120/backward-return-plan.json), [후진 재생 결과](../evidence/t1-stair-ascent-20261120/backward-return-replay.json)

팔 접기 자세의 두 번째 축 목표를 −1.60rad에서 −1.50rad로 옮긴 재생은 바닥 출발, 상부 네 발 지지, 179.94° 회전, 다시 오르막 방향으로 돌아선 뒤 후진 하강까지 734.0초에 이었다. 시작 위치와의 수평 오차는 마지막 0.5초에 최대 28.3mm였다. 그 시간에 최소 발 하중은 68.2N, 최대 몸통 선속은 0.00257m/s, 최대 각속은 0.00469rad/s, 최소 몸통 수직 성분은 0.99755였다. 발 이동 156회의 착지 검사와 바닥 정지 조건을 통과했다. [후진 왕복 경로](../evidence/t1-stair-ascent-20261120/floor-return-with-arm-offset-plan.json), [팔 자세 입력](../evidence/t1-stair-ascent-20261120/floor-return-with-arm-offset-geometry.json), [734초 물리 재생](../evidence/t1-stair-ascent-20261120/floor-return-with-arm-offset-replay.json)

734초 동안 전복은 없었다. 몸통 추종 오차는 최대 34.1mm였고 토크 포화는 7스텝이었다. 접은 팔목과 뒷오른쪽 넓적다리의 접촉은 133.3N, 같은 정강이와는 125.9N까지 높아져 전체 왕복 검사를 통과하지 못했다. 바닥 정지 검사는 처음 재생한 몸통 위치에서 수평 0.15m, 기울기 10°, 몸통 선속 0.05m/s, 각속 0.15rad/s 및 네 발 각각 바닥 하중 1N 이상을 끝 0.5초 동안 적용했다. [정지 복귀 판정에 사용한 기준](../train/eval_target_stairs.py)

## N1 상부 회전과 후진 하강 왕복

RB1의 팔 접기 자세에서 두 번째·세 번째 관절을 함께 조정했다. 두 조합 `−1.20/+0.30rad`와 `−1.40/+0.60rad` 모두 734초 전체 재생에서 전복·토크 포화·발 이외 접촉을 없앴다. 두 재생 모두 바닥에 멈췄고 추종 오차는 각각 34.0mm와 34.1mm였다. 오른쪽 앞발 이동 전 몸통을 하강 방향으로 225mm, 아래로 270mm 옮기는 동작에서 오차가 가장 컸다. 같은 시작 자세에서 이 동작의 시간만 1.5초, 2.5초, 3.5초로 바꿨을 때 최대 추종 오차는 각각 34.1mm, 24.5mm, 21.2mm였다. [팔 자세와 이동 시간 비교](../evidence/t1-stair-ascent-20261120/floor-return-stow-comparison.json)

`−1.40/+0.60rad`와 3.5초 이동을 적용한 736.0초 연속 재생이 N1을 통과했다. 244.2초에 플랫폼에 네 발로 올랐고 367.2초에 179.91° 회전했다. 같은 플랫폼에서 다시 오르막 쪽으로 돌아서 후진으로 내려왔다. 착지 156회에서 하중을 확인했고 최대 토크는 모델 한계의 60.3%, 최대 추종 오차는 24.0mm, 최대 지지 미끄러짐 추정치는 6.75mm였다. 전복·토크 포화·힘 분배 실패·발 이외 접촉은 없었다. [736초 왕복 경로](../evidence/t1-stair-ascent-20261120/target-round-trip-plan.json), [팔 자세와 계단 입력](../evidence/t1-stair-ascent-20261120/target-round-trip-geometry.json), [전체 물리 결과](../evidence/t1-stair-ascent-20261120/target-round-trip-replay.json)

바닥에서 끝 0.5초 동안 출발점과의 수평 오차는 최대 28.2mm, 몸통 선속은 최대 0.00273m/s, 각속은 최대 0.00511rad/s였다. 네 발 각각의 최소 하중은 69.3N이었다. 마지막 하중은 70.0·100.7·133.8·175.7N이었다. `physical_round_trip_verified=true`를 기록했다. [바닥 정지 검사와 마지막 하중](../evidence/t1-stair-ascent-20261120/target-round-trip-replay.json)

하강 방향으로 돌아선 플랫폼 중앙 자세에서 계단 가장자리로 이동한 뒤 두 앞발을 10단에 내렸다. 28.2초 재생에서 마지막 몸통 방향의 절댓값은 179.90°였다. 두 앞발은 10단, 두 뒷발은 플랫폼에 하중을 실었고, 최대 토크는 모델 한계의 46.9%였다. [하강 진입 경로와 물리 재생](../evidence/t1-stair-ascent-20261120/platform-descent-entry.json)

두 앞발 10단 지지에서 시작해 두 뒷발을 플랫폼 가장자리 쪽으로 270mm씩 옮겼다. 이어 두 앞발을 9단에 내렸다. 24초 재생의 마지막 하중은 앞발 51.9·37.1N, 뒷발 202.7·188.5N이었다. 최대 토크는 모델 한계의 53.0%였으며 전복·토크 포화·발 이외 접촉은 없었다. [9단 하강 경로](../evidence/t1-stair-ascent-20261120/tread-nine-descent-plan.json), [9단 하강 재생과 하중](../evidence/t1-stair-ascent-20261120/tread-nine-descent-replay.json)

## 두 뒷발의 10단 착지

앞발 9단·뒷발 플랫폼에서 10단으로 내리는 경로를 탐색했다. 몸통을 스윙 동안 고정한 시험은 양쪽 발에서 중간 자세 392개를 찾았고, 같은 몸통에서 스윙의 모든 구간을 충족한 자세는 없었다. 뒷오른발의 착지점을 10단 중앙에서 하강 방향으로 24mm 옮기고, 발을 드는 동안 몸통도 다섯 자세를 따라 이동하도록 궤적을 이었다. [움직이는 몸통과 발의 경로](../evidence/t1-stair-ascent-20261120/rear-right-tread-ten-plan.json)

같은 몸통 경로와 착지점에서 스윙 2.0초 재생은 종아리와 환경 사이에 458.9N의 접촉이 생겼고 뒷오른발이 10단에 하중을 싣지 못했다. 2.8초 재생은 네 발 지지를 통과했고 최대 토크는 모델 한계의 85.2%였다. 몸통 경로를 바꾼 2.0초 재생은 최대 토크 69.2%와 몸통 추종 오차 39.9mm를 기록했다. 그 경로에서 스윙 시간만 3.3초로 늘린 재생을 선택했다. 최대 토크는 52.6%, 몸통 추종 오차는 19.8mm였다. 마지막 뒷오른발은 10단에 85.9N을 실었고 전복·포화·발 이외 접촉과 힘 분배 제약 완화가 기록되지 않았다. [다섯 재생의 접촉과 토크](../evidence/t1-stair-ascent-20261120/rear-right-tread-ten-comparison.json), [선택한 뒷오른발 10단 재생](../evidence/t1-stair-ascent-20261120/rear-right-tread-ten-replay.json)

뒷왼발을 들 때는 체중의 10% 수평 외력을 포함한 정적 탐색에서 선택한 범위 안에 출발 자세가 없었다. 기본 하중으로 경로 후보를 계산하고 기존 물리 재생 기준으로 검사했다. 뒷왼발을 10단에 내린 세 재생이 지지 검사를 통과했다. 선택한 8.6초 재생의 최대 토크는 51.4%, 몸통 추종 오차는 4.3mm, 최대 지지 미끄러짐 추정치는 1.88mm였다. 앞발 9단과 뒷발 10단에 실린 마지막 하중은 124.2·107.6·135.4·113.1N이었다. [세 뒷왼발 재생의 비교](../evidence/t1-stair-ascent-20261120/rear-left-tread-ten-comparison.json), [선택한 뒷왼발 경로](../evidence/t1-stair-ascent-20261120/rear-left-tread-ten-plan.json), [실제 접촉과 관절 기록](../evidence/t1-stair-ascent-20261120/rear-left-tread-ten-replay.json)

## 바닥 출발부터 상부 회전과 8·10단 하강까지

바닥에서 출발해 상부에 네 발로 올라간 뒤 179.95° 돌고, 두 앞발 9단과 두 뒷발 10단 지지까지 내렸다. 434.0초 연속 재생에서 발 이동 90회의 지지 검사를 통과했다. 마지막 하중은 119.4·105.9·137.5·117.4N이었다. 최대 토크는 모델 한계의 56.8%, 몸통 추종 오차는 최대 23.8mm, 지지 미끄러짐 추정치는 최대 6.72mm였다. 전복·토크 포화·발 이외 접촉은 기록되지 않았다. [연속 경로](../evidence/t1-stair-ascent-20261120/ascent-turn-tread-nine-plan.json), [434초 연속 물리 재생](../evidence/t1-stair-ascent-20261120/ascent-turn-tread-nine-replay.json), [하강 끝 자세](../evidence/t1-stair-ascent-20261120/ascent-turn-tread-nine.png)

하강 순서의 변형으로 뒷오른발 10단·뒷왼발 플랫폼·두 앞발 9단 지지에서 시작해 앞오른발을 8단에 내렸다. 8.6초 재생 끝에 앞오른발이 88.8N을 실었고 최대 토크는 모델 한계의 32.7%였다. 전복·포화·발 이외 접촉은 기록되지 않았다. 이 발 배치에서 반복해서 내려가는 구간을 이어간다. [8단 이동 경로](../evidence/t1-stair-ascent-20261120/front-right-tread-eight-plan.json), [8단 접촉 물리 재생](../evidence/t1-stair-ascent-20261120/front-right-tread-eight-replay.json)

그 움직임에 뒷왼발의 10단 착지를 이어 붙인 시험에서 스윙 시작 2.725초 뒤 앞왼쪽 종아리와 뒷왼발의 접촉력이 53.0N으로 기록됐다. 10ms 기록에서 10N을 처음 넘긴 접촉이었다. 몸통 추종 오차 30mm 초과와 전복이 뒤따랐다. 앞오른발 이동의 계획상 끝 자세에서 시작한 별도 시험도 같은 뒷왼발 경로의 스윙 2.735초 뒤 같은 부위에 16.9N을 기록했다. [실패 경로](../evidence/t1-stair-ascent-20261120/descent-rear-interference-plan.json), [첫 접촉 시각과 하중](../evidence/t1-stair-ascent-20261120/descent-rear-interference-trace.json), [첫 접촉 장면](../evidence/t1-stair-ascent-20261120/descent-rear-first-impact.png)

뒷왼발의 새 착지점을 10단 중앙 X에 맞추고 Y를 바깥쪽으로 60mm 옮겼다. 다시 계산한 몸통 경로를 적용해 앞오른발 8단 이동과 뒷왼발 10단 착지를 한 재생으로 이었다. 16.8초 뒤 앞왼발 9단·앞오른발 8단·두 뒷발 10단의 마지막 하중은 93.6·121.9·133.2·131.6N이었다. 최대 토크는 41.0%, 지지 미끄러짐 추정치는 최대 4.36mm였다. 다리 간 접촉, 전복, 토크 포화는 기록되지 않았다. [착지점을 옮긴 연속 경로](../evidence/t1-stair-ascent-20261120/offset-rear-left-descent-plan.json), [수정 후 하중과 관절 기록](../evidence/t1-stair-ascent-20261120/offset-rear-left-descent-replay.json)

같은 재생에 앞왼발을 8단에 내리는 동작을 더했다. 25.0초 뒤 두 앞발 8단과 두 뒷발 10단의 하중은 84.4·111.3·128.0·156.5N이었다. 최대 토크는 41.0%, 몸통 추종 오차는 최대 8.1mm, 지지 미끄러짐 추정치는 최대 5.34mm였다. 전복·토크 포화·발 이외 접촉은 기록되지 않았다. 이 두 단 간격을 유지한 채 한 단을 더 내려가는 경로를 탐색한다. [두 단 간격 하강 경로](../evidence/t1-stair-ascent-20261120/two-tread-gap-descent-plan.json), [25초 연속 접촉 물리](../evidence/t1-stair-ascent-20261120/two-tread-gap-descent-replay.json)

바닥에서 시작해 상부에 네 발로 올라가 179.95° 돌고, 뒷오른발 10단 도착 뒤 위의 세 발 이동을 이어 재생했다. 450.4초 동안 92회 착지에서 예정한 네 지지면의 하중이 유지됐다. 마지막 두 앞발 8단·두 뒷발 10단의 하중은 84.1·111.3·127.3·157.5N이었다. 최대 토크는 모델 한계의 56.8%, 몸통 추종 오차는 최대 23.8mm, 지지 미끄러짐 추정치는 최대 7.56mm였다. 전복·토크 포화·발 이외 접촉과 힘 분배 제약 완화는 기록되지 않았다. 나머지 하강과 바닥 정지 복귀가 남았다. [450.4초 연속 경로](../evidence/t1-stair-ascent-20261120/ascent-turn-tread-eight-plan.json), [연속 물리 재생과 하중](../evidence/t1-stair-ascent-20261120/ascent-turn-tread-eight-replay.json)

뒷오른발에 10단 하중을 싣기 직전의 몸통과 네 발 위치를 하강 쪽으로 한 단 평행 이동해 정적 자세를 검사했다. 이동 전 자세와 이동 후 자세 모두 세 발의 힘·토크 균형을 계산할 수 있었다. 한 단 아래 자세에서는 뒷왼쪽 정강이가 위쪽 플랫폼과 최대 39.6mm 겹쳤다. 두 단, 세 단 아래에 옮긴 자세에서는 같은 정강이가 위쪽 디딤판과 최대 39.6mm 겹쳤다. 이 값은 기하 검사에서 계산한 침투 깊이다. [평행 이동 정적 자세와 충돌 형상](../evidence/t1-stair-ascent-20261120/translated-downhill-contact-probe.json)

## 좌우 기울기를 적용한 오른쪽 뒷발 9단 착지

몸통을 전후축 주위로 기울이는 해를 탐색해 뒷오른발 9단 착지 경로를 계산했다. 첫 재생에서는 스윙을 시작한 지 2.725초에 몸통 추종 오차가 30.7mm로 기록됐다. 20ms 뒤 오른쪽 앞다리 넓적다리와 뒷오른발 사이에 14.9N 접촉이 생겼고, 이어 전복을 기록했다. [첫 좌우 기울기 경로](../evidence/t1-stair-ascent-20261120/rolled-descent-first-impact-plan.json), [오차·접촉·전복 시각](../evidence/t1-stair-ascent-20261120/rolled-descent-first-impact-trace.json)

9단 착지점을 계단 중심에서 가로 260mm에 두고, 계획 단계에서 같은 쪽 앞·뒷다리 사이에 15mm 이상 간격을 두도록 몸통 경로를 다시 계산했다. 몸통 이동과 발 스윙을 각각 4.0초로 설정한 42.6초 재생에서 마지막 하중은 앞발 161.7·201.2N, 뒷발 19.1·98.1N이었다. 최대 토크는 모델 한계의 75.4%, 몸통 추종 오차는 21.3mm, 지지 미끄러짐 추정치는 최대 5.87mm, 착지 오차는 2.70mm였다. 힘 분배 조건을 완화한 틱은 35회였다. 전복·토크 포화·힘 분배 실패·발 이외 접촉은 없었다. [8·8·10·9단 부분 경로](../evidence/t1-stair-ascent-20261120/rolled-rear-right-nine-plan.json), [42.6초 재생과 하중](../evidence/t1-stair-ascent-20261120/rolled-rear-right-nine-replay.json)

같은 발 이동을 바닥 출발·상승·179.95° 회전이 포함된 경로에 연결해 468.0초간 재생했다. 94회 착지에서 예정한 지지면에 하중이 실렸다. 마지막 앞발 8단 하중은 162.4·201.3N, 왼쪽 뒷발 10단은 18.5N, 오른쪽 뒷발 9단은 98.0N이었다. 최대 토크 75.5%, 몸통 추종 오차 23.8mm, 지지 미끄러짐 추정치 7.56mm, 힘 분배 조건 완화 35틱을 기록했다. 전복·토크 포화·발 이외 접촉은 없었다. [468초 연속 경로](../evidence/t1-stair-ascent-20261120/ascent-turn-rear-nine-roll-plan.json), [물리 재생과 착지별 하중](../evidence/t1-stair-ascent-20261120/ascent-turn-rear-nine-roll-replay.json)

## 재현

```powershell
Set-Location "$HOME/go2-wall-reach-lab"
$py = '.venv/Scripts/python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$run = 'evidence/t1-stair-posture-repeat'
& $py tools/probe_stair_transfer.py "$run/poses.json" --workers 8
& $py tools/replay_stair_transfer.py "$run/poses.json" "$run/floor-first.json"
& $py tools/replay_stair_transfer.py "$run/poses.json" "$run/front-second.json" --move front_left_1_to_2
& $py tools/replay_stair_transfer.py "$run/poses.json" "$run/rear-next.json" --move rear_left_1_to_2
& $py tools/replay_stair_entry.py "$run/poses.json" "$run/front-pair.json" --workers 8
$prefix = 'evidence/t1-stair-posture-20261120'
$entry = 'evidence/t1-stair-entry-20261120'
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$entry/four-feet-plan.json" "$run/four-feet-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$entry/low-round-trip-plan.json" "$run/low-round-trip-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$entry/both-fronts-third-plan.json" "$run/both-fronts-third-replay.json"
$ascent = 'evidence/t1-stair-ascent-20261120'
& $py tools/repeat_stair_cycle.py "$prefix/poses-continuous.json" "$ascent/first-cycle-plan.json" `
    "$run/tread-ten-plan.json" --count 7 --wrap-shift 2.5
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$run/tread-ten-plan.json" "$run/tread-ten-replay.json"
& $py tools/repeat_stair_cycle.py "$prefix/poses-continuous.json" "$ascent/first-cycle-plan.json" `
    "$run/platform-front-plan.json" --count 8 --wrap-shift 2.5
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$run/platform-front-plan.json" "$run/platform-front-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$ascent/platform-all-four-plan.json" "$run/platform-all-four-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" `
    "$ascent/isolated-platform-turn-plan.json" "$run/isolated-platform-turn-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$ascent/ascent-and-turn-plan.json" "$run/ascent-and-turn-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" `
    "$ascent/platform-descent-entry.json" "$run/platform-descent-entry-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" `
    "$ascent/tread-nine-descent-plan.json" "$run/tread-nine-descent-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" `
    "$ascent/rear-right-tread-ten-plan.json" "$run/rear-right-tread-ten-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" `
    "$ascent/rear-left-tread-ten-plan.json" "$run/rear-left-tread-ten-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$ascent/ascent-turn-tread-nine-plan.json" "$run/ascent-turn-tread-nine-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" `
    "$ascent/front-right-tread-eight-plan.json" "$run/front-right-tread-eight-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" `
    "$ascent/offset-rear-left-descent-plan.json" "$run/offset-rear-left-descent-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" `
    "$ascent/two-tread-gap-descent-plan.json" "$run/two-tread-gap-descent-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$ascent/ascent-turn-tread-eight-plan.json" "$run/ascent-turn-tread-eight-replay.json"
& $py tools/replay_stair_sequence.py "$ascent/floor-return-with-arm-offset-geometry.json" `
    "$prefix/front-pair.json" "$ascent/floor-return-with-arm-offset-plan.json" `
    "$run/floor-return-with-arm-offset-replay.json" --round-trip
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" `
    "$ascent/rolled-rear-right-nine-plan.json" "$run/rolled-rear-right-nine-replay.json"
& $py tools/replay_stair_sequence.py "$prefix/poses-continuous.json" "$prefix/front-pair.json" `
    "$ascent/ascent-turn-rear-nine-roll-plan.json" "$run/ascent-turn-rear-nine-roll-replay.json"
& $py tools/replay_stair_sequence.py "$ascent/target-round-trip-geometry.json" `
    "$prefix/front-pair.json" "$ascent/target-round-trip-plan.json" `
    "$run/target-round-trip-replay.json" --round-trip
```

보존된 개별 1→2단 시행을 대조할 때는 `evidence/t1-stair-posture-20261120/poses.json`을 입력으로 사용한다. 보존된 바닥 시작 및 앞발 연속 시행의 입력은 `evidence/t1-stair-posture-20261120/poses-continuous.json`이다.

계단 중간 반복 보행을 재현할 때는 별도 Python 파일의 `if __name__ == '__main__':` 블록에서 `go2_wall.stairs.ARM`을 `rb1`로 지정한 뒤 `plan_cycle(50.0, 0.19, 2)`를 호출한다.
