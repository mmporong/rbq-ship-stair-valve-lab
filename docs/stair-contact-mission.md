# RBQ-10 계단에서 두 패드로 밸브 잠그기

50°·단 높이 190mm에서 RBQ-10+RB1이 바닥에서 열린 손으로 출발해 10N·m 밸브를 잠그고, 놓아 준 레버를 다시 건드리지 않고 출발점 부근에 네 발로 정지했다. MuJoCo 4.0에서 1ms 간격으로 기록한 976.240초 동안 156회 착지 전부에 지지 하중이 실렸다. 지지발 미끄러짐과 종아리 접촉력 등이 한계를 넘어 엄격한 통합 판정은 미달이다. [전체 임무 재판정](../evidence/stair-corridor-20261121/local-repair/full/repeat/raw-check.json) · [원시 배열 독립 검수](../evidence/stair-corridor-20261121/local-repair/full/repeat/independent-review.json)

| 50° 연속 재생 | 물리 기록 |
|---|---|
| 바닥 출발과 조작 | 바닥에서 네 발이 지지했다. 두 패드가 회전 중 5~85°의 16개 구간 모두에서 10N·m 부하를 전달했다. 유지 5초간 실측 각도 89.757~90.068°, 조작 중 최대 발 중심 이동 4.937mm였다. |
| 레버 해제 | 115.565초에 두 패드가 분리되기 시작했다. 분리 확인 뒤 860.176초간 두 패드와 다른 부위에서 레버 접촉력은 모두 0N이었다. |
| 상부 회전과 복귀 | 377.715초에 네 발이 상부 발판을 지지했고 504.815초의 몸통 방향은 179.926°였다. 이후 계단 길이 방향으로 2.708m 후퇴했다. 마지막 0.5초간 시작점과 최대 수평 거리 28.211mm, 네 발 중 최저 하중 68.713N이었다. |
| 첫 위반과 이후 측정 | 111.768초에 왼쪽 뒷발의 누적 접촉점 미끄러짐이 10.000016mm로 한계 10mm를 넘었다. 전체 최대값은 왼쪽 앞발 23.783mm였다. 185.351초에 지지력 제약이 처음 완화됐고 187.346초에 비허용 접촉이 13.08N으로 한계 10N을 넘었다. 왼쪽 앞 종아리와 계단 사이 최대 접촉력은 136.45N이었다. |

조작 중 발 중심의 이동 한계는 5mm이며, 지지 중 접촉점의 누적 미끄러짐 한계는 10mm다. 힘 분배 계산에서 제약 완화 604회, 실패 8회가 있었고, 전복·구동기 포화·상태 초기화·시뮬레이터 외력은 없었다. [시행 설정과 물리 판정](../evidence/stair-corridor-20261121/local-repair/full/repeat/result.json)

왼쪽 앞 종아리가 닿은 곳은 계단의 왼쪽 측면 지지대(`stringerL`)였다. 첫 초과 시각 187.346초와 최대 하중 시각 198.941초의 관절 상태로 접촉 형상을 복원했다. 198.941초에 기록된 값은 136.4542N이고, 그 순간의 접촉력을 다시 풀면 136.4549N이다. 전체 최대 구동 토크는 186.351초 왼쪽 뒷다리 종아리 축의 118.44N·m였다. 발판 반력은 바닥 접근 중 35.205초에 왼쪽 뒷발에서 556.52N까지 올랐다. [접촉 지점과 힘](../evidence/stair-corridor-20261121/local-repair/full/contact-geometry.json) · [전체 하중표](../evidence/stair-corridor-20261121/local-repair/full/first/route-follow.loads.csv) · [기체를 따라가는 왕복 영상](../evidence/stair-corridor-20261121/local-repair/full/first/route-follow.mp4) · [첫 미끄러짐 8초](../evidence/stair-corridor-20261121/local-repair/full/first/slip-follow.mp4)

111.768초의 기록은 네 다리 중 가장 큰 미끄러짐을 담는다. 원래 경로에서 별도로 수행한 129.750초 시험과 전 구간 기록의 34개 배열이 124.850초까지 같았다. 그 단기 시험 전부에서 왼쪽 앞발, 오른쪽 앞발, 오른쪽 뒷발의 최대값은 각각 9.288mm, 8.027mm, 9.541mm였다. 최초로 10mm를 넘은 다리는 왼쪽 뒷발로 확인됐다. [발별 최댓값과 두 기록의 대조](../evidence/stair-corridor-20261121/local-repair/full/slip-attribution.json)

## 50°에서 재디딤이 왕복에 미친 영향

같은 계단과 수정 보행 경로에서 왼쪽 뒷발의 재디딤과 유지 후 모터 보정을 추가했다. 두 제어 모두 밸브에 10N·m 부하를 전달하고 90° 부근에서 5초 유지한 뒤 손을 놓았다. 보행 기준은 지지발 누적 미끄러짐 10mm, 비허용 접촉력 10N이고, 조작 중 발 중심 이동 한계는 5mm다.

| 50° 전 구간의 실측 | 왕복한 경로 | 재디딤을 더한 경로 |
|---|---:|---:|
| 기록 길이 | 976.240초 | 979.040초 |
| 5초 유지각 | 89.757~90.068° | 89.766~90.068° |
| 조작 중 최대 발 중심 이동 | 4.937mm | 4.945mm |
| 지지발 미끄러짐 첫 10mm 초과 | 111.768초, 유지 중 | 127.831초, 보행 재개 후 |
| 지지 하중을 충족한 원래 착지 | 156/156 | 24/156 |
| 같은 단에 추가한 재디딤 | 없음 | 1/1 |
| 상부 회전과 바닥 복귀 | 179.926° 회전 후 안정 정지 | 190.881초에 전복 |
| 엄격한 통합 판정 | 미달 | 미달 |

재디딤 시험은 밸브 해제 뒤 0.5초간 두 패드가 떨어져 있는지 확인한 후 남은 859.937초 동안 레버 재접촉력이 0N이었다. 보행 중에는 189.597초에 몸통 추종 오차가 30mm를 넘었다. 190.830초에 왼쪽 뒷발과 왼쪽 앞 종아리가 부딪혀 575.24N이 걸렸고, 51ms 뒤에 전복됐다. 191.010초에는 몸통이 바닥에 닿아 14.04kN이 기록됐다. 마지막 0.5초의 시작점 수평 오차는 최대 3.886m였다. [전 구간 측정](../evidence/stair-corridor-20261121/replant/full/measurements.json) · [기록 상태에서 복원한 접촉쌍](../evidence/stair-corridor-20261121/replant/full/contact-geometry.json)

전복 뒤 191.071초에 오른쪽 앞 종아리 축이 150N·m에 이르렀다. 이 실패 시행의 팔 장착부 최대 모멘트는 247.656초의 441.79N·m, 발·바닥 간 최대 반력은 248.272초의 1,878.59N이었다. 전체 영상 980프레임과 첫 미끄러짐·전복 영상 각 200프레임의 시각, 미끄러짐, 비허용 접촉력을 원시 기록과 대조했다. [전체 영상](../evidence/stair-corridor-20261121/replant/full/first/route-follow.mp4) · [첫 미끄러짐](../evidence/stair-corridor-20261121/replant/full/first/first-violation.mp4) · [전복 전후 8초](../evidence/stair-corridor-20261121/replant/full/first/fall-follow.mp4) · [하중 히트맵](../evidence/stair-corridor-20261121/replant/full/first/route-follow.heatmap.html) · [발판·관절 하중표](../evidence/stair-corridor-20261121/replant/full/first/route-follow.loads.csv)

왕복할 수 있는 976.240초 경로를 후속 시험의 기준으로 삼는다. 이 경로의 남은 과제는 밸브 유지 중 왼쪽 뒷발의 누적 미끄러짐을 10mm 안에 두고, 상승 시 왼쪽 앞 종아리가 계단 측면 지지대에 가하는 힘을 10N 안으로 줄이며, 힘 분배 제약 완화와 계산 실패 없이 왕복 두 건을 재현하는 것이다.

## 재디딤 뒤 첫 보행 전환

오른쪽 뒷발이 닿은 뒤 78.8초에 왼쪽 뒷발을 들어 같은 발판을 다시 짚었다. 추가 착지 1회와 원래 경로의 착지 156회를 따로 센다. 왼쪽 뒷발 하중은 0N까지 내려갔고, 발이 51.470mm 올라간 뒤 81.6초에 46.60N으로 다시 지지했다. 목표와 실제 발 중심의 수평 차이는 0.815mm였다. 유지 종료 뒤에는 이 다리의 모터에 3,075N/m로 접촉점 변위를 보정했다. [재디딤 원시 재판정](../evidence/stair-corridor-20261121/replant/short-control/frozen-verification.json) · [네 이득의 비교](../evidence/stair-corridor-20261121/replant/short-control/README.md)

같은 재디딤·팔 명령으로 바닥에서 출발해 작업 다음 착지까지 재생한 결과다. 세 시행의 CPU 수치 연산은 한 스레드로 고정했다. 조작 중 발 중심 이동 한계는 5mm, 보행 중 지지 접촉점의 누적 미끄러짐 한계는 10mm다.

| 보행 입력 | 계단 옆 보행 보정 | 밸브 판정 | 보행 판정 | 왼쪽 뒷발 최대 누적 미끄러짐 | 첫 위반 |
|---|---|---|---|---:|---|
| 원래 경로 | 끔 | 통과 | 통과 | 9.968mm | 132.550초까지 없음 |
| 수정 경로 | 끔 | 통과 | 미달 | 11.219mm | 129.694초, 미끄러짐 |
| 수정 경로 | 켬 | 미달 | 미달 | 12.804mm | 127.831초, 미끄러짐 |

보행 보정을 끈 두 입력의 공통 34개 배열은 127.650초까지 같았다. 수정 경로에서 보정을 켜고 끈 두 시행은 127.655초까지 같았다. 보정을 켠 시행도 16개 회전 구간과 5초 유지를 충족했지만 복귀 중 힘 분배 제약을 19회 완화해 밸브 판정의 연속성 기준을 벗어났다. 같은 입력을 수치 연산 스레드 수의 기본 설정으로 재생하면 첫 미끄러짐은 127.791초였다. [네 시행의 소스·원시 해시·최초 분기](../evidence/stair-corridor-20261121/replant/transfer-comparison.json) · [보정을 끈 시행](../evidence/stair-corridor-20261121/replant/no-corridor/raw-check.json) · [보정을 켠 시행](../evidence/stair-corridor-20261121/replant/matched-through-task/raw-check.json)

보정을 켠 127.831초의 표본에서 왼쪽 뒷발은 3단에 210.488N을 실었고 마찰 사용률은 0.291이었다. 네 발에 하중이 남아 있었고 비허용 접촉력은 0N이었다. 이 시점은 유지 종료 후 11.181초, 양패드 하중과 간격이 해제 기준에 든 첫 표본에서 9.227초 뒤다. 시행 당시의 제어 코드 14개와 모델 자산 18개를 별도 디렉터리에 복원해 바닥에서 다시 시작해도 135,715개 물리 표본의 원시 파일, 장면, 판정이 바이트 단위로 일치했다. [고정 소스에서의 재현](../evidence/stair-corridor-20261121/replant/matched-through-task/source-replay.json)

같은 재디딤 제어와 단일 스레드 설정에서 기준·변형·민감도 17조건을 검사했다. 물리 재생 14건에서 조작과 보행의 엄격한 판정은 각각 0/14였다. 세 가지 계단 형상은 경로 생성에서 멈췄다. 접촉 솔버의 마찰 임피던스 비를 10으로 바꾼 시행에서는 4.197초에 지지발이 10.0015mm 미끄러졌고 51.747초에 비허용 접촉력이 575.29N에 이르렀다. 81.600초에는 왼쪽 뒷발 재디딤이 실패했다. 당시 기록기가 앞세운 사건은 전체 요약에서 51.747초, 보행 요약에서 81.600초였다. 수정한 기록기로 새로 계산해 135,715개 표본이 담긴 같은 원시 파일을 얻었고 첫 실패가 4.197초로 보고됐다. [17조건과 실패 장면](../evidence/stair-corridor-20261121/conditions-replant/RESULTS.md) · [CPU 스레드 설정별 대조](../evidence/stair-corridor-20261121/conditions-replant/COMPARISON.md) · [기록기 수정 전후의 물리 재생](../evidence/stair-corridor-20261121/replant/first-failure-order.json)

## 50° 보행 대조와 60° 비교

바닥에서 작업 자세까지 걸어온 뒤 10N·m 밸브를 다룬 129.750초 시행에서 처음으로 두 패드 조작 판정을 통과했다. 팔의 회전 말기 계획각을 92.6°로 두고 2초 동안 유지 자세의 91.4°로 되돌렸다. 실제 힌지는 5초 동안 89.757~90.068°를 유지했다. 패드 이탈 0초, 최대 발 이동 4.937mm, 몸통 이동 4.901mm였다. 해제 뒤에도 관찰한 이 시험에서 왼쪽 뒷발은 111.768초에 보행 미끄러짐 한계인 10mm를 넘었고 최대 13.312mm에 도달했다. 조작 중 발 중심 이동과 착지 뒤 발바닥 접촉점의 누적 미끄러짐은 각각 원래 정의로 계산했다. 독립 실행 두 건에서 1ms 원시 표본 129,750개와 34개 배열이 바이트 단위로 일치했다. [판정](../evidence/stair-corridor-20261121/floor-valve/raw-check.json) · [밸브와 보행 원시 데이터](../evidence/stair-corridor-20261121/floor-valve/raw-parts/parts.json) · [뒷발의 누적 이동 방향](../evidence/stair-corridor-20261121/floor-valve/rl-slip.json)

같은 바닥 접근과 조작 명령을 사용한 별도 시행에서 하중을 1ms마다 기록했다. 94.00~121.84초 영상에는 두 패드의 접촉부터 해제·팔 회수까지 담았다. 이 구간의 최대 지지력은 왼쪽 앞발 248.41N, 팔 장착부의 최대 굽힘·비틀림 모멘트 크기는 28.03N·m였다. 영상 349프레임의 시각과 하중표의 네 극값을 원시 기록과 대조했다. 이 시행은 조작 뒤 측면 보행 경로를 따른다. [조작 영상](../evidence/stair-corridor-20261121/floor-valve/rated-operation.mp4) · [하중 히트맵](../evidence/stair-corridor-20261121/floor-valve/rated-operation.heatmap.html) · [원시 시각 대조](../evidence/stair-corridor-20261121/floor-valve/rated-operation.verify.json)

같은 50°·190mm 보행 참조와 5ms 모터 주기로 밸브 없는 계단을 두 번 재생했다. 물리 간격 1ms와 5ms 모두 736초간 156회 착지 하중, 상부 회전, 바닥 복귀를 충족했다. 마지막 0.5초 동안 출발점과의 최대 수평 거리는 각각 28.21mm와 28.25mm였다. [원시 지지 하중 재판정](../evidence/stair-corridor-20261121/walk-control/raw-check.json)

60°·190mm에서는 기존 참조의 몸통과 3단이 13.60초에 0.684mm 겹쳤다. 선택한 몸통 경로는 그 구간을 통과하고 일곱 번째 착지까지 계획할 수 있었다. 38.06초에 왼쪽 뒷발을 들면서 발 위치 오차가 2.927mm에 이르러 2mm 허용치를 넘었다. 탐색한 몸통 자세 4,374개 중 13개가 성긴 표본에서 하중 이송과 스윙을 충족했지만, 어느 후보도 20ms 간격의 연속 검사를 통과하지 못했다. [자세 탐색](../evidence/stair-corridor-20261121/ship60/swing_pose_search.json) · [두 경계의 재현 코드](../evidence/stair-corridor-20261121/ship60/replay.py)

50°에서 레버 옆으로 지나려는 첫 후보는 작업 자세 이후 발 디딜 곳을 옆으로 옮기고, 상부와 승강장에서는 보행 전체를 72mm 옮긴다. 36,801개 참조 자세에서 닫힌 레버와의 최소 거리는 11.065mm였다. 바뀐 28,104개 자세를 마찰 여유를 포함한 계수 0.42로 다시 계산해 얻은 최대 관절 토크 비율은 0.612였다. 이 수치의 범위는 저장한 계획 표본이다. 표본 사이에서는 별도 보간과 물리 재생이 필요하다. [보행 참조](../evidence/stair-corridor-20261121/lateral/walk50.npz) · [기하 검사](../evidence/stair-corridor-20261121/lateral/geometry.json) · [정역학 검사](../evidence/stair-corridor-20261121/lateral/static.json)

감속 구간에 14,027개 중간 자세를 추가하고 각 자세에서 네 발 위치를 다시 풀었다. 보행 시간이 906.645초인 새 경로는 원래 20ms 격자와 156회 착지 목표를 보존한다. 50,828개 표본에서 발 구와 계단 사이를 제외한 최대 겹침은 0mm, 닫힌 레버와의 최소 거리는 10.302mm, 최대 발 위치 오차는 1.942mm였다. 바뀐 42,132개 자세를 마찰계수 0.42의 지지력 제약으로 다시 풀었을 때 최대 관절 토크 비율은 0.623이었다. [보행 입력](../evidence/stair-corridor-20261121/refined/walk50.npz) · [기하와 정역학 재검사](../evidence/stair-corridor-20261121/refined/geometry.json)

첫 측면 경로만 적용한 913.090초 바닥 출발 시행에서는 111.969초에 지지발 미끄러짐이 10mm를 넘었다. 143.321초에는 비허용 접촉력 59.46N이 기록됐고, 162.093초에 넘어졌다. 156회 중 22회 착지에서 예정한 발판에 필요한 하중이 실렸다. [913,090개 표본의 재판정](../evidence/stair-corridor-20261121/ablations/lateral-only/raw-check.json)

정착 4초에 감속 경로와 지지 다리 보정을 더한 943.895초 시행을 두 번 재생했다. 두 기록에서 34개 공통 배열이 일치했다. 최초 위반은 111.851초의 지지발 미끄러짐이었다. 195.565초에는 오른쪽 앞 넓적다리가 놓아 준 레버에 63.23N으로 닿았고, 217.811초에 전복됐다. 156회 중 26회에서 지지 하중을 충족했다. [전 구간 측정](../evidence/stair-corridor-20261121/ablations/refined-control/measurements.json) · [두 기록의 대조](../evidence/stair-corridor-20261121/ablations/refined-control/reproduction.json)

추적 오차가 컸던 상승·하강 구간에서 몸통과 관절의 경로를 고친 뒤 모터 명령을 다시 감속했다. 이 경로에는 53,020개 자세가 있다. 바뀐 44,324개 표본에서 정적 지지력을 다시 풀었고, 물리 재생에 쓰는 5ms 보간까지 포함해 192,594개 시각의 자세를 검사했다. 닫힌 레버까지의 최소 거리는 6.956mm, 발 위치의 최대 오차는 1.958mm, 비허용 표면의 관통은 0mm였다. 바닥에서 작업 자세에 이르는 89.6초와 156회 착지 일정의 발·발판 조합은 보존했다. [보행 입력과 속도](../evidence/stair-corridor-20261121/tracked/inputs.json) · [정역학](../evidence/stair-corridor-20261121/tracked/geometry.json) · [5ms 보간 검사](../evidence/stair-corridor-20261121/tracked/motor-geometry.json)

같은 경로에서 닫힌 밸브 옆 상승과 하강을 해당 구간의 첫 자세에서 각각 시작했다. 208.895초 상승에서는 19개 중 6개 착지에서 예정한 지지 하중을 충족했고 도중에 넘어졌다. 225.170초 하강에서는 21개 착지 모두 지지했고 밸브는 90°를 유지했다. 비허용 접촉과 구동기 포화는 없었지만, 왼쪽 뒷발 미끄러짐이 18.303mm에 이르렀고 지지력 계산도 9회 실패했다. 두 시험 모두 1ms로 물리를 계산하고 10ms마다 자세·발판별 하중을 저장했다. [상승과 하강의 구간별 판정](../evidence/stair-corridor-20261121/tracked/segments/summary.json) · [구간 재생 코드](../evidence/stair-corridor-20261121/tracked/segments/replay.py)

두 곳의 몸통 위치를 최대 2.0mm와 0.5mm 옮긴 경로에서는 손끝 가정 하중을 전 구간 0N으로 고정했다. 예비 파일에서 손끝 위치가 힘 입력 칸에 들어간 124개 행을 찾아 그 파일을 제외했다. 수정본은 5ms 모터 시각 188,199곳에서 레버와의 최소 간격 0.125mm, 발 위치 최대 오차 1.980mm를 기록했다. [입력과 정역학](../evidence/stair-corridor-20261121/local-repair/geometry.json) · [손끝 힘 검수](../evidence/stair-corridor-20261121/local-repair/input-review.json)

이 경로에서 열린 손과 닫힌 레버로 상승·하강을 따로 시작해 착지 지지 19/19회와 21/21회를 얻었다. 1ms 물리 스텝 412,090개에서 패드 두 개와 기체의 다른 부위를 모두 관측해 레버 접촉 0건을 확인했다. 상승 때 왼쪽 앞 종아리가 계단에 최대 136.58N으로 닿았다. 지지발 미끄러짐은 상승 23.781mm, 하강 16.893mm였다. [구간 결과와 원시 기록](../evidence/stair-corridor-20261121/local-repair/segments/summary.json) · [스텝별 레버 접촉력 대조](../evidence/stair-corridor-20261121/local-repair/segments/contact-audit/summary.json)

아래 명령은 50°의 최신 입력에서 발 목표와 참조 시각을 재구성하고, 원래 계단 제어기의 마찰 여유와 토크 한계로 하중을 다시 계산한다. 마지막 명령은 60°의 원래 경로와 수정 경로에서 각각 처음 한계를 넘긴 표본을 재현한다.

```powershell
cd "$HOME/go2-wall-reach-lab"
$report = Join-Path $env:TEMP ("stair-corridor-" + [guid]::NewGuid() + ".json")
.venv/Scripts/python tools/check_stair_corridor.py evidence/stair-corridor-20261121/local-repair/walk50.npz $report --motor-ticks --check-statics
.venv/Scripts/python evidence/stair-corridor-20261121/ship60/replay.py
```

현재 제어기의 1N·m 대조는 1ms 물리 간격으로 각각 24.250초를 재생했다. 손을 닫은 시행은 양패드 접촉을 0.395초 유지하고 부하 구간에서 21.922° 회전했다. 축 방향 미끄러짐은 0.015mm였다. 손을 연 시행에서는 양패드 파지 시간이 0초이고 부하 구간 최대 각도는 7.199°였다. 두 원시 기록과 해당 제어 소스 해시는 함께 보관했다. [저부하와 열린 손 대조](../evidence/stair-corridor-20261121/tracked/low-control/results.json)

10N·m에서 팔의 접근·회수와 몸통 전환을 합쳐 2.5초 줄이면 왼쪽 뒷발의 누적 미끄러짐은 13.312mm에서 12.736mm로 줄었다. 같은 시행은 유지 시작각이 89.7492°로 하한을 벗어났다. 시간을 5초 줄인 시행에서는 미끄러짐이 12.068mm였고, 왼쪽 앞다리 허벅지가 발판에 닿아 217.97N이 걸렸다. 원래 접근 시간, 변경한 시간표, 뒷발을 다시 디딘 시험을 각각의 실제 결과와 함께 보관했다. [시간표와 지지 조건별 재생 결과](../evidence/stair-corridor-20261121/tracked/approach-comparison.json)

53,020개 자세로 감속 범위를 넓힌 대조 경로에서는 두 번의 998.215초 재생 모두 착지 지지 24/156회, 191.086초 전복을 기록했다. 두 시행의 공통 배열 34개는 물리 스텝 998,215개 전부에서 일치했다. 위에서 156/156회를 기록한 경로는 상승·하강 두 곳의 자세를 보정한 별도 입력을 썼다. [감속 범위를 넓힌 경로의 전 구간 판정](../evidence/stair-corridor-20261121/tracked/full/measurements.json) · [해당 경로의 독립 반복](../evidence/stair-corridor-20261121/tracked/full/reproduction.json)

## 768초 기준 임무와 하중

서로 다른 두 프로세스가 각자 바닥에서 출발해 하나의 물리 상태로 768.000초를 계산했다. 시행당 153,600스텝, 예정 착지 156개를 마지막 관측까지 전부 검사했다. 두 원시 파일은 각각 336,161,212바이트이며 40개 배열과 파일 SHA-256이 일치했다. 초기화와 외력은 0이었다. [첫 시행 재판정](../evidence/stair-mission-20261121/full/first/raw-check.json) · [반복 시행 재판정](../evidence/stair-mission-20261121/full/repeat/raw-check.json) · [두 파일 대조](../evidence/stair-mission-20261121/full/two-process-comparison.json)

| 물리 시각 | 관측 |
|---:|---|
| 97.625초 | 위·아래 패드 하중 90.41N·0N. 아래 패드가 1N 미만인 상태가 25ms 지속돼 허용치 20ms를 처음 넘었다. |
| 111.500초 | 지지발 누적 미끄러짐이 10mm를 처음 넘었다. |
| 132.710초 | 손을 놓고 보행을 재개한 뒤 패드 이외 부위와 레버 사이에 270.46N의 접촉력이 기록됐다. |
| 149.915초 | 오른쪽 뒷다리 엉덩관절이 구동 토크 한계에 도달했다. |
| 172.305초 | 28번째 착지 검사에서 예정된 지지면 하중이 0N이었다. 해당 착지 종료 시 오른쪽 뒷발의 수평 오차는 545.0mm였다. |
| 201.290초 | 전복이 처음 기록됐다. 상부에서 네 발이 모두 지지된 표본과 회전 완료 표본은 없었다. |
| 마지막 0.5초 | 시작점으로부터 최대 수평 오차 463.2mm, 최대 기울기 57.13°, 네 발 중 최소 바닥 하중 0N으로 복귀 기준을 벗어났다. |

5°씩 나눈 회전 16구간 안에서 두 패드가 전달한 부하와 임무 전체의 순간 피크를 구분했다. 유지 중인 111.360초에 패드의 닫힘 토크가 최대 10.753N·m였다. 다리·팔 18축 중 실제 토크 절댓값은 170.345초의 오른쪽 뒷다리 종아리 관절이 가장 컸고 부호를 보존한 값은 −150.0N·m였다. 발–지지면 반력은 전복 뒤 443.125초의 오른쪽 뒷발–바닥에서 1,400.25N이었다. 장착부 합력 8,046.14N과 합모멘트 1,173.06N·m도 전복 뒤 각각 360.850초, 442.145초에 나왔다. [축·지지면별 하중표](../evidence/stair-mission-20261121/full/first/full-trip.loads.csv) · [시간별 하중 히트맵](../evidence/stair-mission-20261121/full/first/full-trip.heatmap.html) · [원시 피크](../evidence/stair-mission-20261121/full/first/full-trip.peaks.json)

전체 영상은 24배속 800프레임, 첫 접촉 이탈을 확대한 영상은 0.5배속 175프레임이다. 각 영상의 모든 자막 시각을 원시 표본과 대조했다. 최대 시각 오차는 모두 5ms였다. [전체 영상 대조](../evidence/stair-mission-20261121/full/first/full-trip.verify.json) · [확대 영상 대조](../evidence/stair-mission-20261121/full/first/pad-loss.verify.json)

## 파지와 회전

작업 자세에서 두 앞발은 5단, 두 뒷발은 3단에 놓였다. 레버 길이는 0.12m, 바닥 마찰계수는 0.6, 패드 마찰계수는 0.8로 두었다. 손가락은 100N 구동기 하나에 연동했다. 힌지는 수동이며 각도와 접촉력은 물리 스텝마다 기록했다. 패드와 레버 이외의 기체 부위도 충돌한다. [고정 조건과 입력 출처](../evidence/stair-mission-20261121/protocol.json) · [모델 구성](../src/go2_wall/stair_valve.py)

- **저부하** 시험에서 1N·m 저항을 받으며 20.422° 회전했다. 두 패드는 0.360초 연속 각각 1N 이상을 유지했고, 레버 길이 방향의 미끄러짐은 최대 0.388mm였다. 열린 손 대조군에서는 한쪽 접촉으로 38.35° 움직였고 양패드 파지 시간은 0초였다. [양성·음성 원시 기록과 판정](../evidence/stair-mission-20261121/low-grip/results.json)
- **정격** 시험에서 5~85°를 5°씩 나눈 16개 구간 전부에 양패드로 저항을 이긴 표본이 있다. 이 조건에 해당하는 1,281개 표본의 패드 전달 토크는 9.99985~10.00027N·m, 접선 방향 합력은 80.93~84.83N이었다. 접선력은 두 패드가 레버에 가한 힘 벡터를 접선 방향으로 투영해 계산했다. 동일한 장면을 별도 프로세스에서 재생해 종전 계측과 공통인 원시 배열 39개가 모두 일치하는지 확인했다. [토크·접선력·접촉 위치](../evidence/stair-mission-20261121/conditions/physical-b/nominal/through-task/raw-check.json) · [두 재생의 원시 배열 대조](../evidence/stair-mission-20261121/conditions/physical-b/nominal/through-task/matched-five-ms.json)
- **첫 접촉 이탈과 해제**를 원시 기록에서 다시 판정했다. 97.600초의 위·아래 패드 법선력은 94.66N과 15.43N이었다. 아래 패드는 97.605초부터 145ms 연속 1N 미만이었다. 97.625초에 허용치 20ms를 처음 넘었다. 5초 유지 시작 시 밸브각은 88.997°였다. 113.605초부터 손을 놓았고, 두 패드가 모두 5mm 이상 떨어진 115.175초에는 87.095°, 기록이 끝난 126.50초에는 69.409°였다. [25,300스텝 재판정](../evidence/stair-mission-20261121/conditions/physical-a/nominal/through-task/raw-check.json) · [하중과 영상 시각 대조](../evidence/stair-mission-20261121/conditions/physical-b/nominal/through-task/pad-loss.verify.json)

조건표에 사용한 제어기로 저부하와 열린 손 대조도 다시 실행했다. 명령을 보낸 시각으로 단계를 구분하면 1N·m에서 양패드가 0.395초 연속 접촉했고 21.660° 회전했다. 열린 손의 양패드 파지 시간은 0초였다. 옛 단독 재생기는 물리 스텝 종료 시각으로 단계명을 붙였다. 두 기록에서 각각 경계 표본 4개의 단계명이 달라지며 저부하 통과와 대조군 실패 판정은 같다. [같은 제어기의 원시 기록](../evidence/stair-mission-20261121/conditions/physical-b/low-control/results.json) · [명령 시각 재판정](../evidence/stair-mission-20261121/conditions/physical-b/low-control/aligned-clock.json)

### 해제 뒤 각도와 물리 스텝

작업 자세에서 시작한 별도 시험은 같은 코드, 10N·m 힌지, 5ms 구동 주기로 물리 스텝만 바꿨다. 5ms와 2.5ms에서는 두 패드가 떨어진 뒤 밸브가 허용 각도를 벗어났다. 1ms에서는 손을 뗀 뒤 5.784초 동안 각도가 유지됐고 마지막 값은 90.001°였다. 회전·유지 중 패드 이탈 시간은 0초였다. 다만 유지 첫 표본이 89.613°로 허용 하한 89.75°보다 낮았고, 최대 발 이동 5.129mm도 한계 5mm를 넘었다. 이 세 시행은 모두 정격 판정에서 실패했다. [세 시간 간격의 원시 기록과 판정](../evidence/stair-mission-20261121/grip-c/results.json)

1ms에서 손가락 목표를 닫힘 접선 방향으로 0.5mm 옮긴 뒤, 바닥부터 129.500초를 연속 재생했다. 5~85° 부하 16구간을 모두 채웠고 패드 이탈은 0초였다. 두 패드가 떨어진 뒤 13.690초를 관찰했으며 마지막 각도는 90.000°였다. 회전·유지 구간의 최고 각도는 89.985°였다. 5초 유지 시작각 89.050°와 최대 발 이동 5.107mm도 기준을 벗어났다. 129,500스텝에서 전복, 구동기 포화, 힘 분배 실패·완화는 모두 0회였다. [판정과 접촉 위치](../evidence/stair-mission-20261121/floor-grip-refinement/raw-check.json) · [입력·원시 복원 정보](../evidence/stair-mission-20261121/floor-grip-refinement/inputs.json)

### 작업 자세에서 통과한 두 패드 정격 조작

앞발 5단·뒷발 3단의 작업 자세에서 회전 명령을 9초로 설정했다. 회전과 유지 중에는 지지발의 평면 이동에 따라 다리 모터 토크를 보정했다. 1ms 물리 스텝·5ms 명령 주기에서 두 번 모두 10N·m 밸브를 잠그고 5초간 90.066~90.068°를 유지했다. 최대 발 이동은 4.981mm, 몸통 이동은 4.952mm였다. 두 패드가 떨어진 뒤 5.767초에 밸브각은 90.001°였다. 두 독립 시행의 원시 파일이 바이트 단위로 같았다. 같은 명령의 5ms 물리 스텝에서는 패드 이탈이 130ms 지속했고 종료각은 35.483°였다. [입력·원시 기록·시간 간격별 판정](../evidence/stair-mission-20261121/grip-refinement/RESULTS.md)

### 바닥에서 출발했을 때의 1ms 정착 시간 비교

다리 모터로 지지발 위치를 보정하는 제어기를 일반 재생 명령에 연결했다. 작업 자세에서 시작한 1ms 재생은 10N·m·5초 유지·해제를 다시 통과했다. 같은 구현으로 1N·m 저부하를 시험하면 부하 구간에서 21.922° 회전했고 양패드 접촉은 0.395초 연속 유지됐다. 손을 계속 연 대조군의 파지 시간은 0초였다. [일반 실행 명령으로 생성한 세 원시 기록의 해시와 판정](../evidence/stair-mission-20261121/stance-feedback/standalone/results.json)

아래 두 시행은 동일한 50°·190mm 참조 경로에서 열린 손으로 바닥을 출발했다. 정착 시간 2초와 4초를 비교했다. 회전 명령 9초와 뒤이은 유지 5초, 물리 스텝 1ms, 명령 주기 5ms, 발 위치 보정 이득 2,500N/m는 같다. 두 조건 모두 부하 16구간과 양패드 연속 접촉을 충족했고, 해제 후 13.685초까지 관찰했다. [입력·자료·실패 직전 장면](../evidence/stair-mission-20261121/stance-feedback/conditions.json)

| 총 정착 시간 | 물리 기록 | 5초 유지각 | 최대 발 이동 | 해제 후 종료각 | 최초 위반 |
|---|---:|---:|---:|---:|---|
| 2초 | 129.750초 | 89.413~90.068° | 왼쪽 앞발 4.972mm | 90.000° | 108.851초 유지 시작각. 하한 89.75° |
| 4초 | 131.750초 | 89.792~90.068° | 왼쪽 앞발 5.329mm | 90.000° | 111.851초 누적 지지발 미끄러짐 10mm 초과. 114.008초에는 왼쪽 앞발 이동도 5mm를 넘음 |

정착을 늘린 시행에서 왼쪽 앞발은 유지 끝까지 시작점에서 좌우 방향으로 4.988mm 이동했다. 앞뒤·높이 성분을 합친 이동 거리는 5.329mm였다. 보행 기준은 발이 지지하는 동안 접선 속도를 적분해 미끄러짐을 계산한다. 네 발 중 이 값이 가장 큰 왼쪽 뒷발에서 각각 13.293mm와 13.738mm를 기록했다. [2초 정착 재판정](../evidence/stair-mission-20261121/stance-feedback/settle-2s/raw-check.json) · [4초 정착 재판정](../evidence/stair-mission-20261121/stance-feedback/settle-4s/raw-check.json)

4초 정착 조건은 바닥부터 마지막 예정 착지까지 773.250초를 별도 프로세스에서 재생했다. 최초 위반은 111.851초의 보행 미끄러짐이었다. 137.554초에 패드 이외 부위가 레버에 90.63N을 가했다. 해당 시점의 원시 자세에서 레버에 닿은 기체 부위는 몸통이었다. 167.751초에는 26번째 착지에서 예정 지지 하중이 0N으로 떨어졌고, 175.050초에 해제된 밸브가 89.749°로 허용 하한을 벗어났다. 181.259초에 전복됐다. [접촉 전후 자세·힘과 전체 판정](../evidence/stair-mission-20261121/stance-feedback/inputs.json)

156개 예정 착지 중 네 발의 지지 하중을 충족한 횟수는 25회였다. 플랫폼에서 네 발이 지지된 표본과 상부 회전 완료 표본은 모두 0개였다. 마지막 0.5초의 출발점 수평 오차는 최대 2.273m, 기울기는 최대 125.88°, 네 발 중 최소 바닥 하중은 0N이었다. 최종 밸브각은 73.348°였다. 독립 검수에서 156개 착지 판정과 해제 후 재접촉·전복·복귀 수치를 원시 배열로 재현했다. [773,250스텝 원시 재판정](../evidence/stair-mission-20261121/stance-feedback/full/raw-check.json) · [독립 검수 기록](../evidence/stair-mission-20261121/stance-feedback/full/independent-review.json)

이 시행은 34개 채널을 기록했다. 지지면별 3축 반력과 장착부 하중을 포함한 영상·히트맵은 위 768초 기준 시험에서 얻었다. [상세 하중과 영상 대조](../evidence/stair-mission-20261121/full/first/full-trip.verify.json)

### 잠긴 밸브 옆에서 다시 걷기

닫힌 레버 옆을 통과할 공간이 있는지 확인하려고 밸브를 처음부터 90°로 놓고 바닥에서 출발했다. 기존 보행은 100.307초에 몸통이 레버와 90.62N으로 충돌했다. 몸통을 낮추고 위치·기울기를 조정한 경로는 107.400초까지 예상 밖 접촉과 전복이 없었다. 이 구간의 최대 몸통 추종 오차는 18.87mm, 지지발 미끄러짐은 6.77mm였다. 원래 보행으로 돌아갈 때 107.412초에 추종 오차가 32.50mm로 30mm 한계를 넘었다. 따로 계획한 연장 구간도 107.270초에 추종 기준을 넘었다. [바닥 출발 우회 경로와 접촉 기록](../evidence/stair-mission-20261121/bypass/floor-route/README.md)

명목 보행의 112.02초 자세에서 시작해 뒷발 순서를 바꾼 별도 시험에서는 열린 손을 유지한 526개 계획 자세가 정적 검사를 통과했다. 같은 구간을 물리 재생하자 3.521초에 오른쪽 뒷발 지지력이 1N 아래로 내려갔고, 7.333초에 레버와 101.42N으로 충돌했다. 두 시각은 구간 시작 뒤 경과한 물리 시간이다. 바닥부터 이 시작 자세까지 연속으로 도달한 기록은 없다. [순서 변경 계획·물리 재생·독립 반복](../evidence/stair-mission-20261121/bypass/swing-order/README.md)

## 50°와 60°를 포함한 조건 비교

53,020개 자세의 보행 경로와 같은 제어기로 시험한 물리 14조건 중 6개가 밸브 조작을 통과했다. 통과한 조건은 기준값, 발판 마찰계수 0.5와 0.8, 레버 길이 0.14m, 패드 마찰계수 0.6, 손가락 슬라이드 마찰 손실 15N이었다. 보행까지 충족한 조건은 0개였다. 기준 조건에서 최초 위반은 111.768초의 지지발 미끄러짐 10.000016mm였다. 경사 또는 단 높이를 바꾼 3개 조건은 계획 단계에서 멈췄다. 각 물리 시행의 보행·조작 판정을 원시 기록에서 다시 계산했다. [17조건과 첫 실패 장면](../evidence/stair-corridor-20261121/conditions-tracked/RESULTS.md) · [무결성 검사](../evidence/stair-corridor-20261121/conditions-tracked/verification.json)

### 초기 5ms 제어기 대조

경사 비교는 사용자가 선택한 60° 선박형 계단에서 단 높이를 190mm로 유지했다. 각 행에서 변수 하나만 바꿨다. 동일한 계획기에서 세 형상은 기하 검사를 통과하지 못했다. 나머지 14개는 각각 새 CPU 프로세스로 바닥에서 시작했고, 모두 작업 직후까지의 연속 재생에서 이미 필수 조건을 위반했다. 이 제어기의 전체 임무 성공은 **0/17**이다. 성공·실패가 맞닿은 두 조건이 없어 전환 경계의 범위는 산출하지 않았다. [조건별 입력·최초 실패·장면 17개](../evidence/stair-mission-20261121/conditions/physical-b/condition-results.json)

| 주 조건 | 첫 위반 시각 | 측정값과 판정 기준 |
|---|---:|---|
| 기준 50°·190mm·10N·m | 97.625초 | 아래 패드 하중 1N 미만이 25ms 지속. 허용 20ms |
| 60°·190mm | 13.600초 | 계획 자세의 몸통이 3단에 0.684mm 겹침. 허용 0.10mm |
| 50°·165mm | 55.960초 | 오른쪽 뒷발 역기구학 오차 2.214mm. 허용 2mm |
| 50°·200mm | 34.520초 | 왼쪽 뒷발 역기구학 오차 6.969mm. 허용 2mm |
| 바닥 마찰 0.5 / 0.8 | 두 조건 모두 97.625초 | 패드 접촉 끊김 25ms. 허용 20ms |
| 밸브 저항 5N·m | 97.860초 | 몸통 이동 10.160mm. 허용 10mm |
| 밸브 저항 15N·m | 97.600초 | 양패드 하중 연속 유지 95ms. 최소 200ms |

| 민감도 조건 | 첫 위반 시각 | 측정값과 판정 기준 |
|---|---:|---|
| 레버 0.10m | 97.600초 | 양패드 하중 연속 유지 135ms. 최소 200ms |
| 레버 0.14m | 108.605초 | 유지 구간 시작각 89.366°. 허용 89.75~90.25° |
| 패드 마찰 0.6 / 1.0 | 두 조건 모두 97.625초 | 패드 접촉 끊김 25ms. 허용 20ms |
| 손가락 구동력 상한 75N | 97.600초 | 양패드 하중 연속 유지 160ms. 최소 200ms |
| 손가락 슬라이드 마찰 손실 각 15N | 97.625초 | 패드 접촉 끊김 25ms. 허용 20ms |
| LIMS-EX 팔 축별 한계 | 97.605초 | 요청 토크가 적용한 모델 한계의 216.08%. 허용 100% |
| 물리 스텝 2.5ms | 97.6025초 | 조작 시작각 0.275°. 허용 ±0.25° |
| 접촉 솔버 마찰 임피던스 비 10 | 4.195초 | 추정 지지발 미끄러짐 10.0023mm. 허용 10mm |

기준 물리 스텝은 5ms다. 2.5ms 시험에서도 18축과 손가락의 명령은 5ms 동안 유지했다. 시계열에서 두 명령 갱신 사이의 값이 같은지 검사했다. RB1 축별 한계와 패드 마찰계수에는 모델 가정이 들어가며, 60° 결과는 이번에 고정한 보행 경로 생성 규칙에 대한 판정이다. [시간 간격 재판정](../evidence/stair-mission-20261121/conditions/physical-b/sensitivity_timestep_s_0_0025/through-task/raw-check.json) · [모델 한계 출처](valve-capacity.md)

물리 스텝이 5ms인 13개 조건은 계측 코드를 바꾸기 전후 두 프로세스에서 공통 원시 배열이 모두 일치했다. 2.5ms 물리 스텝에서 구동기까지 2.5ms마다 갱신한 예비 시행은 시작각이 0.373°였다. 조건표에는 5ms 동안 명령을 유지해 얻은 0.275°를 사용했다. [13조건 원시 대조와 제어 주기 영향](../evidence/stair-mission-20261121/conditions/physical-b/fixed-rate-crosscheck.json)

## 976.240초 연속 재생과 원시 기록 복원

두 CPU 프로세스가 각각 바닥에서 시작했다. 한 시행은 구조 하중 여섯 항목을 추가로 관측했다. 두 시행의 공통 34개 배열이 976,240스텝 전체에서 바이트 단위로 일치했다. 구조 하중을 포함한 원본은 1,960,454,163바이트이며 39조각으로 저장했다. 반복 시행은 758,489,872바이트다. 종료 시 밸브각은 90.000°였다. [두 시행 대조](../evidence/stair-corridor-20261121/local-repair/full/reproduction.json) · [입력과 원시 SHA-256](../evidence/stair-corridor-20261121/local-repair/full/inputs.json) · [전체 영상](../evidence/stair-corridor-20261121/local-repair/full/first/full-trip.mp4) · [첫 미끄러짐 영상](../evidence/stair-corridor-20261121/local-repair/full/first/first-slip.mp4) · [관절·발판 하중](../evidence/stair-corridor-20261121/local-repair/full/first/full-trip.heatmap.html)

아래 명령은 파일을 복원하고 두 시행을 각각 다시 판정한다. 세 번째로 복원하는 132,915스텝은 같은 보행 참조로 별도로 실행한 기준 조건의 작업 직후까지다. 끝난 위치와 저장된 물리 배열은 전 구간의 앞부분과 일치한다. [기준 조건의 독립 실행 대조](../evidence/stair-corridor-20261121/local-repair/full/nominal-prefix.json)

```powershell
cd "$HOME/go2-wall-reach-lab"
$suite = 'evidence/stair-corridor-20261121'
$full = "$suite/local-repair/full"
$walk = "$suite/local-repair/walk50.npz"
$verification = Join-Path $env:TEMP ('rbq-976-' + [guid]::NewGuid())
foreach ($run in 'first','repeat') {
    New-Item -ItemType Directory -Path "$verification/$run" | Out-Null
    Copy-Item -LiteralPath "$full/$run/scene.json","$full/$run/result.json" -Destination "$verification/$run"
}
$py = (Resolve-Path '.venv/Scripts/python.exe').Path
& $py tools/stair_trace_parts.py restore "$full/first/raw-parts" "$verification/first/mission.npz"
$repeatHash = (Get-Content -Raw "$full/repeat/result.json" | ConvertFrom-Json).raw_sha256
& $py tools/stair_trace_parts.py restore-without-loads "$verification/first/mission.npz" "$verification/repeat/mission.npz" --sha256 $repeatHash
foreach ($run in 'first','repeat') {
    & $py tools/verify_stair_replay.py "$verification/$run" "$verification/$run/verdict.json" --walk $walk --require-full-endpoint
}
& $py tools/stair_trace_parts.py restore-prefix "$verification/repeat/mission.npz" "$verification/nominal.npz" --samples 132915 --sha256 c39e16bb2061001ed6ed502fa8d17d56e7189fa3bf854868fe0c8a6b9a670db2
```

물리를 새로 계산할 때는 아래와 같이 기록에 사용한 소스 커밋과 같은 상대경로에 입력을 둔다. 13개 소스 파일의 해시와 RBQ 모델 자산 18개를 대조했다. 이 구성으로 바닥 출발부터 작업 다음 착지까지 별도 재생한 파일이 같은 환경의 원시 파일과 일치했다. [실행 환경과 자산](../evidence/stair-corridor-20261121/local-repair/inputs.json) · [고정 소스로 다시 실행한 결과](../evidence/stair-corridor-20261121/local-repair/source-replay.json)

```powershell
cd "$HOME/go2-wall-reach-lab"
$repo = (Get-Location).Path
$py = (Resolve-Path '.venv/Scripts/python.exe').Path
$relative = 'evidence/stair-corridor-20261121'
$spec = Get-Content -Raw "$relative/local-repair/inputs.json" | ConvertFrom-Json
$trial = Join-Path $env:TEMP ('rbq-976-sim-' + [guid]::NewGuid())
$src = Join-Path $trial 'source'
New-Item -ItemType Directory -Path $trial | Out-Null
$paths = @($spec.frozen_source_reproduction.input_data)
git archive --format=zip -o "$trial/source.zip" $spec.controller_source_commit -- @paths
Expand-Archive -LiteralPath "$trial/source.zip" -DestinationPath $src
New-Item -ItemType Directory -Path "$src/.assets" | Out-Null
Copy-Item -LiteralPath "$repo/.assets/rbq" -Destination "$src/.assets/rbq" -Recurse
foreach ($inputFile in "$relative/local-repair/walk50.npz","$relative/tracked/low-control/loaded.json.gz") {
    $dest = Join-Path $src $inputFile
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $repo $inputFile) -Destination $dest
}
$env:PYTHONPATH = "$src/src"
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$arguments = @('--walk',"$src/$relative/local-repair/walk50.npz",'--extent','full','--corridor-feedback',
    '--timestep','0.001','--motion-seconds','9','--jaw-load-bias-m','0.0005',
    '--stance-spring-gain-n-per-m','2500','--terminal-overtravel-deg','1.2',
    '--low-run',"$src/$relative/tracked/low-control/loaded.json.gz")
& $py "$src/tools/replay_stair_mission.py" "$trial/first" @arguments --capture-loads
& $py "$src/tools/replay_stair_mission.py" "$trial/repeat" @arguments
& $py "$src/tools/compare_stair_replays.py" "$trial/first" "$trial/repeat" "$trial/comparison.json" --require-same-source --allow-load-observer
```

## 재디딤 전 구간의 원시 기록과 재생

독립 검수에서 밸브의 부하 회전·유지·해제와 같은 단 재디딤을 확인했다. 두 전 구간 모두 보행과 통합 기준은 미달이었다. 원격 커밋과 일치하는 Git 객체 37개에서 복원한 두 기록을 각각 다시 판정했고, 최초 실패 시각부터 마지막 예정 착지까지 원본 판정과 일치했다. [독립 검수](../evidence/stair-corridor-20261121/replant/full/independent-review.json) · [Git 조각·복원 후 재판정](../evidence/stair-corridor-20261121/replant/full/archive-review.json)

두 CPU 프로세스가 바닥에서 시작해 각각 979,040개 시점을 기록했다. 공통 배열 34개가 전부 일치했다. 구조 하중 6개를 더 관측한 첫 원본은 1,817,100,836바이트이며 37조각으로 저장했다. 반복 원본은 670,097,686바이트다. 두 번째 파일을 복원할 때는 공통 배열의 압축 데이터를 그대로 복사한 뒤 ZIP 중앙 디렉터리를 갱신한다. 재압축으로 달라졌던 반복 파일의 SHA-256이 이 방식으로 독립 원본과 일치했다. 다음 명령은 두 파일을 임시 디렉터리에 복원하고 보행 입력 해시와 마지막 착지까지 다시 판정한다. [입력·기록 해시](../evidence/stair-corridor-20261121/replant/full/inputs.json) · [공통 배열 대조](../evidence/stair-corridor-20261121/replant/full/reproduction.json)

```powershell
cd "$HOME/go2-wall-reach-lab"
$py = (Resolve-Path '.venv/Scripts/python.exe').Path
$suite = 'evidence/stair-corridor-20261121'
$full = "$suite/replant/full"
$walk = "$suite/local-repair/walk50.npz"
$trial = Join-Path $env:TEMP ('rbq-replant-' + [guid]::NewGuid())
$info = Get-Content -Raw "$full/inputs.json" | ConvertFrom-Json
foreach ($run in 'first','repeat') {
    New-Item -ItemType Directory -Path "$trial/$run" | Out-Null
    Copy-Item -LiteralPath "$full/$run/scene.json","$full/$run/result.json" -Destination "$trial/$run"
}
& $py tools/stair_trace_parts.py restore "$full/first/raw-parts" "$trial/first/mission.npz"
& $py tools/stair_trace_parts.py restore-without-loads "$trial/first/mission.npz" "$trial/repeat/mission.npz" --sha256 $info.raw[1].sha256
foreach ($run in 'first','repeat') {
    & $py tools/verify_stair_replay.py "$trial/$run" "$trial/$run/verdict.json" --walk $walk --require-full-endpoint
}
```

물리를 새로 계산하려면 같은 터미널에서 다음 명령을 실행한다. Git에서 시행 당시의 코드와 입력을 꺼내고, 저장소에 이미 준비된 RBQ 자산을 사용한다. 두 번째 실행도 바닥 초기 자세에서 시작한다.

```powershell
$repo = (Get-Location).Path
$tree = Join-Path $trial 'source'
$sourceList = (Get-Content -Raw "$suite/local-repair/inputs.json" | ConvertFrom-Json).frozen_source_reproduction.input_data
$paths = @($sourceList) + @($walk,"$suite/tracked/low-control/loaded.json.gz")
git archive --format=zip -o "$trial/source.zip" $info.controller_commit -- @paths
Expand-Archive -LiteralPath "$trial/source.zip" -DestinationPath $tree
New-Item -ItemType Directory -Path "$tree/.assets" | Out-Null
Copy-Item -LiteralPath "$repo/.assets/rbq" -Destination "$tree/.assets/rbq" -Recurse
$env:PYTHONPATH = "$tree/src"
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
Push-Location $tree
try {
    $settings = @('--extent','full','--walk',$walk,'--corridor-feedback','--control-profile','rl-mid-replant-k3075-v1',
        '--low-run',"$suite/tracked/low-control/loaded.json.gz")
    & $py tools/replay_stair_mission.py "$trial/sim-first" @settings --capture-loads
    & $py tools/replay_stair_mission.py "$trial/sim-repeat" @settings
    & $py tools/compare_stair_replays.py "$trial/sim-first" "$trial/sim-repeat" "$trial/physics-comparison.json" --require-same-source --allow-load-observer
    foreach ($i in 0,1) {
        $folder = @('sim-first','sim-repeat')[$i]
        if ((Get-FileHash -LiteralPath "$trial/$folder/mission.npz").Hash -ne $info.raw[$i].sha256) { throw "원시 기록 불일치 $folder" }
    }
} finally {
    Pop-Location
}
```

## 768초 기준 임무 재현

보행 계획 원본 커밋은 `25117a003b785a0358ae77eaf2dd61628edf6d84`이고, 정격 조건표의 물리 재생 원본은 `874f3bee820322502f0025180c4f279308d353c0`이다. 보행 계획 10개 파일과 재생 11개 파일의 SHA-256을 시행 매니페스트와 대조했다. 아래 명령은 각 커밋의 체크아웃에서 실행한다. 먼저 자산을 준비하고 계획 원본에서 참조 경로를 생성한 뒤, 물리 재생 원본에서 같은 경로를 읽는다. 출력 경로는 새 이름을 사용한다. [체크아웃과 조건별 정확한 인자](../evidence/stair-mission-20261121/conditions/physical-b/inputs.json)

```powershell
cd "$HOME/go2-wall-reach-lab"
.venv/Scripts/python tools/fetch_assets.py
.venv/Scripts/python tools/stair_mission_route.py "$env:TEMP/rbq-walk50.npz"
.venv/Scripts/python tools/replay_stair_mission.py "$env:TEMP/rbq-rated-nominal" --walk "$env:TEMP/rbq-walk50.npz" --case nominal --extent through-task --capture-loads
.venv/Scripts/python tools/verify_stair_replay.py "$env:TEMP/rbq-rated-nominal" "$env:TEMP/rbq-rated-check.json"
```

전체 왕복 재생은 별도 체크아웃에 고정한다. `e835c8b`에는 누적 시간 반올림으로 빠지던 마지막 5ms 물리 스텝이 반영돼 있다. 다음 명령은 같은 계획·저부하 원본으로 두 프로세스를 차례로 시작하고 마지막 예정 착지까지 대조한다. 계획 파일을 생성한 커밋은 `25117a0`이다.

```powershell
cd "$HOME/go2-wall-reach-lab"
$repo = (Get-Location).Path
$suite = Join-Path $repo 'evidence/stair-mission-20261121'
$trial = Join-Path $env:TEMP ('rbq-full-' + [guid]::NewGuid())
$source = Join-Path $trial 'source'
$python = Join-Path $repo '.venv/Scripts/python.exe'
New-Item -ItemType Directory -Path $trial | Out-Null
git worktree add --detach $source e835c8b8406ee4e76085d94a98ac620e2c22ac7b
Copy-Item -LiteralPath "$repo/.assets" -Destination "$source/.assets" -Recurse
$env:PYTHONPATH = "$source/src"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
foreach ($run in 'first','repeat') {
    & $python "$source/tools/replay_stair_mission.py" "$trial/$run" --walk "$suite/full/walk50.npz" --case nominal --extent full --capture-loads --low-run "$suite/conditions/physical-b/low-control/frozen-grip-loaded.json.gz"
    if ($LASTEXITCODE -ne 0) { throw "재생 실패 $run" }
}
$env:PYTHONPATH = "$repo/src"
foreach ($run in 'first','repeat') {
    & $python tools/verify_stair_replay.py "$trial/$run" "$trial/$run-check.json" --require-full-endpoint
    if ($LASTEXITCODE -ne 0) { throw "검증 실패 $run" }
}
& $python tools/compare_stair_replays.py "$trial/first" "$trial/repeat" "$trial/comparison.json"
```

시뮬레이션을 다시 돌리지 않고 판정을 확인할 때는 공개 기록 7조각에서 원시 파일을 복원한다. 각 조각과 합친 파일의 SHA-256을 복원 중에 검사한다. 독립 재판정기는 두 원본 파일을 별개로 생성한 위의 재생 결과를 비교할 때도 쓸 수 있다. [실행 인자와 두 프로세스의 입력·출력 해시](../evidence/stair-mission-20261121/full/inputs.json) · [원시 조각별 해시](../evidence/stair-mission-20261121/full/raw-parts/parts.json)

```powershell
cd "$HOME/go2-wall-reach-lab"
$suite = 'evidence/stair-mission-20261121'
$restored = Join-Path $env:TEMP ('rbq-record-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $restored | Out-Null
Copy-Item -LiteralPath "$suite/full/first/scene.json","$suite/full/first/result.json" -Destination $restored
.venv/Scripts/python tools/stair_trace_parts.py restore "$suite/full/raw-parts" "$restored/mission.npz"
.venv/Scripts/python tools/verify_stair_replay.py $restored "$restored/verdict.json" --require-full-endpoint
.venv/Scripts/python tools/verify_stair_artifacts.py $restored "$suite/full/first/full-trip.mp4" "$restored/video-check.json" --speed 24
```

닫힌 밸브 보행 진단 두 묶음은 커밋 `e1c37833738ee8bf1ee34da2b19f7016ff3bd305`에 소스와 입력이 함께 있다. 바닥 출발 우회 시험은 BLAS·OpenMP 스레드 환경변수를 비워 두고, 뒷발 순서 변경 시험은 두 값을 모두 1로 고정했다. 각 묶음의 매니페스트에 기록한 환경에서 원시 결과를 비교한다. [바닥 우회 실행법](../evidence/stair-mission-20261121/bypass/floor-route/README.md) · [뒷발 순서 변경 실행법](../evidence/stair-mission-20261121/bypass/swing-order/inputs.json)

### 1ms 지지발 보정 제어기 재생

이 비교는 `0db7b29e7130fbf7ad416dfb1986a32bb4f71c75`의 제어기와 공개된 `walk50.npz`를 사용한다. `--extra-settle-seconds 0`은 기본 정착 2초, `--extra-settle-seconds 2`는 총 정착 4초에 해당한다. 아래 명령은 새 임시 경로에 해당 버전의 모델 입력과 코드를 풀고 세 시행을 각각 새 프로세스로 시작한다. 재판정 명령은 현재 체크아웃에서 실행한다.

```powershell
cd "$HOME/go2-wall-reach-lab"
$repo = (Get-Location).Path
$suite = Join-Path $repo 'evidence/stair-mission-20261121'
$trial = Join-Path $env:TEMP ('rbq-stance-' + [guid]::NewGuid())
$src = Join-Path $trial 'source'
$python = Join-Path $repo '.venv/Scripts/python.exe'
$files = @('src', 'tools', 'configs', 'pyproject.toml', 'evidence/stair-mission-20261121/protocol.json', 'evidence/t1-stair-ascent-20261120/target-round-trip-geometry.json', 'evidence/t1-stair-ascent-20261120/target-round-trip-plan.json', 'evidence/t1-stair-posture-20261120/front-pair.json', 'evidence/t3-valve-20261120/fixed-body-candidates.json')
New-Item -ItemType Directory -Path $trial | Out-Null
git archive --format=zip -o "$trial/source.zip" 0db7b29e7130fbf7ad416dfb1986a32bb4f71c75 -- @files
Expand-Archive -LiteralPath "$trial/source.zip" -DestinationPath $src
New-Item -ItemType Directory -Path "$src/.assets" | Out-Null
Copy-Item -LiteralPath "$repo/.assets/rbq" -Destination "$src/.assets/rbq" -Recurse
$env:PYTHONPATH = "$src/src"
$env:OPENBLAS_NUM_THREADS = $env:OMP_NUM_THREADS = $env:MKL_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$control = @('--walk', "$suite/full/walk50.npz", '--case', 'nominal', '--timestep', '0.001', '--motion-seconds', '9', '--jaw-load-bias-m', '0.0005', '--stance-spring-gain-n-per-m', '2500')
foreach ($extra in 0,2) {
    & $python "$src/tools/replay_stair_mission.py" "$trial/settle-$($extra + 2)s" @control --extent through-task --extra-settle-seconds $extra
    if ($LASTEXITCODE -ne 0) { throw "정착 비교 재생 오류 $extra" }
}
& $python "$src/tools/replay_stair_mission.py" "$trial/full" @control --extent full --extra-settle-seconds 2 --low-run "$suite/conditions/physical-b/low-control/frozen-grip-loaded.json.gz"
if ($LASTEXITCODE -ne 0) { throw '전체 재생 오류' }
$env:PYTHONPATH = "$repo/src"
& $python tools/verify_stair_replay.py "$trial/full" "$trial/full-check.json" --require-full-endpoint
```

기록만 확인하려면 아래 명령으로 11조각에서 513,334,086바이트 원본을 복원한다. 처음 131,750스텝은 별도로 실행한 4초 정착 시험과 바이트 단위로 일치한다. 재판정 출력의 `verdict comparison: True`는 저장한 실패 판정을 재현했다는 뜻이다. [원본 조각 해시](../evidence/stair-mission-20261121/stance-feedback/full/raw-parts/parts.json) · [입력·두 파일 대조](../evidence/stair-mission-20261121/stance-feedback/inputs.json)

```powershell
cd "$HOME/go2-wall-reach-lab"
$suite = 'evidence/stair-mission-20261121/stance-feedback'
$restored = Join-Path $env:TEMP ('rbq-stance-record-' + [guid]::NewGuid())
$full = Join-Path $restored 'full'
$prefix = Join-Path $restored 'settle-4s'
New-Item -ItemType Directory -Path $full,$prefix -Force | Out-Null
$env:PYTHONPATH = (Resolve-Path 'src').Path
Copy-Item -LiteralPath "$suite/full/scene.json","$suite/full/result.json" -Destination $full
Copy-Item -LiteralPath "$suite/settle-4s/scene.json","$suite/settle-4s/result.json" -Destination $prefix
.venv/Scripts/python tools/stair_trace_parts.py restore "$suite/full/raw-parts" "$full/mission.npz"
.venv/Scripts/python tools/stair_trace_parts.py restore-prefix "$full/mission.npz" "$prefix/mission.npz" --samples 131750 --sha256 801f7bc4e263367b5931ba3289c07e8b64f4ef4d3221ba29f585b1143ee9e77a
.venv/Scripts/python tools/verify_stair_replay.py $full "$full/verdict.json" --require-full-endpoint
.venv/Scripts/python tools/verify_stair_replay.py $prefix "$prefix/verdict.json"
```
