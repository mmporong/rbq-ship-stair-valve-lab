# 왼쪽 뒷발을 다시 디딘 뒤의 밸브 조작

원래 50°·190mm 보행 경로에서 바닥 출발부터 밸브 해제 뒤 첫 착지까지 보행과 조작이 모두 통과했다. 132.550초 동안 물리 상태를 1ms마다 기록하고 모터 명령을 5ms마다 갱신했다. 최초 두 프로세스와 고정된 제품 코드가 같은 원시 파일을 만들었다. [고정 소스 재판정](frozen-verification.json) · [실행 설정](frozen-scene.json)

78.8초에 오른쪽 뒷발이 착지한 뒤 왼쪽 뒷발의 하중을 빼고 같은 발판에 다시 디뎠다. 왼쪽 뒷발은 51.470mm 올라갔고 79.205~81.221초에는 해당 발판의 하중이 1N 미만이었다. 81.6초 재착지 때 하중은 46.600N, 목표 대비 발 중심의 수평 위치 오차는 0.815mm였다. 제품 판정은 관측 구간에서 원래 착지 19회와 이 보조 착지 1회를 검사한다. 전체 일정은 원래 156회와 보조 착지 1회다. [재디딤 하중과 실제 위치](frozen-result.json)

## 보정 이득별 측정

네 시험은 같은 원래 경로를 사용했다. 팔의 회전 말기 목표는 1.3° 더 진행하고 손가락 명령에 0.5mm를 더했다. 왼쪽 뒷발의 접촉점 누적 변위에 따른 복원력은 유지 종료 뒤 116.65~129.4초에 그 다리의 모터 토크로 적용했다. 손끝 가정 하중과 시뮬레이터 외력은 0이었다.

| 뒷발 보정 이득 | 밸브 | 보행 | 누적 미끄러짐 FL / FR / RL / RR (mm) | 첫 위반 |
|---:|---|---|---|---|
| 600N/m | 통과 | 미달 | 9.191 / 8.038 / 11.282 / 9.667 | 122.840초 왼쪽 뒷발 |
| 3,000N/m | 통과 | 미달 | 9.186 / 9.176 / 10.001 / 8.970 | 127.765초 왼쪽 뒷발 |
| 3,075N/m | 통과 | 통과 | 9.174 / 9.579 / 9.968 / 8.928 | 관측 구간 내 없음 |
| 3,300N/m | 통과 | 미달 | 9.172 / 10.451 / 9.944 / 8.794 | 130.377초 오른쪽 앞발 |

3,075N/m에서 조작 중 최대 발 중심 이동은 4.945mm였다. 밸브는 90.068°까지 회전했고 패드 이탈 없이 5초 동안 유지됐다. 전복, 포화, 지지력 제약 완화와 계산 실패는 모두 0회였다. 그 뒤 레버 옆을 지나 상부를 회전하고 바닥으로 돌아오는 부분은 별도 전 구간 시험에서 판정한다. [네 시험의 측정값과 원시 해시](gain-study.json) · [다리 모터에 적용한 보정량](k3075-control.json) · [첫 시행](k3075-result.json) · [독립 반복](k3075-repeat-result.json)

## 원시 기록 복원과 재생

세 조각을 복원한 파일은 106,378,299바이트다. SHA-256은 `b39bd2e9a55501a3ee225595c89bd6ba4e506443fa04ed8717d687beffb11d73`이다. 원본을 다시 계산할 때는 커밋 `2c91d83ff3a222bc2919c3156b93920ebcff5ec4`에 고정한 제어 코드와 원래 보행 입력을 사용한다. 이 시험에서 레버 옆 구간의 추가 보행 피드백은 꺼져 있었다. 현재 판정기는 실패 사건을 시간순으로 보고한다. 보관한 장면의 소스 해시까지 재현하려면 해당 커밋에서 아래 재생 명령을 실행한다. [조각별 해시](raw-parts/parts.json) · [실행별 지문과 소스 이력](manifest.json) · [고정 소스 복원 예시](../../../../docs/stair-contact-mission.md#재디딤-전-구간의-원시-기록과-재생)

```powershell
cd "$HOME/go2-wall-reach-lab"
$py = (Resolve-Path '.venv/Scripts/python.exe').Path
$evidence = 'evidence/stair-corridor-20261121/replant/short-control'
$walk = 'evidence/stair-mission-20261121/full/walk50.npz'
$restored = Join-Path $env:TEMP ('rbq-replant-record-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $restored | Out-Null
Copy-Item -LiteralPath "$evidence/frozen-scene.json" -Destination "$restored/scene.json"
Copy-Item -LiteralPath "$evidence/frozen-result.json" -Destination "$restored/result.json"
& $py tools/stair_trace_parts.py restore "$evidence/raw-parts" "$restored/mission.npz"
& $py tools/verify_stair_replay.py $restored "$restored/verdict.json" --walk $walk
$trial = Join-Path $env:TEMP ('rbq-replant-sim-' + [guid]::NewGuid())
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='-1'
& $py tools/replay_stair_mission.py $trial --extent through-task --walk $walk --control-profile rl-mid-replant-k3075-v1
```
