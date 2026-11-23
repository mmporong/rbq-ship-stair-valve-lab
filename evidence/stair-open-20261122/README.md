# 50° 계단에서 밸브를 연 채 정지

RBQ-10+RB1이 MuJoCo 4.0에서 바닥을 출발해 단 높이 190mm의 50° 계단을 올랐다. 앞발은 5단, 뒷발은 3단에 둔 상태로 닫힌 밸브를 열었다. 10N·m 수동 힌지의 0.12m 레버를 두 패드로 돌려 열린 각도를 5초간 유지했다. 그 뒤 손을 놓고 팔을 회수한 채 같은 발판에 5초 더 서 있었다. 같은 기준 조건을 새 프로세스 두 개에서 실행했고 모두 통과했다. [127.75초 전체 임무, 4배속](nominal/mission.mp4) · [파지·열기·놓기 확대, 1.5배속](nominal/valve.mp4)

## 물리 기록

| 항목 | 측정값 | 판정 기준 |
|---|---:|---|
| 예정 착지 | 18/18회 지지 | 도중 착지와 작업 발판 지지 |
| 시작과 끝의 레버 방향 | 월드 XZ에서 180° → 270.00006° | 설치된 밸브의 닫힘 → 열림, 끝 오차 ≤0.25° |
| 회전 중 패드 토크 | 5~85°의 16개 구간에서 9.9996~10.0004N·m | 각 구간에서 저항의 9.98N·m 이상 전달 |
| 손으로 열린 각도 유지 | 90.0480~90.0489°, 5.000초 | 90° ±0.25°로 5초, 허용 패드 이탈 20ms |
| 회전·유지 중 패드 이탈 | 0ms | 양쪽 패드 하중 조건 |
| 조작 중 최대 몸통 이동 / 발 중심 이동 | 5.025mm / 4.440mm | ≤10mm / ≤5mm |
| 조작 중 최소 발 지지력 / 최대 마찰 사용률 | 29.83N / 0.496 | ≥1N / ≤0.80 |
| 전체 구간 최대 지지발 미끄러짐 | 9.769mm, 왼쪽 앞발 | 접촉면 미끄럼 누적 변위 ≤10mm |
| 팔 회수 뒤 발판 정지 | 5.000초, 발별 최소 95.81N | 네 발이 지정 발판에 계속 하중 전달 |
| 정지 중 손과 레버 | 패드 최소 거리 598.44mm, 접촉력 0N | 손가락 개방, 패드 분리 |
| 정지 중 최대 몸통 선속도 / 각속도 | 0.299mm/s / 0.000566rad/s | 정지 판정 통과 |
| 낙상 / 구동기 포화 / 접촉력 계산 실패·제약 완화 | 모두 0회 | 모두 0회 |

각도는 밸브의 이동량이며, 월드 방향은 손잡이 위치와 힌지축으로 복원했다. 기록은 1ms 간격의 127,750스텝이며 모터 명령 주기는 5ms다. 하중을 추가 계측한 실행과 일반 실행에서 공통 상태·센서 34개 배열의 모든 표본이 바이트 단위로 일치했다. 원시 기록을 다시 읽어 보행·조작·정지 판정을 재현했다. 두 실행은 같은 조건의 재현성을 확인한다. [127.681초에 촬영된 정지 자세](nominal/station.png) · [첫 실행 재판정](nominal/raw-check.json) · [반복 실행 재판정](repeat/raw-check.json) · [두 실행의 대조](same-source.json)

결과 파일에서 `station.verified`는 지정 발판의 반력, 몸통 속도와 자세, 밸브 각도 및 손의 분리를 검사한다. 공용 기록 항목 `metrics.feet_down_at_end`는 네 발바닥이 세계 높이 0의 평지에서 10mm 이내인지 검사하므로 이 계단 끝 자세에서는 거짓이다. 임무 종합 판정은 `outcome`에 기록했다. [시행 결과](nominal/result.json)

## 발 이동을 줄인 제어 변경

| 기준 조건에서의 제어 | 조작 중 최대 발 이동 | 조작 중 최소 발 하중 | 보행 / 조작 / 정지 |
|---|---:|---:|---|
| 파지 시점의 다리 관절각으로 네 발 지지 | 7.972mm | 2.09N | 통과 / 미달 / 통과 |
| 뒷발 두 개를 같은 3단에서 40mm 뒤로 옮기고 몸통 목표·이득 조정 | 6.824mm | 26.78N | 통과 / 미달 / 통과 |
| 같은 발 위치에서 회전 시간과 끝단 힘 조정 | 4.440mm | 29.83N | 통과 / 통과 / 통과 |

최종 제어는 7.5초의 회전 궤적을 사용한다. 궤적 종료 2초 뒤인 107.350초에 실제 밸브 이동각은 90.0585°였다. 이 시점부터 팔·다리 모터의 토크를 계산할 때 사용하는 접선 하중 명령을 0.8초에 걸쳐 20%로 줄였다. 힌지의 물리 저항은 10N·m로 유지했고, 두 패드는 손을 놓는 단계까지 닫혔다. 전체 구간의 지지발 미끄러짐은 세 설정에서 8.646, 9.488, 9.769mm였다. 이 변화량은 각 설정 전체를 비교한 값이다. [설정별 코드 버전·해시·원시 측정](controller-comparison.json)

## 장착판 응력

매 스텝에서 RB1 장착부의 힘·모멘트를 몸통 좌표계로 변환하고, 가정한 볼트선 단면에서 굽힘과 비틀림을 계산했다. 80×6mm 직사각 단면, 외팔 길이 150mm, 6061-T6 항복강도 276MPa는 이전 Go2+D1 연구의 가정이다. 실제 RBQ 장착판의 치수와 재질은 미확인이다. 전체 추정 등가응력 상한은 **계단 접근 35.205초에 76.775MPa**, 밸브 회전 중에는 **101.550초에 41.535MPa**였다. [무지개색 히트맵](stress/mount-stress.html) · [단면과 두께별 비교](stress/mount-stress.png) · [최대 응력 시점의 자세](stress/mount-stress-pose.png) · [1ms별 계산값](stress/mount-stress.csv)

단면 그림의 색은 피크 시각의 절대 굽힘응력 0~73.68MPa를 나타낸다. 영상의 장착점 색과 고정 시간축 범례는 등가응력 상한을 0~276MPa로 표시한다. HTML의 확대 시간축에서는 같은 기록의 작은 변화도 확인할 수 있다. [계산 성분·범례·원시 파일 해시](stress/mount-stress.json) · [전체 CSV 재계산](stress/raw-check.json)

## 60° 계단과 RL 정책

같은 190mm 높이의 60° 선박형 계단은 접근 계획 13.600초에 몸통과 3단이 최대 0.684mm 겹쳤다. 허용치는 0.100mm였다. 이 조건은 작업 자세에 도달하기 전에 계획이 중단돼 밸브 조작 결과가 없다. [60° 접촉 형상과 재현 기록](ship60-approach.json)

위 50° 개방 임무에는 계획 궤적과 힘 제어기를 썼다. 동결한 RL 정책은 별도 실험에서 같은 작업 발판 도달을 평가했다. 원래 RL Triage Agent 저장소는 텔레메트리를 읽었고, 해당 RBQ 과제가 등록 범위 밖이어서 개입을 보류했다. RBQ용 호스트 어댑터에서는 로컬 Gemma가 개발 조건 세 개를 보고 보행 위상 설정을 골랐다. 별도로 보류한 여섯 시작 조건에서 작업 발판에 0.5초간 도달한 횟수는 무개입·고정 감속·모델 선택 모두 0/6이었다. 모델 선택은 무개입보다 평균 최고 지지 단을 0.67단 높였고 최대 피치는 6.28° 커졌다. 몸통 충돌은 두 경우 모두 4/6이었다. [원래 저장소 실행 결과](triage/README.md) · [사전 고정 설계와 18회 대조](triage/rbq-host-ab-v1/README.md)

## 재현

저장소의 MuJoCo 환경과 자산을 준비한 뒤 PowerShell에서 실행한다. 두 실행은 새로운 출력 폴더를 사용한다.

```powershell
Set-Location (Join-Path $HOME 'go2-wall-reach-lab')
$py = '.venv/Scripts/python.exe'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$walk = 'evidence/stair-mission-20261121/full/walk50.npz'
$out = '.omx/reproduce-open-station'
& $py tools/replay_stair_mission.py "$out/nominal" --walk $walk --extent open-at-station --capture-loads
& $py tools/replay_stair_mission.py "$out/repeat" --walk $walk --extent open-at-station
& $py tools/verify_stair_replay.py "$out/nominal" "$out/nominal/raw-check.json" --walk $walk --require-open-station
& $py tools/verify_stair_replay.py "$out/repeat" "$out/repeat/raw-check.json" --walk $walk --require-open-station
& $py tools/compare_stair_replays.py "$out/nominal" "$out/repeat" "$out/same-source.json" --require-same-source --allow-load-observer
& $py tools/stair_stress_heatmap.py "$out/nominal/mission.npz" "$out/stress/mount-stress.html"
& $py tools/render_stair_mission.py "$out/nominal" "$out/nominal/mission.mp4" --follow-base --speed 4
& $py tools/render_stair_mission.py "$out/nominal" "$out/nominal/valve.mp4" --valve-view --start 95 --end 127.7 --speed 1.5
```

저장된 첫 원시 파일은 여섯 조각으로 복원할 수 있다. 거기서 추가 하중 채널 여섯 개를 제거하면 별도로 실행한 반복 기록과 같은 SHA-256이 나온다. 위와 같은 PowerShell에서 다음 명령으로 두 파일을 복원하고 재판정한다. [첫 기록의 조각별 해시](nominal/raw-parts/parts.json) · [반복 기록의 해시](repeat/result.json)

```powershell
$e = 'evidence/stair-open-20261122'
$repeatSha = (Get-Content "$e/repeat/result.json" -Raw | ConvertFrom-Json).raw_sha256
& $py tools/stair_trace_parts.py restore "$e/nominal/raw-parts" "$e/nominal/mission.npz"
& $py tools/stair_trace_parts.py restore-without-loads "$e/nominal/mission.npz" "$e/repeat/mission.npz" --sha256 $repeatSha
& $py tools/verify_stair_replay.py "$e/nominal" "$e/nominal/recheck.json" --walk $walk --require-open-station
& $py tools/verify_stair_replay.py "$e/repeat" "$e/repeat/recheck.json" --walk $walk --require-open-station
```
