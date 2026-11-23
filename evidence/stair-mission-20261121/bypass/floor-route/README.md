# 닫힌 밸브 옆을 지나는 50° 바닥 출발 보행

밸브 힌지를 시작할 때 한 번 90°로 놓고 바닥에서 출발했다. 보행 제어 이득은 1.5이고 외력은 0이다. 표의 시각은 물리 스텝이 끝난 뒤의 관측 시각이다. [입력·원시 해시·전체 정밀도 측정값](manifest.json)

| 보행 입력 | 첫 위반과 도달 범위 |
|---|---|
| 기존 경로 | 100.307초에 몸통과 레버가 90.62N으로 충돌했다. |
| 몸통 높이·기울기·위치 조정 | 107.400초까지 최대 추종 오차 18.87mm, 지지발 미끄러짐 6.77mm, 요청 관절 토크는 한계의 70.87%였다. 예상 밖 접촉, 전복, 포화, 힘 분배 실패는 0회였다. 원래 보행에 다시 합류하면서 107.412초에 추종 오차가 32.50mm로 30mm 한계를 넘었다. |
| 111.82초까지 연장한 경로 | 107.270초에 추종 오차를 처음 넘겼고 107.981초에 힘 분배 계산이 실패했다. 108.675초에 전복된 뒤 108.989초에 오른쪽 뒷다리 종아리와 3단이 682.29N으로 부딪혔다. 최대 접촉력은 109.073초에 같은 다리 허벅지와 오른쪽 측판 사이의 7,770.10N이었다. |

## 기록 형식

`verified-prefix-control.npz`와 `extension-attempt-control.npz`에는 시각·몸통·발·팔 목표·지지 상태·토크 입력이 들어 있다. 각 `*-trace.npz`는 전체 1ms 표본에서 해석에 사용한 채널을 담는다. 명령 시각은 `time`, 그 명령을 적용한 뒤 관측한 시각은 `physics_time_s`다. 각 `*-result.json`에는 모든 채널을 포함한 원본의 SHA-256이 있다. 아래 재생 명령으로 전체 원본도 생성할 수 있다.

`extra`의 7열은 밸브 각도, 각속도, 위·아래 패드 법선력, 닫힘 방향 패드 토크, 예상 밖 레버 접촉 최대값, 전체 레버 접촉 최대값 순이다. `attempt`는 힘 분배 해를 구하지 못했으면 −1, 명목 해이면 0, 완화된 해이면 양수다. 구동기 명령·기록 주기는 1ms, 힘 분배 계산 주기는 10ms다.

허용 접촉은 네 발과 지형 사이다. 손가락의 +100N 명령은 열림 방향이다. 위·아래 손가락의 슬라이드 변위와 레버 접촉력은 물리 스텝마다 기록했다.

## 재현

저장소 루트에서 실행한다. 출력은 매번 `$env:TEMP` 아래 새 폴더를 사용한다. 기존 출력 폴더가 있으면 도구는 오류로 끝나며 덮어쓰지 않는다.

```powershell
cd "$HOME/go2-wall-reach-lab"
$env:OPENBLAS_NUM_THREADS = $null
$env:OMP_NUM_THREADS = $null
$prefixOut = Join-Path $env:TEMP ("go2-floor-prefix-" + [guid]::NewGuid())
.\.venv\Scripts\python.exe tools\replay_valve_bypass.py `
  evidence\stair-mission-20261121\bypass\swing-order\walk50.npz `
  evidence\stair-mission-20261121\bypass\floor-route\verified-prefix-control.npz `
  $prefixOut --end-s 107.5 `
  --manifest evidence\stair-mission-20261121\bypass\floor-route\manifest.json

$extensionOut = Join-Path $env:TEMP ("go2-floor-extension-" + [guid]::NewGuid())
.\.venv\Scripts\python.exe tools\replay_valve_bypass.py `
  evidence\stair-mission-20261121\bypass\swing-order\walk50.npz `
  evidence\stair-mission-20261121\bypass\floor-route\extension-attempt-control.npz `
  $extensionOut --accepted-only --end-s 111.82 `
  --manifest evidence\stair-mission-20261121\bypass\floor-route\manifest.json
```

단일 스레드 감사 재생은 baseline에서 접촉 시각과 실패 종류가 같았고 힘 차이는 0.006424 N이었다. 연장 입력도 실패 종류는 같았지만 전복 뒤 최대 접촉력은 1277.850318 N 달랐다. 이 차이는 `manifest.json`에 남겼으며 기본 재생 수치를 공개 대표값으로 사용한다.
