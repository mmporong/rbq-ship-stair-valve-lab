# 작업 자세에서 10N·m 밸브 잠금·유지·해제

50°·단 높이 190mm 계단의 앞발 5단·뒷발 3단 자세에서 두 패드로 수동 밸브를 돌렸다. 1ms 물리 스텝에서 두 시행 모두 정격 기준을 통과했다. 두 프로세스가 쓴 111,985,270바이트의 원시 JSON은 SHA-256까지 같았다. 움직임과 접촉은 MuJoCo 4.0에서 측정했다. [시행별 입력과 판정](results.json)

| 물리 스텝 | 독립 시행 | 양패드 최대 이탈 | 몸통 / 발 중심 최대 이동 | 5초 유지 중 각도 | 마지막 각도 | 정격 판정 |
|---|---:|---:|---:|---:|---:|---|
| 1ms | 2회 | 0ms | 4.952 / 4.981mm | 90.0662~90.0684° | 90.0013° | 두 번 통과 |
| 5ms | 1회 | 130ms | 10.933 / 5.286mm | 90.0644~90.0646° | 35.4826° | 실패 |

세 시행 모두 모터 명령 주기 5ms, 회전 명령 9초, 손가락 목표 접선 보정 0.5mm, 레버 0.12m, 힌지 저항 10N·m를 사용했다. 회전 시작 시 지지발 중심의 평면 위치를 저장하고, 유지가 끝날 때까지 이 위치에서 벗어난 거리에 2,500N/m를 곱했다. 발의 자코비안을 통해 다리 모터 토크를 보정했다. 1ms에서 지지발의 최대 보정력은 12.150N, 관절에 더한 최대 토크는 2.342N·m였다. 발 중심과 접촉 속도는 물리 기록에서 읽었다. [시험 코드](run_torque_stance_spring.py)

1ms에서는 5~85°의 부하 구간 16개를 모두 채웠다. 네 발 중 가장 작은 수직 하중은 44.126N, 최대 마찰 사용률은 0.4481이었다. 패드가 완전히 떨어진 뒤 5.767초까지 관찰했다. 전복, 토크 포화, 힘 분배 실패·완화, 외력과 출발 뒤 상태 재설정은 모두 0이었다. 두 시행의 원시 SHA-256은 `12902a184d05acdc035a4146346d5d038dc48a9e9042a59a4491c7d9a7a72aaa`다. 압축을 풀면 두 시행의 원본을 같은 바이트로 복원할 수 있다. [1ms 원시 기록](torque-k2500-1ms.json.gz) · [5ms 원시 기록](torque-k2500-5ms.json.gz)

앞선 감쇠 보정 탐색에서는 지지발 속도를 바꿔 미끄러짐 누적값까지 달라졌다. 위 표에는 모터 토크만 보정한 재생의 값을 실었다. 이 시험의 출발점은 계단 위 작업 자세다. 바닥에서 출발한 8.75초 회전 명령의 시행은 별도로 보관했다. [바닥 출발 시행](../floor-grip-refinement/raw-check.json)

## 재현

고정 커밋 `b704ccf5f19a2ab87efd102fbb7a263f34221f7d`의 시험에 필요한 파일은 소스 압축본에 들어 있다. 저장소에 준비한 자산을 임시 폴더에 복사한 뒤 실행한다. 기록에는 사용한 소스·모터 보정 함수의 해시와 원시 시계열이 들어간다. [고정 소스](source-b704ccf.zip)

```powershell
cd "$HOME/go2-wall-reach-lab"
$kit = 'evidence/stair-mission-20261121/grip-refinement'
$trial = Join-Path $env:TEMP ('rbq-valve-' + [guid]::NewGuid())
Expand-Archive -LiteralPath "$kit/source-b704ccf.zip" -DestinationPath $trial
Copy-Item -LiteralPath '.assets' -Destination (Join-Path $trial '.assets') -Recurse
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
foreach ($run in 'first','repeat') {
    .venv/Scripts/python "$kit/run_torque_stance_spring.py" $trial "$trial/$run.json" --spring-gain 2500 --timestep 0.001 --motion-seconds 9
}
Get-FileHash -Algorithm SHA256 "$trial/first.json","$trial/repeat.json"
```
