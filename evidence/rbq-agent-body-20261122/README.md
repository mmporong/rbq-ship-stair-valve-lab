# 50° 선박형 계단에서의 밸브 개방과 RL 에이전트 평가

## 시험 조건

RBQ-10과 RB1이 높이 190mm의 계단을 오른다. 앞발은 5단, 뒷발은 3단에 있어야 한다. 네 발 각각이 해당 단에 1N 이상의 수직 하중을 연속 0.5초 전달한 첫 구간을 작업 자세 도달로 판정했다. 전체 임무는 보행 중 18회 착지, 10N·m의 수동 힌지를 두 패드로 90±0.25°까지 돌려 5초 유지, 손을 놓고 열린 밸브 앞에서 추가 5초 정지를 모두 요구한다. 보행 중 재질 미끄러짐 10mm와 조작 중 발 중심 이동 5mm 제한도 적용했다. [사전 고정한 조건과 판정 기준](protocol.json)

| 시행 환경 | 시작 자세와 제어 | 해석 범위 |
|---|---|---|
| 체크포인트 9100의 원래 보행 환경 | 계단 1.10m 앞, 동결한 PPO의 50Hz 위치 명령, 19.96초 | 원래 학습된 보행과 실제 지지 접촉. 팔이 고정된 모델 |
| 팔과 밸브가 움직이는 전체 임무 환경 | 계단 0.405m 앞에 선 바닥 자세, 물리 1ms, 모터 5ms, 최대 127.75초 | 같은 출발점에서 무개입 PPO·고정 감속·위상 변경·시범·접촉 보조의 발판 도달과 밸브 개방 |

각 환경 안에서는 모든 제어기에 동일한 횡위치·방향·관절 교란을 적용했다. 두 환경의 시작 자세와 모터 모델이 달라 성공률을 환경 사이에서 빼지 않았다. 전체 임무 환경의 PPO는 실제 측정 상태에서 입력 234개를 계산하고 이름이 일치하는 모터에 학습 당시의 위치 서보 이득을 적용한다. 접촉 보조에는 이전 성공 시범의 몸통·발 궤적과 모터 토크를 제공했다. 매 제어 시점에 하중이 실린 발로 몸통 자세를 보정하고 공중에 든 발의 실제 위치를 목표에 맞춘다. 이 모드가 선택되면 다리 제어권은 시범 기반 제어기로 넘어간다. [제어 코드](../../tools/rbq_articulated_control.py) · [시행 코드](../../tools/rbq_articulated_study.py)

## 개발 중 관측한 실패

원래 보행 환경에서 기본 PPO는 개발 조건 9개 중 7번 차체 접촉으로 종료했고 지정 발판에 도달한 횟수는 0이었다. 보행 위상을 25스텝 이동한 설정은 0/9, 보행 위상과 횡방향 피드백을 함께 바꾼 설정도 0/9였다. 이 세 설정에서 보고된 불법 접촉 16건의 센서는 모두 차체 하부였다. [원래 정책의 개발 시험](frozen-ppo/README.md)

전체 임무 환경의 개발 조건 3개에서 기본 PPO·감속·위상 변경은 엄격한 임무 성공이 각각 0/3이었다. 시범 토크 재생 및 관절각 보정은 각각 1/3, 접촉 보조는 3/3이었다. 무교란 시행은 세 시범 기반 설정에서 모두 성공했다. 두 교란 시행의 접촉 보조에서는 조작 중 발 중심 이동이 각각 4.440mm, 보행 중 최대 재질 미끄러짐이 각각 9.774mm였다. [개발 시행별 기록](articulated/development.json) · [모터·접촉·밸브 재판정](articulated/development-strict.json)

첫 에이전트는 조기 종료 횟수를 원본과 다르게 인용해 검증기에서 보류됐다. 후속 구현은 모델이 고른 관측 ID에서 원래 수치를 조회하고, 선택한 개입의 개발 시행만 공개한다. 시간 제한으로 끝나지 않은 충돌 시행은 원래 저장소의 시간 제한 진단에 넣지 않았다. [첫 실행의 검증 실패](decision/attempt-invalid.json) · [후보 목록](catalog.json)

두 번째 호출은 후보를 골랐지만 해당 환경에 등록되지 않은 진단 순위를 반복해 검증에서 탈락했다. 세 번째 구현은 개입과 관측의 ID를 따로 검사했다. 실제 CPU 모델 Gemma 3 4B가 차체 하부 충돌 3건의 원본 관측을 인용해 첫 번째로 접촉 보조를 제안했다. 자동 검증기가 해당 후보의 개발 시행에서 작업 자세 3/3과 충돌 0건을 확인해 채택했다. 관측을 가린 모델도 같은 후보를 골랐고, 개발 성적이 가장 높은 제어기를 고정하는 규칙도 같은 결과였다. 이후 두 보류 집합을 열기 전에 선택과 제어 소스를 커밋했다. [두 번째 호출 원문](decision-v2/) · [세 번째 호출과 물리 검증](decision-v3/) · [동결 기록](selection-freeze.json)

## 서로 다른 16개 출발 조건에서 두 번 평가한 결과

| 같은 관절형 로봇에 적용한 제어 | 1차 작업 자세 | 1차 엄격 임무 | 2차 작업 자세 | 2차 엄격 임무 | 차체 충돌 |
|---|---:|---:|---:|---:|---:|
| 에이전트가 고른 접촉 보조 | 16/16 | 12/16 | 16/16 | 12/16 | 0/32 |
| 원래 PPO | 0/16 | 0/16 | 0/16 | 0/16 | 32/32 |
| 고정 감속 | 0/16 | 0/16 | 0/16 | 0/16 | 32/32 |
| 보행 위상 25스텝 이동 | 0/16 | 0/16 | 0/16 | 0/16 | 32/32 |
| 시범 토크와 관절각 보정 | 0/16 | 0/16 | 0/16 | 0/16 | 32/32 |
| 시범 토크 재생 | 0/16 | 0/16 | 0/16 | 0/16 | 32/32 |

각 16쌍에서 무개입·고정 감속 대비 작업 자세 도달률은 100%p, 엄격 임무 성공률은 75%p 높았다. 엄격 성공률 차이의 보수적인 95% 구간은 각 집합에서 +20.2~+94.1%p였다. 양측 정확 McNemar 검정은 각 집합에서 p=0.000488이었다. 에이전트의 선택과 개발 최고 고정 선택은 같은 제어이므로 그 둘 사이의 실측 차이는 0%p다. 접촉 보조의 다리 토크는 시범과 접촉 피드백에서 나온다. [쌍별 비교와 통계](paired-results.json)

접촉 보조 32건 모두 밸브를 열어 5초 유지하고, 손을 놓은 뒤 발판에서 추가 5초 정지했다. 완주했지만 엄격 판정에서 탈락한 8건은 모두 출발 약 19.1초에 누적 발 미끄러짐이 10mm를 넘었다. 최초 초과량은 0.021~0.636mm였다. 원래 PPO의 팔 고정 환경에서도 작업 자세 도달은 0/32였다. 이 수치는 PPO용 원래 출발점에서 따로 집계했다. [1차 물리 판정](articulated/holdout-strict.json) · [독립 재시험 판정](articulated/confirmation-strict.json) · [원래 PPO 환경](frozen-ppo/holdout.json)

## 보상 학습이 바꾼 부분

체크포인트 9100을 고정하고, 매 순간 몸통 오차로 접촉 보조의 비율을 정하는 계수 2개와 PPO 토크 비율을 정하는 계수 1개를 CPU에서 학습했다. 교차 엔트로피 방법(CEM)으로 세 세대에 걸쳐 매 세대 8개 후보를 개발 출발점 3곳에서 평가했다. 총 72회의 20초 물리 시행에서 얻은 보상으로 계수를 선택했다. 전체 밸브 임무는 127.75초이므로 학습 구간 뒤의 밸브 성능은 별도로 측정했다. [학습 계수와 전체 보상 기록](reward-policy/training.json) · [독립 시험 규약](reward-policy/attribution-protocol.json)

| 세 번째 시험의 12개 출발 조건 | 작업 자세 | 엄격 임무 | 차체 충돌 |
|---|---:|---:|---:|
| 학습 전 계수, PPO 끔 | 0/12 | 0/12 | 10/12 |
| 보상으로 학습한 계수, PPO 끔 | 12/12 | 9/12 | 0/12 |
| 보상으로 학습한 계수, PPO 토크 약 0.30% 적용 | 12/12 | 0/12 | 0/12 |
| 접촉 보조 100% 고정, PPO 끔 | 12/12 | 9/12 | 0/12 |
| 접촉 보조 끔, 학습된 PPO 비율 유지 | 0/12 | 0/12 | 12/12 |

PPO를 양쪽 모두 끈 비교에서 학습한 계수는 엄격 성공률을 75%p 올렸다. 평균 접촉 보조 비율은 67.3%였고 고정 100%와 성공 수가 같았다. 보조 비율은 제어식의 가중치다. 에너지나 모터 부하 절감률로 해석할 수 없다. 학습한 PPO 토크를 남겨둔 경우에는 12건 모두 작업 자세에 도달했으나 9건의 첫 실패가 밸브 각도, 3건의 첫 실패가 보행 미끄러짐이었다. 시범의 토크와 궤적을 사용했고 원래 PPO 가중치는 동결했다. [물리 제어와 보상식](../../train/rbq_reward_residual.py) · [개발 엄격 판정](reward-policy/development-strict/)

이 잔여 실패를 익명 후보 6개 중 한 번만 고르도록 원본 에이전트에 맡겼다. 실패 관측을 받은 모델은 실패한 학습 제어를 다시 골랐고, 관측을 가린 모델은 PPO 토크를 뺀 제어를 골랐다. 사전에 선택을 고정한 뒤 확인한 엄격 성공 수는 각각 0/12와 9/12였다. [사전 고정한 프롬프트와 후보](reward-decision/protocol.json) · [모델의 실제 선택](reward-decision/summary.json) · [12쌍의 학습·제거 실험 결과](reward-policy/holdout-summary.json)

## 진단 에이전트의 수리 동작

원본 `rl-triage-agent`의 복구 모듈에서 이미 검증에 실패한 설정을 다음 제안 목록에서 제외하고, 새 후보의 엄격 물리 시험 결과로 채택 여부를 정하도록 바꿨다. 단일 제안 당시의 원본 커밋은 `5e9bfe8`, 수정된 원본 커밋은 `f3234ea`다. 자동 제안은 CPU의 Gemma 3 4B가 수행했다. 후보 순서는 고정했고 최대 3개까지 시도할 수 있었다. 매 제안 전에는 다른 후보의 개발 성공률을 공개하지 않았다. 개발 3건에서 엄격 성공 3/3과 차체 충돌 0건을 모두 측정해야 채택했다. [원본 모듈과 연결된 호스트](../../tools/rbq_validated_residual_repair.py) · [사전 시험 조건](repair-confirmation/protocol.json)

실패 기록을 받은 에이전트는 첫 후보로 초기 계수를 시험했다. 작업 자세 0/3, 차체 충돌 2건이어서 거절하고, 두 번째 후보인 학습 계수와 PPO 출력 제거 조합을 채택했다. 초기 진단을 가린 모델은 첫 시도에 접촉 보조를 꺼서 3번 충돌한 뒤 두 번째에 같은 조합을 채택했다. 두 팔 모두 실제로 두 후보를 시험했다. 물리 결과를 확인해 재시도하는 절차와 프롬프트에 실패 관측을 넣는 효과는 구분해 읽어야 한다. [실제 제안과 거절·채택 영수증](validated-repair/summary.json) · [새 출발 조건 공개 전 고정한 선택](repair-confirmation/selection-freeze.json)

네 번째 독립 시험에서는 동일한 출발점 12곳에서 수정 전 단일 제안과 수정 후 채택 제어를 비교했다. 두 설정 모두 12/12번 작업 자세에 도달하고 차체 충돌은 0건이었다. 전체 임무는 단일 제안 0/12에서 검증형 복구 10/12로 늘었다. 쌍별 차이는 +83.3%p였고 보수적인 95% 구간은 +16.9~+98.6%p, 양측 정확 McNemar p값은 0.001953이었다. 두 탈락은 보행 중 발 미끄러짐이 처음으로 10.118mm와 10.022mm에 이르렀을 때였다. 고정 100% 접촉 보조와 초기 진단을 가린 에이전트도 같은 10건을 통과했다. 이 시험에서 확인된 에이전트의 역할은 실패한 제안을 거절하고 실효성 있는 제어를 물리 시험으로 골라 채택하는 것이다. 프롬프트에 실패 관측을 추가한 경우의 성공률 차이는 0%p였다. [12개 출발 조건의 쌍별 결과](repair-confirmation/results.json) · [72회 물리 시험의 엄격 판정](repair-confirmation/strict.json)

## 동작 중 응력 표시

첫 보류 시행의 실제 몸통 위치·관절 속도·모터 출력을 물리 기록에서 읽어 매 영상 프레임의 하중을 다시 구했다. 깊이와 메시 식별자로 현재 카메라에 보이는 몸통·다리·팔 표면을 찾는다. 몸통은 네 고관절과 RB1 장착부를 통해서만 하중을 받으므로, 그 반력과 몸통 자체 무게(16.76kg)를 몸통 길이 방향으로 옮겨 단면 응력을 구했다. 단면은 고정 모델의 차체 충돌 상자 0.8×0.2×0.148m에 벽 두께 3mm를 가정했다. 정지 구간에서 다섯 반력의 합이 몸통 무게 164.4N과 맞아 하중 부호를 확인했다. 다리와 팔의 18개 링크는 링크 하나를 한 색으로 칠하지 않고 표면 픽셀마다 색을 정한다. 링크 양 끝의 6축 반력(종아리 끝은 발 접지력)을 링크 축을 따라 옮겨 단면별 축력·굽힘·비틀림·전단을 구하고, 각 픽셀이 굽힘 방향의 어느 면에 있는지에 따라 가정 단면의 등가응력을 계산한다. 그래서 종아리는 무릎 쪽이 붉고 발 쪽으로 갈수록 푸르며, 굽힘 중립축 쪽은 옅게 나온다. 두 끝에서 각각 옮긴 모멘트는 링크 중간에서 중앙값 0.5~1.6N·m 차이였고(관절 모멘트 중앙값 6~31N·m), 이 차이는 링크 자체 무게와 관성으로 보고 두 값을 축 방향으로 선형 혼합했다. 오른쪽 패널은 영상 전체에서 최대 응력이 큰 두 링크를 옆에서 확대하고, 아래에 현재 응력이 큰 링크 다섯 개를 막대로 보여 준다. 하늘·바둑판 바닥·그림자 조명은 렌더링에만 추가했고 물체·관절·접촉 설정은 그대로다. 밸브 레버는 원래 빨간색이라 고응력으로 오해되지 않도록 회색으로 그렸다. 몸통·다리·팔 모두 같은 범례 0–20MPa를 쓰고, 색은 파랑에서 청록·초록·노랑·빨강 순으로 변한다. 재료·벽 두께를 가정했으므로 유한요소 응력장이나 실물 설계 허용치로 사용할 수 없다.

2026-11-23 이전 판은 몸통을 다른 방식으로 칠했다. 네 고관절 링크 값과 Go2+D1용 장착판 가정(80×6mm 판, 150mm 외팔보)을 5점 보간했고, 그 장착판 값을 실제 RB1 장착 위치(몸통 중심 앞 0.238m)가 아닌 몸통 중앙에 두었다. 몸통 가운데가 순간적으로 붉게 보인 것은 이 장착판 가정값(최대 70.3MPa)이 뒷발을 차는 약 12ms 충격과 겹친 프레임이었다. 이 판은 위의 방식으로 바꿨다. [이전 판 단면·시점별 하중](heatmap/link-stress.json)

| 첫 평가 시행 | 보기 |
|---|---|
| 바닥 출발부터 밸브 개방·해제·정지까지 8배속 | [전체 동작](heatmap/agent-holdout.mp4) · [등반 장면](heatmap/agent-holdout.png) · [최대 응력 순간 66.241초, 뒤왼 종아리 무릎 쪽 21.98MPa](heatmap/agent-holdout-peak.png) |
| 같은 시행에서 팔 접근부터 마지막 정지까지 2배속 | [밸브 확대](heatmap/agent-holdout-valve.mp4) · [정착 107.321초](heatmap/agent-holdout-valve.png) |

35.201초에는 보이는 몸통 14,685픽셀에서 0.00–0.55MPa, 66.241초에는 13,862픽셀에서 0.00–0.68MPa였다. 영상 400프레임에서 몸통 최대는 0.84MPa로, 같은 범례의 다리 최대 21.98MPa보다 훨씬 낮다. 두 장면 모두 원래 몸통 메시의 픽셀만 사용했다. 영상·프레임별 물리 시점·자막·최종 상태를 같은 시행의 해시로 추적할 수 있다. [실제 메시 측정](heatmap/agent-holdout.surface.json) · [전체 영상과 물리 상태 대응](heatmap/agent-holdout.receipt.json) · [프레임별 링크 최대 응력과 위치](heatmap/agent-holdout.links.json) · [별도 기준 시범](heatmap/body-stress-follow.mp4)

## 재현

PowerShell에서 아래 명령은 보관한 체크포인트 9100의 CPU 번들과 원본 시범을 사용한다. 각 실행의 원시 상태는 새 디렉터리에 기록한다. 두 보류 집합의 조건 파일, 행별 해시와 엄격한 판정은 이 디렉터리에 보관한다.

보관한 시범 조각 여섯 개에서 259,115,531바이트를 복원해 첫 보류 조건의 무개입과 접촉 보조를 다시 실행했다. 두 시행 모두 전체 물리 상태 SHA-256과 엄격 판정의 측정값이 원래 실행과 일치했다. 재생성한 판정 캐시에는 입력 파일 해시 세 개가 추가되어 압축 파일의 해시가 달라진다. 계산에 쓰인 시각·구간·자세·지지발의 네 배열은 원본과 비트 단위로 일치했다. [재현 대조 기록](articulated/reproduction.json)

```powershell
Set-Location "$HOME/go2-wall-reach-lab"
$py = '.venv/Scripts/python.exe'
$e = 'evidence/rbq-agent-body-20261122'
$nom = 'evidence/stair-open-20261122/nominal'
$walk = 'evidence/stair-mission-20261121/full/walk50.npz'
$out = '.omx/reproduce-rbq-agent-body'
New-Item -ItemType Directory -Force "$out/nominal" | Out-Null
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
Copy-Item -LiteralPath "$nom/scene.json","$nom/result.json" -Destination "$out/nominal"
& $py tools/stair_trace_parts.py restore "$nom/raw-parts" "$out/nominal/mission.npz"
& $py tools/rbq_articulated_study.py prepare "$out/nominal" $walk "$out/servo.npz"
& $py tools/rbq_articulated_verdict.py prepare-reference "$nom/scene.json" "$out/nominal/mission.npz" $walk "$out/judge.npz"
$protocol = Get-Content -Raw "$e/protocol.json" | ConvertFrom-Json
foreach ($split in @('holdout','confirmation')) {
    $hash = $protocol.case_set_commitments.$split.sha256
    $cases = "$e/articulated/$split-cases.json"
    & $py tools/rbq_articulated_study.py run "$nom/scene.json" "$out/servo.npz" $cases "$out/$split.json" --bundle "$e/frozen-ppo/t1-v8-cpu-9100" --modes contact_gated_reference no_intervention phase_25 edge_slow_rule joint_reference_pd teacher_open_loop --expected-sha256 $hash --workers 4 --state-dir "$out/$split-states"
    & $py tools/score_rbq_agent_cohort.py "$nom/scene.json" "$out/judge.npz" $walk "$out/$split.json" "$out/$split-states" "$out/$split-strict.json" --workers 4
    & $py tools/triage_rbq_host_ab.py "$e/frozen-ppo" $cases "$out/$split-native.json" --candidates no_intervention phase_25 edge_slow_rule --expected-cases-sha256 $hash
}
& $py tools/verify_rbq_agent_study.py $e
```

보상 정책의 학습 전후·PPO 제거 비교는 같은 시범과 CPU 번들을 사용한다. 네 번째 독립 시험은 첫 프롬프트와 두 번째 제안의 선택을 모두 고정한 뒤 여섯 가지 제어를 실행했다. 위에서 만든 입력으로 물리를 다시 실행하거나, 보관된 판정에서 마지막 줄만 실행해 쌍별 성공률을 다시 계산할 수 있다. [학습 비교 재현 명령](reward-policy/README.md)

```powershell
$repair = "$e/repair-confirmation"
$hash = (Get-Content -Raw "$repair/protocol.json" | ConvertFrom-Json).case_set_sha256
& $py train/rbq_reward_residual.py evaluate "$nom/scene.json" "$out/servo.npz" "$e/frozen-ppo/t1-v8-cpu-9100" "$repair/cases.json" "$out/repair-physics.json" --policy "$e/reward-policy/training.json" --state-dir "$out/repair-states" --expected-cases-sha256 $hash --workers 6
& $py tools/score_rbq_agent_cohort.py "$nom/scene.json" "$out/judge.npz" $walk "$repair/strict-input.json" "$out/repair-states" "$out/repair-strict.json" --workers 6
& $py tools/summarize_rbq_reward_confirmation.py verify $e
```
