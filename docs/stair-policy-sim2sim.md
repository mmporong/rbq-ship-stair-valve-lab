# RBQ 계단 정책과 MuJoCo 4.0 CPU 검증

T1의 7판 정책(`model_8997.pt`)을 MuJoCo 4.0.0 Windows CPU로 옮겼다. 같은 장면과 초기 상태를 MuJoCo 3.5.0 WSL CPU에서도 평가했다. 열린 계단에서는 speculative contacts와 시간 스텝도 바꿔 봤다. 원본 체크포인트, 장면과 실험 조건의 해시는 [실측 JSON](../evidence/t1-mujoco4-20261120/)에 기록했다.

## 열린 발판 접촉

RBQ 공개 모델의 발 충돌체는 반지름 30mm 구다. T1의 열린 발판은 두께 30mm인 상자다. 이 두 형상을 써서 중력 없는 단일 발 낙하 실험을 만들었다. 발 질량 1kg, 초기 하강 속도 1·3·6·12·24m/s, 초기 발바닥 간격은 한 스텝 이동량의 10·50·90%로 설정했다. 질량과 속도는 기능을 시험하기 위한 입력이다. 발이 발판 아래로 완전히 나가는지, 그 전에 해결기에 들어간 접촉(`efc_address >= 0`)이 있었는지를 셌다.

| 시간 스텝 | 끔 · 완전 통과 | 끔 · 그중 접촉 없이 통과 | 켬 · 완전 통과 |
|---|---:|---:|---:|
| 5ms | 6/15 | 1/15 | 0/15 |
| 10ms | 5/15 | 3/15 | 0/15 |
| 20ms | 8/15 | 5/15 | 0/15 |

Euler와 implicitfast에서 위 결과가 같았다. 켠 조건에서도 발의 최하단은 발판 윗면 아래로 들어갔다. 최대 높이 차이는 5·10·20ms에서 각각 21.31·13.20·8.80mm였다. 발판 옆으로 떨어뜨린 12건은 설정과 관계없이 접촉 없이 지나갔다. [발판 실험 JSON](../evidence/t1-mujoco4-20261120/contacts.json)

이 기능은 스텝 시작 속도로 볼록 지오메트리의 접촉을 예측한다. 6판에서는 발판 아래에 바닥을 추가해 빈 공간으로 빠진 발을 받쳤다. T1 학습의 MuJoCo Warp 3.5는 이 플래그를 지원하지 않는다. [MuJoCo 4.0.0 접촉 문서](https://mujoco.readthedocs.io/en/4.0.0/computation/speculative.html), [MuJoCo 4.0 발표](https://github.com/google-deepmind/mujoco/discussions/3773)

## 액추에이터 힘 제한

1kg 슬라이더에 위치 이득 20N/m, 속도 이득 4N·s/m을 적용했다. 제어값 1m를 넣고, 20ms implicitfast 스텝에서 속도 미분과 다음 속도를 측정했다. 포화가 없는 0.1m 입력과 Euler 대조군도 결과에 담았다.

| 제한 위치 | 관절 힘 | 3.5 / 4.0 속도 미분 (N·s/m) | 3.5 / 4.0 다음 속도 (m/s) |
|---|---:|---:|---:|
| 제한 없음 | 20N | −4 / −4 | 0.37037 / 0.37037 |
| 제어 입력 `ctrlrange` | 2N | −4 / −4 | 0.03704 / 0.03704 |
| 개별 출력 `forcerange` | 5N | 0 / 0 | 0.10000 / 0.10000 |
| 관절 합산 `actuatorfrcrange` | 5N | −4 / 0 | 0.09259 / 0.10000 |

4.0의 변경은 관절·텐던 합산 제한에 걸린 액추에이터의 암시적 속도 미분을 제외한다. 개별 `forcerange`가 포화될 때는 3.5에서도 이미 제외했다. [4.0.0 변경 기록](https://mujoco.readthedocs.io/en/4.0.0/changelog.html#version-4-0-0-november-6-2026), [4.0.0 엔진](https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_derivative.c#L3377-L3431), [3.5.0 엔진](https://github.com/google-deepmind/mujoco/blob/3.5.0/src/engine/engine_derivative.c#L983-L1055)

컴파일한 Go2+D1, Go2+Z1, RBQ+RB1 임무 모델과 7판 계단 모델에서 관절·텐던 합산 힘 제한은 0개였다. 임무의 토크 모터는 `ctrlrange`를 쓰고, Z1 그리퍼와 7판의 위치 구동기 12개는 개별 `forcerange`를 쓴다. 이 모델 설정에서 4.0의 합산 포화 변경이 작동할 경로는 없다. [3.5 측정](../evidence/t1-mujoco4-20261120/actuators-3.5.0.json), [4.0 측정](../evidence/t1-mujoco4-20261120/actuators-4.0.0.json)

## 7판 정책 검증 조건

- 고정 체크포인트의 234개 관측과 위치 행동 12개를 사용했다. WSL의 `unitree_rl_mjlab` `1425b15`, `mjlab` 1.2.0, MuJoCo Warp 3.5.0 환경에서 MJCF, ELU 신경망, 관측 정규화, 초기 상태를 내보냈다. CPU 평가기는 기존 MuJoCo와 NumPy로 실행한다.
- 같은 10×10 지형, 열당 초기 상태 3개, 단계당 30개를 두 버전에 썼다. 단계별 단 높이는 지형 시드에 따라 달라진다. 원래와 같은 0.6m/s 전진과 방향각 0 목표로 998회(19.96초) 제어했다. 물리 기본 스텝은 5ms, 제어는 50Hz다.
- 원본과 같은 위치 목표, PD 구동, 기울기 70°와 베이스·팔의 지형 접촉력 10N 종료 조건을 썼다. 시간 스텝을 바꿀 때는 한 제어 주기에서 관측된 힘을 사용했다. 생존율은 전체 시행, 이동 거리·종료 높이의 중앙값은 생존 시행 기준이다. 종료 높이 `dz`는 출발 지형에서 몸통까지의 높이라 평지에서도 약 0.53m다.
- 두 CPU 버전의 초기 상태에는 평가기의 위치·방향 초기화가 들어 있다. 관측 잡음, 외부 밀기, 마찰·엔코더·질량중심 무작위화는 껐다. 기존 7판 GPU 결과는 1,000개씩 평가한 기록이다. 짝을 맞춘 CPU 비교에는 같은 초기 상태를 쓰며, GPU 결과는 기존 동작을 비교하는 데 사용했다. [기존 7판 평가](../evidence/t1-train/rbq_v7_eval_straight.json)

900개 기준 상태에서 부동소수점 배열을 비교했다. Windows 4.0에서 관측의 최대 절대 차이는 `1.01×10⁻⁶`, 같은 관측을 넣었을 때 정책 출력 차이는 `1.32×10⁻⁶`, 관측부터 다시 계산한 관절 목표값 차이는 `2.54×10⁻⁵ rad`였다. 지형 스캔은 고정 상자에 대해 일괄 계산하며, 같은 표본에서 MuJoCo의 `mj_ray`와 차이가 `7×10⁻¹⁷` 이하였다. 3.5에서 MJCF를 읽어들였을 때의 모델 배열 오차도 기록했다.

## 7판 정책의 교차 평가

5ms에서 3.5는 278/300, 4.0은 282/300이 생존했다. 272개는 양쪽에서 생존했고 12개는 양쪽에서 종료됐다. 3.5에서만 생존한 경우가 6개, 4.0에서만 생존한 경우가 10개였다. 양쪽에서 생존한 272개를 짝지었을 때 거리 차이의 중앙값은 1.60mm, 최대는 2.59m였다. 종료 높이 차이의 중앙값은 0.20mm, 최대는 1.14m였다. [3.5 CPU 평가](../evidence/t1-mujoco4-20261120/sim2sim-3.5.0.json), [4.0 CPU 평가](../evidence/t1-mujoco4-20261120/sim2sim-4.0.0.json)

| 지형 | 3.5 CPU 생존 | 4.0 CPU 생존 |
|---|---:|---:|
| 평지 | 30/30 | 30/30 |
| 닫힌 오르막 | 55/60 | 57/60 |
| 닫힌 내리막 | 60/60 | 60/60 |
| 열린 오르막 | 74/90 | 75/90 |
| 열린 내리막 | 59/60 | 60/60 |

| 조건 | 3.5 CPU 생존 · 거리 · 종료 높이 | 4.0 CPU 생존 · 거리 · 종료 높이 | 기존 3.5 Warp 생존 · 거리 · 종료 높이 |
|---|---|---|---|
| 열린 오르막 5단계 | 9/9 · 4.31m · 1.742m | 9/9 · 4.28m · 1.743m | 95.7% · 4.55m · 1.595m |
| 닫힌 오르막 6단계 | 6/6 · 4.86m · 1.977m | 6/6 · 4.88m · 1.976m | 98.5% · 5.46m · 1.951m |
| 닫힌 내리막 6단계 | 6/6 · 0.79m · 0.539m | 6/6 · 0.79m · 0.539m | 99.5% · 0.89m · 0.523m |
| 열린 오르막 6단계 | 8/9 · 4.16m · 2.038m | 7/9 · 4.14m · 2.043m | 46.7% · 1.60m · 0.888m |
| 열린 내리막 6단계 | 6/6 · 0.81m · 0.538m | 6/6 · 0.81m · 0.538m | 95.5% · 0.80m · 0.530m |
| 열린 오르막 9단계 | 4/9 · 0.82m · 0.562m | 3/9 · 1.04m · 0.667m | 9.0% · 1.01m · 0.638m |

6~9단계 내리막에서는 두 CPU 버전 모두 평판 근처에 멈췄고, 3m를 넘긴 시행은 0개였다. 9단계 열린 오르막에서도 3m를 넘긴 시행이 없었다. CPU에서 7판의 난도가 높은 구간이 재현됐다. 6단계 열린 오르막의 생존율 차이는 표본 수, 지형 시드와 꺼 둔 무작위화가 있는 조건에서 관찰한 값이다. 3.5 WSL과 4.0 Windows를 비교했으므로 버전 변화에는 플랫폼과 수치 구현의 차이도 포함된다.

## 보행 중 5ms와 10ms 비교

MuJoCo 4.0에서 열린 계단 5~9단계의 같은 초기 상태 75개를 네 조건에 썼다. 오르막 45개, 내리막 30개다. 제어는 50Hz로 고정했다. 5ms에서 제어 주기당 4단계, 10ms에서 2단계씩 물리를 진행했다.

| 물리 스텝 | Speculative contacts | 전체 생존 | 오르막 생존 | 9단계 열린 오르막 생존 | 생존하며 3m 초과 |
|---|---|---:|---:|---:|---:|
| 5ms | 끔 | 60/75 | 30/45 | 3/9 | 21/75 |
| 5ms | 켬 | 56/75 | 26/45 | 3/9 | 21/75 |
| 10ms | 끔 | 56/75 | 26/45 | 1/9 | 21/75 |
| 10ms | 켬 | 53/75 | 23/45 | 0/9 | 21/75 |

5ms 기본 설정에서 생존이 가장 많았고, 네 조건의 전진 시행 수는 같았다. 7판 CPU 평가 기준을 5ms로 둔다. 10ms에 사전 접촉을 켠 9단계 열린 오르막의 15번 시행은 4.00초에 기울기 제한으로 종료됐다. 종료 때 출발 지형 중심에서 0.99m 떨어져 있었다. [5ms 켬](../evidence/t1-mujoco4-20261120/sim2sim-4.0.0-open-5ms-speculative.json), [10ms 끔](../evidence/t1-mujoco4-20261120/sim2sim-4.0.0-open-10ms.json), [10ms 켬](../evidence/t1-mujoco4-20261120/sim2sim-4.0.0-open-10ms-speculative.json), [종료 장면](../evidence/t1-train/failures/q1-v7-mujoco4-10ms-open-up-l9.png)

## 재현

WSL에서 기존 학습 환경으로 내보내고 기준 CPU 결과를 만든다. 반복 실행에는 새 `bundle`과 `run` 경로를 사용한다.

```bash
windows_home=$(powershell.exe -NoProfile -Command '$env:USERPROFILE' | tr -d '\r')
repo="$(wslpath "$windows_home")/go2-wall-reach-lab"
bundle="$repo/.omx/v7-cpu-repeat"
run="$repo/evidence/t1-mujoco4-repeat"
cd ~/go2-train/unitree_rl_mjlab
checkpoint=logs/rsl_rl/rbq10_rb1_stairs/2026-11-10_17-45-51_v7/model_8997.pt
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  "$repo/train/export_stairs_cpu.py" "$checkpoint" "$bundle" --samples-per-column 3
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  "$repo/train/eval_stairs_cpu.py" "$bundle" "$run/sim2sim-3.5.0.json" --workers 3
CUDA_VISIBLE_DEVICES= .venv/bin/python "$repo/tools/mujoco4_checks.py" actuators \
  "$run/actuators-3.5.0.json" --probes-only --rbq-scene "$bundle/scene.xml"
```

Windows PowerShell에서 같은 번들을 사용한다. 체크포인트 가중치와 내보낸 MJCF는 `.omx`에 둔다.

```powershell
Set-Location "$HOME/go2-wall-reach-lab"
$py = '.venv/Scripts/python.exe'
$bundle = '.omx/v7-cpu-repeat'
$run = 'evidence/t1-mujoco4-repeat'
$env:OPENBLAS_NUM_THREADS = '1'
& $py tools/mujoco4_checks.py contacts "$run/contacts.json"
& $py tools/mujoco4_checks.py actuators "$run/actuators-4.0.0.json" --rbq-scene "$bundle/scene.xml"
& $py train/eval_stairs_cpu.py $bundle "$run/sim2sim-4.0.0.json" --workers 4
$open = @('--levels', '5', '6', '7', '8', '9', '--terrain', 'open_stairs_up', 'open_stairs_down')
& $py train/eval_stairs_cpu.py $bundle "$run/sim2sim-4.0.0-open-5ms-speculative.json" @open --timestep 0.005 --speculative --workers 3
& $py train/eval_stairs_cpu.py $bundle "$run/sim2sim-4.0.0-open-10ms.json" @open --timestep 0.01 --workers 3
& $py train/eval_stairs_cpu.py $bundle "$run/sim2sim-4.0.0-open-10ms-speculative.json" @open --timestep 0.01 --speculative --workers 3
```
