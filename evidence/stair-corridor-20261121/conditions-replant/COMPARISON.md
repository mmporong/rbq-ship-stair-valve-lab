# 기본 환경과 1-thread 환경 비교

발표용 17행은 1-thread CPU 환경만 사용한다. 아래 표는 같은 frozen source·명령에서 실행 환경에 따른 바이트·수치·판정 차이를 분리해 기록한다.

| 조건 | raw 동일 | 판정 동일 | 기본 최초 위반 | 1-thread 최초 위반 | 시각 차이 | 측정값 차이 |
|---|---:|---:|---|---|---:|---:|
| 기준 조건 | 다름 | 같음 | stance_slip_m @ 127.791 s | stance_slip_m @ 127.831 s | +0.040000 s | -0.000125 mm |
| 발–계단 마찰 0.50 | 다름 | 같음 | qp_attempt @ 127.656 s | qp_attempt @ 127.656 s | +0.000000 s | +0 |
| 발–계단 마찰 0.80 | 다름 | 같음 | stance_slip_m @ 127.830 s | stance_slip_m @ 127.830 s | +0.000000 s | -0.000171 mm |
| 밸브 저항 5 N·m | 다름 | 같음 | max_base_shift_m @ 100.620 s | max_base_shift_m @ 100.620 s | +0.000000 s | -0.000032 mm |
| 밸브 저항 15 N·m | 다름 | 같음 | max_base_shift_m @ 102.579 s | max_base_shift_m @ 102.579 s | +0.000000 s | +0.000057 mm |
| 레버 0.10 m | 다름 | 같음 | pad_contact_loss_s @ 100.558 s | pad_contact_loss_s @ 100.558 s | +0.000000 s | +0.000000 s |
| 레버 0.14 m | 다름 | 같음 | stance_slip_m @ 125.717 s | stance_slip_m @ 125.939 s | +0.222000 s | -0.000090 mm |
| 패드 마찰 0.6 | 다름 | 같음 | stance_slip_m @ 127.859 s | stance_slip_m @ 127.873 s | +0.014000 s | +0.000709 mm |
| 패드 마찰 1.0 | 다름 | 같음 | angle_deg @ 111.651 s | angle_deg @ 111.651 s | +0.000000 s | -0.000075° |
| 손가락 힘 한계 75 N | 다름 | 같음 | pad_contact_loss_s @ 100.460 s | pad_contact_loss_s @ 100.460 s | +0.000000 s | +0.000000 s |
| 손가락 미끄럼 손실 15 N | 다름 | 같음 | stance_slip_m @ 127.792 s | stance_slip_m @ 127.831 s | +0.039000 s | +0.000234 mm |
| 팔 관절 한계 LIMS-EX | 다름 | 같음 | requested_torque_ratio @ 100.401 s | requested_torque_ratio @ 100.401 s | +0.000000 s | +6.28657422e-06 |
| 물리 간격 2.5 ms | 다름 | 같음 | start_angle_deg @ 100.402 s | start_angle_deg @ 100.402 s | +0.000000 s | -0.007545° |
| 접촉 임피던스 비 10 | 다름 | 같음 | stance_slip_m @ 4.197 s | stance_slip_m @ 4.197 s | +0.000000 s | -0.000001 mm |

원시는 두 `.omx` run-root에 보존하며 evidence 디렉터리에 복사하지 않았다.
