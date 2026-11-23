# 1 ms 레버 접촉 감사

정식 상승·하강 replay의 `mj_step` 직후 MuJoCo가 이미 계산한 contact/constraint만 읽는다. 추가 `mj_forward`, 상태·힘·명령 쓰기는 없다. `contacts.npz`에는 레버 접촉이 존재한 tick만 희소 저장한다.

```powershell
cd "$HOME/go2-wall-reach-lab"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$auditRoot = Join-Path $env:TEMP ("go2-contact-audit-" + [guid]::NewGuid())

.\.venv\Scripts\python.exe `
  evidence\stair-corridor-20261121\local-repair\segments\contact-audit\replay_contact_audit.py `
  up (Join-Path $auditRoot 'up') `
  --walk evidence\stair-corridor-20261121\local-repair\walk50.npz `
  --reference-segment evidence\stair-corridor-20261121\local-repair\segments\up\segment.npz

.\.venv\Scripts\python.exe `
  evidence\stair-corridor-20261121\local-repair\segments\contact-audit\replay_contact_audit.py `
  down (Join-Path $auditRoot 'down') `
  --walk evidence\stair-corridor-20261121\local-repair\walk50.npz `
  --reference-segment evidence\stair-corridor-20261121\local-repair\segments\down\segment.npz
```

`time_index × physics_step_s`가 접촉 관측 시각이다. `contact_mask`의 bit 0·1·2는 각각 upper pad, lower pad, nonpad 접촉 존재를 뜻한다. `*_n`은 해당 tick/category의 `mj_contactForce(...)[0:3]` 합력 크기 최대값이고 `*_normal_n`은 `abs(force[0])` 최대값이다.
