# Frozen PPO 9100 position-servo development

This directory contains CPU-only development runs on the exact RBQ-10 + RB1,
50-degree, 190 mm stair model. The former six-case holdout was released for
development before these runs. No current holdout or confirmation cases were
read or executed.

The frozen actor is checkpoint 9100 with SHA-256
`6d39b7c28c9cdda9c3964c3b37493c3bcd24f81ca46f5e367618a7f264cfc757`.
Every run uses the exported actor's 998 control steps at 0.02 s, or 19.96 s.
Success requires FL/FR on tread 5 and RL/RR on tread 3 continuously for 0.5 s.

## Development result

Across all nine released development cases, every candidate reached the work
stance in 0/9 trials.

| Candidate | Mean highest tread | Base collision rate | Interpretation |
| --- | ---: | ---: | --- |
| Frozen PPO | 1.33 | 7/9 | Unmodified baseline |
| Phase offset 25 | 2.00 | 5/9 | Best progress, but no work stance |
| Phase 25 + lateral/tread-3 feedback | 1.67 | 4/9 | Fewer collisions, less progress |

Every identified illegal-contact sensor was
`nonfoot_ground_touch_base_link_collision1_force`; none was an RB1 arm sensor.
The saved demonstration residuals also reached 0/3. The `demo_teacher` group is
a leg-target replacement after activation and is not PPO recovery. The
`demo_r35` and `demo_r60` groups retain the actor and add 35% or 60% joint-target
residuals, respectively.

## Frozen baseline command (do not execute before release)

After a private case set is released by the evaluation owner, run only the
unmodified `ppo` candidate. The command below is a recipe and was not executed
for either pending set.

```powershell
cd C:\Users\LIMMM\go2-wall-reach-lab
$casesPath = "C:\absolute\private\released-cases.json"
$outPath = "C:\absolute\new-output.json"
uv run python train\rbq_stair_interventions.py .omx $casesPath $outPath --candidates ppo
```

Use a new output path for the holdout and confirmation runs. The evaluator must
keep the private case files inaccessible until each preregistered release.

## Required implementation and test

- `train/rbq_stair_interventions.py`
- `tests/test_rbq_stair_interventions.py`

The focused unit command is:

```powershell
uv run pytest -q tests\test_rbq_stair_interventions.py
```
