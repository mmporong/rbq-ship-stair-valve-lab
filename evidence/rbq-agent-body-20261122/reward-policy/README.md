# Reward-trained residual policy

This auxiliary CPU study asks whether rollout-return optimization finds useful
authority for the frozen checkpoint-9100 actor when an expert trajectory and
measured-state contact controller are available. It does not change the actor,
scene, motor limits, mission endpoint or 10 N m passive valve.

The policy has three learned scalars. Two map measured body tracking error to a
contact-feedback authority, and one sets the checkpoint-9100 leg-torque
authority. CEM used three generations of eight candidates, three development
starts and an exact 20 s physical rollout return. The fixed seed was 20261122.
The training consumed 72 episodes in 414.27 wall seconds on CPU.

## Development result

The frozen parameters are `[0.7173154614, -1.9024451180, -4.4043169076]`.
They use about 67.4% mean contact-feedback authority and 0.302% mean PPO
authority over the three 127.75 s development runs.

| Variant | Work stance | Proxy full mission | Strict full mission | Collision |
| --- | ---: | ---: | ---: | ---: |
| Learned residual | 3/3 | 3/3 | 0/3 | 0/3 |
| Learned residual with PPO removed | 3/3 | 3/3 | 3/3 | 0/3 |
| Fixed 100% contact feedback | 3/3 | 3/3 | 3/3 | 0/3 |
| Initial untrained policy | 0/3 | 0/3 | not applicable | 2/3 |
| Expert torque without contact feedback | 1/3 | 1/3 | not rescored | 2/3 |
| Learned PPO authority with contact feedback removed | 0/3 | 0/3 | not applicable | 3/3 |

The proxy endpoint only checked maximum valve travel. The independent strict
judge found that the learned policy's small PPO contribution let the hold angle
fall to 89.7405--89.7406 degrees, just outside the fixed 90 +/- 0.25 degree
band. Removing PPO passed all three strict development runs. Therefore this
development set supports learned reduction of model-based assistance, but does
not support a beneficial checkpoint-9100 contribution.

The 12-case reward-policy holdout was generated before these outcomes were
read. Its public canonical commitment is in `protocol.json`; its contents stay
outside this evidence directory. Code, reward, parameters, comparisons and
thresholds are frozen before that set is evaluated.

## Frozen holdout result

The preregistered 12-case holdout was evaluated for all six original arms and
the matched initial-parameter/no-PPO attribution arm. Every 127.75 s trajectory
was scored with the common articulated verdict code. The scorer verified each
raw state SHA-256 before judging it, and treated an early body contact or fall
as a strict failure.

| Variant | Work stance | Proxy full mission | Strict full mission | Collision or fall | Mean contact authority | Mean PPO authority |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned residual | 12/12 | 12/12 | 0/12 | 0/12 | 67.44% | 0.302% |
| Learned residual with PPO removed | 12/12 | 12/12 | 9/12 | 0/12 | 67.34% | 0% |
| Fixed 100% contact feedback | 12/12 | 12/12 | 9/12 | 0/12 | 100% | 0% |
| Initial parameters with PPO removed | 0/12 | 0/12 | 0/12 | 12/12 | 32.14% | 0% |
| Initial untrained policy | 0/12 | 0/12 | 0/12 | 12/12 | 33.52% | 1.186% |
| Expert torque without contact feedback | 0/12 | 0/12 | 0/12 | 12/12 | 0% | 0% |
| Learned PPO authority with contact feedback removed | 0/12 | 0/12 | 0/12 | 12/12 | 0% | 0.302% |

The matched learning contrast supports the reward-trained contact policy. With
PPO disabled on both sides, optimization improved strict success from 0/12 to
9/12. All nine discordant pairs favored the trained parameters, giving an
exact two-sided McNemar p-value of 0.00390625. The trained no-PPO policy and
fixed-contact controller succeeded and failed on exactly the same cases, while
the trained policy used 32.66% less mean contact authority.

The checkpoint-9100 contribution is negative in this experiment. Adding its
small learned torque contribution changed strict success from 9/12 to 0/12.
Nine runs left the required valve-hold band, while the other three shared the
same pre-walk stance-slip failure as the fixed-contact and no-PPO controls. The
rollout return still ranked these trajectories highly because its proxy and
20 s training horizon did not represent the later strict valve hold. This is a
measured reward-design failure, not evidence of a useful PPO contribution.

The preregistered triage model chose `learned` when it observed the development
failure and `learned_no_ppo` when that observation was removed. The observed
choice therefore scored 0/12 against 9/12 for the observation-removed choice.
This trial does not validate that triage agent.

Teacher assistance remains part of every residual-policy result. The learned
controller composes frozen expert torque, measured-state contact feedback and,
when enabled, checkpoint torque. It is not a standalone PPO policy. The
teacher-only open-loop arm and PPO-without-contact arm both collided or fell in
12/12 cases, isolating the useful contribution to the reward-trained contact
authority parameters.

## Reproduction

First restore the demonstration and generate the servo and judge inputs using the
[main reproduction steps](../README.md#재현). From the repository root, use the
published CPU bundle and case manifest with those inputs.

```powershell
Set-Location "$HOME/go2-wall-reach-lab"
New-Item -ItemType Directory -Force .omx\reproduce-rbq-agent-body\reward | Out-Null
.venv\Scripts\python.exe -u train\rbq_reward_residual.py evaluate `
  evidence\stair-open-20261122\nominal\scene.json `
  .omx\reproduce-rbq-agent-body\servo.npz `
  evidence\rbq-agent-body-20261122\frozen-ppo\t1-v8-cpu-9100 `
  evidence\rbq-agent-body-20261122\reward-policy\holdout-cases.json `
  .omx\reproduce-rbq-agent-body\reward\reproduce-holdout-evaluation.json `
  --policy evidence\rbq-agent-body-20261122\reward-policy\training.json `
  --state-dir .omx\reproduce-rbq-agent-body\reward\reproduce-holdout-states `
  --expected-cases-sha256 f0d31ec70f45427d4164e23ac82358120b0f83ca12eaa4bab50b6eacaa0c4a3d `
  --workers 12

.venv\Scripts\python.exe -u tools\rbq_initial_no_ppo.py `
  evidence\stair-open-20261122\nominal\scene.json `
  .omx\reproduce-rbq-agent-body\servo.npz `
  evidence\rbq-agent-body-20261122\frozen-ppo\t1-v8-cpu-9100 `
  evidence\rbq-agent-body-20261122\reward-policy\holdout-cases.json `
  .omx\reproduce-rbq-agent-body\reward\reproduce-initial-no-ppo-evaluation.json `
  --state-dir .omx\reproduce-rbq-agent-body\reward\reproduce-initial-no-ppo-states `
  --expected-cases-sha256 f0d31ec70f45427d4164e23ac82358120b0f83ca12eaa4bab50b6eacaa0c4a3d `
  --workers 12

.venv\Scripts\python.exe -u tools\score_rbq_agent_cohort.py `
  evidence\stair-open-20261122\nominal\scene.json `
  .omx\reproduce-rbq-agent-body\judge.npz `
  evidence\stair-mission-20261121\full\walk50.npz `
  evidence\rbq-agent-body-20261122\reward-policy\holdout-strict-input.json `
  .omx\reproduce-rbq-agent-body\reward\reproduce-holdout-states `
  .omx\reproduce-rbq-agent-body\reward\reproduce-holdout-strict.json `
  --workers 12

.venv\Scripts\python.exe -u tools\score_rbq_agent_cohort.py `
  evidence\stair-open-20261122\nominal\scene.json `
  .omx\reproduce-rbq-agent-body\judge.npz `
  evidence\stair-mission-20261121\full\walk50.npz `
  evidence\rbq-agent-body-20261122\reward-policy\initial-no-ppo-strict-input.json `
  .omx\reproduce-rbq-agent-body\reward\reproduce-initial-no-ppo-states `
  .omx\reproduce-rbq-agent-body\reward\reproduce-initial-no-ppo-strict.json `
  --workers 12
```

## Sources

- `training.json` contains every sampled parameter, per-rollout return and the
  selected frozen parameter vector.
- `development-evaluation.json` contains paired endpoint and authority data.
- `development-strict/` contains the independent full physical verdicts for
  all six development arms across all three development starts.
- `holdout-evaluation.json` and `initial-no-ppo-evaluation.json` contain the
  sealed paired rollouts and raw state commitments.
- `holdout-cases.json` is the post-evaluation public copy of the once-sealed
  12-case set. Its canonical case-set commitment is
  `f0d31ec70f45427d4164e23ac82358120b0f83ca12eaa4bab50b6eacaa0c4a3d`;
  its separate raw-file SHA-256 is
  `43f764b887a439277c6e13e77a43ae6d1b311c7f8aa9007ce2a6fc0ad5a7c537`.
  Both values match their respective preregistered fields in `protocol.json`.
- `holdout-strict-input.json` and `initial-no-ppo-strict-input.json` bind the
  strict scorer to the exact trial records and state hashes.
- `holdout-strict.json` and `initial-no-ppo-strict.json` contain the common
  strict verdicts after state-hash verification.
- `holdout-summary.json` contains arm-level metrics, paired contrasts, exact
  tests, source artifact hashes and the frozen triage choices.
- `train/rbq_reward_residual.py` is the learner and evaluator.
- `tests/test_rbq_reward_residual.py` freezes the policy and study contract.
