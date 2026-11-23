# RBQ host A/B result

This directory records an independent Go2 host experiment. It is not an NVIDIA
registered `rl-triage-agent` task, Nemotron result, approved adaptation, or policy
update. The original registered path remains `HOLD_without_execution` for this RBQ
case.

## Frozen design

- Actor: checkpoint 9100, identical frozen CPU bundle in every arm.
- Stair: 50 degrees, 190 mm rise, 10 treads, friction 0.6.
- Primary endpoint: FL/FR on tread 5 and RL/RR on tread 3 for 0.5 continuous seconds.
- Development: 3 paired initial conditions, all four candidate settings.
- Holdout: 6 paired initial conditions committed by SHA-256 before selection and
  revealed only after the local model decision.
- Arms: no intervention, a pre-fixed edge-slowdown rule, and the development-selected
  candidate. Every arm has the same maximum control-step budget; physical safety
  termination can end an individual replay early.

The selector was local Ollama `gemma3:4b`, temperature 0, seed 42, `num_gpu=0`.
Development summaries were converted to citable facts by the original
`rl_triage.active_diagnosis.receipt_facts` function. Candidate authorization and
execution are supplied by the new RBQ host contract, not by the NVIDIA registry.

## Development selection

| Candidate | Work stance | Mean highest tread | Body collision |
| --- | ---: | ---: | ---: |
| No intervention | 0/3 | 1.67 | 3/3 |
| Phase 15 | 0/3 | 1.67 | 3/3 |
| Phase 25 | 0/3 | 2.33 | 1/3 |
| Fixed edge-slow rule | 0/3 | 0.33 | 3/3 |

The local selector chose `phase_25` and cited its exact collision and highest-tread
receipt facts.

## Paired holdout result

| Arm | Work stance | Mean highest tread | Body collision | Mean max forward x |
| --- | ---: | ---: | ---: | ---: |
| No intervention | 0/6 | 1.17 | 4/6 | -0.08 m |
| Fixed edge-slow rule | 0/6 | 0.83 | 5/6 | -0.05 m |
| Agent-selected phase 25 | 0/6 | 1.83 | 4/6 | -0.01 m |

Agent-selected minus no-intervention paired effects:

- Work-stance rate: 0.00 (0/6 versus 0/6).
- Highest supported tread: mean +0.67; 5/6 pairs improved, two-sided exact sign
  test p=0.21875.
- Maximum forward x: mean +0.077 m; 4/6 pairs improved, p=0.6875.
- Maximum body height increased by 0.039 m on average across the six pairs. Four
  pairs increased, p=0.6875.
- Body-collision rate: 0.00 difference.
- Maximum absolute pitch: mean +6.28 degrees (worse); 6/6 pairs increased,
  p=0.03125. These secondary tests are exploratory and unadjusted for multiplicity.

The pre-registered mission endpoint did not improve. The experiment therefore does
not demonstrate that the local agent meaningfully solved the RBQ work-position
problem. It found a setting with better exploratory tread reach than no intervention
and the fixed simple rule, but that gain came with greater pitch excursion and no
reduction in collision rate.
