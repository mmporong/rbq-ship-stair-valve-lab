# S5 학습 처리량

2026-11-08에 RTX 3060에서 Unitree Go2 평지 보행 학습(`Unitree-Go2-Flat`)의 처리량을 측정했다. 정본은 `evidence/s5-train-throughput/summary.json`이다.

## 환경

| 항목 | 값 |
|---|---|
| GPU | RTX 3060 12GB, 드라이버 610.62, WSL2 Ubuntu |
| 학습 코드 | `unitree_rl_mjlab` `1425b15` (2026-04-13) |
| 라이브러리 | `mjlab` 1.2.0, `mujoco-warp` 3.5.0, MuJoCo 3.5.0, `warp-lang` 1.12.0, `torch` 2.15.0+cu132, `rsl-rl-lib` 5.0.1 |
| 학습 설정 | PPO, 신경망 512-256-128, 반복당 환경마다 24 step |
| 측정 | 환경 수마다 반복 30회, 앞의 5회는 커널 컴파일 구간이라 빼고 중앙값 |

## 설치 고정

`mjlab` 1.2.0과 `mujoco-warp` 3.5.0은 의존성 하한만 적어 둔다. 그대로 설치하면 최신판이 들어와 두 군데서 시작하자마자 실패한다.

| 패키지 | 그대로 설치한 판 | 증상 | 고정 |
|---|---|---|---|
| `mujoco` | 4.0.0 | `mujoco_warp`가 4.0에서 없어진 `mjENBL_MULTICCD`를 참조해 import 실패 | 3.5 계열 |
| `warp-lang` | 1.19.0 | `mjlab`이 쓰는 `wp.context`가 없어 환경 생성 실패 | 1.12.0 |
| `scipy` | 없음 | `mjlab`이 import하지만 의존성에 없다 | 추가 설치 |

고정 버전은 `mjlab` v1.2.0 저장소의 `uv.lock`을 따랐다. 설치 명령과 제약은 `configs/train_constraints.txt`에 있다.

## 결과

| 환경 수 | 초당 step | 반복 시간 | 수집 | 학습 | 최대 VRAM 증가 |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 20,590 | 1.19s | 1.02s | 0.17s | 1,057MiB |
| 2,048 | 34,833 | 1.41s | 1.23s | 0.19s | 1,067MiB |
| 4,096 | 40,180 | 2.45s | 2.10s | 0.32s | 2,265MiB |
| 8,192 | 49,682 | 3.96s | 3.49s | 0.47s | 5,548MiB |

- 초당 step은 반복당 step을 반복 시간으로 나눈 값과 0.3% 안에서 맞는다.
- VRAM 증가는 측정 전 사용량 2,004MiB를 뺀 값이다. 8,192개에서도 전체 12,288MiB 중 7,552MiB까지만 썼다.
- 반복 시간의 86~88%가 시뮬레이션 수집이다. 환경 수를 두 배로 늘릴 때 처리량은 1.15~1.69배 는다.
- 평균 GPU 사용률(27~65%)은 시작과 컴파일 구간을 포함한 값이라 비교에만 쓴다.

## 학습 시간 환산

| 환경 수 | 반복 1,000회 | 10억 step |
|---:|---:|---:|
| 4,096 | 0.68시간 | 6.9시간 |
| 8,192 | 1.10시간 | 5.6시간 |

이 값은 보행 작업 기준이다. 벽 짚기 작업은 팔 6관절과 벽 접촉이 늘어 step당 시간이 더 걸린다. S6에서 결합 모델로 다시 잰다.

## 판단

- 로컬 RTX 3060으로 보행 규모 학습을 하룻밤 안에 돌릴 수 있다. S6의 학습 시험은 과금 없이 로컬에서 시작한다.
- 기본 환경 수는 8,192로 둔다. 4,096보다 처리량이 24% 높고 VRAM 여유가 있다.
- 학습한 정책은 Windows의 MuJoCo 4.0 하중 분석 모델에서 다시 검증한다(`PLAN.md` S5).

## 재현

```bash
# WSL
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git ~/go2-train/unitree_rl_mjlab
cd ~/go2-train/unitree_rl_mjlab && git checkout 1425b15
uv venv --python 3.11 .venv
uv pip install -p .venv/bin/python -e . -c /mnt/c/Users/LIMMM/go2-wall-reach-lab/configs/train_constraints.txt scipy
bash /mnt/c/Users/LIMMM/go2-wall-reach-lab/tools/train_bench.sh . ~/go2-train/bench 30 1024 2048 4096 8192
python3 /mnt/c/Users/LIMMM/go2-wall-reach-lab/tools/parse_train_bench.py ~/go2-train/bench
```
