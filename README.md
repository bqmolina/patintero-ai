# patintero-ai

Short CLI guide for training, playing, replaying trajectories, and live metrics.

## Quick start

Run from the project folder.

### 1) Train headless (fast)

python main.py --mode train --no-render --episodes 3000 --model-path mappo_patintero.pt

### 2) Train with trajectories + metrics

python main.py --mode train --no-render --episodes 3000 --model-path mappo_patintero.pt --log-trajectories --trajectory-checkpoint-every 100 --trajectory-episodes 5 --trajectory-format jsonl --trajectory-dir trajectory_logs --log-metrics --metrics-format both --metrics-log-step 10 --metrics-window 200 --metrics-dir metrics_logs

### 3) Open TensorBoard

tensorboard --logdir metrics_logs

### 4) Play a trained model

python main.py --mode play --model-path mappo_patintero.pt --render --play-episodes 5

## Replay trajectories

### Replay one checkpoint file

python main.py --mode replay --trajectory-file trajectory_logs/checkpoint_0002700.jsonl --trajectory-episode 1

### Replay all checkpoints from N onward in one window

python main.py --mode replay --trajectory-dir trajectory_logs --replay-from-checkpoint 2700 --trajectory-episode 1

Optional upper bound:

python main.py --mode replay --trajectory-dir trajectory_logs --replay-from-checkpoint 2700 --replay-to-checkpoint 3000 --trajectory-episode 1

## Useful flags

- --mode train|play|replay
- --model-path PATH
- --episodes N
- --no-render (headless)
- --render --render-every N
- --seconds-per-15-frames X
- --log-metrics --metrics-format tensorboard|jsonl|both --metrics-log-step N --metrics-window N --metrics-dir DIR
- --log-trajectories --trajectory-checkpoint-every N --trajectory-episodes N --trajectory-format json|jsonl --trajectory-dir DIR