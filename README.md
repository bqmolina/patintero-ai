# patintero-ai

Patintero 1v1 multi-agent reinforcement learning with MAPPO (PyTorch).

## What the policy learns

This project trains two agents together:

- Attacker: a continuous policy that outputs a movement angle (mapped to 0 to 360 degrees).
- Defender: a discrete policy that chooses one of three actions: move left, stay, move right.

Both agents are trained with MAPPO (Multi-Agent PPO):

- Decentralized actors: each agent has its own actor network and acts from its own local observation.
- Centralized critics: both critics are conditioned on the joint state (attacker obs + defender obs) for more stable advantage estimates.
- PPO objective: clipped policy-ratio loss + value loss + entropy bonus.
- Temporal context: frame stacking (current frame + previous frames) helps each policy infer motion.

## Model architecture (short)

- Backbone: MLP (2 hidden layers, Tanh activations).
- Attacker actor head: Gaussian policy over an unconstrained latent action, squashed and mapped to angle space.
- Defender actor head: categorical policy over 3 classes.
- Critics: value MLPs (one per agent) over stacked joint-state features.

## Quick start

Run commands from the project root.

### 1) Train headless (fast)

```bash
python main.py --mode train --no-render --episodes 3000 --model-path mappo_patintero.pt
```

### 2) Train with trajectories and metrics

```bash
python main.py --mode train --no-render --episodes 3000 --model-path mappo_patintero.pt --log-trajectories --trajectory-checkpoint-every 100 --trajectory-episodes 5 --trajectory-format jsonl --trajectory-dir trajectory_logs --log-metrics --metrics-format both --metrics-log-step 10 --metrics-window 200 --metrics-dir metrics_logs
```

### 3) Open TensorBoard

```bash
tensorboard --logdir metrics_logs
```

### 4) Play a trained model

```bash
python main.py --mode play --model-path mappo_patintero.pt --render --play-episodes 5
```

## Replay trajectories

### Replay one evaluation episode from one checkpoint file

```bash
python main.py --mode replay --trajectory-file trajectory_logs/checkpoint_0002700.jsonl --trajectory-episode 1
```

### Replay all evaluation episodes in one checkpoint file

```bash
python main.py --mode replay --trajectory-file trajectory_logs/checkpoint_0002700.jsonl --replay-all-episodes
```

### Replay checkpoint files from N onward in one window

```bash
python main.py --mode replay --trajectory-dir trajectory_logs --replay-from-checkpoint 2700 --trajectory-episode 1
```

Optional upper bound:

```bash
python main.py --mode replay --trajectory-dir trajectory_logs --replay-from-checkpoint 2700 --replay-to-checkpoint 3000 --trajectory-episode 1
```

Replay every evaluation episode for each file in the range:

```bash
python main.py --mode replay --trajectory-dir trajectory_logs --replay-from-checkpoint 2700 --replay-to-checkpoint 3000 --replay-all-episodes
```

## Useful flags

- --mode train|play|replay
- --model-path PATH
- --episodes N
- --max-steps N
- --render and --no-render
- --render-every N
- --seconds-per-15-frames X
- --log-metrics --metrics-format tensorboard|jsonl|both --metrics-log-step N --metrics-window N --metrics-dir DIR
- --log-trajectories --trajectory-checkpoint-every N --trajectory-episodes N --trajectory-format json|jsonl --trajectory-dir DIR
- --trajectory-file PATH --trajectory-episode N --replay-all-episodes
- --replay-from-checkpoint N --replay-to-checkpoint N