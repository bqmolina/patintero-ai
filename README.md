# patintero-ai

Patintero 5v5 multi-agent reinforcement learning with MAPPO (PyTorch).

The board follows the Patintero layout with 4 crosswise lines and 1 lengthwise line.
Four defenders move crosswise on the horizontal lanes, and one defender moves along the lengthwise lane between the outer crosswise lines.

## Game mechanics

- Each round starts with 5 attackers and 5 defenders placed on the board.
- Attacker positions are randomized inside the starting area (below the last crosswise line).
- Attackers move with continuous angle actions and must first reach the return area (above the first crosswise line), then return to the starting area.
- Defenders move on fixed lanes: 4 defenders patrol the crosswise lines and 1 defender patrols the lengthwise line between the first and last crosswise lines.
- The game advances one frame at a time. Environment timeout is 900 steps unless changed via `--max-steps`.

## Win condition

- Attacker win: at least one attacker reaches the return area and then comes back to the starting area.
- Defender win: a defender tags an attacker.
- Defender win: an attacker illegally recrosses a previously crossed crosswise line before reaching the return area.
- Defender win by timeout: if no one wins before the 900-frame limit.
- After a win, the round resets immediately and the score is updated for the winning team.

## Reward policy

Rewards are computed per attacker and per defender every frame.

- Base living reward: each agent receives `-0.01` every step.
- Attacker progress shaping: each attacker gets
	`+0.4 * (previous_target_distance - current_target_distance)`.
	Before reaching the return area, the target is the return-area boundary; after reaching it, the target switches to the starting-area boundary.
- Return-area milestone reward: an attacker gets `+0.5` on the first step it reaches the return area.
- Attacker distance reward: each attacker gets `+0.1 * (normalized_distance_to_nearest_defender)` every step, rewarding them for maintaining distance from defenders.
- Return-area lingering penalty: an attacker in the return area accumulates a frame counter; each step they linger, they receive `-0.05 * (lingering_frames / 100.0)`.
- Defender tracking shaping: each defender gets
	`+0.2 * ((previous_nearest_attacker_distance - current_nearest_attacker_distance) / board_width)`.
	This rewards closing distance to nearby attackers.

Terminal rewards:

- `tag` (defender catches attacker):
	all attackers `-2.0`, all defenders `+1.0`, plus tagged attacker `-1.0` and tagging defender `+1.0`.
- `return` (attacker reaches return area, then returns to starting area):
	all attackers `+3.0`, all defenders `-1.0`, plus successful attacker `+2.0`.
- `invalid_recross` (illegal recross on outbound or return):
	all attackers `-1.0`, all defenders `+1.0`.
- `timeout` (frame limit reached):
	all attackers `-1.0`, all defenders `+1.0`.

During training logs, attacker and defender episode returns are team sums across all 5 agents.

## What the agents see

The agents do not see raw pixels. They receive structured observations from the environment.

- Attacker input: each attacker sees its own normalized position and direction, the relative positions of all defenders, the relative positions of the other attackers, the board line layout, and a small score/episode context.
- Defender input: each defender sees its own normalized position, whether it is a crosswise or lengthwise defender, its lane position, the relative positions of all attackers, the relative positions and lane roles of the other defenders, the board line layout, and a small score/episode context.
- Temporal input: the policy frame-stacks the current observation with previous frames, so each agent can infer motion over time.

## What the agents learn

This project trains two teams of five agents each:

- Attackers: a shared continuous policy for 5 agents that outputs a movement angle (mapped to 0 to 360 degrees).
- Defenders: a shared discrete policy for 5 agents that chooses one of three actions: move left, stay, move right.

Both agents are trained with MAPPO (Multi-Agent PPO):

- Decentralized actors: each team shares one actor network, and each agent acts from its own local observation.
- Centralized critics: both critics are conditioned on the joint state of the board and both teams for more stable advantage estimates.
- PPO objective: clipped policy-ratio loss + value loss + entropy bonus.
- Temporal context: frame stacking (current frame + previous frames) helps each policy infer motion.

## Model architecture

- Agent backbone: MLP (2 hidden layers, ReLU activations).
- Attacker actor head: one shared Gaussian policy for the 5 attackers, squashed and mapped to angle space.
- Defender actor head: one shared categorical policy for the 5 defenders.
- Critics: value MLPs over stacked joint-state features.

## Hyperparameters

Environment and game defaults:

- Teams: 5 attackers, 5 defenders.
- FPS: 15.
- Environment timeout: `time_limit_seconds=60` -> `max_frames=900`.
- Reward shaping scales: attacker progress `0.4`, attacker distance `0.1`, return-area lingering penalty `0.05`, defender tracking `0.2`.

MAPPO and optimization defaults:

- Discount factor (`gamma`): `0.99`.
- GAE lambda (`gae_lambda`): `0.95`.
- PPO clip epsilon (`clip_eps`): `0.2`.
- Value loss coefficient (`vf_coef`): `0.5`.
- Entropy coefficient (`ent_coef`): `0.01`.
- Actor learning rate (`actor_lr`): `3e-4`.
- Critic learning rate (`critic_lr`): `1e-3`.
- Frame stack: `5`.
- Minibatch size: `256`.
- PPO update epochs per episode (`--update-epochs`): `10`.

Common training/runtime knobs (CLI):

- `--episodes`: number of additional training episodes (default `1500`).
- `--max-steps`: max steps per episode (default `900`).
- `--update-epochs`: PPO optimization passes per collected episode (default `10`).
- `--render-every`: render cadence during training (default `1`).
- `--metrics-log-step`: metric logging interval (default `1`).
- `--metrics-window`: rolling metrics window (default `100`).

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

### Replay one episode from one checkpoint file

```bash
python main.py --mode replay --trajectory-file trajectory_logs/checkpoint_0002700.jsonl --trajectory-episode 1
```

### Replay all episodes in one checkpoint file

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

Replay every episode for each file in the range:

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