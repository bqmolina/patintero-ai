from collections import deque
from datetime import datetime

import numpy as np

from metrics_utils import compute_rolling_metrics, get_terminal_outcome, log_training_metrics
from policy import RolloutBuffer
from trajectory_utils import write_checkpoint_trajectories


def _merge_buffers(buffers):
    """Merge multiple RolloutBuffer objects into one."""
    merged = RolloutBuffer()
    for buffer in buffers:
        merged.attacker_obs.extend(buffer.attacker_obs)
        merged.defender_obs.extend(buffer.defender_obs)
        merged.attacker_state_obs.extend(buffer.attacker_state_obs)
        merged.defender_state_obs.extend(buffer.defender_state_obs)
        merged.attacker_action.extend(buffer.attacker_action)
        merged.defender_action_idx.extend(buffer.defender_action_idx)
        merged.attacker_logp.extend(buffer.attacker_logp)
        merged.defender_logp.extend(buffer.defender_logp)
        merged.attacker_value.extend(buffer.attacker_value)
        merged.defender_value.extend(buffer.defender_value)
        merged.attacker_reward.extend(buffer.attacker_reward)
        merged.defender_reward.extend(buffer.defender_reward)
        merged.done.extend(buffer.done)
    return merged


def run_training(args, env, renderer, policy, trained_episodes, metrics_writer, metrics_path):
    rolling_history = deque(maxlen=args.metrics_window)
    accumulate_episodes = getattr(args, "accumulate_episodes", 1)
    minibatch_size = getattr(args, "minibatch_size", 256)

    if trained_episodes > 0:
        print(
            f"Resuming training from checkpoint: trained_episodes={trained_episodes}, "
            f"episode_number={env.episode_number}, attacker_score={env.attacker_score}, defender_score={env.defender_score}"
        )

    print(f"Training MAPPO on device={policy.device} for {args.episodes} additional episodes...")
    print(f"  Accumulate {accumulate_episodes} episode(s) before each update.")
    print(f"  Using minibatch size {minibatch_size}.")
    completed_in_run = 0

    ep = 1
    while ep <= args.episodes:
        global_ep = trained_episodes + ep
        accumulated_buffers = []
        accumulated_stats = []

        for accum_idx in range(accumulate_episodes):
            if ep > args.episodes:
                break

            episode_renderer = None
            if renderer is not None and (global_ep % args.render_every == 0):
                episode_renderer = renderer

            buffer, rollout_stats = policy.collect_episode(
                env,
                max_steps=args.max_steps,
                deterministic=False,
                renderer=episode_renderer,
            )
            accumulated_buffers.append(buffer)
            accumulated_stats.append(rollout_stats)
            completed_in_run = ep

            winner, terminal_reason = get_terminal_outcome(env)
            rolling_history.append(
                {
                    "winner": winner,
                    "terminal_reason": terminal_reason,
                    "episode_length": int(rollout_stats["episode_len"]),
                }
            )

            if renderer is not None and not renderer.running:
                print("Training interrupted by window close.")
                break

            ep += 1
            global_ep = trained_episodes + ep

        merged_buffer = _merge_buffers(accumulated_buffers)
        policy.update(merged_buffer, update_epochs=args.update_epochs, minibatch_size=minibatch_size)

        last_stats = accumulated_stats[-1] if accumulated_stats else {"attacker_return": 0.0, "defender_return": 0.0, "episode_len": 0}
        last_ep = global_ep - 1
        if last_ep % 25 == 0 or last_ep == 1:
            timestamp = datetime.now().isoformat(timespec="seconds")
            print(
                f"[{timestamp}] Episode {last_ep:4d} | "
                f"Attacker return: {last_stats['attacker_return']:+.3f} | "
                f"Defender return: {last_stats['defender_return']:+.3f} | "
                f"Len: {int(last_stats['episode_len'])}"
            )

        if args.log_metrics and (last_ep % args.metrics_log_step == 0):
            rolling_metrics = compute_rolling_metrics(rolling_history)
            last_winner, last_terminal_reason = get_terminal_outcome(env)
            metrics_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "episode": int(last_ep),
                "winner": last_winner,
                "terminal_reason": last_terminal_reason,
                "attacker_score": int(env.attacker_score),
                "defender_score": int(env.defender_score),
                "attacker_return": float(last_stats["attacker_return"]),
                "defender_return": float(last_stats["defender_return"]),
                "episode_length": int(last_stats["episode_len"]),
                **rolling_metrics,
            }
            log_training_metrics(metrics_record, args.metrics_format, metrics_writer, metrics_path)

        if args.log_trajectories and (last_ep % args.trajectory_checkpoint_every == 0):
            output_path = write_checkpoint_trajectories(
                env,
                policy,
                checkpoint_episode=last_ep,
                num_episodes=args.trajectory_episodes,
                max_steps=args.max_steps,
                output_dir=args.trajectory_dir,
                output_format=args.trajectory_format,
            )
            print(f"Saved trajectory checkpoint: {output_path}")

        if last_ep % args.autosave_every == 0:
            policy.save(
                args.model_path,
                extra_state={
                    "trained_episodes": last_ep,
                    "episode_number": int(env.episode_number),
                    "attacker_score": int(env.attacker_score),
                    "defender_score": int(env.defender_score),
                },
            )
            print(f"Autosaved model at episode {last_ep}: {args.model_path}")

        if renderer is not None and not renderer.running:
            print("Training interrupted by window close.")
            break

    total_trained_episodes = trained_episodes + completed_in_run
    policy.save(
        args.model_path,
        extra_state={
            "trained_episodes": total_trained_episodes,
            "episode_number": int(env.episode_number),
            "attacker_score": int(env.attacker_score),
            "defender_score": int(env.defender_score),
        },
    )
    print(f"Saved model to: {args.model_path}")

    return total_trained_episodes