from collections import deque
from datetime import datetime

from metrics_utils import compute_rolling_metrics, get_terminal_outcome, log_training_metrics
from trajectory_utils import write_checkpoint_trajectories


def run_training(args, env, renderer, policy, trained_episodes, metrics_writer, metrics_path):
    rolling_history = deque(maxlen=args.metrics_window)

    if trained_episodes > 0:
        print(
            f"Resuming training from checkpoint: trained_episodes={trained_episodes}, "
            f"episode_number={env.episode_number}, attacker_score={env.attacker_score}, defender_score={env.defender_score}"
        )

    print(f"Training MAPPO on device={policy.device} for {args.episodes} additional episodes...")
    completed_in_run = 0

    for ep in range(1, args.episodes + 1):
        global_ep = trained_episodes + ep
        episode_renderer = None
        if renderer is not None and (global_ep % args.render_every == 0):
            episode_renderer = renderer

        buffer, rollout_stats = policy.collect_episode(
            env,
            max_steps=args.max_steps,
            deterministic=False,
            renderer=episode_renderer,
        )
        policy.update(buffer, update_epochs=10, minibatch_size=64)
        completed_in_run = ep

        winner, terminal_reason = get_terminal_outcome(env)
        rolling_history.append(
            {
                "winner": winner,
                "terminal_reason": terminal_reason,
                "episode_length": int(rollout_stats["episode_len"]),
            }
        )

        if args.log_metrics and (global_ep % args.metrics_log_step == 0):
            rolling_metrics = compute_rolling_metrics(rolling_history)
            metrics_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "episode": int(global_ep),
                "winner": winner,
                "terminal_reason": terminal_reason,
                "attacker_score": int(env.attacker_score),
                "defender_score": int(env.defender_score),
                "attacker_return": float(rollout_stats["attacker_return"]),
                "defender_return": float(rollout_stats["defender_return"]),
                "episode_length": int(rollout_stats["episode_len"]),
                **rolling_metrics,
            }
            log_training_metrics(metrics_record, args.metrics_format, metrics_writer, metrics_path)

        if global_ep % 25 == 0 or global_ep == 1:
            print(
                f"Episode {global_ep:4d} | "
                f"Attacker return: {rollout_stats['attacker_return']:+.3f} | "
                f"Defender return: {rollout_stats['defender_return']:+.3f} | "
                f"Len: {int(rollout_stats['episode_len'])}"
            )

        if renderer is not None and not renderer.running:
            print("Training interrupted by window close.")
            break

        if args.log_trajectories and (global_ep % args.trajectory_checkpoint_every == 0):
            output_path = write_checkpoint_trajectories(
                env,
                policy,
                checkpoint_episode=global_ep,
                num_episodes=args.trajectory_episodes,
                max_steps=args.max_steps,
                output_dir=args.trajectory_dir,
                output_format=args.trajectory_format,
            )
            print(f"Saved trajectory checkpoint: {output_path}")

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