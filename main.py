import argparse
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from environment import Environment, Renderer
from policy import MAPPOPolicy
from runtime_utils import compute_render_fps, select_device
from training import run_training
from trajectory_utils import (
    discover_trajectory_files,
    load_trajectory_episode,
    replay_trajectory,
    run_episode_with_policy,
    write_checkpoint_trajectories,
)


MAX_FRAMES_PER_ROUND = 150


def parse_args():
    parser = argparse.ArgumentParser(description="Train and run MAPPO policy for Patintero 1v1")
    parser.add_argument("--mode", choices=["train", "play", "replay"], default="train")
    parser.add_argument("--episodes", type=int, default=1500, help="Training episodes for MAPPO")
    parser.add_argument("--max-steps", type=int, default=MAX_FRAMES_PER_ROUND, help="Max steps per episode")
    parser.add_argument("--model-path", type=str, default="mappo_patintero.pt")
    parser.add_argument("--play-episodes", type=int, default=5, help="Evaluation episodes in play mode")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.set_defaults(render=True)
    parser.add_argument("--render", dest="render", action="store_true", help="Enable rendering")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Disable rendering for faster training")
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="When rendering is enabled, render every N training episodes (1 renders all episodes)",
    )
    parser.add_argument(
        "--seconds-per-15-frames",
        type=float,
        default=0.3,
        help="Render pacing: how many seconds 15 frames should take (0.5 => 30 FPS, 1.0 => 15 FPS)",
    )
    parser.add_argument("--log-trajectories", action="store_true", help="Log per-frame trajectories at periodic checkpoints during training")
    parser.add_argument("--trajectory-checkpoint-every", type=int, default=500, help="Log trajectories every N training episodes")
    parser.add_argument("--trajectory-episodes", type=int, default=5, help="Number of evaluation episodes to store per trajectory checkpoint")
    parser.add_argument("--trajectory-dir", type=str, default="trajectory_logs", help="Directory for saved checkpoint trajectory JSON files")
    parser.add_argument(
        "--trajectory-format",
        choices=["json", "jsonl"],
        default="json",
        help="Trajectory export format: json for readable files, jsonl for compact line-oriented logs",
    )
    parser.add_argument("--trajectory-file", type=str, default=None, help="Trajectory file (.json or .jsonl) to replay")
    parser.add_argument("--trajectory-episode", type=int, default=1, help="Evaluation episode index inside the trajectory file to replay")
    parser.add_argument("--replay-from-checkpoint", type=int, default=None, help="Replay all checkpoint files from this checkpoint number onward")
    parser.add_argument("--replay-to-checkpoint", type=int, default=None, help="Optional upper bound for batch replay checkpoint number")
    parser.add_argument("--log-metrics", action="store_true", help="Log training metrics for live monitoring")
    parser.add_argument("--metrics-log-step", type=int, default=1, help="Log metrics every N training episodes")
    parser.add_argument("--metrics-window", type=int, default=100, help="Rolling window size for win-rate and episode-length metrics")
    parser.add_argument(
        "--metrics-format",
        choices=["tensorboard", "jsonl", "both"],
        default="tensorboard",
        help="Metrics output format for training logs",
    )
    parser.add_argument("--metrics-dir", type=str, default="metrics_logs", help="Directory for metrics output files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.render_every <= 0:
        raise ValueError("--render-every must be >= 1")
    if args.trajectory_checkpoint_every <= 0:
        raise ValueError("--trajectory-checkpoint-every must be >= 1")
    if args.trajectory_episodes <= 0:
        raise ValueError("--trajectory-episodes must be >= 1")
    if args.trajectory_episode <= 0:
        raise ValueError("--trajectory-episode must be >= 1")
    if args.metrics_log_step <= 0:
        raise ValueError("--metrics-log-step must be >= 1")
    if args.metrics_window <= 0:
        raise ValueError("--metrics-window must be >= 1")
    if args.mode == "replay" and not args.trajectory_file and args.replay_from_checkpoint is None:
        raise ValueError("For replay mode, provide --trajectory-file or --replay-from-checkpoint")
    if args.replay_from_checkpoint is not None and args.replay_from_checkpoint <= 0:
        raise ValueError("--replay-from-checkpoint must be >= 1")
    if args.replay_to_checkpoint is not None and args.replay_to_checkpoint <= 0:
        raise ValueError("--replay-to-checkpoint must be >= 1")
    if (
        args.replay_from_checkpoint is not None
        and args.replay_to_checkpoint is not None
        and args.replay_to_checkpoint < args.replay_from_checkpoint
    ):
        raise ValueError("--replay-to-checkpoint must be >= --replay-from-checkpoint")

    env = Environment()
    render_fps = compute_render_fps(args.seconds_per_15_frames)
    renderer = None
    if args.render or args.mode == "replay":
        renderer = Renderer(env, fps=render_fps)
        renderer.running = True

    device = select_device(args.device)
    policy = MAPPOPolicy(device=device)
    checkpoint_state = {}
    trained_episodes = 0
    metrics_writer = None
    metrics_path = os.path.join(args.metrics_dir, "training_metrics.jsonl")

    if args.mode == "replay":
        replay_files = []
        if args.replay_from_checkpoint is not None:
            replay_files = discover_trajectory_files(
                args.trajectory_dir,
                from_checkpoint=args.replay_from_checkpoint,
                to_checkpoint=args.replay_to_checkpoint,
            )
            if not replay_files:
                raise FileNotFoundError(
                    f"No trajectory files found in {args.trajectory_dir} for checkpoint range "
                    f"[{args.replay_from_checkpoint}, {args.replay_to_checkpoint}]"
                )
        else:
            replay_files = [args.trajectory_file]

        print(f"Replaying {len(replay_files)} trajectory file(s) in one session")
        for path in replay_files:
            trajectory_data = load_trajectory_episode(path, args.trajectory_episode)
            print(f"Loaded trajectory from: {path}")
            stopped = replay_trajectory(env, renderer, trajectory_data)
            if stopped:
                break

        if renderer is not None:
            renderer.quit()
        print("Game closed.")
        raise SystemExit(0)

    if args.log_metrics and args.metrics_format in {"tensorboard", "both"}:
        if SummaryWriter is None:
            print("TensorBoard logging requested but tensorboard is not installed. Falling back to JSONL metrics only.")
            if args.metrics_format == "tensorboard":
                args.metrics_format = "jsonl"
            else:
                args.metrics_format = "jsonl"
        else:
            os.makedirs(args.metrics_dir, exist_ok=True)
            metrics_writer = SummaryWriter(log_dir=args.metrics_dir)

    if os.path.exists(args.model_path):
        checkpoint_state = policy.load(args.model_path)
        print(f"Loaded model from: {args.model_path}")

        env.attacker_score = int(checkpoint_state.get("attacker_score", env.attacker_score))
        env.defender_score = int(checkpoint_state.get("defender_score", env.defender_score))
        env.episode_number = int(checkpoint_state.get("episode_number", env.episode_number))
        trained_episodes = int(checkpoint_state.get("trained_episodes", 0))
    elif args.mode == "play":
        raise FileNotFoundError(f"Model not found at {args.model_path}. Train first or provide a valid path.")

    if args.render:
        print(f"Render speed: 15 frames per {args.seconds_per_15_frames:.3f}s ({render_fps:.2f} FPS)")
        print(f"Training render cadence: every {args.render_every} episode(s)")
    else:
        print("Rendering disabled (--no-render).")

    if args.log_metrics:
        print(
            f"Metrics logging enabled: format={args.metrics_format}, step={args.metrics_log_step}, "
            f"window={args.metrics_window}, dir={args.metrics_dir}"
        )

    if args.mode == "train":
        run_training(args, env, renderer, policy, trained_episodes, metrics_writer, metrics_path)

    episodes_to_play = args.play_episodes if args.mode == "play" else 3
    played = 0

    while played < episodes_to_play:
        a_ret, d_ret, stopped = run_episode_with_policy(
            env,
            policy,
            renderer,
            deterministic=True,
            max_steps=args.max_steps,
        )
        print(f"Episode {played + 1}: attacker_return={a_ret:+.3f}, defender_return={d_ret:+.3f}")
        played += 1

        if stopped:
            break

    if renderer is not None:
        renderer.quit()
    if metrics_writer is not None:
        metrics_writer.close()

    print("Game closed.") 