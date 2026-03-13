import glob
import json
import os
import re
from datetime import datetime

import numpy as np


def run_episode_with_policy(
    env,
    policy,
    renderer=None,
    deterministic=True,
    max_steps=150,
    collect_trace=False,
):
    obs = env.reset()
    policy.reset_history()
    attacker_total = 0.0
    defender_total = 0.0
    frames = []
    terminal_reason = None

    for t in range(max_steps):
        attacker_position = env.attacker.position.copy()
        defender_position = env.defender.position.copy()
        attacker_forward = env.attacker.forward.copy()

        attacker_action, defender_action = policy.act(obs, deterministic=deterministic)
        obs, (attacker_reward, defender_reward), done = env.step(attacker_action, defender_action)

        step_info = getattr(env, "last_step_info", {})
        terminal_reason = step_info.get("terminal_reason", terminal_reason)

        attacker_total += attacker_reward
        defender_total += defender_reward

        if collect_trace:
            frames.append(
                {
                    "frame_idx": int(t),
                    "attacker_x": float(attacker_position[0]),
                    "attacker_y": float(attacker_position[1]),
                    "attacker_forward_x": float(attacker_forward[0]),
                    "attacker_forward_y": float(attacker_forward[1]),
                    "defender_x": float(defender_position[0]),
                    "defender_y": float(defender_position[1]),
                    "attacker_action": float(attacker_action),
                    "defender_action": int(defender_action),
                    "attacker_reward": float(attacker_reward),
                    "defender_reward": float(defender_reward),
                    "done": bool(done),
                    "terminal_reason": step_info.get("terminal_reason"),
                }
            )

        if renderer is not None:
            renderer.render()
            if not renderer.running:
                if collect_trace:
                    return attacker_total, defender_total, True, frames, terminal_reason
                return attacker_total, defender_total, True

        if done:
            break

    if collect_trace:
        return attacker_total, defender_total, False, frames, terminal_reason
    return attacker_total, defender_total, False


def write_checkpoint_trajectories(env, policy, checkpoint_episode, num_episodes, max_steps, output_dir, output_format):
    os.makedirs(output_dir, exist_ok=True)

    saved_scores = (int(env.attacker_score), int(env.defender_score), int(env.episode_number))
    training_state = {
        "attacker_score": int(env.attacker_score),
        "defender_score": int(env.defender_score),
        "episode_number": int(env.episode_number),
    }

    episodes_payload = []
    for eval_idx in range(1, num_episodes + 1):
        a_ret, d_ret, stopped, frames, terminal_reason = run_episode_with_policy(
            env,
            policy,
            renderer=None,
            deterministic=True,
            max_steps=max_steps,
            collect_trace=True,
        )
        episodes_payload.append(
            {
                "evaluation_episode": int(eval_idx),
                "attacker_return": float(a_ret),
                "defender_return": float(d_ret),
                "stopped": bool(stopped),
                "terminal_reason": terminal_reason,
                "num_frames": int(len(frames)),
                "frames": frames,
            }
        )
        if stopped:
            break

    env.attacker_score, env.defender_score, env.episode_number = saved_scores
    env.reset()

    environment_metadata = {
        "width": int(env.width),
        "height": int(env.height),
        "fps": int(env.fps),
        "time_limit_seconds": int(env.time_limit_seconds),
        "max_frames": int(env.max_frames),
    }
    created_at = datetime.utcnow().isoformat() + "Z"

    if output_format == "jsonl":
        output_path = os.path.join(output_dir, f"checkpoint_{checkpoint_episode:07d}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            metadata_record = {
                "record_type": "metadata",
                "created_at": created_at,
                "checkpoint_episode": int(checkpoint_episode),
                "environment": environment_metadata,
                "training_state": training_state,
            }
            f.write(json.dumps(metadata_record, separators=(",", ":")) + "\n")

            for episode_payload in episodes_payload:
                episode_summary = {
                    "record_type": "episode",
                    "checkpoint_episode": int(checkpoint_episode),
                    "evaluation_episode": int(episode_payload["evaluation_episode"]),
                    "attacker_return": float(episode_payload["attacker_return"]),
                    "defender_return": float(episode_payload["defender_return"]),
                    "stopped": bool(episode_payload["stopped"]),
                    "terminal_reason": episode_payload["terminal_reason"],
                    "num_frames": int(episode_payload["num_frames"]),
                }
                f.write(json.dumps(episode_summary, separators=(",", ":")) + "\n")

                for frame in episode_payload["frames"]:
                    frame_record = {
                        "record_type": "frame",
                        "checkpoint_episode": int(checkpoint_episode),
                        "evaluation_episode": int(episode_payload["evaluation_episode"]),
                        **frame,
                    }
                    f.write(json.dumps(frame_record, separators=(",", ":")) + "\n")
    else:
        payload = {
            "created_at": created_at,
            "checkpoint_episode": int(checkpoint_episode),
            "environment": environment_metadata,
            "training_state": training_state,
            "episodes": episodes_payload,
        }

        output_path = os.path.join(output_dir, f"checkpoint_{checkpoint_episode:07d}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return output_path


def load_trajectory_episode(trajectory_file, trajectory_episode):
    if trajectory_file.endswith(".json"):
        with open(trajectory_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for episode_payload in payload.get("episodes", []):
            if int(episode_payload.get("evaluation_episode", -1)) == trajectory_episode:
                return {
                    "checkpoint_episode": int(payload.get("checkpoint_episode", 0)),
                    "environment": payload.get("environment", {}),
                    "training_state": payload.get("training_state", {}),
                    "episode": episode_payload,
                }

        raise ValueError(f"Trajectory episode {trajectory_episode} not found in {trajectory_file}")

    if trajectory_file.endswith(".jsonl"):
        metadata = None
        episode_summary = None
        frames = []

        with open(trajectory_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_type = record.get("record_type")

                if record_type == "metadata":
                    metadata = record
                elif record_type == "episode" and int(record.get("evaluation_episode", -1)) == trajectory_episode:
                    episode_summary = record
                elif record_type == "frame" and int(record.get("evaluation_episode", -1)) == trajectory_episode:
                    frame = dict(record)
                    frame.pop("record_type", None)
                    frame.pop("checkpoint_episode", None)
                    frame.pop("evaluation_episode", None)
                    frames.append(frame)

        if episode_summary is None:
            raise ValueError(f"Trajectory episode {trajectory_episode} not found in {trajectory_file}")

        return {
            "checkpoint_episode": int((metadata or {}).get("checkpoint_episode", 0)),
            "environment": (metadata or {}).get("environment", {}),
            "training_state": (metadata or {}).get("training_state", {}),
            "episode": {
                "evaluation_episode": trajectory_episode,
                "attacker_return": float(episode_summary.get("attacker_return", 0.0)),
                "defender_return": float(episode_summary.get("defender_return", 0.0)),
                "stopped": bool(episode_summary.get("stopped", False)),
                "terminal_reason": episode_summary.get("terminal_reason"),
                "num_frames": int(episode_summary.get("num_frames", len(frames))),
                "frames": frames,
            },
        }

    raise ValueError("Trajectory file must end with .json or .jsonl")


def list_trajectory_episode_indices(trajectory_file):
    if trajectory_file.endswith(".json"):
        with open(trajectory_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        indices = [int(ep.get("evaluation_episode")) for ep in payload.get("episodes", []) if "evaluation_episode" in ep]
        return sorted(set(indices))

    if trajectory_file.endswith(".jsonl"):
        indices = set()
        with open(trajectory_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("record_type") == "episode" and "evaluation_episode" in record:
                    indices.add(int(record["evaluation_episode"]))
        return sorted(indices)

    raise ValueError("Trajectory file must end with .json or .jsonl")


def replay_trajectory(env, renderer, trajectory_data, hydrate_from_training_state=True):
    episode_payload = trajectory_data["episode"]
    checkpoint_episode = int(trajectory_data.get("checkpoint_episode", 0))
    training_state = trajectory_data.get("training_state", {})
    frames = episode_payload.get("frames", [])
    evaluation_episode = int(episode_payload.get("evaluation_episode", 1))

    if hydrate_from_training_state:
        env.attacker_score = int(training_state.get("attacker_score", 0))
        env.defender_score = int(training_state.get("defender_score", 0))
        base_episode_number = int(training_state.get("episode_number", checkpoint_episode or 1))
        # Evaluation episodes are generated sequentially after checkpoint state.
        env.episode_number = base_episode_number + max(evaluation_episode - 1, 0)
    env.done = False
    env.frame_count = 0

    for frame in frames:
        env.attacker.position = np.array([frame["attacker_x"], frame["attacker_y"]], dtype=np.float32)
        env.attacker.forward = np.array(
            [frame.get("attacker_forward_x", 0.0), frame.get("attacker_forward_y", -1.0)], dtype=np.float32
        )
        env.defender.position = np.array([frame["defender_x"], frame["defender_y"]], dtype=np.float32)
        env.frame_count = int(frame.get("frame_idx", 0)) + 1

        renderer.render()
        if not renderer.running:
            return True

    print(
        f"Replayed checkpoint {checkpoint_episode}, evaluation episode {evaluation_episode} | "
        f"terminal_reason={episode_payload.get('terminal_reason')} | "
        f"attacker_return={episode_payload.get('attacker_return', 0.0):+.3f} | "
        f"defender_return={episode_payload.get('defender_return', 0.0):+.3f}"
    )
    return False


def _extract_checkpoint_number(path):
    base = os.path.basename(path)
    match = re.search(r"checkpoint_(\d+)", base)
    if not match:
        return None
    return int(match.group(1))


def discover_trajectory_files(trajectory_dir, from_checkpoint, to_checkpoint=None):
    candidates = glob.glob(os.path.join(trajectory_dir, "checkpoint_*.json")) + glob.glob(
        os.path.join(trajectory_dir, "checkpoint_*.jsonl")
    )

    filtered = []
    for path in candidates:
        checkpoint_num = _extract_checkpoint_number(path)
        if checkpoint_num is None:
            continue
        if checkpoint_num < from_checkpoint:
            continue
        if to_checkpoint is not None and checkpoint_num > to_checkpoint:
            continue
        filtered.append((checkpoint_num, path))

    filtered.sort(key=lambda item: item[0])
    return [path for _, path in filtered]