import json
import os


def get_terminal_outcome(env):
    terminal_reason = env.last_step_info.get("terminal_reason")
    if terminal_reason == "return":
        winner = "attacker"
    elif terminal_reason in {"tag", "timeout", "invalid_recross"}:
        winner = "defender"
    else:
        winner = None
    return winner, terminal_reason


def compute_rolling_metrics(history_window):
    total = len(history_window)
    if total == 0:
        return {
            "rolling_attacker_win_rate": 0.0,
            "rolling_defender_win_rate": 0.0,
            "rolling_timeout_rate": 0.0,
            "rolling_avg_episode_length": 0.0,
        }

    attacker_wins = sum(1 for item in history_window if item["winner"] == "attacker")
    defender_wins = sum(1 for item in history_window if item["winner"] == "defender")
    timeout_wins = sum(1 for item in history_window if item["terminal_reason"] == "timeout")
    avg_episode_length = sum(item["episode_length"] for item in history_window) / total

    return {
        "rolling_attacker_win_rate": attacker_wins / total,
        "rolling_defender_win_rate": defender_wins / total,
        "rolling_timeout_rate": timeout_wins / total,
        "rolling_avg_episode_length": avg_episode_length,
    }


def append_metrics_jsonl(metrics_path, metrics_record):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics_record, separators=(",", ":")) + "\n")


def log_training_metrics(metrics_record, metrics_format, writer, metrics_path):
    if metrics_format in {"jsonl", "both"}:
        append_metrics_jsonl(metrics_path, metrics_record)

    if metrics_format in {"tensorboard", "both"} and writer is not None:
        step = int(metrics_record["episode"])
        writer.add_scalar("scoreboard/attacker_score", metrics_record["attacker_score"], step)
        writer.add_scalar("scoreboard/defender_score", metrics_record["defender_score"], step)
        writer.add_scalar("returns/attacker_return", metrics_record["attacker_return"], step)
        writer.add_scalar("returns/defender_return", metrics_record["defender_return"], step)
        writer.add_scalar("episode/length", metrics_record["episode_length"], step)
        writer.add_scalar("rolling/attacker_win_rate", metrics_record["rolling_attacker_win_rate"], step)
        writer.add_scalar("rolling/defender_win_rate", metrics_record["rolling_defender_win_rate"], step)
        writer.add_scalar("rolling/timeout_rate", metrics_record["rolling_timeout_rate"], step)
        writer.add_scalar("rolling/avg_episode_length", metrics_record["rolling_avg_episode_length"], step)