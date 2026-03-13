import torch


def compute_render_fps(seconds_per_15_frames: float) -> float:
    if seconds_per_15_frames <= 0:
        raise ValueError("--seconds-per-15-frames must be > 0")
    return 15.0 / seconds_per_15_frames


def select_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        try:
            x = torch.tensor([1.0], device="cuda")
            _ = x * 2.0
            return "cuda"
        except RuntimeError as err:
            print(f"CUDA requested but failed to initialize ({err}). Falling back to CPU.")
            return "cpu"

    if torch.cuda.is_available():
        try:
            x = torch.tensor([1.0], device="cuda")
            _ = x * 2.0
            return "cuda"
        except RuntimeError as err:
            print(f"CUDA detected but unusable ({err}). Using CPU instead.")
            return "cpu"
    return "cpu"