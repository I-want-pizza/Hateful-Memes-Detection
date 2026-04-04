from pathlib import Path

import torch


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, cfg: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics":         metrics,
            "config":          cfg,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, device: str = "cpu"):
    """
    Load model (and optionally optimizer) state from a checkpoint file.
    Returns the checkpoint dict for further inspection.
    """
    checkpoint = torch.load(path, map_location=device)

    # Handle plain state-dict files for backward compatibility
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
    else:
        model.load_state_dict(checkpoint)

    return checkpoint
