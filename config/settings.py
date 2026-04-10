"""
Centralized project settings via Pydantic.

Reads from .env file (or environment variables).
Usage anywhere in the codebase:

    from config.settings import settings

    model = load_data(settings.data_dir)
    wandb.init(project=settings.wandb_project)
"""
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Датасет ───────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")

    # ── Weights & Biases ──────────────────────────────────────────────────────
    wandb_api_key: str = ""
    wandb_project: str = "hateful-memes-clf"

    # ── Обучение ──────────────────────────────────────────────────────────────
    device: str = "cuda"
    num_workers: int = 4

    # ── Remote машина ─────────────────────────────────────────────────────────
    remote_user: str = "root"
    remote_ip: str = ""
    remote_port: int = 22
    remote_path: str = "~/hateful_memes"

    # ── Локальные пути ────────────────────────────────────────────────────────
    local_results_dir: Path = Path("results")

    # ── Валидация ─────────────────────────────────────────────────────────────
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        import torch
        if v == "cuda" and not torch.cuda.is_available():
            print("[settings] CUDA not available — falling back to CPU")
            return "cpu"
        return v

    @field_validator("data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v) -> Path:
        return Path(v)

    @field_validator("local_results_dir", mode="before")
    @classmethod
    def resolve_results_dir(cls, v) -> Path:
        return Path(v)


# Singleton — импортируй везде этот объект
settings = Settings()
