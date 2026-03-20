import os
from pathlib import Path
from typing import Optional
import warnings


def load_env_file(env_path: str = ".env") -> None:
    env_file = Path(env_path)
    if not env_file.exists():
        return
    with env_file.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


class Config:
    def __init__(self) -> None:
        load_env_file()
        self._openai_api_key: Optional[str] = None
        self._hf_token: Optional[str] = None
        self._wandb_project: Optional[str] = None

    @property
    def openai_api_key(self) -> str:
        if self._openai_api_key is None:
            self._openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not self._openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Set it in .env or export it before running judge-based evaluation."
                )
        return self._openai_api_key

    @property
    def hf_token(self) -> str:
        if self._hf_token is None:
            self._hf_token = os.environ.get("HF_TOKEN")
            if not self._hf_token:
                raise ValueError(
                    "HF_TOKEN not found in environment variables. "
                    "Set it in .env or export it before loading gated Hugging Face models."
                )
        return self._hf_token

    @property
    def wandb_project(self) -> str:
        if self._wandb_project is None:
            self._wandb_project = os.environ.get("WANDB_PROJECT", "emergent-misalignment-followup-gpu")
        return self._wandb_project

    def setup_environment(
        self,
        require_openai: bool = True,
        require_hf: bool = True,
        require_wandb: bool = False,
    ) -> None:
        if require_openai:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if require_hf:
            os.environ["HF_TOKEN"] = self.hf_token
        if require_wandb:
            os.environ["WANDB_PROJECT"] = self.wandb_project

    def validate_credentials(
        self,
        require_openai: bool = True,
        require_hf: bool = True,
    ) -> bool:
        try:
            if require_openai:
                _ = self.openai_api_key
            if require_hf:
                _ = self.hf_token
            return True
        except ValueError as exc:
            warnings.warn(f"Credential validation failed: {exc}")
            return False


config = Config()


def setup_credentials(
    require_openai: bool = True,
    require_hf: bool = True,
    require_wandb: bool = False,
) -> Config:
    config.setup_environment(
        require_openai=require_openai,
        require_hf=require_hf,
        require_wandb=require_wandb,
    )
    if not config.validate_credentials(
        require_openai=require_openai,
        require_hf=require_hf,
    ):
        raise RuntimeError("Failed to validate required credentials")
    return config
