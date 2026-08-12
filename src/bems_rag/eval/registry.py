"""Model registry: versioned RAG configs with champion/challenger aliases.

Wraps the MLflow model registry so each evaluated RAG configuration becomes a numbered
version, and promotion is an alias flip (champion / challenger) rather than a code
change. Backed by a local MLflow store here; the same API maps onto a remote registry
in production.

A "model" in this project is a RAG configuration (embedding backend, k, chunking) plus
its eval metrics -- not a trained network. What is versioned and promoted is that
configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "bems-rag"


@dataclass
class RegisteredVersion:
    version: str
    metrics: dict[str, float]
    aliases: list[str] = field(default_factory=list)


class Registry:
    """Thin wrapper over the MLflow model registry for champion/challenger flow."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.client = MlflowClient()

    def register(self, run_id: str, metrics: dict[str, float]) -> str:
        """Register the model produced by an MLflow run; return the new version id."""
        # Ensure the registered model exists.
        try:
            self.client.create_registered_model(self.model_name)
        except mlflow.exceptions.MlflowException:
            pass  # already exists
        mv = self.client.create_model_version(
            name=self.model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        # Persist eval metrics as tags on the version for later comparison.
        for k, v in metrics.items():
            self.client.set_model_version_tag(self.model_name, mv.version, k, str(v))
        return mv.version

    def set_alias(self, alias: str, version: str) -> None:
        """Point an alias (e.g. 'champion' / 'challenger') at a version."""
        self.client.set_registered_model_alias(self.model_name, alias, version)

    def get_alias_version(self, alias: str) -> str | None:
        try:
            mv = self.client.get_model_version_by_alias(self.model_name, alias)
            return mv.version
        except mlflow.exceptions.MlflowException:
            return None

    def promote(self, challenger_version: str) -> str | None:
        """Make the challenger the champion; keep the old champion as previous_champion.

        Returns the version that was demoted (the old champion), or None if there was
        no champion yet.
        """
        old = self.get_alias_version("champion")
        if old is not None:
            self.set_alias("previous_champion", old)
        self.set_alias("champion", challenger_version)
        return old

    def rollback(self) -> str | None:
        """Flip champion back to previous_champion. Returns the restored version."""
        prev = self.get_alias_version("previous_champion")
        if prev is None:
            return None
        current = self.get_alias_version("champion")
        self.set_alias("champion", prev)
        if current is not None:
            self.set_alias("previous_champion", current)
        return prev
