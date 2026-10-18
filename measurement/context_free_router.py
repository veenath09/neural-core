
"""C3-10: Context-free, low-rank prompt router.

The router:
- Learns from single-prompt text and binary model outcomes.
- Uses TF-IDF, TruncatedSVD and logistic regression.
- Selects a model using predicted success and training-set costs.
- Does not inspect episode traces or test correctness labels.

This is a RouteLLM-inspired baseline, not an exact
implementation of RouteLLM matrix factorisation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer

from controller.contracts import Strategy, TaskRep


class ContextFreeRouter:
    """Train a context-free router and expose decide(TaskRep)."""

    def __init__(
        self,
        *,
        model_ids: Sequence[str],
        prompt_text_by_id: Mapping[str, str],
        model_costs: Mapping[str, float],
        vectorizer: TfidfVectorizer,
        reducer: TruncatedSVD,
        normalizer: Normalizer,
        predictors: Mapping[str, object],
        cost_ref_usd: float,
        default_lam: float = 0.25,
    ) -> None:
        self.model_ids = tuple(model_ids)

        self.prompt_text_by_id = {
            str(key): str(value)
            for key, value in prompt_text_by_id.items()
        }

        self.model_costs = {
            str(key): float(value)
            for key, value in model_costs.items()
        }

        self.vectorizer = vectorizer
        self.reducer = reducer
        self.normalizer = normalizer
        self.predictors = dict(predictors)

        self.cost_ref_usd = float(cost_ref_usd)
        self.default_lam = float(default_lam)

        if not self.model_ids:
            raise ValueError("Model pool cannot be empty")

        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("Model IDs must be unique")

        if not math.isfinite(self.cost_ref_usd) or self.cost_ref_usd <= 0:
            raise ValueError("Cost reference must be positive and finite")

        if not math.isfinite(self.default_lam) or self.default_lam < 0:
            raise ValueError("Lambda must be finite and nonnegative")

        for model_id in self.model_ids:
            if model_id not in self.model_costs:
                raise ValueError(f"Missing model cost: {model_id}")

            if model_id not in self.predictors:
                raise ValueError(f"Missing predictor: {model_id}")

            cost = self.model_costs[model_id]

            if not math.isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid model cost: {model_id}")

    @classmethod
    def fit(
        cls,
        train_frame: pd.DataFrame,
        prompt_text_by_id: Mapping[str, str],
        model_ids: Sequence[str],
        *,
        cost_ref_usd: float,
        n_components: int = 64,
        max_features: int = 20000,
        random_state: int = 1234,
        default_lam: float = 0.25,
    ) -> "ContextFreeRouter":
        """Fit using ONLY the caller-supplied training rows.

        The future training script must filter train_frame using
        data/clean/splits.parquet before calling this method.

        prompt_text_by_id may contain train/val/test TEXT
        for inference-time lookup, but must not contain
        test correctness labels.
        """

        if train_frame.empty:
            raise ValueError("Training frame is empty")

        if n_components < 1 or max_features < 2:
            raise ValueError("Invalid feature dimensions")

        model_ids = tuple(model_ids)

        if not model_ids or len(set(model_ids)) != len(model_ids):
            raise ValueError("Model IDs must be nonempty and unique")

        required = {"prompt_id"}

        for model_id in model_ids:
            required.add(f"{model_id}_correct")
            required.add(f"{model_id}_cost")

        missing = required - set(train_frame.columns)

        if missing:
            raise ValueError(
                f"Missing training columns: {sorted(missing)}"
            )

        prompt_ids = (
            train_frame["prompt_id"]
            .astype(str)
            .tolist()
        )

        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("Duplicate training prompt IDs")

        text_lookup = {
            str(key): str(value)
            for key, value in prompt_text_by_id.items()
        }

        missing_text = [
            prompt_id
            for prompt_id in prompt_ids
            if prompt_id not in text_lookup
            or not text_lookup[prompt_id].strip()
        ]

        if missing_text:
            raise ValueError(
                f"Missing prompt text for {len(missing_text)} "
                "training prompts"
            )

        texts = [
            text_lookup[prompt_id]
            for prompt_id in prompt_ids
        ]

        # TF-IDF vocabulary is fitted exclusively on train text.
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1 if len(texts) < 10 else 2,
            max_features=max_features,
            sublinear_tf=True,
        )

        sparse_features = vectorizer.fit_transform(texts)

        # Low-rank representation, fitted only on train text.
        rank = min(
            n_components,
            sparse_features.shape[0] - 1,
            sparse_features.shape[1] - 1,
        )

        if rank < 1:
            raise ValueError(
                "Insufficient training data for low-rank features"
            )

        reducer = TruncatedSVD(
            n_components=rank,
            random_state=random_state,
        )

        reduced_features = reducer.fit_transform(
            sparse_features
        )

        normalizer = Normalizer()

        features = normalizer.fit_transform(
            reduced_features
        )

        predictors = {}
        model_costs = {}

        for model_id in model_ids:
            correct_column = f"{model_id}_correct"
            cost_column = f"{model_id}_cost"

            labels = pd.to_numeric(
                train_frame[correct_column],
                errors="raise",
            ).to_numpy(dtype=float)

            costs = pd.to_numeric(
                train_frame[cost_column],
                errors="raise",
            ).to_numpy(dtype=float)

            if (
                not np.isfinite(labels).all()
                or not np.isin(labels, [0.0, 1.0]).all()
            ):
                raise ValueError(
                    f"Invalid binary outcomes: {model_id}"
                )

            if (
                not np.isfinite(costs).all()
                or (costs < 0).any()
            ):
                raise ValueError(
                    f"Invalid training costs: {model_id}"
                )

            model_costs[model_id] = float(costs.mean())

            # A constant-outcome model cannot fit logistic
            # regression; retain its training success rate.
            if len(np.unique(labels)) == 1:
                predictors[model_id] = float(labels[0])
                continue

            classifier = LogisticRegression(
                max_iter=600,
                random_state=random_state,
            )

            classifier.fit(
                features,
                labels.astype(int),
            )

            predictors[model_id] = classifier

        return cls(
            model_ids=model_ids,
            prompt_text_by_id=text_lookup,
            model_costs=model_costs,
            vectorizer=vectorizer,
            reducer=reducer,
            normalizer=normalizer,
            predictors=predictors,
            cost_ref_usd=cost_ref_usd,
            default_lam=default_lam,
        )

    def predict_probabilities(
        self,
        prompt_id: str,
    ) -> dict[str, float]:
        """Predict model success from this prompt's text only.

        These are raw logistic-regression estimates;
        calibration has not yet been established.
        """

        key = str(prompt_id)

        if key not in self.prompt_text_by_id:
            raise KeyError(f"Unknown prompt ID: {key}")

        text = self.prompt_text_by_id[key]

        features = self.vectorizer.transform([text])

        features = self.reducer.transform(features)

        features = self.normalizer.transform(features)

        predictions = {}

        for model_id in self.model_ids:
            predictor = self.predictors[model_id]

            if isinstance(
                predictor,
                (float, np.floating),
            ):
                probability = float(predictor)
            else:
                probability = float(
                    predictor.predict_proba(features)[0, 1]
                )

            predictions[model_id] = min(
                1.0,
                max(0.0, probability),
            )

        return predictions

    def decide(self, task: TaskRep) -> Strategy:
        """Select a model without accessing execution history."""

        probabilities = self.predict_probabilities(
            task.step_emb_key
        )

        lam = (
            self.default_lam
            if task.lam is None
            else float(task.lam)
        )

        if not math.isfinite(lam) or lam < 0:
            raise ValueError("Invalid task lambda")

        budget = float(task.budget_left_usd)

        if not math.isfinite(budget):
            raise ValueError("Invalid remaining budget")

        affordable = [
            model_id
            for model_id in self.model_ids
            if self.model_costs[model_id] <= budget
        ]

        if not affordable:
            # Let the simulator enforce budget exhaustion.
            selected = min(
                self.model_ids,
                key=lambda model_id: (
                    self.model_costs[model_id],
                    model_id,
                ),
            )
        else:
            def utility(model_id: str) -> float:
                return (
                    probabilities[model_id]
                    - lam
                    * self.model_costs[model_id]
                    / self.cost_ref_usd
                )

            selected = min(
                affordable,
                key=lambda model_id: (
                    -utility(model_id),
                    self.model_costs[model_id],
                    model_id,
                ),
            )

        return Strategy(
            step_id=task.step_id,
            round=task.round,
            model_id=selected,
            halt="stop",
            confidence=probabilities[selected],
        )
