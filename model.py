"""Train and use the Student Focus Analyzer's scikit-learn classifier.

The model is trained on a deterministic synthetic teaching dataset. Its output is
an estimate, not a measurement of a student's mental state or a clinical result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

FEATURES = ["study_minutes", "distraction_minutes", "break_minutes", "concentration"]
LABELS = ["Low focus", "Medium focus", "High focus"]
LABEL_ORDER = {name: index for index, name in enumerate(LABELS)}


def make_sample_dataset(rows: int = 900, seed: int = 42) -> pd.DataFrame:
    """Create reproducible example data with a transparent labeling recipe."""
    rng = np.random.default_rng(seed)
    study = rng.integers(15, 181, size=rows)
    distraction = np.minimum(rng.gamma(2.1, 7.0, size=rows), study * 0.75)
    breaks = np.minimum(rng.gamma(1.7, 8.0, size=rows), 60)
    concentration = rng.integers(1, 11, size=rows)

    # Teaching-label recipe: combine self-rating and observable session ratios.
    # This is used only to create a labeled sample for training, not at inference.
    distraction_ratio = distraction / np.maximum(study, 1)
    break_ratio = breaks / np.maximum(study, 1)
    score = (
        (concentration - 1) / 9 * 0.55
        + (1 - np.clip(distraction_ratio, 0, 1)) * 0.32
        + (1 - np.clip(np.abs(break_ratio - 0.18) / 0.65, 0, 1)) * 0.13
        + rng.normal(0, 0.035, size=rows)
    )
    focus = np.where(score < 0.43, "Low focus", np.where(score < 0.69, "Medium focus", "High focus"))
    return pd.DataFrame({
        "study_minutes": study.astype(int),
        "distraction_minutes": np.round(distraction, 1),
        "break_minutes": np.round(breaks, 1),
        "concentration": concentration.astype(int),
        "focus_category": focus,
    })


def train_and_save(dataset_path: str | Path, model_path: str | Path, metrics_path: str | Path) -> dict[str, Any]:
    """Train a Random Forest on a stratified hold-out split and persist artifacts."""
    dataset_path, model_path, metrics_path = map(Path, (dataset_path, model_path, metrics_path))
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        data = pd.read_csv(dataset_path)
    else:
        data = make_sample_dataset()
        data.to_csv(dataset_path, index=False)
    if not set(FEATURES + ["focus_category"]).issubset(data.columns):
        raise ValueError("Dataset is missing required model columns.")

    X = data[FEATURES]
    y = data["focus_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=220, max_depth=9, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "test_rows": int(len(y_test)),
        "dataset_rows": int(len(data)),
        "classification_report": classification_report(
            y_test, predictions, labels=LABELS, output_dict=True, zero_division=0
        ),
        "feature_importance": {
            feature: round(float(value), 4)
            for feature, value in zip(FEATURES, model.feature_importances_)
        },
        "dataset_note": "Synthetic sample dataset; generated with a documented teaching-label recipe.",
    }
    joblib.dump(model, model_path)
    import json
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path: str | Path):
    return joblib.load(model_path)


def predict_focus(model, study_minutes: float, distraction_minutes: float,
                  break_minutes: float, concentration: int) -> dict[str, Any]:
    """Return category, class-probability confidence, and a human-readable explanation."""
    values = {
        "study_minutes": float(study_minutes),
        "distraction_minutes": float(distraction_minutes),
        "break_minutes": float(break_minutes),
        "concentration": int(concentration),
    }
    frame = pd.DataFrame([values], columns=FEATURES)
    probabilities = model.predict_proba(frame)[0]
    best_index = int(np.argmax(probabilities))
    category = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    reasons = []
    distraction_ratio = values["distraction_minutes"] / max(values["study_minutes"], 1)
    break_ratio = values["break_minutes"] / max(values["study_minutes"], 1)
    if values["concentration"] >= 8:
        reasons.append("your self-reported concentration was high")
    elif values["concentration"] <= 4:
        reasons.append("your self-reported concentration was low")
    if distraction_ratio >= 0.25:
        reasons.append("distraction time was a notable share of study time")
    elif distraction_ratio <= 0.08:
        reasons.append("distraction time was relatively low")
    if break_ratio > 0.55:
        reasons.append("break time was long relative to this study block")
    elif 0.08 <= break_ratio <= 0.35:
        reasons.append("break time was in a moderate range")
    explanation = "The model leaned on " + (", and ".join(reasons) if reasons else "the combined pattern of session duration, distractions, breaks, and your concentration rating") + "."
    return {
        "category": category,
        "confidence": confidence,
        "probabilities": {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)},
        "explanation": explanation,
        "self_reported_concentration": int(concentration),
    }


def build_recommendations(category: str, study_minutes: float, distraction_minutes: float,
                          distraction_count: int, break_minutes: float, concentration: int) -> list[str]:
    """Create personalized, deterministic coaching tips from the saved session data."""
    tips: list[str] = []
    distraction_ratio = distraction_minutes / max(study_minutes, 1)
    if category == "Low focus":
        tips.extend(["Try a shorter 20–25 minute study block, then reset before the next one.",
                     "Put your phone out of reach and close unrelated tabs for the next session."])
    elif category == "Medium focus":
        tips.extend(["Choose one small, finishable goal before you start your next block.",
                     "Use a planned 5–10 minute break instead of switching tasks mid-session."])
    else:
        tips.extend(["Repeat the setup that worked well for this session and protect that time slot.",
                     "Keep your breaks balanced so a productive rhythm stays sustainable."])

    if distraction_ratio >= 0.2 or distraction_count >= 5:
        tips.append(f"You logged {distraction_count} distractions over {distraction_minutes:g} minutes; note the top trigger and remove it before studying.")
    elif distraction_minutes <= 2:
        tips.append("Distractions were minimal; keep the same low-friction study environment.")
    if concentration <= 4:
        tips.append("Your concentration rating was {}/10; try a quick reset (water, stretch, or a clearer task) before continuing.".format(concentration))
    if break_minutes == 0 and study_minutes >= 45:
        tips.append("For a longer block, schedule a short screen-free break next time.")
    elif break_minutes > study_minutes * 0.5:
        tips.append("Your break time was long compared with study time; try setting a gentle break timer.")
    return tips[:5]
