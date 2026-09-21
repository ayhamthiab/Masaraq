"""
Model training module for the Masarak University Recommendation System.

This module handles the training of machine learning models for major recommendation.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import warnings
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
    classification_report,
    make_scorer,
)
from sklearn.utils.class_weight import compute_class_weight

from utils import (
    ROOT, DATA_DIR, RAW_DIR, PROC_DIR, INTERIM_DIR,
    MODELS_DIR, OUTPUTS_DIR, RECS_DIR,
    ensure_dirs, set_seed, load_json, save_json,
    save_model, load_model, find_column
)

# =========================================================
# Data loading / splitting / preprocessing
# =========================================================

def load_ml_dataset() -> pd.DataFrame:
    """Load the machine learning dataset."""
    ml_df = pd.read_csv(PROC_DIR / "merged_dataset.csv")
    return ml_df


def split_data(
    df: pd.DataFrame,
    target_col: str = 'label',
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split the data into training, validation, and test sets."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size,
        random_state=random_state, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Basic numeric-only preprocessing + imputation."""
    drop_cols = ['student_id', 'student_name', 'top_interests', 'branch']
    X_train_proc = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns]).copy()
    X_val_proc   = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns]).copy()
    X_test_proc  = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns]).copy()

    for df in (X_train_proc, X_val_proc, X_test_proc):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    train_means = X_train_proc.mean(numeric_only=True)
    X_train_proc = X_train_proc.fillna(train_means)
    X_val_proc   = X_val_proc.fillna(train_means)
    X_test_proc  = X_test_proc.fillna(train_means)

    all_nan_cols = X_train_proc.columns[X_train_proc.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"Dropping {len(all_nan_cols)} all-NaN train columns")
        X_train_proc.drop(columns=all_nan_cols, inplace=True)
        X_val_proc.drop(columns=[c for c in all_nan_cols if c in X_val_proc.columns], inplace=True)
        X_test_proc.drop(columns=[c for c in all_nan_cols if c in X_test_proc.columns], inplace=True)

    return X_train_proc, X_val_proc, X_test_proc

# =========================================================
# Helpers (proba alignment, eligibility mask, etc.)
# =========================================================

def _get_model_classes(model) -> np.ndarray:
    """Get classes_ from a model or from a pipeline's classifier step."""
    if hasattr(model, "classes_"):
        return model.classes_
    try:
        clf = getattr(model, "named_steps", {}).get("classifier", None)
        if clf is not None and hasattr(clf, "classes_"):
            return clf.classes_
    except Exception:
        pass
    raise AttributeError("Model has no classes_ attribute.")


def _predict_proba(model, X) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified predict_proba:
      - if model has predict_proba -> use it
      - else if decision_function -> softmax to probabilities
    Returns (proba, classes).
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = _get_model_classes(model)
        if isinstance(proba, list):  # rare multilabel cases
            proba = np.vstack([p[:, 1] for p in proba]).T
        return np.asarray(proba), np.asarray(classes)

    if hasattr(model, "decision_function"):
        dec = np.asarray(model.decision_function(X))
        classes = _get_model_classes(model)
        dec = dec - dec.max(axis=1, keepdims=True)
        proba = np.exp(dec)
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
        return proba, np.asarray(classes)

    raise AttributeError("Model does not support predict_proba or decision_function.")


def _align_proba_to_classes(proba: np.ndarray, model_classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    """Align model's probability matrix to a unified class order."""
    col_map = {c: i for i, c in enumerate(model_classes)}
    aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    for j, cls in enumerate(target_classes):
        if cls in col_map:
            aligned[:, j] = proba[:, col_map[cls]]
        else:
            aligned[:, j] = 0.0
    return aligned


def _apply_eligibility_mask(y_proba: np.ndarray, X_feat: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    """
    Zero out probabilities for ineligible programs per-student, then renormalize.
    Fallback: if a row becomes all-zero, revert to original y_proba for that row.
    """
    masks = []
    for cls in classes:
        col = f"eligible_{cls}"
        masks.append(X_feat[col].astype(int).values if hasattr(X_feat, "columns") and col in X_feat.columns
                     else np.ones(y_proba.shape[0], dtype=int))
    mask = np.vstack(masks).T
    masked = y_proba * mask
    row_sums = masked.sum(axis=1, keepdims=True)
    fallback = (row_sums.squeeze() == 0)
    if fallback.any():
        masked[fallback] = y_proba[fallback]
        row_sums = masked.sum(axis=1, keepdims=True)
    return masked / np.maximum(row_sums, 1e-12)


def _soft_vote_proba(X, models_list: List[Any], weights: List[float], classes: np.ndarray) -> np.ndarray:
    """Soft-voting after aligning each model's proba to a common class order."""
    aligned_list = []
    for m in models_list:
        p, mc = _predict_proba(m, X)
        aligned_list.append(_align_proba_to_classes(p, mc, classes))
    stacked = np.stack(aligned_list, axis=0)  # (n_models, n_samples, n_classes)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.tensordot(w, stacked, axes=(0, 0))  # (n_samples, n_classes)

# =========================================================
# Models (LR / RF / GB)
# =========================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a Logistic Regression (OVR) baseline and print Top-3/Top-5 on validation.
    """
    print("Training Logistic Regression (OVR) baseline...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', OneVsRestClassifier(
            LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced'
            ),
            n_jobs=-1
        ))
    ])

    param_grid = {
        'classifier__estimator__C': [0.1, 1.0, 10.0],
        'classifier__estimator__solver': ['lbfgs', 'liblinear'],
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, y_train)

    best_lr = grid_search.best_estimator_
    print(f"Logistic Regression best parameters: {grid_search.best_params_}")

    metrics: Dict[str, float] = {}
    if hasattr(best_lr, "predict_proba"):
        y_proba = best_lr.predict_proba(X_val)
        classes = best_lr.named_steps['classifier'].classes_
        top3 = top_k_accuracy_score(y_val, y_proba, k=3, labels=classes)
        top5 = top_k_accuracy_score(y_val, y_proba, k=min(5, len(classes)), labels=classes)
        print(f"Top-3 validation accuracy (LR): {top3:.4f}")
        print(f"Top-5 validation accuracy (LR): {top5:.4f}")
        metrics.update({'top3_accuracy': top3, 'top5_accuracy': top5})
    else:
        print("Model does not support predict_proba; cannot compute Top-K.")

    return best_lr, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    print("Training Random Forest model...")

    base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        oob_score=False
    )

    param_grid = {
        'n_estimators': [300, 600],
        'max_depth': [None, 20, 40],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', 0.5],
    }

    grid_search = GridSearchCV(
        base, param_grid,
        cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"Random Forest best parameters: {grid_search.best_params_}")

    y_pred = best_rf.predict(X_val)
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision_weighted': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_val, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_val, y_pred, average='weighted', zero_division=0),
    }
    if hasattr(best_rf, "predict_proba"):
        y_proba = best_rf.predict_proba(X_val)
        metrics['top3_accuracy'] = top_k_accuracy_score(y_val, y_proba, k=3, labels=best_rf.classes_)
        metrics['top5_accuracy'] = top_k_accuracy_score(y_val, y_proba, k=min(5, len(best_rf.classes_)), labels=best_rf.classes_)

    print(f"Validation metrics: {metrics}")
    try:
        print(classification_report(y_val, y_pred)[:1000])
    except Exception:
        pass

    return best_rf, metrics


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    use_eligibility_mask: bool = True,
    verbose: int = 1
) -> Tuple[GradientBoostingClassifier, Dict[str, float]]:
    """
    Gradient Boosting + GridSearch (refit by Top-3). Prints Top-3/Top-5 (+ eligibility-masked).
    """
    print("Training Gradient Boosting model...")

    gb = GradientBoostingClassifier(random_state=42)

    param_grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "subsample": [0.8, 1.0],
        "max_features": [None, "sqrt"],
    }

    classes = np.unique(y_train)
    class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_w_map = {c: w for c, w in zip(classes, class_w)}
    sample_weight = y_train.map(class_w_map).values

    top3_scorer = make_scorer(lambda yt, yp: top_k_accuracy_score(yt, yp, k=3), needs_proba=True)
    scoring = {"acc": "accuracy", "top3": top3_scorer}

    grid_search = GridSearchCV(
        gb, param_grid, cv=3, scoring=scoring, refit="top3", n_jobs=-1, verbose=verbose
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, y_train, **{"sample_weight": sample_weight})

    best_gb = grid_search.best_estimator_
    print(f"Gradient Boosting best parameters (refit=top3): {grid_search.best_params_}")

    metrics: Dict[str, float] = {}
    if hasattr(best_gb, "predict_proba"):
        y_proba = best_gb.predict_proba(X_val)
        classes_out = best_gb.classes_

        top3 = top_k_accuracy_score(y_val, y_proba, k=3, labels=classes_out)
        top5 = top_k_accuracy_score(y_val, y_proba, k=min(5, len(classes_out)), labels=classes_out)
        print(f"Top-3 (GB): {top3:.4f}")
        print(f"Top-5 (GB): {top5:.4f}")
        metrics.update({"top3_accuracy": top3, "top5_accuracy": top5})

        if use_eligibility_mask:
            y_proba_m = _apply_eligibility_mask(y_proba, X_val, classes_out)
            top3_m = top_k_accuracy_score(y_val, y_proba_m, k=3, labels=classes_out)
            top5_m = top_k_accuracy_score(y_val, y_proba_m, k=min(5, len(classes_out)), labels=classes_out)
            print(f"Top-3 (GB, elig): {top3_m:.4f}")
            print(f"Top-5 (GB, elig): {top5_m:.4f}")
            metrics.update({"top3_accuracy_elig": top3_m, "top5_accuracy_elig": top5_m})
    else:
        print("Model does not support predict_proba; cannot compute Top-K.")

    return best_gb, metrics

# =========================================================
# Pickle-friendly Ensemble
# =========================================================

class SoftVotingEnsemble:
    """
    Soft-voting ensemble (picklable). Aligns model probabilities to a unified class order,
    optionally applies eligibility mask before returning probabilities.
    """
    def __init__(self, models: List[Any], weights: List[float], classes: np.ndarray, use_eligibility_mask: bool = True):
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.sum() <= 0:
            self.weights = np.ones_like(self.weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.classes_ = np.asarray(classes)
        self.use_eligibility_mask = bool(use_eligibility_mask)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = _soft_vote_proba(X, self.models, self.weights, self.classes_)
        if self.use_eligibility_mask:
            proba = _apply_eligibility_mask(proba, X, self.classes_)
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


def create_ensemble(models_list: List[Any], X_val: pd.DataFrame, y_val: pd.Series, use_eligibility_mask: bool = True) -> Tuple[SoftVotingEnsemble, Dict[str, float]]:
    """
    Build a soft-voting ensemble; search simple weight grids; pick best by Hit@3 on validation.
    """
    classes = _get_model_classes(models_list[0])

    # Weight grid allowing zeros (can drop weak model)
    grid = [
        (1,0,0), (0,1,0), (0,0,1),
        (1,1,0), (1,0,1), (0,1,1),
        (2,1,0), (1,2,0), (2,0,1), (0,2,1), (1,0,2), (0,1,2),
        (1,1,1), (2,1,1), (1,2,1), (1,1,2), (3,2,1), (3,1,2), (2,3,1), (2,1,3), (1,3,2), (1,2,3)
    ]

    best = {"w": None, "top3": -1, "top5": -1}
    for w in grid:
        proba = _soft_vote_proba(X_val, models_list, w, classes)
        if use_eligibility_mask:
            proba = _apply_eligibility_mask(proba, X_val, classes)
        top3 = top_k_accuracy_score(y_val, proba, k=3, labels=classes)
        top5 = top_k_accuracy_score(y_val, proba, k=min(5, len(classes)), labels=classes)
        if top3 > best["top3"]:
            best = {"w": w, "top3": top3, "top5": top5}

    weights = (np.asarray(best["w"], dtype=float) / sum(best["w"])).tolist()
    ensemble = SoftVotingEnsemble(models_list, weights, classes, use_eligibility_mask=use_eligibility_mask)
    metrics = {"val_top3": float(best["top3"]), "val_top5": float(best["top5"]), "weights": weights}
    print(f"[Ensemble] chosen weights (val): {weights} | val_top3={metrics['val_top3']:.4f} | val_top5={metrics['val_top5']:.4f}")
    return ensemble, metrics



def evaluate_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate models on test set (Top-1 + Top-3 + Top-5).
    For non-ensemble models, we apply eligibility mask before Top-K for fairness.
    """
    print("Evaluating models on test set...")
    evaluation: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        # Top-1 metrics
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        # Top-K metrics
        proba, classes = _predict_proba(model, X_test)
        if not isinstance(model, SoftVotingEnsemble):
            proba = _apply_eligibility_mask(proba, X_test, classes)

        metrics['top3_accuracy'] = top_k_accuracy_score(y_test, proba, k=3, labels=classes)
        metrics['top5_accuracy'] = top_k_accuracy_score(y_test, proba, k=min(5, len(classes)), labels=classes)

        evaluation[name] = {k: float(v) for k, v in metrics.items()}
        print(f"{name} test metrics: {evaluation[name]}")

    return evaluation

# =========================================================
# Orchestration
# =========================================================

def train_models() -> None:
    """
    1) Load dataset
    2) Split Train/Val/Test
    3) Preprocess
    4) Train RF + GB + LR
    5) Build soft-voting ensemble with validation-weight search
    6) Evaluate on test (Top-1/Top-3/Top-5)
    7) Save models & evaluation
    """
    ensure_dirs()
    set_seed(42)

    try:
        print("Loading ML dataset...")
        ml_df = load_ml_dataset()

        print("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(ml_df)

        print("Preprocessing data...")
        X_train_proc, X_val_proc, X_test_proc = preprocess_data(X_train, X_val, X_test)

        # Train base models
        gb_model, gb_metrics = train_gradient_boosting(X_train_proc, y_train, X_val_proc, y_val)
        save_model(gb_model, MODELS_DIR / "gradient_boosting_model.pkl")
        rf_model, rf_metrics = train_random_forest(X_train_proc, y_train, X_val_proc, y_val)
        save_model(rf_model, MODELS_DIR / "random_forest_model.pkl")
        lr_model, lr_metrics = train_logistic_regression(X_train_proc, y_train, X_val_proc, y_val)
        save_model(lr_model, MODELS_DIR / "logistic_regression_model.pkl")
       
        

        # Build ensemble (allow dropping weak model via zero weight)
        models_list = [rf_model, gb_model, lr_model]
        ensemble, ensemble_metrics = create_ensemble(models_list, X_val_proc, y_val, use_eligibility_mask=True)

        # Evaluate on test
        models = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'logistic_regression': lr_model,
            'ensemble': ensemble
        }
        evaluation = evaluate_models(models, X_test_proc, y_test)

        # Save models
        print("Saving models...")
        
       
        
        save_model(ensemble, MODELS_DIR / "ensemble_model.pkl")  # picklable class

        # Save evaluation results
        print("Saving evaluation results...")
        evaluation_results = {
            'validation': {
                'random_forest': rf_metrics,
                'gradient_boosting': gb_metrics,
                'logistic_regression': lr_metrics,
                'ensemble': ensemble_metrics
            },
            'test': evaluation
        }
        save_json(evaluation_results, OUTPUTS_DIR / "model_evaluation.json")

        # Feature importances (if available)
        try:
            outs = []
            if hasattr(rf_model, 'feature_importances_'):
                outs.append(("rf_feature_importances.csv", rf_model.feature_importances_))
            if hasattr(gb_model, 'feature_importances_'):
                outs.append(("gb_feature_importances.csv", gb_model.feature_importances_))
            for fname, importances in outs:
                feature_importance = pd.DataFrame({
                    'feature': X_train_proc.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                feature_importance.to_csv(OUTPUTS_DIR / fname, index=False)
        except Exception:
            pass

        print("Model training completed successfully!")

    except Exception as e:
        print(f"Error during model training: {e}")
        raise



if __name__ == "__main__":
    train_models()
