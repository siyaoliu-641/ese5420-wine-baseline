from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 5420
DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "winequality-white.csv"


def selected_cv_scores(search):
    """Return the validation scores for the selected hyperparameter."""
    score_columns = [
        name
        for name in search.cv_results_
        if name.startswith("split") and name.endswith("_test_score")
    ]
    return np.array(
        [search.cv_results_[name][search.best_index_] for name in score_columns]
    )


def print_result(name, scores, setting=""):
    """Print mean ROC-AUC and its spread across validation folds."""
    mean = scores.mean()
    standard_deviation = scores.std(ddof=1)
    standard_error = standard_deviation / np.sqrt(len(scores))
    print(
        f"{name}{setting}: mean ROC-AUC = {mean:.4f}, "
        f"SD = {standard_deviation:.4f}, SE = {standard_error:.4f}"
    )


# Load, audit, and deduplicate before any split.
data = pd.read_csv(DATA_PATH, sep=";")
raw_rows = len(data)
duplicate_rows = data.duplicated().sum()
assert data.isna().sum().sum() == 0
data = data.drop_duplicates()

X = data.drop(columns="quality")
y = (data["quality"] >= 7).astype(int)

print(f"Raw rows: {raw_rows}")
print(f"Exact duplicate rows removed: {duplicate_rows}")
print(f"Rows after deduplication: {len(data)}")
print(f"Positive observations: {y.sum()} ({y.mean():.1%})")


# Create one fixed test set. It is deliberately not evaluated in this checkpoint.
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)
print(f"Development rows: {len(X_dev)}")
print(f"Held-out test rows: {len(X_test)} (not evaluated)\n")
del X_test, y_test


# Use the same repeated stratified folds for every development-set comparison.
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=SEED,
)


# Course baseline: scaling is inside the pipeline, so it is fitted within folds.
logistic = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=10_000, random_state=SEED)),
    ]
)
logistic_search = GridSearchCV(
    logistic,
    {"model__C": np.logspace(-4, 4, 9)},
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
)
logistic_search.fit(X_dev, y_dev)
logistic_scores = selected_cv_scores(logistic_search)


# Comparison: standardized kNN with odd k values requested in the feedback.
knn = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ]
)
knn_search = GridSearchCV(
    knn,
    {"model__n_neighbors": range(1, 152, 2)},
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
)
knn_search.fit(X_dev, y_dev)
knn_scores = selected_cv_scores(knn_search)


# Simple sanity check: a constant score should have ROC-AUC 0.5.
dummy_scores = cross_val_score(
    DummyClassifier(strategy="prior"),
    X_dev,
    y_dev,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
)


print_result(
    "Logistic regression",
    logistic_scores,
    f" (C = {logistic_search.best_params_['model__C']:g})",
)
print_result(
    "kNN",
    knn_scores,
    f" (k = {knn_search.best_params_['model__n_neighbors']})",
)
print_result("Majority-class check", dummy_scores)
