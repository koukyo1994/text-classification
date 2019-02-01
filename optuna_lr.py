import argparse

import optuna
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf

from core.loader import DataLoader
from core.preprocessing import get_stopwords, tokenizer
from core.util import timer, get_logger

X = None
X_test = None
y = None
y_test = None


def optimal_params(trial: optuna.Trial):
    params = {}
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    params["penalty"] = penalty

    tol = trial.suggest_uniform("tol", 1e-5, 1e-3)
    params["tol"] = tol
    C = trial.suggest_uniform("C", 1e-4, 10)
    params["C"] = C
    params["random_state"] = 42
    if penalty == "l2":
        solver = trial.suggest_categorical(
            "solver", ["newton-cg", "lbfgs", "sag", "saga"])
        params["solver"] = solver
    else:
        params["solver"] = "saga"
    params["max_iter"] = 10000
    params["multi_class"] = "auto"

    oof_preds = np.zeros((X.shape[0], ))
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for (trn_index, val_index) in fold.split(X, y):
        X_train, X_val = X[trn_index], X[val_index]
        y_train = y[trn_index]
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        oof_preds[val_index] = y_pred
    f1 = f1_score(y, oof_preds, average="macro")
    return 1.0 - f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp")
    parser.add_argument("--ntrial", default=200, type=int)
    parser.add_argument("--n_jobs", default=3, type=int)

    args = parser.parse_args()
    logger = get_logger(exp=args.exp)
    with timer("Load Data", logger):
        loader = DataLoader()

    with timer("tokenize", logger):
        loader.tokenize(tokenizer, {
            "stopwords": get_stopwords(),
            "include_verb": True
        })

    train, test = loader.load()
    X = train["tokenized"].fillna("")
    X_test = test["tokenized"].fillna("")
    y = train["label"].values
    y_test = test["label"].values

    with timer("vectorize", logger):
        tv = Tfidf(max_features=20000, ngram_range=(1, 3))
        X = tv.fit_transform(X)
        X_test = tv.transform(X_test)

    with timer("optimize", logger):
        study = optuna.create_study()
        study.optimize(
            optimal_params, n_trials=args.ntrial, n_jobs=args.n_jobs)

    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best trial: {study.best_trial}")
