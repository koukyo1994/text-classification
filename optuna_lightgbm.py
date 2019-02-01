import argparse

import optuna
import numpy as np
import lightgbm as lgb

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf

from core.loader import DataLoader
from core.preprocessing import get_stopwords, tokenizer
from core.util import timer, get_logger

X = None
X_test = None
y = None
y_test = None


def optimal_params(trial: optuna.Trial):
    boosting = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
    learning = trial.suggest_uniform("learning_rate", 1e-5, 0.2)
    num_leaves = trial.suggest_int("num_leaves", 5, 1000)
    sub_sample = trial.suggest_int("subsample_for_bin", 100, 200000)
    min_split_gain = trial.suggest_uniform("min_split_gain", 0.0, 1e-1)
    min_data = trial.suggest_int("min_child_samples", 2, 50)
    min_child_weight = trial.suggest_uniform("min_child_weight", 1e-5, 0.2)
    bagging_frac = trial.suggest_uniform("bagging_fraction", 0.4, 0.99)
    bagging_freq = trial.suggest_int("bagging_freq", 1, 20)
    feature_frac = trial.suggest_uniform("feature_fraction", 0.4, 0.99)
    lambda_l1 = trial.suggest_uniform("lambda_l1", 1e-6, 1.0)
    lambda_l2 = trial.suggest_uniform("lambda_l2", 1e-6, 1.0)
    max_bin = trial.suggest_int("max_bin", 16, 2048)
    params = {
        "objective": "multiclass",
        "boosting_type": boosting,
        "n_estimators": 2000,
        "learning_rate": learning,
        "num_leaves": num_leaves,
        "subsample_for_bin": sub_sample,
        "min_split_gain": min_split_gain,
        "min_child_samples": min_data,
        "min_child_weight": min_child_weight,
        "subsample": bagging_frac,
        "subsample_freq": bagging_freq,
        "colsample_bytree": feature_frac,
        "reg_alpha": lambda_l1,
        "reg_lambda": lambda_l2,
        "max_bin": max_bin,
        "n_jobs": 1
    }
    model = lgb.LGBMClassifier(**params)
    oof_preds = np.zeros((X.shape[0], ))
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for (trn_index, val_index) in fold.split(X, y):
        X_train, X_val = X[trn_index], X[val_index]
        y_train, y_val = y[trn_index], y[val_index]
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=100)
        y_pred = model.predict(X_val)
        oof_preds[val_index] = y_pred
    f1 = f1_score(y, oof_preds, average="macro")
    return (1.0 - f1)


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
