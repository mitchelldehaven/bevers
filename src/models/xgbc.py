"""
Script for training gradient boosting classifier via XGBoost.
Expects that previous pipeline steps have been ran.
"""
import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.data.utils import dump_pickle, load_pickle
from src.paths import DATASET, MODELS_DIR, PROCESSED_DATA_DIR

if DATASET == "fever":
    # FEVER's dev set is large enough for support fairly large XGBC models.
    XGBC_PARAMS = {
        "max_depth": [2, 4, 6, 8],
        "lambda": [1],
        "alpha": [0],
        "n_estimators": [20, 40, 60, 80, 100],
        "learning_rate": [0.1, 0.3],
        "use_label_encoder": [False],
        "eval_metric": ["mlogloss"],
    }
elif DATASET == "scifact":
    # SciFact's dev set is quite small, so potential models are smaller than FEVER supports.
    XGBC_PARAMS = {
        "max_depth": [1, 2],
        "lambda": [1, 2, 3, 4, 5],
        "alpha": [0, 0.25, 0.5, 0.75, 1],
        "n_estimators": [20],
        "learning_rate": [0.1, 0.3],
        "use_label_encoder": [False],
        "eval_metric": ["mlogloss"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim_scores_file", type=Path)
    args = parser.parse_args()
    return args


def reorder_by_score_filter_add_softmax(data, claims, min_ret_score=0.0):
    """
    Reorders the score matrix in descending order. If `min_ret_score` set,
    filter out any results with ret_score `min_ret_score`.
    """
    new_data = []
    new_labels = []
    for sample, claim in zip(data, claims):
        matrix_shape = sample.shape[0]
        concat_score = sample[-1].reshape((1, -1))
        sample = sample[:-1]
        label = claim.get_label_num()
        idx_order = np.argsort(sample[:, -1])
        sample = sample[idx_order]
        sample = sample[sample[:, -1] >= min_ret_score]
        sample = np.concatenate([sample, concat_score])
        nan_padded = np.ones((matrix_shape, sample.shape[1]))
        nan_padded[nan_padded == 1] = np.nan
        if sample.shape[0] >= 1:
            nan_padded[-sample.shape[0] :] = sample
        new_data.append(nan_padded)
        new_labels.append(label)
    return np.array(new_data), new_labels


def train(args, xgbc_params=XGBC_PARAMS):
    """
    Function for training XGBoost classifier. Use dev results to train
    an aggregation model. The dev set is used, since it isn't finetuned
    on, thus the scores are not biases.

    Uses 4 fold cross validation. Best parameter set is chosen and the
    final model is retrained with all available data.
    """
    dev_claims = load_pickle(PROCESSED_DATA_DIR / "dev" / "claims.pkl")
    dev_labels = [claim.get_label_num() for claim in dev_claims]
    dev_claim_scores = np.load(PROCESSED_DATA_DIR / "dev" / args.claim_scores_file)
    dev_claim_scores, dev_labels = reorder_by_score_filter_add_softmax(
        dev_claim_scores, dev_claims
    )
    dev_claim_scores = dev_claim_scores.reshape((len(dev_claim_scores), -1))
    xgbc = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    gridcv = GridSearchCV(
        xgbc,
        xgbc_params,
        "accuracy",
        n_jobs=10,
        cv=4,
        verbose=2,
        refit=True,
        return_train_score=True,
    )
    gridcv.fit(dev_claim_scores, dev_labels)
    print(gridcv.cv_results_.keys())
    print(gridcv.best_params_)
    print(
        "Best Mean Train Accuracy Across Folds:",
        gridcv.cv_results_["mean_train_score"].mean(),
    )
    print(
        "Best Mean Test Accuracy Across Folds:",
        gridcv.cv_results_["mean_test_score"].mean(),
    )
    dump_pickle(gridcv.best_estimator_, MODELS_DIR / "xgbc.pkl")


if __name__ == "__main__":
    args = parse_args()
    train(args)
