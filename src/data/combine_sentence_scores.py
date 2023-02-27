import argparse
import numpy as np
from pathlib import Path
from src.data.utils import load_pickle
from src.eval.my_scorer import fever_score
from src.paths import PROCESSED_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--datasets", type=Path, nargs="+", default=["train", "valid", "dev", "test"])
    parser.add_argument("--discount_factor", type=float)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    discount_iterable = [args.discount_factor] if args.discount_factor else list(i / 20 for i in range(21))
    best_discount = 0
    best_recall = 0
    print(args.datasets, discount_iterable)
    for dataset in args.datasets:
        print(dataset)
        for discount_factor in discount_iterable:
            claims = load_pickle(args.processed_data_dir / dataset / "claims.pkl")
            dev_sentence_scores = np.load(args.processed_data_dir / dataset / "sentence_scores.npy")
            dev_expanded_sentence_scores = np.load(
                args.processed_data_dir / dataset / "expanded_evidence_sentence_scores.npy"
            )
            combined_sentence_scores = []
            for base_scores, expanded_scores in zip(
                dev_sentence_scores, dev_expanded_sentence_scores
            ):
                if np.isnan(expanded_scores.sum()):
                    combined_scores = base_scores
                else:
                    expanded_scores[:, 2] = expanded_scores[:, 2] * discount_factor
                    combined_scores = np.concatenate([base_scores, expanded_scores], axis=0)
                sorted_idx = np.argsort(
                    -combined_scores[:, 2]
                )  # actual score value in 3rd column
                combined_sentence_scores.append(combined_scores[sorted_idx][:5])

            int2sym = {0: "REFUTES", 1: "NOT ENOUGH INFO", 2: "SUPPORTS"}
            instances = []
            for claim, scores in zip(claims, combined_sentence_scores):
                predicted_evidence = [
                    [int(xx) for xx in x[:2].astype(np.int32)] for x in scores if x[2] >= 0.0
                ]
                instance = {
                    "idx": 0,
                    "claim": claim.claim,
                    "label": int2sym[claim.get_label_num()],
                    "predicted_label": "NOT ENOUGH INFO",
                    "predicted_evidence": predicted_evidence,
                    "evidence": claim.evidence,
                    "preds": None,
                    "pred_proba": None,
                    "pred_proba": None,
                    "docs": None,
                    "near_miss": False,
                }
                instances.append(instance)
            strict_score, label_accuracy, precision, recall, f1 = fever_score(
                instances, max_evidence=5
            )
            if recall > best_recall:
                best_recall = recall
                best_discount = discount_factor

            np.save(
                args.processed_data_dir / dataset / "combined_sentence_scores.npy",
                np.array(combined_sentence_scores),
            )

            print(discount_factor)
            print(recall)
            print("=" * 100)


if __name__ == "__main__":
    main()

# discount_factor = 0.0
# dataset = "dev"
# best_discount = 0
# best_recall = 0
# while discount_factor <= 1.0:
#     claims = load_pickle(PROCESSED_DATA_DIR / dataset / "claims.pkl")
#     dev_sentence_scores = np.load("data/processed/dev/sentence_scores.npy")
#     dev_expanded_sentence_scores = np.load(
#         "data/processed/dev/expanded_evidence_sentence_scores.npy"
#     )
#     combined_sentence_scores = []
#     for base_scores, expanded_scores in zip(
#         dev_sentence_scores, dev_expanded_sentence_scores
#     ):
#         if np.isnan(expanded_scores.sum()):
#             combined_scores = base_scores
#         else:
#             expanded_scores[:, 2] = expanded_scores[:, 2] * discount_factor
#             combined_scores = np.concatenate([base_scores, expanded_scores], axis=0)
#         sorted_idx = np.argsort(
#             -combined_scores[:, 2]
#         )  # actual score value in 3rd column
#         combined_sentence_scores.append(combined_scores[sorted_idx][:5])

#     int2sym = {0: "REFUTES", 1: "NOT ENOUGH INFO", 2: "SUPPORTS"}
#     instances = []
#     for claim, scores in zip(claims, combined_sentence_scores):
#         predicted_evidence = [
#             [int(xx) for xx in x[:2].astype(np.int32)] for x in scores if x[2] >= 0.0
#         ]
#         instance = {
#             "idx": 0,
#             "claim": claim.claim,
#             "label": int2sym[claim.get_label_num()],
#             "predicted_label": "NOT ENOUGH INFO",
#             "predicted_evidence": predicted_evidence,
#             "evidence": claim.evidence,
#             "preds": None,
#             "pred_proba": None,
#             "pred_proba": None,
#             "docs": None,
#             "near_miss": False,
#         }
#         instances.append(instance)
#     strict_score, label_accuracy, precision, recall, f1 = fever_score(
#         instances, max_evidence=5
#     )
#     if recall > best_recall:
#         best_recall = recall
#         best_discount = discount_factor

#     np.save(
#         "data/processed/dev/combined_sentence_scores.npy",
#         np.array(combined_sentence_scores),
#     )
#     discount_factor += 0.05

# print(best_discount)
# print(best_recall)
# print("=" * 100)
