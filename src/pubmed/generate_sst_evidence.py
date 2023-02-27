import argparse
from pathlib import Path

import numpy as np

from src.data.utils import Evidence, dump_pickle, load_pickle
from src.paths import PROCESSED_DATA_DIR


def create_sst_evidence(claims, topk_predicted_sentences, threshold, debug):
    new_claims = []
    scores = []
    for i, (claim, top_sentences) in enumerate(zip(claims, topk_predicted_sentences)):
        claim_page_id = claim.evidence[0][0].page_id
        top_sentences = top_sentences[np.argsort(-1 * top_sentences[:, 2])]
        pred_page_id, pred_line_num, score = top_sentences[0]
        for top_sentence in top_sentences:
            pred_page_id, pred_line_num, score = top_sentence
            if pred_page_id == claim_page_id:
                break
        if pred_page_id == claim_page_id and score >= threshold:
            if args.debug:
                print(i, claim_page_id, pred_line_num, score)
            new_evidence = {"page_id": claim_page_id, "line_number": pred_line_num}
            evidence = [[Evidence(**new_evidence)]]
            claim.evidence = evidence
            new_claims.append(claim)
            scores.append(score)
    np.array(scores)
    return new_claims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", type=Path, nargs="+", default=["train", "valid", "dev"]
    )
    parser.add_argument("--processed_data_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        dataset_claims = load_pickle(
            args.processed_data_dir / dataset / "claims_no_evidence.pkl"
        )
        init_len = len(dataset_claims)
        dataset_topk_sentences = np.load(
            args.processed_data_dir / dataset / "sentence_scores.npy"
        )
        dataset_claims_w_evidence = create_sst_evidence(
            dataset_claims, dataset_topk_sentences, args.threshold, args.debug
        )
        post_len = len(dataset_claims_w_evidence)
        print(init_len, post_len)
        dump_pickle(
            dataset_claims_w_evidence, args.processed_data_dir / dataset / "claims.pkl"
        )
