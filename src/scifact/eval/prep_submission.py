import argparse
import json
from pathlib import Path

import numpy as np

from src.data.utils import load_pickle
from src.models.xgbc import reorder_by_score_filter_add_softmax
from src.paths import DB_PATH, MODELS_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR
from src.scifact.data.prep_scifact_data import get_idmap

PREDICTION_MAP = {0: "CONTRADICT", 1: "NOT ENOUGH INFO", 2: "SUPPORT"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["test"]
    )
    parser.add_argument("--processed_data_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--models_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--db_path", type=Path, default=DB_PATH)
    args = parser.parse_args()
    xgbc = load_pickle(args.models_dir / "xgbc.pkl")
    idmap = get_idmap(args.db_path)
    for dataset in args.datasets:
        claims = load_pickle(args.processed_data_dir / dataset / "claims.pkl")
        claim_scores = np.load(args.processed_data_dir / dataset / "claim_scores.npy")
        sentence_scores = np.load(
            args.processed_data_dir / dataset / "sentence_scores.npy"
        )
        reordered_claim_scores, _ = reorder_by_score_filter_add_softmax(
            claim_scores, claims
        )
        reordered_claim_scores = reordered_claim_scores.reshape(
            (len(reordered_claim_scores), -1)
        )
        predictions = xgbc.predict(reordered_claim_scores)
        output = []
        for claim, sentence_score, claim_score, prediction in zip(
            claims, sentence_scores, claim_scores, predictions
        ):
            idx_order = np.argsort(-1 * sentence_score[:, 2])
            sorted_sentence_score = sentence_score[idx_order]
            evidence_map = {}
            label = PREDICTION_MAP[prediction]
            # print(sorted_sentence_score.shape)
            # if dataset in ["dev"] and claim.id == 1:
            # print(sorted_sentence_score)
            # print()
            for row in sorted_sentence_score:
                doc_id, row_num, ret_score = row
                if ret_score < 0.6:
                    continue
                row_num = int(row_num)
                og_doc_id = str(idmap[doc_id])
                if og_doc_id in evidence_map:
                    evidence_map[og_doc_id]["sentences"].append(row_num)
                else:
                    evidence_map[og_doc_id] = {"sentences": [row_num]}
                    evidence_map[og_doc_id]["label"] = label
            # for doc_id, doc_map in evidence_map.items():
            #     sentences = doc_map["sentences"]
            #     if len(sentences) < 3:
            #         sentences = doc_map["sentences"]
            #         diff = 3 - len(doc_map["sentences"])
            #         c = 0
            #         while diff > 0:
            #             if c not in sentences:
            #                 sentences.append(c)
            #                 diff -= 1
            #             c += 1
            #         doc_map["sentences"] = sentences
            #     evidence_map[doc_id] = doc_map

            # if dataset in ["dev"] and claim.id == 1:
            #     print(evidence_map)
            evidence_map = {} if label == "NOT ENOUGH INFO" else evidence_map
            sample = {
                "id": claim.id,
                "evidence": evidence_map,
            }
            output.append(sample)
        output = sorted(output, key=lambda x: x["id"])
        with open(OUTPUTS_DIR / "predictions.jsonl", "w") as f:
            for sample in output:
                print(json.dumps(sample), file=f)
