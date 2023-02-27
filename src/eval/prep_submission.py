import json
import sqlite3

import numpy as np

from src.data.utils import load_pickle
from src.paths import DB_PATH, MODELS_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR
from src.models.xgbc import reorder_by_score_filter_add_softmax


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    d = dict(list(conn.execute("SELECT page_id, og_page_name FROM texts").fetchall()))
    conn.close()
    dataset = "test"
    claims = load_pickle(PROCESSED_DATA_DIR / dataset / "claims.pkl")
    claim_labels = [claim.get_label_num() if claim.label else -1 for claim in claims]
    top5_sentences = np.load(
        PROCESSED_DATA_DIR / dataset / "combined_sentence_scores.npy"
    )
    top5_predictions = np.load(
        PROCESSED_DATA_DIR / dataset / "combined_claim_scores.npy"
    )
    top5_predictions, _ = reorder_by_score_filter_add_softmax(top5_predictions, claims)
    rfc = load_pickle(MODELS_DIR / "xgbc.pkl")
    preds = rfc.predict(top5_predictions.reshape((len(top5_predictions), -1)))
    int2sym = {0: "REFUTES", 1: "NOT ENOUGH INFO", 2: "SUPPORTS"}
    instances = []
    for i, (claim, pred, label) in enumerate(zip(claims, preds, claim_labels)):
        new_instance_evidence = []
        if claim.evidence:
            for j, evidence_set in enumerate(claim.evidence):
                new_evidence_set = []
                for evidence in evidence_set:
                    new_evidence_set.append(
                        [None, None, d[evidence.page_id], evidence.line_number]
                    )
                new_instance_evidence.append(new_evidence_set)
        sym_pred = int2sym[pred]
        predicted_evidence = []
        for top_sentence in top5_sentences[i]:
            doc_id, line_num = top_sentence[:2].astype(np.int32)
            if doc_id > 0:
                predicted_evidence.append([d[int(doc_id)], int(line_num)])
        instance = {
            "id": claim.id,
            "predicted_label": sym_pred,
            "predicted_evidence": predicted_evidence,
        }
        instances.append(instance)
    with open(OUTPUTS_DIR / "predictions.jsonl", "w") as f:
        for instance in instances:
            print(json.dumps(instance), file=f)
