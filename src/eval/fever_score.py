"""
Compute FEVER Score for FEVER Dev set.
"""
import numpy as np
from sklearn.metrics import confusion_matrix

from src.data.utils import load_pickle
from src.eval.my_scorer import fever_score
from src.paths import PROCESSED_DATA_DIR, MODELS_DIR
from src.models.xgbc import reorder_by_score_filter_add_softmax


def main():
    # Hard code dataset, as dev is only set this makes sense for.
    dataset = "dev"
    claims = load_pickle(PROCESSED_DATA_DIR / dataset / "claims.pkl")
    claim_labels = [claim.get_label_num() for claim in claims]
    top5_sentences = np.load(
        PROCESSED_DATA_DIR / dataset / "combined_sentence_scores.npy"
    )
    top5_retrieval_scores = top5_sentences[:, :, 2]
    # top5_sentences = top5_sentences
    top5_predictions = np.load(
        PROCESSED_DATA_DIR / dataset / "combined_claim_scores.npy"
    )
    top5_predictions, _ = reorder_by_score_filter_add_softmax(top5_predictions, claims)
    top_docs = load_pickle(PROCESSED_DATA_DIR / dataset / "expanded_doc_ids.pkl")
    rfc = load_pickle(MODELS_DIR / "xgbc.pkl")
    preds = rfc.predict(top5_predictions.reshape((len(top5_predictions), -1)))
    pred_probas = rfc.predict_proba(
        top5_predictions.reshape((len(top5_predictions), -1))
    )
    pred_inputs = top5_predictions
    int2sym = {0: "REFUTES", 1: "NOT ENOUGH INFO", 2: "SUPPORTS"}
    instances = []
    misses = 0
    bad_misses = 0
    multi_step_misses = 0
    for i, (claim, pred, label) in enumerate(zip(claims, preds, claim_labels)):
        # if claim.label == "NOT ENOUGH INFO":
        #     continue
        # if i == 19975:
        #     print(top5_sentences[i])
        new_instance_evidence = []
        if claim.evidence:
            max_len = 0
            for j, evidence_set in enumerate(claim.evidence):
                new_evidence_set = []
                for evidence in evidence_set:
                    if [
                        None,
                        None,
                        evidence.page_id,
                        evidence.line_number,
                    ] not in new_evidence_set:
                        new_evidence_set.append(
                            [None, None, evidence.page_id, evidence.line_number]
                        )
                max_len = max(max_len, len(new_evidence_set))
                new_instance_evidence.append(new_evidence_set)
        sym_label = int2sym[label]
        sym_pred = int2sym[pred]
        near_miss = False
        if sym_label != sym_pred and sym_label != "NOT ENOUGH INFO":
            if min([len(evidence_set) for evidence_set in claim.evidence]) == 2:
                multi_step_misses += 1
            misses += 1
            sample = pred_inputs[i]
            weighted_scores = sample[:, :-1] * sample[:, -1].reshape((-1, 1))
            weighted_label_score = np.nanmax(weighted_scores[:, label])
        idx_order = np.argsort(top5_sentences[i][:, -1])
        top5_sentences[i] = top5_sentences[i][idx_order]
        predicted_evidence = [
            [int(xx) for xx in x[:2].astype(np.int32)]
            for x in top5_sentences[i]
            if x[2] >= 0.0
        ]
        top5_retrieval_scores[i]
        instance = {
            "idx": i,
            "claim": claim.claim,
            "label": sym_label,
            "predicted_label": sym_pred,
            "predicted_evidence": predicted_evidence,
            "evidence": claim.evidence,
            "preds": top5_predictions[i],
            "pred_proba": None,
            "pred_proba": pred_probas[i],
            "docs": top_docs[i],
            "near_miss": near_miss,
        }
        instances.append(instance)
    strict_score, label_accuracy, precision, recall, f1 = fever_score(
        instances, max_evidence=5
    )
    print(len(instances))
    print(strict_score)
    print(label_accuracy)
    print(precision)
    print(recall)
    print(f1)

    print(confusion_matrix(claim_labels, preds))
    print(misses)
    print(bad_misses)
    print(multi_step_misses)


if __name__ == "__main__":
    main()
