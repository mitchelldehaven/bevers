"""
Evaluation script from SciFact task.
Computes abstact-level and sentence-level scores as described in:
https://arxiv.org/abs/2004.14974.
"""


import json
import re
from argparse import ArgumentParser
from collections import Counter

# Utility functions.


def load_jsonl(fname):
    return [json.loads(x) for x in open(fname)]


def safe_divide(num, denom):
    if denom == 0:
        return 0.0
    else:
        return num / denom


def unify_label(gold):
    "For gold instances, put the label at abstract-level rather than rationale."
    evidence = {}
    for doc, ev in gold["evidence"].items():
        sents = [entry["sentences"] for entry in ev]
        labels = [entry["label"] for entry in ev]
        if len(set(labels)) > 1:
            raise ValueError("Conflicting labels.")
        label = labels[0]
        evidence[doc] = {"label": label, "rationales": sents}

    res = {"id": gold["id"], "evidence": evidence}

    return res


def compute_f1(gold, retrieved, correct, title):
    precision = safe_divide(correct, retrieved)
    recall = safe_divide(correct, gold)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        f"{title}_precision": precision,
        f"{title}_recall": recall,
        f"{title}_f1": f1,
    }


########################################

# Evaluator class.


class Evaluator:
    def __init__(self, verbose=False):
        self.reset()
        self.allowed_labels = ["SUPPORT", "CONTRADICT"]
        # For abstract evaluation, keep at most 3 rationale sentences.
        self.max_abstract_sents = 3
        # Whether to return all metrics or just F1.
        self.verbose = verbose

    def reset(self):
        self.counts_abstract = Counter()
        self.counts_sentence = Counter()

    def evaluate(self, golds, preds):
        "Evaluate a list of predictions against a list of gold claims."
        self.reset()
        self.check_ordering(golds, preds)

        # Get counts for all predictions.
        for gold, pred in zip(golds, preds):
            self.evaluate_claim(gold, pred)

        # Summarize the counts.
        res = {}

        # Abstract evaluation, label-only.
        res.update(
            compute_f1(
                self.counts_abstract["relevant"],
                self.counts_abstract["retrieved"],
                self.counts_abstract["correct_label_only"],
                "abstract_label_only",
            )
        )
        # Abstract evaluation, rationalized.
        res.update(
            compute_f1(
                self.counts_abstract["relevant"],
                self.counts_abstract["retrieved"],
                self.counts_abstract["correct_rationalized"],
                "abstract_rationalized",
            )
        )
        # Sentence evaluation, selection-only
        res.update(
            compute_f1(
                self.counts_sentence["relevant"],
                self.counts_sentence["retrieved"],
                self.counts_sentence["correct_selection"],
                "sentence_selection",
            )
        )
        print(self.counts_sentence)
        # Sentence evaluation, selection + label
        res.update(
            compute_f1(
                self.counts_sentence["relevant"],
                self.counts_sentence["retrieved"],
                self.counts_sentence["correct_label"],
                "sentence_label",
            )
        )

        # If not verbose, only keep the f1 metrics.
        if not self.verbose:
            res = {k: v for k, v in res.items() if re.match(".*_f1$", k)}

        return res

    @staticmethod
    def check_ordering(golds, preds):
        "Make sure the predictions are ordered correctly."
        gold_ids = [entry["id"] for entry in golds]
        pred_ids = [entry["id"] for entry in preds]
        if gold_ids != pred_ids:
            raise ValueError("Predicted claims do not match gold.")

    def evaluate_claim(self, gold, pred):
        "Evaluate a single claim."
        # Count gold sentences and abstracts.
        self.counts_abstract["relevant"] += len(gold["evidence"])
        for gold_doc in gold["evidence"].values():
            self.counts_sentence["relevant"] += sum(
                [len(x) for x in gold_doc["rationales"]]
            )

        # Loop over predicted documents and evaluate.
        for doc_id, doc_pred in pred["evidence"].items():
            # Make sure the label is legal.
            if doc_pred["label"] not in self.allowed_labels:
                raise ValueError(f"Unallowed label f{doc_pred['label']} predicted.")

            self.evaluate_abstract_level(gold["evidence"], doc_id, doc_pred)
            self.evaluate_sentence_level(gold["evidence"], doc_id, doc_pred)

    ####################

    # Abstract-level evaluation.

    def evaluate_abstract_level(self, gold_ev, doc_id, doc_pred):
        "Score a single abstract as correct or incorrect."
        self.counts_abstract["retrieved"] += 1

        # If the document isn't in the gold set, we're done.
        if doc_id not in gold_ev:
            return

        # If the label's wrong, we're done.
        gold_label = gold_ev[doc_id]["label"]
        if doc_pred["label"] != gold_label:
            return
        # Otherwise, the model gets credit for getting the label right.
        self.counts_abstract["correct_label_only"] += 1

        # If correctly rationalized, give credit.
        gold_rationales = [set(x) for x in gold_ev[doc_id]["rationales"]]
        # print(gold_ev[doc_id])

        shortest_rationale = min([len(x) for x in gold_rationales])
        max_abstract_sents = max(self.max_abstract_sents, shortest_rationale)

        # Truncate to only keep the first 3 predicted rationales.
        pred_rationales = set(doc_pred["sentences"][:max_abstract_sents])
        # print(gold_ev[doc_id])
        # print(pred_rationales)
        if self.contains_evidence(gold_rationales, pred_rationales):
            self.counts_abstract["correct_rationalized"] += 1
        #     print("GOOD")
        #     print(self.counts_abstract["correct_rationalized"])
        # print("-"*100)

    @staticmethod
    def contains_evidence(gold, pred):
        # If any of gold are contained in predicted, we're good.
        for gold_rat in gold:
            if gold_rat.issubset(pred):
                return True
        # If we get to the end, didn't find one.
        return False

    ####################

    # Sentence-level evaluation.

    def evaluate_sentence_level(self, gold_ev, doc_id, doc_pred):
        "Count all the gold rationale sentences for this claim."
        self.counts_sentence["retrieved"] += len(doc_pred["sentences"])

        # If the document isn't in the gold set, we're done.
        if doc_id not in gold_ev:
            return

        # Count the number of correct predicted rationale sentences.
        gold_rationales = [set(x) for x in gold_ev[doc_id]["rationales"]]
        # No need to truncate here, since the metric penalizes over-prediction.
        pred_rationales = set(doc_pred["sentences"])
        n_correct, n_incorrect = self.count_rationale_sents(
            gold_rationales, pred_rationales
        )
        # print(n_correct, n_incorrect)

        # Add the correct sentences to the count.
        self.counts_sentence["correct_selection"] += n_correct
        self.counts_sentence["incorrect_selection"] += n_incorrect
        # If the label is correct, add to the count.
        gold_label = gold_ev[doc_id]["label"]
        if gold_label == doc_pred["label"]:
            self.counts_sentence["correct_label"] += n_correct

    @staticmethod
    def count_rationale_sents(gold, predicted):
        "Count the correct rationale sentences."
        n_correct = 0
        n_incorrect = 0
        for ix in predicted:
            gold_sets = [entry for entry in gold if ix in entry]
            # print(gold_sets)
            assert len(gold_sets) < 2  # Can't be in two rationales.
            # If it's not in a gold set, no dice.
            if len(gold_sets) == 0:
                # print("continuing")
                continue
            # If it's in a gold set, make sure the rest got retrieved.
            gold_set = gold_sets[0]
            # print(gold_set)
            # print(predicted)
            # print(gold_set.issubset(predicted))
            # print()
            if gold_set.issubset(predicted):
                n_correct += 1
            else:
                n_incorrect += 1
        return n_correct, n_incorrect


########################################

# Main entry point.


def get_args():
    parser = ArgumentParser(description="SciFact evaluation.")
    parser.add_argument("--labels_file", type=str, help="File with gold evidence.")
    parser.add_argument("--preds_file", type=str, help="File with predictions.")
    parser.add_argument(
        "--metrics_output_file",
        type=str,
        help="Location of output metrics file",
        default="metrics.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If given, store all metrics; not just F1.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    evaluator = Evaluator(args.verbose)
    golds = load_jsonl(args.labels_file)
    golds = [unify_label(entry) for entry in golds]
    preds = load_jsonl(args.preds_file)
    metrics = evaluator.evaluate(golds, preds)
    if args.metrics_output_file == "stdout":
        print(json.dumps(metrics, indent=2))
    else:
        with open(args.metrics_output_file, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
