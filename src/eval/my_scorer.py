import six
import json
import sqlite3 
from src.paths import DB_PATH
import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


conn = sqlite3.connect(DB_PATH)
curs = conn.cursor()
d = dict(list(curs.execute("SELECT page_id, og_page_name FROM texts").fetchall()))


def check_predicted_evidence_format(instance):
    if 'predicted_evidence' in instance.keys() and len(instance['predicted_evidence']):
        assert all(isinstance(prediction, list)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(len(prediction) == 2
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

#         assert all(isinstance(prediction[0], six.string_types)
#                     for prediction in instance["predicted_evidence"]), \
#             "Predicted evidence must be a list of (page<string>,line<int>) lists"

#         assert all(isinstance(prediction[1], int)
#                    for prediction in instance["predicted_evidence"]), \
#             "Predicted evidence must be a list of (page<string>,line<int>) lists"


def is_correct_label(instance):
    correct = instance["label"].upper() == instance["predicted_label"].upper()
    # print("-"*100)
    # print(instance["claim"])
    # print("Label:", instance["label"])
    # print("Pred:", instance["predicted_label"])
    # print("Near Miss:", instance["near_miss"])
    # strictly_correct = False
    # # print([[d[page_id], page_id, line_num] for page_id, line_num in instance["predicted_evidence"]])
    # query = "SELECT line FROM lines WHERE page_id=? and line_num=?"
    # for page_id, line_num in instance["predicted_evidence"]:
    #     print(d[page_id], page_id, line_num)
    #     print(curs.execute(query, (page_id, line_num)).fetchall()[0][0])
    # print(instance["preds"])
    # print()
    # print(instance["pred_proba"])
    # if instance["label"] != "NOT ENOUGH INFO":
        # print([[[d[e.page_id], e.page_id, e.line_number] for e in evidence_group] for evidence_group in instance["evidence"]])
        # if max(len(evidence_group) for evidence_group in instance["evidence"]) == 1:
        #     evidence = instance["evidence"][0][0]
        #     print(curs.execute(query, (evidence.page_id, evidence.line_number)).fetchall()[0][0])
    if not correct:
        # print("-"*100)
        # print(instance["claim"])
        # print("Label:", instance["label"])
        # print("Pred:", instance["predicted_label"])
        # strictly_correct = False
        # print([[d[page_id], page_id, line_num] for page_id, line_num in instance["predicted_evidence"]])
        # print(instance["preds"])
        if instance["label"] != "NOT ENOUGH INFO":
            strictly_correct = False
            for evience_group in instance["evidence"]:
                actual_sentences = [[e.page_id, e.line_number] for e in evience_group]
                #Only return true if an entire group of actual sentences is in the predicted sentences
                if all([actual_sent in instance["predicted_evidence"] for actual_sent in actual_sentences]):
                    strictly_correct = True
            if strictly_correct:
                a = 9
            else:
                b = 0
            # print("-"*100)
            # print(instance["claim"])
            # print("Label:", instance["label"])
            # print("Pred:", instance["predicted_label"])
            # print("Near Miss:", instance["near_miss"])
            # strictly_correct = False
            # # print([[d[page_id], page_id, line_num] for page_id, line_num in instance["predicted_evidence"]])
            # query = "SELECT line FROM lines WHERE page_id=? and line_num=?"
            # for page_id, line_num in instance["predicted_evidence"]:
            #     print(d[page_id], page_id, line_num)
            #     print(curs.execute(query, (page_id, line_num)).fetchall()[0][0])
            # print(instance["preds"])
            # print()
            # print(instance["pred_proba"])
            # print([[[d[e.page_id], e.page_id, e.line_number] for e in evidence_group] for evidence_group in instance["evidence"]])
            # if max(len(evidence_group) for evidence_group in instance["evidence"]) == 1:
            #     evidence = instance["evidence"][0][0]
        else:
            x = 0
            # print("-"*100)
            # print(instance["claim"])
            # print("Label:", instance["label"])
            # print("Pred:", instance["predicted_label"])
            # print("Near Miss:", instance["near_miss"])
            # strictly_correct = False
            # # print([[d[page_id], page_id, line_num] for page_id, line_num in instance["predicted_evidence"]])
            # query = "SELECT line FROM lines WHERE page_id=? and line_num=?"
            # for page_id, line_num in instance["predicted_evidence"]:
            #     print(d[page_id], page_id, line_num)
            #     print(curs.execute(query, (page_id, line_num)).fetchall()[0][0])
            # print(instance["preds"])
            # print()
            # print(instance["pred_proba"])
            # print([[[d[e.page_id], e.page_id, e.line_number] for e in evidence_group] for evidence_group in instance["evidence"]])
            # if max(len(evidence_group) for evidence_group in instance["evidence"]) == 1:
            #     evidence = instance["evidence"][0][0]
            #     print(curs.execute(query, (evidence.page_id, evidence.line_number)).fetchall()[0][0])
            # print(strictly_correct) 
    return correct


def is_strictly_correct(instance, max_evidence=None):
    #Strict evidence matching is only for NEI class
    check_predicted_evidence_format(instance)

    if instance["label"].upper() != "NOT ENOUGH INFO" and is_correct_label(instance):
        assert 'predicted_evidence' in instance, "Predicted evidence must be provided for strict scoring"

        if max_evidence is None:
            max_evidence = len(instance["predicted_evidence"])

        # print("="*100)
        # print(instance["evidence"])
        for evience_group in instance["evidence"]:
            actual_sentences = [[e.page_id, e.line_number] for e in evience_group]
            #Only return true if an entire group of actual sentences is in the predicted sentences
            if all([actual_sent in instance["predicted_evidence"][:max_evidence] for actual_sent in actual_sentences]):
                return True
    #     if max(len(evidence_group) for evidence_group in instance["evidence"]) == 1:
    #         print("-"*100)
    #         print(instance["idx"])
    #         print(instance["claim"])
    #         print(instance["label"])
    #         print([[d[page_id], line_num] for page_id, line_num in instance["predicted_evidence"]])
    #         print()
    #         print([[[d[e.page_id], e.line_number] for e in evidence_group] for evidence_group in instance["evidence"]])
    #         print(any([all([e.page_id in instance["docs"] for e in evidence_group]) for evidence_group in instance["evidence"]]))
    #         print([d[page_id] for page_id in instance["docs"]])
    # #If the class is NEI, we don't score the evidence retrieval component
    elif instance["label"].upper() == "NOT ENOUGH INFO" and is_correct_label(instance):
        return True

    return False


def evidence_macro_precision(instance, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e.page_id, e.line_number] for eg in instance["evidence"] for e in eg if e.page_id is not None]

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

def evidence_macro_recall(instance, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
           return 1.0, 1.0

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for evidence_group in instance["evidence"]:
            evidence = [[e.page_id, e.line_number] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0


# Micro is not used. This code is just included to demostrate our model of macro/micro
def evidence_micro_precision(instance):
    this_precision = 0
    this_precision_hits = 0

    # We only want to score Macro F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e.page_id, e.line_number] for eg in instance["evidence"] for e in eg if e.page_id is not None]

        for prediction in instance["predicted_evidence"]:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

    return this_precision, this_precision_hits


def fever_score(predictions,actual=None, max_evidence=5):
    correct = 0
    strict = 0

    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0

    for idx,instance in enumerate(predictions):
        # print("="*100)
        # for k, v in instance.items():
        #     print(k, v)
        # print(json.dumps(instance, indent=2))
        assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

        #If it's a blind test set, we need to copy in the values from the actual data
        if 'evidence' not in instance or 'label' not in instance:
            assert actual is not None, 'in blind evaluation mode, actual data must be provided'
            assert len(actual) == len(predictions), 'actual data and predicted data length must match'
            assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
            instance['evidence'] = actual[idx]['evidence']
            instance['label'] = actual[idx]['label']

        assert 'evidence' in instance.keys(), 'gold evidence must be provided'

        if is_correct_label(instance):
            correct += 1.0

            if is_strictly_correct(instance, max_evidence):
                strict+=1.0

        macro_prec = evidence_macro_precision(instance, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(instance, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

    total = len(predictions)

    strict_score = strict / total
    acc_score = correct / total

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec)

    return strict_score, acc_score, pr, rec, f1