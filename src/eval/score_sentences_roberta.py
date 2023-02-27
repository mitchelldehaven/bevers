"""
Script for producing top-5 sentences for claims for each dataset defined by --datasets
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from tqdm import tqdm
from transformers import RobertaTokenizerFast, AutoTokenizer

from src.data.utils import load_pickle, untokenize
from src.models import RoBERTa
from src.paths import DB_PATH, FEATURES_DIR, PROCESSED_DATA_DIR


def parse_args():
    """
    Parse arguments for script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--datasets", type=str, nargs="+", default=["train", "valid", "dev", "test"])
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--features_dir", type=Path, default=FEATURES_DIR / "tfidf")
    parser.add_argument("--db_file", type=Path, default=DB_PATH)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--exclude_fuzzy", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reretrieval", action="store_true")
    parser.add_argument("--claims_file", type=Path, default="claims.pkl")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def load_initial_data(args, dataset, k=None):
    """
    Load initial data for scoring:
        - Use title and document retrieval for base set.
        - Also uses document from fuzzy string search if not args.exclude_fuzzy.
    Returns:
        - Dataset claims.
        - Top document ids.
        - Discount map - this is empty in this function as discount map is used
          for re-retrieval which is handle by `load_scored_data`.
    """
    dataset_claims = load_pickle(args.processed_dir / dataset / args.claims_file)
    k = args.k if k is None else k
    dataset_titles = np.load(
        args.features_dir / f"{dataset}_title_scores-{k}.npy"
    )
    dataset_docs = np.load(
        args.features_dir / f"{dataset}_document_scores-{k}.npy"
    )
    # actual idxs are in the 0th column
    dataset_topids = np.concatenate((dataset_titles[:, 0], dataset_docs[:, 0]), axis=1)
    dataset_topids2 = []
    if args.exclude_fuzzy:
        for topids in dataset_topids:
            dataset_topids2.append(list(topids))
    else:
        dataset_fuzzy_docs = load_pickle(
            args.processed_dir / dataset / "fuzzy_docs.pkl"
        )
        for topids, fuzzy_docs in zip(dataset_topids, dataset_fuzzy_docs):
            dataset_topids2.append(list(topids) + fuzzy_docs)

    dataset_topids = dataset_topids2
    discount_maps = [{}] * len(dataset_topids)
    return dataset_claims, dataset_topids, discount_maps


def load_scored_data(args, dataset):
    """
    Load scored data for doing re-retrieval.
    Returns:
        - Dataset claims.
        - Top document ids.
        - Discount map - this is empty in this function as discount map is used
          for re-retrieval which is handle by `load_scored_data`.
    """
    dataset_claims = load_pickle(args.processed_dir / dataset / args.claims_file)
    dataset_topids = load_pickle(
        args.processed_dir / dataset / "expanded_evidence_doc_ids.pkl"
    )
    discount_maps = load_pickle(
        args.processed_dir / dataset / "expanded_discount_maps.pkl"
    )
    return dataset_claims, dataset_topids, discount_maps


def load_data(args, dataset):
    """
    Simple function for conditionally loading data based on args.reretrieval.
    """
    if args.reretrieval:
        return load_scored_data(args, dataset)
    return load_initial_data(args, dataset)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    model = RoBERTa(args.model_type, args.num_labels, tokenizer=tokenizer)
    model.expand_embeddings()
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt["state_dict"])
    model.cuda()
    model.eval()
    conn = sqlite3.connect(args.db_file)
    curs = conn.cursor()
    for dataset in args.datasets:
        dataset_claims, dataset_topids, discount_maps = load_data(args, dataset)
        hits = 0
        hits_1 = 0
        possible_count = 0
        possible_count_1 = 0
        all_count = 0
        all_count_1 = 0
        topk_sentences = []
        total_needed = 0
        total_got = 0
        total_retrieved = 0
        loop = tqdm(
            zip(dataset_claims, dataset_topids, discount_maps),
            total=len(dataset_claims),
        )
        for claim, top_ids, discount_map in loop:
            top_ids = list(set(top_ids))  # no duplicate queries
            query = (
                "SELECT DISTINCT t.page_name, l.page_id, l.line_num, l.line FROM lines as l "
                "JOIN texts as t ON l.page_id = t.page_id WHERE l.page_id IN ({})"
            )
            query = query.format(",".join([str(doc_id) for doc_id in top_ids]))
            top_ids = {int(x) for x in top_ids}
            curs.execute(query)
            res = list(
                filter(lambda x: x[2] <= 100 and x[3] != "", list(curs.fetchall()))
            )
            evidence_sets = (
                [
                    {
                        (evidence.page_id, evidence.line_number)
                        for evidence in evidence_set
                    }
                    for evidence_set in claim.evidence
                ]
                if claim.evidence
                else [set()]
            )
            evidence_docs = (
                [
                    {evidence.page_id for evidence in evidence_set}
                    for evidence_set in claim.evidence
                ]
                if claim.evidence
                else [set()]
            )
            evidence_list = [(r[1], r[2]) for r in res]
            lines = [[r[0], untokenize(r[3])] for r in res]
            all_preds = []
            if len(lines) < 5:
                topk_sentences.append(np.array([[np.nan] * 3] * 5))
                continue
            i = 0
            with autocast(dtype=torch.bfloat16, device_type="cuda"):
                with torch.no_grad():
                    while i < len(lines):
                        batch_lines = lines[i : i + args.batch_size]
                        batch_inputs = [
                            ((title + " -- " + line), claim.claim)
                            for title, line in batch_lines
                        ]
                        batch_input_ids = tokenizer(
                            batch_inputs,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=256,
                        )
                        batch_input_ids["input_ids"] = batch_input_ids[
                            "input_ids"
                        ].cuda()
                        batch_input_ids["attention_mask"] = batch_input_ids[
                            "attention_mask"
                        ].cuda()
                        preds = model(batch_input_ids).logits.float().detach().cpu()
                        del batch_input_ids
                        all_preds += [preds]
                        i = i + args.batch_size
            all_preds = torch.cat(all_preds, dim=0)
            softmax_scores = F.softmax(all_preds, dim=1)
            if args.num_labels == 3:
                score_criteria = 1 - softmax_scores[:, 1]
            else:
                score_criteria = softmax_scores[:, 1]
            if args.reretrieval:
                for e_idx, evidence in enumerate(evidence_list):
                    doc_id, line_num = evidence
                    if doc_id in discount_map:
                        score_criteria[e_idx] *= discount_map[doc_id]
                    else:
                        print(doc_id, " not in discount map")
                        print(discount_map)
                        exit(1)
            args_sort = torch.argsort(score_criteria, descending=True)
            topk = args_sort[:5]
            topk_evidence = []
            for top in topk:
                _, doc_id, line_num, _ = res[top]
                topk_evidence.append((doc_id, line_num, float(score_criteria[top])))
            topk_evidence = set(topk_evidence)
            if claim.get_label_num() != 1:
                is_one = min([len(evidence_set) for evidence_set in evidence_sets]) == 1
                if is_one:
                    all_count_1 += 1
                topk_lines = set([(line[0], line[1]) for line in topk_evidence])
                hit = False
                for evidence_set in evidence_sets:
                    total_needed += len(evidence_set)
                total_got += sum(
                    [
                        evidence_set.issubset(topk_lines)
                        for evidence_set in evidence_sets
                    ]
                )
                total_retrieved += len(topk_lines)
                if any(
                    evidence_set.issubset(topk_lines) for evidence_set in evidence_sets
                ):
                    hit = True
                    hits += 1
                    if is_one:
                        hits_1 += 1
                if any(
                    evidence_doc.issubset(top_ids) for evidence_doc in evidence_docs
                ):
                    possible_count += 1
                    if is_one:
                        possible_count_1 += 1
                    if not hit and args.debug:
                        print("=" * 100)
                        print("Gold evidence scores:")
                        print(claim)
                        print()
                        print(claim.claim)
                        print()
                        for evidence_set in evidence_sets:
                            for evidence in evidence_set:
                                try:
                                    evidence_idx = evidence_list.index(evidence)
                                except Exception:
                                    continue
                                evidence_score = score_criteria[evidence_idx]
                                evidence_place = list(args_sort).index(evidence_idx)
                                evidence_line = lines[evidence_idx]
                                print(
                                    f"{evidence_place} - {evidence_score :.4f} - {evidence_line}"
                                )
                            print()
                        print("-" * 100)
                        for i, top in enumerate(topk):
                            pred_score = score_criteria[top]
                            evidence_line = lines[top]
                            print(f"{i} - {pred_score :.4f} - {evidence_line}")
                all_count += 1
            topk_evidence = [list(x) for x in topk_evidence]
            topk_sentences.append(topk_evidence)
            # print(
            #     "Hit ratio: {}/{} = {:.3f}".format(
            #         hits, all_count, hits / (all_count + 1e-6)
            #     )
            # )
            # print(
            #     "Can hit ratio: {}/{} = {:.3f}".format(
            #         hits, possible_count, hits / (possible_count + 1e-6)
            #     )
            # )
        print(
            "Hit ratio: {}/{} = {:.3f}".format(
                hits, all_count, hits / (all_count + 1e-6)
            )
        )
        print(
            "Can hit ratio: {}/{} = {:.3f}".format(
                hits, possible_count, hits / (possible_count + 1e-6)
            )
        )
        print(
            "Hit ratio: {}/{} = {:.3f}".format(
                hits_1, all_count_1, hits_1 / (all_count_1 + 1e-6)
            )
        )
        print(
            "Can hit ratio: {}/{} = {:.3f}".format(
                hits_1, possible_count_1, hits_1 / (possible_count_1 + 1e-6)
            )
        )
        if args.save_results:
            if args.reretrieval:
                np.save(
                    args.processed_dir
                    / dataset
                    / "expanded_evidence_sentence_scores.npy",
                    np.array(topk_sentences),
                )
            else:
                np.save(
                    args.processed_dir / dataset / "sentence_scores.npy",
                    np.array(topk_sentences),
                )


if __name__ == "__main__":
    main()
