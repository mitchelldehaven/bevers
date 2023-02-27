import argparse
import os
import random
import sqlite3
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.utils import load_pickle, untokenize
from src.eval.score_sentences_roberta import load_initial_data
from src.paths import DB_PATH, FEATURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR


RANDOM_SEED = random.Random(0)
if os.environ["DATASET"] == "fever":
    NEGATIVE_SAMPLE_MULTIPLIER = 3
else:
    NEGATIVE_SAMPLE_MULTIPLIER = 10


def get_tfidf_scores(model, claim_text, line_texts):
    tfidf_claim_text = model.transform([claim_text]).tocsr()
    tfidf_line_texts = model.transform(line_texts).tocsr().T
    scores = (tfidf_claim_text * tfidf_line_texts).toarray()
    return scores[0]


def get_pos_neg_samples(
    i, top_docs, claim, num_neg_samples, db_path=DB_PATH, use_mnli_labels=True
):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT t.page_name, l.line FROM lines as l JOIN texts as t ON l.page_id = t.page_id WHERE l.page_id=? AND l.line_num=?"
    positive_results = []
    claim_evidence = set()
    positive_label = claim.get_label_num() if use_mnli_labels else 1
    negative_label = 1 if use_mnli_labels else 0
    if claim.evidence is not None:
        for evidence_group in claim.evidence:
            for evidence in evidence_group:
                evidence_tuple = (evidence.page_id, evidence.line_number)
                if evidence_tuple not in claim_evidence:
                    claim_evidence.add(evidence_tuple)
                else:
                    continue
                query_results = conn.execute(query, evidence_tuple).fetchall()
                if claim.get_label_num() == 0:  # over sample refutes
                    query_results = query_results * NEGATIVE_SAMPLE_MULTIPLIER
                positive_results += query_results
    positives = [
        [claim.claim, sample[0], untokenize(sample[1]), positive_label]
        for sample in positive_results
    ]
    neg_docs = top_docs
    query = "SELECT l.page_id, l.line_num, t.page_name, l.line FROM lines as l JOIN texts as t ON l.page_id=t.page_id WHERE l.page_id IN ({})".format(
        neg_docs
    )
    query = query.replace("[", "").replace("]", "")
    negatives_results = conn.execute(query).fetchall()
    filtered_negatives = list(
        set(
            [
                (untokenize(result[2]), untokenize(result[3]), result[0])
                for result in negatives_results
                if (result[0], result[1]) not in claim_evidence
                and len(untokenize(result[3])) > 5
            ]
        )
    )
    weights = [len(filtered_negative[1]) for filtered_negative in filtered_negatives] # use length of sentence as a weight for random sampling
    # weights = [1 for filtered_negative in filtered_negatives]
    choices = RANDOM_SEED.choices(
        filtered_negatives, weights, k=min(len(filtered_negatives), num_neg_samples * 5)
    )
    unique_choices = set()
    for choice in choices:
        if choice not in unique_choices:
            unique_choices.add(choice)
        if len(unique_choices) == num_neg_samples:
            break
    negatives = [
        [claim.claim, sample[0], sample[1], negative_label] for sample in unique_choices
    ]
    conn.close()
    return positives + negatives


def helper(data, num_neg_samples):
    # print(*data)
    # print(num_neg_samples)
    return get_pos_neg_samples(*data, num_neg_samples)


def create_dataset(
    db_name,
    top_docs,
    claims,
    tfidf_model,
    num_neg_samples,
    random_seed=0,
    use_mnli_labels=True,
):
    random.Random(random_seed)
    pool = Pool(8)
    zipped_data = zip(list(range(len(claims))), top_docs, claims)
    clumped_dataset = list(
        tqdm(
            pool.imap(partial(helper, num_neg_samples=num_neg_samples), zipped_data),
            total=len(claims),
        )
    )
    dataset = []
    for d in clumped_dataset:
        dataset += d
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=Path, default=FEATURES_DIR / "tfidf")
    parser.add_argument(
        "--tfidf_model", type=Path, default=MODELS_DIR / "document_vectorizer.pkl"
    )
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--exclude_fuzzy", action="store_true")
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument(
        "--dataset_dest_dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "sentence_selection",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--claims_file", type=Path, default="claims.pkl")
    args = parser.parse_args()
    datasets = ["train", "valid", "dev"]
    for dataset in datasets:
        for k in args.ks:
            dataset_dir = args.dataset_dest_dir / f"k_{k}_random_expanded"
            dataset_dir.mkdir(exist_ok=True, parents=True)
            claims = load_pickle(PROCESSED_DATA_DIR / dataset / "claims.pkl")
            if args.exclude_fuzzy:
                _, top_docs, _ = load_initial_data(args, dataset, k=k)
            else:
                top_docs = load_pickle(PROCESSED_DATA_DIR / dataset / "expanded_doc_ids.pkl")
            sentence_selection_dataset = create_dataset(
                DB_PATH, top_docs, claims, None, k
            )
            sent_expt_dataset = dataset_dir / f"{dataset}.csv"
            df = pd.DataFrame(
                sentence_selection_dataset,
                columns=["claim", "candidate_title", "candidate_sentence", "label"],
            )
            df.to_csv(sent_expt_dataset)
