import sqlite3

import numpy as np
from tqdm import tqdm

from src.data.utils import load_pickle
import pickle
from src.paths import PROCESSED_DATA_DIR, ROOT_DIR


def measure_tfidf(claims, title_topk=None, document_topk=None, extra_docs=None):
    if title_topk is None and document_topk is None:
        print("Both inputs cannot be None")
        exit(1)
    sqlite3.connect(PROCESSED_DATA_DIR / "data.db")
    if title_topk is not None and document_topk is not None:
        iterable = zip(claims, title_topk, document_topk)
    else:
        iterable = title_topk if title_topk is not None else document_topk
        iterable = zip(claims, iterable)
    hits = 0
    total = 0
    label_map = {0: 0, 1: 0, 2: 0}
    for i, instance in enumerate(iterable):
        claim = instance[0]
        if claim.label != "NOT ENOUGH INFO":
            topk_page_ids = []
            # faiss index accountting
            try:
                for topk in instance[1:]:
                    topk_page_ids += list(topk[0])
            except:
                topk_page_ids = list(topk)
            if extra_docs:
                topk_page_ids += extra_docs[i]
            topk_page_ids = set(topk_page_ids)
            evidence_groups = claim.evidence
            page_id_groups = [
                [evidence.page_id for evidence in evidence_group]
                for evidence_group in evidence_groups
            ]
            all_hit = False
            for page_id_group in page_id_groups:
                all_hit = all_hit or all(
                    [page_id in topk_page_ids for page_id in page_id_group]
                )
            if all_hit:
                label_map[claim.get_label_num()] += 1
                hits += 1
            total += 1
    return hits / total


if __name__ == "__main__":
    dev_claims = load_pickle(PROCESSED_DATA_DIR / "dev" / "claims.pkl")
    title_topks = [
        np.load(ROOT_DIR / f"expts/data/expt{i}_title_5.npy") for i in range(32)
    ]
    document_topks = [
        np.load(ROOT_DIR / f"expts/data/expt{i}_document_5.npy") for i in range(32)
    ]
    recall_scores = []
    with open("fever/data/processed/dev/fuzzy_docs.pkl", "rb") as f:
        extra_docs = pickle.load(f)

    # Test concatenated approach to TF-IDF
    for i in tqdm(range(32)):
        topk = np.load(ROOT_DIR / f"expts/data/expt{i}_concat_10.npy")
        recall = measure_tfidf(dev_claims, title_topk=topk, extra_docs=extra_docs)
        recall_scores.append((i, recall))

    # For generatint LaTeX tables used in my thesis.
    rows = [recall_scores[int(i) : int(i + 4)] for i in range(0, 32, 4)]
    formatted_strings = " \\\ \hline\n".join(
        [
            " & ".join(
                [
                    f"{expt_id} & {recall_score*100:.2f} \%"
                    for expt_id, recall_score in row
                ]
            )
            for row in rows
        ]
    )
    print(formatted_strings)
    print("=" * 100)
    print("=" * 100)
    print("=" * 100)
    recall_scores = []
    # Test title document separated approach to TF-IDF.
    for i in tqdm(range(32)):
        for j in range(32):
            title_topk = title_topks[i]
            document_topk = document_topks[j]
            recall = measure_tfidf(
                dev_claims,
                title_topk=title_topk,
                document_topk=document_topk,
                extra_docs=extra_docs,
            )
            recall_scores.append((i, j, recall))

    # For generatint LaTeX tables used in my thesis.
    rows = [recall_scores[int(i) : int(i + 4)] for i in range(0, 1024, 4)]
    formatted_strings = " \\\ \hline\n".join(
        [
            " & ".join(
                [
                    f"{title_id},{document_id} & {recall_score*100:.2f} \%"
                    for title_id, document_id, recall_score in row
                ]
            )
            for row in rows
        ]
    )
    print(formatted_strings)
