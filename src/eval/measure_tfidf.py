import argparse
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import spacy
import sqlite_spellfix
from tqdm import tqdm

from src.data.utils import dump_pickle, load_pickle
from src.paths import DB_PATH, FEATURES_DIR, PROCESSED_DATA_DIR

# spacy.require_gpu()
nlp = spacy.load("en_core_web_trf")
conn = sqlite3.connect(DB_PATH)
curs = conn.cursor()
conn.enable_load_extension(True)
conn.load_extension(sqlite_spellfix.extension_path())

# 92.6, 99.1
# 93.8


@lru_cache(maxsize=20000)
def expand_documents(nlp, sentence, curs, minimum_distance=300, query_limit=7):
    additional_ids = []
    ignore_labels = ["QUANTITY", "DATE", "ORDINAL"]
    doc = nlp(sentence)
    tokens = [
        str(token).replace('"', '""')
        for token in doc
        if ("subj" in token.dep_ or (not token.is_lower and not token.is_punct))
        and not token.is_stop
    ]
    ents = [
        str(ent).replace('"', '""')
        for ent in doc.ents
        if ent.label_ not in ignore_labels
    ]
    fitlered_tokens = []
    for token in tokens:
        if not any([token in ent for ent in ents]) and not all(
            [char.isnumeric() for char in str(token)]
        ):
            fitlered_tokens.append(token)
    for ent in set(ents + fitlered_tokens):
        if len(ent) < 3 or all([char.isnumeric() for char in str(ent)]):
            continue
        query = f"SELECT rowid, word, distance FROM clean_titles_default_cost WHERE word MATCH ? AND distance <= {minimum_distance * 3} ORDER BY distance"
        results = curs.execute(query, (ent,)).fetchall()
        filtered_results = [
            result
            for i, result in enumerate(results)
            if result[2] <= minimum_distance or i < query_limit
        ]
        for result in filtered_results[: query_limit * 3]:
            additional_ids.append(result[0])
    return additional_ids


def get_doc_recall(claims, topk_titles, topk_documents, loaded_fuzzy_docs=[], report_misses=False):
    hits = 0
    total = 0
    hits_1 = 0
    total_1 = 0
    all_top_docs = []
    all_fuzzy_docs = []
    for i, (claim, topk_title, topk_document) in tqdm(
        enumerate(zip(claims, topk_titles, topk_documents)), total=len(claims)
    ):
        topk_title, topk_title_scores = topk_title
        topk_document, topk_document_scores = topk_document
        if loaded_fuzzy_docs:
            fuzzy_docs = loaded_fuzzy_docs[i]
        else:
            fuzzy_docs = expand_documents(nlp, claim.claim, curs)
        topk_page_ids = set(list(topk_title) + list(topk_document) + fuzzy_docs)
        all_top_docs.append(list(topk_page_ids))
        all_fuzzy_docs.append(fuzzy_docs)
        if (
            claim.get_label_num() == 0 or claim.get_label_num() == 2
        ) and claim.evidence:
            total += 1
            evidence_groups = claim.evidence
            evidence_1 = max([len(evidence) for evidence in evidence_groups]) == 1
            if evidence_1:
                total_1 += 1
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
                hits += 1
                if evidence_1:
                    hits_1 += 1

    if total:
        print(hits / total)
    if total_1:
        print(hits_1 / total_1)
    return all_top_docs, all_fuzzy_docs


def get_tfidf_scores(model, claim_text, doc_ids):
    query = "SELECT DISTINCT line, page_id, line_num FROM lines where page_id IN ({})".format(
        ", ".join(str(i) for i in doc_ids)
    )
    lines, page_ids, line_nums = zip(*curs.execute(query).fetchall())
    all_evidence = list(zip(page_ids, line_nums))
    tfidf_claim_text = model.transform([claim_text]).tocsr()
    tfidf_line_texts = model.transform(lines).tocsr().T
    scores = (tfidf_claim_text * tfidf_line_texts).toarray()[0]
    kth = min(1000, len(scores) - 1)
    topk_idxs = np.argpartition(-scores, kth)[:kth]
    return set([all_evidence[topk_idx] for topk_idx in topk_idxs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fever_db", type=Path, default=DB_PATH)
    parser.add_argument("--features_dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--use_saved_fuzzy", action="store_true")
    parser.add_argument("--report_misses", action="store_true")
    args = parser.parse_args()
    datasets = ["train", "valid", "dev", "test"]
    all_topk_docs = []
    conn = sqlite3.connect(DB_PATH)
    for dataset in datasets:
        dataset_claims = load_pickle(args.processed_dir / dataset / "claims.pkl")
        claims_topk_titles = np.load(
            args.features_dir / "tfidf" / f"{dataset}_title_scores-{args.k}.npy"
        ).astype(np.int32)
        claims_topk_documents = np.load(
            args.features_dir / "tfidf" / f"{dataset}_document_scores-{args.k}.npy"
        ).astype(np.int32)
        # use for cases where fuzzy string is available and we want to swap out tf-idf.
        fuzzy_docs = []
        if args.use_saved_fuzzy:
            fuzzy_docs = load_pickle(args.processed_dir / dataset / "fuzzy_docs.pkl")
        topk_docs, fuzzy_docs = get_doc_recall(
            dataset_claims,
            claims_topk_titles,
            claims_topk_documents,
            loaded_fuzzy_docs=fuzzy_docs,
            report_misses=args.report_misses,
        )
        dump_pickle(topk_docs, args.processed_dir / dataset / "expanded_doc_ids.pkl")
        dump_pickle(fuzzy_docs, args.processed_dir / dataset / "fuzzy_docs.pkl")
