import sqlite3

import numpy as np
import sqlite_spellfix
from scipy import sparse
from tqdm import tqdm

from src.data.utils import dump_pickle, load_pickle, untokenize
from src.paths import DB_PATH, PROCESSED_DATA_DIR, DATA_DIR, MODELS_DIR, FEATURES_DIR
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=Path, default=DB_PATH)
    parser.add_argument("--processed_data_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--models_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--features_dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--datasets", type=str, nargs="+", default=["train", "valid", "dev", "test"])
    args = parser.parse_args()
    return args


def get_topk_docs(claim_feats, evidence_feats, k):
    distance = claim_feats * evidence_feats
    k = min(len(distance.data), k)
    top_k = np.argpartition(distance.data, k - 1)[:k]
    top_k = distance.indices[top_k]
    sorted_idx = np.argsort(top_k)
    return top_k[sorted_idx]


def expand_documents(
    sentences, line_extras, sentence_scores, curs, title_tfidf, title_feats, minimum_distance=150, query_limit=7
):
    additional_ids = []
    discount_maps = []
    for line_extra, sentence_score in zip(line_extras, sentence_scores):
        discount_map = {}
        for ent in line_extra:
            if len(ent) < 3 or all([char.isnumeric() for char in str(ent)]):
                continue
            ent_feats = title_tfidf.transform([ent])
            docs = get_topk_docs(ent_feats, title_feats, query_limit)
            for doc_id in docs:
                if doc_id not in discount_map:
                    additional_ids.append(doc_id)
                    discount_map[doc_id] = sentence_score
                elif discount_map[doc_id] < sentence_score:
                    additional_ids.append(doc_id)
                    discount_map[doc_id] = sentence_score
        discount_maps.append(discount_map)
    final_discount_map = {}
    for discount_map in discount_maps:
        for doc_id, discount_score in discount_map.items():
            if (
                doc_id in final_discount_map
                and final_discount_map[doc_id] < discount_score
            ):
                final_discount_map[doc_id] = discount_score
            elif doc_id not in final_discount_map:
                final_discount_map[doc_id] = discount_score
    return additional_ids, final_discount_map


def no_parens(string):
    return string and not string[0] == "(" and not string[-1] == ")"


def main():
    args = parse_args()
    conn = sqlite3.connect(args.db_path, timeout=1)
    curs = conn.cursor()
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_spellfix.extension_path())

    doc_query = (
        "SELECT DISTINCT t.page_name, l.page_id, l.line_num, l.line FROM lines as l "
        "JOIN texts as t ON l.page_id = t.page_id WHERE l.page_id IN ({})"
    )
    sentence_query = (
        "SELECT t.page_name, l.line, l.line_extra FROM lines AS l JOIN texts AS t "
        "ON l.page_id = t.page_id WHERE l.page_id=? and l.line_num=?"
    )
    title_feats = ((-1 * sparse.load_npz(args.features_dir / "tfidf/title_feats.npz")).T).tocsr()
    title_tfidf = load_pickle(args.models_dir / "title_vectorizer.pkl")

    for dataset in args.datasets:
        # dev_sentence_scores = np.load(f"data/processed/{dataset}/sentence_scores.npy")
        # dev_doc_ids = load_pickle(f"data/processed/{dataset}/expanded_doc_ids.pkl")
        dev_sentence_scores = np.load(args.processed_data_dir / dataset / "sentence_scores.npy")
        dev_doc_ids = load_pickle(args.processed_data_dir / dataset / "expanded_doc_ids.pkl")
        document_expanded_ids = []
        discount_maps = []
        for queried_doc_ids, sentence_scores in tqdm(
            zip(dev_doc_ids, dev_sentence_scores), total=len(dev_doc_ids)
        ):
            queried_doc_ids = set(queried_doc_ids)
            sentence_ids = []
            discount_scores = []
            for score in sentence_scores:
                sentence_ids.append((score[0], score[1]))
                discount_scores.append(score[2])
            sentences = [
                untokenize(curs.execute(sentence_query, ids).fetchone()[1])
                for ids in sentence_ids
            ]
            # This is a bit over aggresive, as it grabs both the href and the displayed string
            line_extras = [
                list(
                    set(
                        [
                            x.lower()
                            for x in curs.execute(sentence_query, ids)
                            .fetchone()[2]
                            .split("\t")
                        ]
                    )
                )
                for ids in sentence_ids
            ]
            new_docs = []
            all_docs, discount_map = expand_documents(
                sentences,
                line_extras,
                discount_scores,
                conn,
                title_tfidf,
                title_feats,
                minimum_distance=50,
                query_limit=1,
            )
            for new_doc in all_docs:
                if new_doc not in queried_doc_ids:
                    new_docs.append(new_doc)
            document_expanded_ids.append(list(set(new_docs)))
            discount_maps.append(discount_map)

        dump_pickle(
            document_expanded_ids,
            PROCESSED_DATA_DIR / dataset / "expanded_evidence_doc_ids.pkl",
        )
        dump_pickle(
            discount_maps,
            PROCESSED_DATA_DIR / dataset / "expanded_discount_maps.pkl",
        )


if __name__ == "__main__":
    main()
