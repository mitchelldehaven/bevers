import argparse
import gc
import json
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tfidf import (
    generate_document_tfidf_features,
    generate_tfidf_features,
    generate_title_tfidf_features,
)
from tqdm import tqdm

from src.data.utils import load_pickle
from src.paths import DB_PATH, PROCESSED_DATA_DIR, ROOT_DIR


def compute_tfidf_distance(i):
    k_val = k
    batch_claims = claim_feats[i : (i + 1)]
    batch_distances = batch_claims * evidence_feats
    if batch_distances.data.shape[0] < k_val:
        k_val = batch_distances.data.shape[0]
    top_k = np.argpartition(batch_distances.data, k_val - 1)[:k_val]
    top_k_dist = batch_distances.data[top_k]
    top_k = batch_distances.indices[top_k]
    sorted_idx = np.argsort(top_k_dist)
    sorted_top_k_dist = -1 * top_k_dist[sorted_idx]
    sorted_top_k = top_k[sorted_idx]
    if sorted_top_k.shape[0] < k:
        zeros = np.zeros(k - sorted_top_k.shape[0])
        neg_ones = -1 * np.ones(k - sorted_top_k.shape[0])
        sorted_top_k_dist = np.concatenate([sorted_top_k_dist, zeros])
        sorted_top_k = np.concatenate([sorted_top_k, neg_ones])

    assert sorted_top_k.shape[0] == k, "shape is {}".format(sorted_top_k.shape[0])
    return sorted_top_k, sorted_top_k_dist


def get_dataset_evidence_scores(threads):
    pool = Pool(threads)
    indices = list(range(claim_feats.shape[0]))
    results = list(
        tqdm(
            pool.imap(compute_tfidf_distance, indices), total=len(indices), leave=False
        )
    )
    results = np.array(results)
    return results


def get_param_options(db_path):
    param_set = {
        # "min_df": [1],
        # "token_pattern": [r"(?u)\b\w+\b"],
        # "dtype": [np.float32],
        "max_features": [50_000_000],
        # "standard_idf": [True],
        "db_path": [db_path],
        "save": [False],
        "max_df": [1.0],
        "lowercase": [True, False],
        "max_ngrams": [1, 2],
        "norm": ["l2", None],
        "strip_accents": ["ascii", None],
        "sublinear_tf": [True, False],
    }
    param_options = {}
    for i, param_values in enumerate(product(*list(param_set.values()))):
        param_option = dict(zip(list(param_set.keys()), param_values))
        param_options[i] = param_option
    return param_options


import gc


def main(args):
    expt_results_dir = args.expt_dir / "data"
    expt_conf_dir = args.expt_dir / "conf"
    expt_results_dir.mkdir(parents=True, exist_ok=True)
    expt_conf_dir.mkdir(parents=True, exist_ok=True)
    param_options = get_param_options(args.db_path)
    iterable = [
        ("title", args.k // 2, generate_title_tfidf_features),
        ("document", args.k // 2, generate_document_tfidf_features),
        ("concat", args.k, generate_tfidf_features),
    ]
    # Only running tests against dev
    claims = load_pickle(args.processed_data_dir / "dev" / "claims.pkl")
    claims_text = [claim.claim for claim in claims]
    for i, param_option in tqdm(param_options.items()):
        if i < args.restart:
            print("Skipping expt", i)
            continue
        param_option["db_path"] = str(param_option["db_path"])
        conf_file = expt_conf_dir / f"expt{i}.json"
        with open(conf_file, "w") as f:
            print(json.dumps(param_option, indent=2), file=f)
        global evidence_feats, claim_feats, k
        for save_string, k, tfidf_function in iterable:
            if save_string not in args.tfidf_types:
                continue
            param_args = SimpleNamespace(**param_option)
            print(save_string)
            gc.collect()
            evidence_feats, vectorizer = tfidf_function(param_args)
            evidence_feats = (-1 * evidence_feats).T.tocsr()
            claim_feats = vectorizer.transform(claims_text).tocsr()
            # print(evidence_feats.shape)
            # print(claim_feats.shape)
            topk_results = get_dataset_evidence_scores(args.threads)
            feats_file = expt_results_dir / f"expt{i}_{save_string}_{k}.npy"
            np.save(feats_file, topk_results)
            evidence_feats, claim_feats, vectorizer, topk_results = (
                None,
                None,
                None,
                None,
            )
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generating TF-IDF features for evidence and claims"
    )
    parser.add_argument("--db_path", type=Path, default=DB_PATH)
    parser.add_argument("--expt_dir", type=Path, default=ROOT_DIR / "expts")
    parser.add_argument("--processed_data_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--restart", type=int, default=-1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tfidf_types", nargs="+", default=["title", "document", "concat"])
    args = parser.parse_args()
    main(args)
