"""
Script for computing TF-IDF distance between claim representations and document representations.
Some of the functions should probably be dundered, as they aren't workable outside of this script
due to the fact that the multiprocessing is using global variables.
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm

from src.paths import FEATURES_DIR


def compute_title_distance(i):
    k_val = k
    batch_claims = claims_title_feats[i : (i + 1)]
    batch_distances = batch_claims * evidence_title_feats
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


def compute_document_distance(i):
    k_val = k
    batch_claims = claims_document_feats[i : (i + 1)]
    batch_distances = batch_claims * evidence_document_feats
    if batch_distances.data.shape[0] <= k_val:
        k_val = batch_distances.data.shape[0] - 1
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


def get_separated_dataset_evidence_scores(feats_dir, threads, dataset):
    global claims_title_feats, claims_document_feats
    claims_title_feats = sparse.load_npz(
        feats_dir / f"{dataset}_claims_title_feats.npz"
    ).tocsr()
    claims_document_feats = sparse.load_npz(
        feats_dir / f"{dataset}_claims_doc_feats.npz"
    ).tocsr()
    pool = Pool(threads)
    print(f"Starting to compute {dataset} evidence scores...")
    t0 = time.time()
    indices = list(range(claims_title_feats.shape[0]))
    title_results = list(
        tqdm(pool.imap(compute_title_distance, indices), total=len(indices))
    )
    print("Title done")
    print("{}: {:.2f}s".format(0, (time.time() - t0)))
    title_results = np.array(title_results)
    np.save(feats_dir / f"{dataset}_title_scores-{k}.npy", title_results)

    print("Starting to compute document scores...")
    t0 = time.time()
    indices = list(range(claims_document_feats.shape[0]))
    document_results = list(
        tqdm(pool.imap(compute_document_distance, indices), total=len(indices))
    )
    print("Document done")
    print("{}: {:.2f}s".format(0, (time.time() - t0)))
    document_results = np.array(document_results)
    np.save(feats_dir / f"{dataset}_document_scores-{k}.npy", document_results)


def get_concat_dataset_evidence_scores(feats_dir, threads, dataset):
    global claims_title_feats, claims_document_feats
    claims_title_feats = sparse.load_npz(
        feats_dir / f"{dataset}_claims_cat_feats.npz"
    ).tocsr()
    pool = Pool(threads)
    print(f"Starting to compute {dataset} evidence scores...")
    t0 = time.time()
    indices = list(range(claims_title_feats.shape[0]))
    concat_results = list(
        tqdm(pool.imap(compute_title_distance, indices), total=len(indices))
    )
    print("Title done")
    print("{}: {:.2f}s".format(0, (time.time() - t0)))
    concat_results = np.array(concat_results)
    print(feats_dir / f"{dataset}_title_scores-{k}.npy")
    np.save(feats_dir / f"{dataset}_title_scores-{k}.npy", concat_results)
    np.save(feats_dir / f"{dataset}_document_scores-{k}.npy", concat_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generating TF-IDF features for evidence and claims"
    )
    parser.add_argument("--feature_dir", type=Path, default=FEATURES_DIR / "tfidf")
    parser.add_argument(
        "--datasets", type=Path, nargs="+", default=["train", "valid", "dev", "test"]
    )
    parser.add_argument(
        "--k", type=int, default=5
    )  # 5 if using separated tf-idf, 10 if using concatenated.
    parser.add_argument("--tfidf_type", type=str, choices=["cat", "sep"], default="sep")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    global evidence_title_feats, evidence_document_feats, k
    k = args.k
    if args.tfidf_type == "sep":
        evidence_title_feats = (-1 * sparse.load_npz(args.feature_dir / "title_feats.npz").T).tocsr()
        evidence_document_feats = (-1 * sparse.load_npz(args.feature_dir / "doc_feats.npz").T).tocsr()
        for dataset in args.datasets:
            get_separated_dataset_evidence_scores(args.feature_dir, args.threads, dataset)
    else:
        evidence_title_feats = (-1 * sparse.load_npz(args.feature_dir / "concat_feats.npz").T).tocsr()
        for dataset in args.datasets:
            get_concat_dataset_evidence_scores(args.feature_dir, args.threads, dataset)
