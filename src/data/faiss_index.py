"""
A script for building a FAISS index and generating candidates via dense retrieval.
This was primarily intended for SciFact. Although using dense retrieval improved
SciFact's retrieval mechanism marginally, we decided to exclude it as it provided
no improvement in terms of prediction accuracy. The resulting candidates are saved
to `fuzzy_docs.pkl` somewhat erroneously, as we do not use this approach in conjunction
with fuzzy string search, so we reuse the filename here.
"""

import argparse
import os
import sqlite3
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer

from src.data.utils import dump_pickle, load_pickle, untokenize
from src.paths import DB_PATH, FEATURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR


os.environ["OMP_NUM_THREADS"] = "1"


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=Path, default=DB_PATH)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--models_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=FEATURES_DIR / "dense")
    args = parser.parse_args()
    return args


def untokenize_helper(data_tuple, max_line_len=300):
    """
    Untokenize and return clipped data, defined my `max_line_len`.
    """
    page_name, line = data_tuple
    untokenized_line = untokenize(line)
    return (page_name[:100], untokenized_line[:max_line_len])


def main():
    args = parse_args()
    conn = sqlite3.connect(args.db_path)
    curs = conn.cursor()
    query = "SELECT page_name, text FROM texts"
    results = [untokenize_helper(res) for res in curs.execute(query).fetchall()]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        encoded_data = model.encode(
            results, normalize_embeddings=True, show_progress_bar=True
        )
    index = faiss.IndexFlatL2(encoded_data.shape[1])  # build the index
    index.verbose = True
    index.add(encoded_data)  # add vectors to the index
    faiss.write_index(index, str(args.models_dir / "faiss.index"))
    del encoded_data
    with torch.no_grad():
        for dataset in ["train", "valid", "dev", "test"]:
            dataset_claims = load_pickle(args.processed_dir / dataset / "claims.pkl")
            dataset_claims_text = [claim.claim for claim in dataset_claims]
            encoded_dataset_claims = model.encode(
                dataset_claims_text, normalize_embeddings=True, show_progress_bar=True
            )
            topk_documents = index.search(encoded_dataset_claims, args.k)[1]
            dump_pickle(
                topk_documents.tolist(),
                PROCESSED_DATA_DIR / dataset / "fuzzy_docs.pkl",
            )


if __name__ == "__main__":
    main()
