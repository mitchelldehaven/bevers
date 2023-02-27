"""
Script for generating TF-IDF representations and building associated top-k documents.
"""
import argparse
import json
import pickle
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.utils import load_pickle
from src.paths import DB_PATH, FEATURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR

stopwords = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
]


TITLE_TFIDF_PARAMS = {
    "max_features": 50000000,
    "max_df": 1.0,
    "lowercase": True,
    "max_ngrams": 2,
    "norm": "l2", 
    "strip_accents": "ascii",
    "sublinear_tf": True,
}


DOCUMENT_TFIDF_PARAMS = {
    "max_features": 50000000,
    "max_df": 1.0,
    "lowercase": False,
    "max_ngrams": 2,
    "norm": None,
    "strip_accents": "ascii",
    "sublinear_tf": True,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generating TF-IDF features for evidence and claims"
    )
    parser.add_argument("--db_path", type=Path, default=DB_PATH)
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DATA_DIR)
    # TODO: fix with str2bool function
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--dest_path", type=Path, default=FEATURES_DIR / "tfidf")
    parser.add_argument("--model_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--config_file", type=Path)
    parser.add_argument("--tfidf_type", type=str, choices=["cat", "sep"], default="sep")
    parser.add_argument("--claims_file", type=str, default="claims.pkl")
    parser.add_argument("--datasets", type=Path, nargs="+", default=["train", "valid", "dev", "test"])
    args = parser.parse_args()
    return args


def save_tfidf_model(model, model_filename):
    model_path = MODELS_DIR / model_filename
    print("About to save", model_filename)
    joblib.dump(model, model_path)
    print("Done saving", model_filename)


def generate_title_tfidf_features(args):
    """
    Function for generating title TF-IDF features.
    Resulting "model" and featurized documents are
    saved to the args.dest_path, which by default is
    FEATURES_DIR / "tfidf".
    """
    conn = sqlite3.connect(args.db_path)
    curs = conn.cursor()
    query = curs.execute("SELECT page_name FROM texts")
    titles = [title[0] for title in query.fetchall()]
    cleaned_titles = [re.sub(r"\([^)]*\)", "", title).strip() for title in titles]
    title_vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=args.max_df,
        lowercase=args.lowercase,
        ngram_range=(1, args.max_ngrams),
        dtype=np.float32,
        max_features=args.max_features,
        norm=args.norm,
        strip_accents=args.strip_accents,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=args.sublinear_tf,
        standard_idf=True,
    )
    title_vectorizer.fit(cleaned_titles)
    if args.save:
        with open(args.model_dir / "title_vectorizer.pkl", "wb") as f:
            pickle.dump(title_vectorizer, f)
    title_vecs = title_vectorizer.transform(titles).tocsr()
    if args.save:
        scipy.sparse.save_npz(args.dest_path / "title_feats.npz", title_vecs)
    return title_vecs, title_vectorizer


def generate_document_tfidf_features(args):
    """
    Function for generating document TF-IDF features.
    Resulting "model" and featurized documents are
    saved to the args.dest_path, which by default is
    FEATURES_DIR / "tfidf".
    """
    conn = sqlite3.connect(args.db_path)
    curs = conn.cursor()
    query = curs.execute("SELECT text FROM texts")
    texts = [text[0] for text in query.fetchall()]
    document_vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=args.max_df,
        lowercase=args.lowercase,
        ngram_range=(1, args.max_ngrams),
        dtype=np.float32,
        max_features=args.max_features,
        norm=args.norm,
        strip_accents=args.strip_accents,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=args.sublinear_tf,
        standard_idf=True,
    )
    document_vectorizer.fit(texts)
    if args.save:
        with open(args.model_dir / "document_vectorizer.pkl", "wb") as f:
            pickle.dump(document_vectorizer, f)
    document_vecs = document_vectorizer.transform(texts)
    if args.save:
        scipy.sparse.save_npz(args.dest_path / "doc_feats.npz", document_vecs)
    return document_vecs, document_vectorizer


def generate_tfidf_features(args):
    """
    Function for generating concatenated TF-IDF features.
    Resulting "model" and featurized documents are
    saved to the args.dest_path, which by default is
    FEATURES_DIR / "tfidf".
    """
    conn = sqlite3.connect(args.db_path)
    curs = conn.cursor()
    query = curs.execute("SELECT page_name, text FROM texts")
    results = query.fetchall()
    titles, documents = zip(*results)
    cleaned_titles = [re.sub(r"\([^)]*\)", "", title).strip() for title in titles]
    concat_documents = [
        title + " --- " + document for title, document in zip(cleaned_titles, documents)
    ]
    del cleaned_titles
    concat_vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=1.0,
        lowercase=args.lowercase,
        ngram_range=(1, args.max_ngrams),
        dtype=np.float32,
        max_features=args.max_features,
        norm=args.norm,
        strip_accents=args.strip_accents,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=args.sublinear_tf,
        standard_idf=True,
    )
    concat_vectorizer.fit(concat_documents)
    del concat_documents
    if args.save:
        with open(MODELS_DIR / "concat_vectorizer.pkl", "wb") as f:
            pickle.dump(concat_vectorizer, f)
    concat_vecs = concat_vectorizer.transform(titles) + concat_vectorizer.transform(
        documents
    )
    if args.save:
        scipy.sparse.save_npz(args.dest_path / "concat_feats.npz", concat_vecs)
    return concat_vecs, concat_vectorizer


def apply_document_tfidf(
    sentences, tfidf_model_path=MODELS_DIR / "document_vectorizer.pkl"
):
    tfidf_model = load_pickle(tfidf_model_path)
    sentences_features = tfidf_model.transform(sentences).tocsr()
    return sentences_features


def apply_title_tfidf(sentences, tfidf_model_path=MODELS_DIR / "title_vectorizer.pkl"):
    tfidf_model = load_pickle(tfidf_model_path)
    sentences_features = tfidf_model.transform(sentences).tocsr()
    return sentences_features


def main_sep(title_args, document_args, args):
    """
    Main function for running separated TF-IDF feature generation.
    """
    title_args.dest_path.mkdir(exist_ok=True, parents=True)
    _, doc_vectorizer = generate_document_tfidf_features(document_args)
    _, title_vectorizer = generate_title_tfidf_features(title_args)
    for dataset in args.datasets:
        claims = load_pickle(PROCESSED_DATA_DIR / dataset / args.claims_file)
        claims_text = [claim.claim for claim in claims]
        claims_title_feats = title_vectorizer.transform(claims_text).tocsr()
        claims_doc_feats = doc_vectorizer.transform(claims_text).tocsr()
        scipy.sparse.save_npz(
            args.dest_path / f"{dataset}_claims_title_feats.npz", claims_title_feats
        )
        scipy.sparse.save_npz(
            args.dest_path / f"{dataset}_claims_doc_feats.npz", claims_doc_feats
        )


def main_cat(cat_args, args):
    """
    Main function for running concatenated TF-IDF feature generation.
    """
    cat_args.dest_path.mkdir(exist_ok=True, parents=True)
    _, cat_vectorizer = generate_tfidf_features(cat_args)
    for dataset in args.datasets:
        claims = load_pickle(PROCESSED_DATA_DIR / dataset / args.claims_file)
        claims_text = [claim.claim for claim in claims]
        claims_feats = cat_vectorizer.transform(claims_text).tocsr()
        scipy.sparse.save_npz(
            args.dest_path / f"{dataset}_claims_cat_feats.npz", claims_feats
        )


def main():
    args = parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            config = json.load(f)
        TITLE_TFIDF_PARAMS.update(config)
    for param in [TITLE_TFIDF_PARAMS, DOCUMENT_TFIDF_PARAMS]:
        param["db_path"] = args.db_path
        param["save"] = args.save
        param["dest_path"] = args.dest_path
        param["model_dir"] = args.model_dir
    title_args = SimpleNamespace(**TITLE_TFIDF_PARAMS)
    document_args = SimpleNamespace(**DOCUMENT_TFIDF_PARAMS)
    if args.tfidf_type == "cat":
        main_cat(title_args, args)
    else:
        main_sep(title_args, document_args, args)


if __name__ == "__main__":
    main()
