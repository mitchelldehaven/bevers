from src.data.utils import load_pickle
from src.paths import MODELS_DIR


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
