"""
Script for organizing various components of the BEVERS pipeline into a single class.
"""
import sqlite3

import numpy as np
import spacy
import sqlite_spellfix
import torch
from scipy import sparse
from transformers import RobertaTokenizerFast

from src.data.utils import load_pickle, untokenize
from src.paths import DB_PATH, FEATURES_DIR, MODELS_DIR

### DEFAULTS
TITLE_TFIDF = MODELS_DIR / "title_vectorizer.pickle"
TITLE_TFIDFF_FEATS = (
    -1 * sparse.load_npz(FEATURES_DIR / "tfidf" / "title_feats.npz")
).T.tocsr()
DOCUMENT_TFIDF = MODELS_DIR / "document_vectorizer.pickle"
DOCUMENT_TFIDF_FEATS = (
    -1 * sparse.load_npz(FEATURES_DIR / "tfidf" / "doc_feats.npz")
).T.tocsr()
SS_MODEL = (
    MODELS_DIR
    / "sentence_selection"
    / "ss_roberta-large_k_20_random_expanded_epoch=00-valid_loss=0.08616.ckpt"
)
CC_MODEL = (
    MODELS_DIR
    / "claim_classification"
    / "cc_roberta_concat_epoch=00-valid_loss=0.68141.ckpt"
)
AGGREGATOR_MODEL = None


def strip_bad_stopwords(entity):
    first_index = 0
    last_index = len(entity) - 1
    for i in range(first_index, last_index):
        if not entity[i].is_stop:
            first_index = i
            break
    for i in range(last_index, first_index, -1):
        if not entity[i].is_stop:
            last_index = i
            break
    return entity[first_index : last_index + 1]


# duplicated -- refactor needed
def expand_documents(nlp, sentence, curs, minimum_distance=150, query_limit=7):
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
        str(strip_bad_stopwords(ent)).replace('"', '""')
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
        query = (
            "SELECT rowid, word, distance FROM clean_titles_default_cost "
            f"WHERE word MATCH ? AND distance <= {minimum_distance * 3} ORDER BY distance"
        )
        results = curs.execute(query, (ent,)).fetchall()
        filtered_results = [
            result
            for i, result in enumerate(results)
            if result[2] <= minimum_distance or i < query_limit
        ]
        for result in filtered_results[: query_limit * 3]:
            additional_ids.append(result[0])
    return additional_ids


class FeverModel:
    def __init__(
        self,
        db_path=DB_PATH,
        title_tfidf_path=TITLE_TFIDF,
        document_tfidf_path=DOCUMENT_TFIDF,
        title_tfidf_feats=TITLE_TFIDFF_FEATS,
        document_tfidf_feats=DOCUMENT_TFIDF_FEATS,
        ss_model_path=SS_MODEL,
        cc_model_path=CC_MODEL,
        aggregator_path=AGGREGATOR_MODEL,
        device="cpu",
        dtype=torch.bfloat16,
        batch_size=4,
    ):
        self.db_path = db_path
        self.conn = self.__init_db__()
        self.title_tfidf_path = title_tfidf_path
        self.title_tfidf = load_pickle(title_tfidf_path)
        self.document_tfidf_path = document_tfidf_path
        self.document_tfidf = load_pickle(document_tfidf_path)
        self.title_feats = title_tfidf_feats
        self.doc_feats = document_tfidf_feats.T.tocsr()
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.ss_model_path = ss_model_path
        self.ss_model = self.load_model(SS_RoBERTa, ss_model_path)
        self.cc_model_path = cc_model_path
        self.cc_model = self.load_model(CC_RoBERTa, cc_model_path)
        self.nlp = spacy.load("en_core_web_trf")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")

    def __init_db__(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_spellfix.extension_path())
        return conn

    def load_model(self, model_type, model_path):
        ckpt_state_dict = torch.load(
            model_path, map_location=torch.device(self.device)
        )["state_dict"]
        model = model_type("roberta-large", 3)
        if "roberta.loss_fct.weight" in ckpt_state_dict:
            del ckpt_state_dict["roberta.loss_fct.weight"]
        model.load_state_dict(ckpt_state_dict)
        model.eval()
        return model

    def retrieve_docs(self, claim, k):
        claim_title_feats = self.title_tfidf.transform([claim])
        self.document_tfidf.transform(claim)
        top_title_docs = self.get_topk_docs(claim_title_feats, self.title_feats, k)
        self.get_topk_docs(claim_doc_feats, self.doc_feats, k)
        expanded_ids = expand_documents(self.nlp, claim, self.conn)
        return set(top_title_docs + expanded_ids)

    def get_topk_docs(self, claim_feats, evidence_feats, k):
        distance = claim_feats * evidence_feats
        k = min(len(distance.data), k)
        top_k = np.argpartition(distance.data, k - 1)[:k]
        top_k = distance.indices[k]
        sorted_idx = np.argsort(top_k)
        return sorted_idx

    def get_sentences(self, topk_docs, limit=1e6):
        doc_query = (
            "SELECT DISTINCT t.page_name, l.page_id, l.line_num, l.line FROM lines as l "
            "JOIN texts as t ON l.page_id = t.page_id WHERE l.page_id IN ({})"
        )
        doc_query = doc_query.format(",".join([str(doc_id) for doc_id in topk_docs]))
        query_results = list(self.conn.execute(doc_query))
        sentences = [
            (untokenize(res[0]), untokenize(res[3]))
            for res in query_results
            if untokenize(res[3]).strip()
        ]
        sentence_ids = [
            (res[1], res[2]) for res in query_results if untokenize(res[3]).strip()
        ]
        return sentences, sentence_ids

    def get_topk_sentences(self, claim, sentences, ids, k=5):
        all_preds = []
        i = 0
        with torch.no_grad():
            while i < len(sentences):
                batch_sentences = sentences[i : i + self.batch_size]
                batch_inputs = [
                    (title + " -- " + line, claim) for title, line in batch_sentences
                ]
                batch_input_ids = self.tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=256,
                )
                preds = self.ss_model(batch_input_ids).logits.float().detach().cpu()
                del batch_input_ids
                all_preds += [preds]
                i = i + self.batch_size
        all_preds = torch.cat(all_preds, dim=0)
        softmax_scores = torch.nn.functional.softmax(all_preds, dim=1)
        sentence_scores = 1 - softmax_scores[:, 1]  #
        args_sort = torch.argsort(sentence_scores, descending=True)
        topk = args_sort[:k]
        return (
            [sentences[top] for top in topk],
            [ids[top] for top in topk],
            [sentence_scores[top].item() for top in topk],
        )

    def predict_claim_with_sentences(self, claim, topk_sentences):
        formatted_evidence = " </s></s> ".join(
            [page_name + " -- " + line for page_name, line in topk_sentences[:2]]
        )
        tokenized_inputs = self.tokenizer(
            [(claim, formatted_evidence)],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=368,
        )
        with torch.no_grad():
            pred = self.cc_model(tokenized_inputs).logits.detach().float().cpu()
            softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        return softmax_pred

    def predict_claim(self, claim):
        topk_docs = self.retrieve_docs(claim, 5)
        sentences, sentence_ids = self.get_sentences(topk_docs)
        (
            topk_sentences,
            topk_sentence_ids,
            topk_sentence_scores,
        ) = self.get_topk_sentences(sentences, sentence_ids)
        return self.predict_claim_with_sentences(claim, topk_sentences)
