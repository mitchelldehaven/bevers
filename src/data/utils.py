"""
Module containing several help functions and dataclasses used throughout the system.
"""
import pickle
from dataclasses import dataclass
from typing import List


@dataclass
class Evidence:
    annotator_id: int = None
    evidence_id: int = None
    page_title: str = None
    line_number: int = None
    untok_page_title: str = None
    page_id: int = None
    og_page_id: int = None

    def __post_init__(self):
        if self.page_title:
            self.untok_page_title = untokenize(self.page_title, replace_underscore=True)


@dataclass
class DataSample:
    id: str = None
    claim: str = None
    verifiable: str = None
    label: str = None
    evidence: List[Evidence] = None
    cited_doc_ids: int = None

    def get_verifiable_num(self):
        if self.verifiable == "VERIFIABLE":
            return 1
        return 0

    def get_label_num(self):
        if self.label in ["REFUTES", "CONTRADICT"]:
            return 0
        if self.label in ["SUPPORT", "SUPPORTS"]:
            return 2
        return 1

    def correct_evidence(self, pred_evidence):
        for evidence_group in self.evidence:
            actual_sentences = [[e.page_id, e.line_number] for e in evidence_group]
            # Only return true if an entire group of actual sentences is in the predicted sentences
            if all([actual_sent in pred_evidence for actual_sent in actual_sentences]):
                return True
        return False


def dump_pickle(obj, filename, open_type="wb", protocol=pickle.HIGHEST_PROTOCOL):
    """
    Simple function for dumping pickles.
    """
    with open(filename, open_type) as pickle_fp:
        pickle.dump(obj, pickle_fp, protocol=protocol)


def load_pickle(filename, open_type="rb"):
    """
    Simple function for loading pickles.
    """
    with open(filename, open_type) as pickle_fp:
        return pickle.load(pickle_fp)


replacements = {
    "-LRB-": "(",
    "-LSB-": "[",
    "-LCB-": "{",
    "-RCB-": "}",
    "-RRB-": ")",
    "-RSB-": "]",
    "-COLON-": ":",
}

# KGAT's evidence lines for some reason have the hyphens stripped out of the special characters
# so we need to have a second set to clean these up as well
kgat_replacements = {
    "LRB": "(",
    "LSB": "[",
    "LCB": "{",
    "RCB": "}",
    "RRB": ")",
    "RSB": "]",
    "COLON": ":",
}


def untokenize(sentence, replace_underscore=False, replace_kgat=False):
    """
    Function to untokenize data from the processing setup for data.db.
    """
    for bad, good in replacements.items():
        sentence = sentence.replace(bad, good)
    if replace_underscore:
        sentence = sentence.replace("_", " ")
    if replace_kgat:
        for bad, good in kgat_replacements.items():
            sentence = sentence.replace(bad, good)
    return sentence


def get_title_line(conn, evidence):
    """
    Function to return title and line given evidence page_id and line_number.
    """
    page_id, line_number = evidence
    query = (
        "SELECT t.page_name, l.line FROM lines AS l JOIN texts AS t "
        "ON l.page_id = t.page_id WHERE l.page_id=? and l.line_num=?"
    )
    results = conn.execute(query, (page_id, line_number)).fetchone()
    title, line = results
    return title, untokenize(line)
