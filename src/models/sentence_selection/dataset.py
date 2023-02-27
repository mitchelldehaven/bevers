import sqlite3

import torch
from torch.utils import data
from tqdm import tqdm

from src.data.utils import untokenize


def collate_fn(batch, tokenizer, max_length):
    batch_x, batch_y = [list(x) for x in zip(*batch)]
    batch_x = tokenizer(
        batch_x,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    batch_y = torch.tensor(batch_y)
    return batch_x, batch_y


class SentenceDatasetRoBERTa(data.Dataset):
    def __init__(self, claims, data, tokenizer, binary_label=False, claim_second=False):
        self.claims = claims
        self.data = data
        self.tokenizer = tokenizer
        self.binary_label = binary_label
        self.claim_second = claim_second

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        claim, candidate_title, candidate_sentence, label = self.data[idx]
        if self.binary_label:
            label = 0 if label == 1 else 1
        claim = str(claim)
        claim = claim[0].upper() + claim[1:]
        candidate_title = str(candidate_title)
        candidate_sentence = str(candidate_sentence)
        if candidate_title == "nan":
            return (candidate_sentence, claim), label
        else:
            return (candidate_title + " -- " + candidate_sentence, claim), label


# Define static dataloader to test various negative sampling approaches
class SentenceDatasetRoBERTaDev(data.Dataset):
    def __init__(
        self,
        claims,
        claims_top_docs,
        tokenizer,
        db_file,
        binary_label=False,
        claim_second=False,
    ):
        self.claims = claims
        self.claims_top_docs = claims_top_docs
        self.tokenizer = tokenizer
        self.binary_label = binary_label
        self.claim_second = claim_second
        self.db_file = db_file
        self.dataset = []
        self.__init_dataset__()

    def __init_dataset__(self):
        conn = sqlite3.connect(self.db_file)
        doc_query = "SELECT DISTINCT t.page_name, l.page_id, l.line_num, l.line FROM lines as l JOIN texts as t ON l.page_id = t.page_id WHERE l.page_id IN ({})"
        evidence_query = "SELECT t.page_name, l.line FROM lines as l JOIN texts as t ON l.page_id = t.page_id WHERE l.page_id=? AND l.line_num=?"
        for claim, top_docs in tqdm(zip(self.claims, self.claims_top_docs)):
            gold_evidence = set()
            gold_results = []
            if claim.evidence is not None:
                for evidence_group in claim.evidence:
                    for evidence in evidence_group:
                        evidence_tuple = (evidence.page_id, evidence.line_number)
                        if evidence_tuple not in gold_evidence:
                            gold_evidence.add(evidence_tuple)
                        else:
                            continue
                        gold_results += conn.execute(
                            evidence_query, evidence_tuple
                        ).fetchall()
            for (page_name, line) in gold_results:
                self.dataset.append(
                    (
                        untokenize(page_name),
                        untokenize(line),
                        claim.claim,
                        claim.get_label_num(),
                    )  # 1 is NEI label
                )
            this_query = (
                doc_query.format(list(top_docs)).replace("[", "").replace("]", "")
            )
            results = conn.execute(this_query).fetchall()
            for (page_name, page_id, line_num, line) in results:
                if (
                    len(line) > 5
                    and "list of" not in page_name.lower()
                    and "index of" not in page_name.lower()
                    and (page_id, line_num) not in gold_evidence
                ):
                    self.dataset.append(
                        (
                            untokenize(page_name),
                            untokenize(line),
                            claim.claim,
                            1,
                        )  # 1 is NEI label
                    )
        conn.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        page_name, line, claim, label = self.dataset[idx]
        return (page_name + " -- " + line, claim), label
