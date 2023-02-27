import random
import sqlite3

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from src.data.utils import get_title_line, untokenize


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


class ClaimClassificationDataset(data.Dataset):
    def __init__(
        self,
        claims,
        topk_sentences,
        db_path,
        claim_second=False,
        train=False,
        static_order=True,
        over_sample=True,
        shuffle_evidence_p=0.3,
    ):
        self.claims = claims
        self.topk_sentences = topk_sentences
        self.db_path = db_path
        self.data = []
        self.data_labels = []
        self.train = train
        self.claim_second = claim_second
        self.static_order = static_order
        self.over_sample = over_sample
        self.shuffle_evidence_p = shuffle_evidence_p
        self.processed_dict = {}
        # if self.train:
        # self.__initdatasetv1__()
        self.__initdataset__()

    def __initdatasetv2__(self):  ### seemingly more principled approach
        conn = sqlite3.connect(self.db_path)
        curs = conn.cursor()
        query = "SELECT t.page_name, l.line FROM lines AS l JOIN texts AS t ON l.page_id = t.page_id WHERE l.page_id=? and l.line_num=?"
        # Initializing the dataset in this way should happen outside of worker, because work gets done multiple times.
        # for i, claim in enumerate(tqdm(self.claims, total=len(self.claims), desc="Initializing Dataset")):
        for i, claim in enumerate(self.claims, total=len(self.claims)):
            gold_claim_evidences = set()
            if claim.label != "NOT ENOUGH INFO":
                gold_claim_evidences = [
                    (evidence.page_id, evidence.line_number)
                    for evidence_group in claim.evidence
                    for evidence in evidence_group
                    if len(evidence_group) < 3
                ]
                gold_claim_evidences = set(gold_claim_evidences)
            pred_claim_evidences = self.topk_sentences[
                i, :, :2
            ]  # ignore retreival scores
            pred_claim_evidences = [
                tuple([int(xx) for xx in list(x)]) for x in pred_claim_evidences
            ]
            pred_claim_evidences = [
                pred_evidence
                for pred_evidence in pred_claim_evidences
                if pred_evidence not in gold_claim_evidences
            ]
            # gold_evidence = [curs.execute(query, claim_evidence).fetchone() for claim_evidence in gold_claim_evidences] if gold_claim_evidences else []
            pred_evidence = [
                curs.execute(query, claim_evidence).fetchone()
                for claim_evidence in pred_claim_evidences
            ]
            if claim.label != "NOT ENOUGH INFO":
                used_evidence = set()
                # for evidence_gr in claim.evidence[:3]:
                #     unique_evidence.add(evidence)
                for evidence_group in claim.evidence[:3]:
                    if len(evidence_group) == 1:
                        evidence = evidence_group[0]
                        if (evidence.page_id, evidence.line_number) in used_evidence:
                            continue
                        used_evidence.add((evidence.page_id, evidence.line_number))
                        evidence_title, evidence_line = get_title_line(
                            conn, (evidence.page_id, evidence.line_number)
                        )
                        all_evidence = [(evidence_title, evidence_line)]
                        # formatted_evidence = "{} -- {}".format(evidence_title, evidence_line)
                    elif len(evidence_group) == 2:
                        first_evidence, second_evidence = evidence_group
                        first_title, first_line = get_title_line(
                            conn, (first_evidence.page_id, first_evidence.line_number)
                        )
                        second_title, second_line = get_title_line(
                            conn, (second_evidence.page_id, second_evidence.line_number)
                        )
                        all_evidence = [
                            (first_title, first_line),
                            (second_title, second_line),
                        ]
                        # formatted_evidence = "{} -- {} </s></s> {} -- {}".format(first_title, first_line, second_title, second_line)
                    else:
                        continue
                    sample = (claim.claim, all_evidence)
                    self.data.append(sample)
                    self.data_labels.append(claim.get_label_num())
                    if claim.label == "REFUTES":
                        self.data.append(sample)
                        self.data.append(sample)
                        self.data_labels.append(claim.get_label_num())
                        self.data_labels.append(claim.get_label_num())
            num_nei_samples = 1  # if claim.label != "NOT ENOUGH INFO" else 3
            num_nei_samples = min(len(pred_evidence), num_nei_samples)
            sampled_pred_evidence = random.sample(pred_evidence, k=num_nei_samples)
            for i, (page_name, evidence_line) in enumerate(sampled_pred_evidence):
                if (
                    random.random() < 0.1
                    and len(sampled_pred_evidence) > 1
                    and i < len(sampled_pred_evidence) - 1
                ):
                    random_page_name, random_evidence_line = sampled_pred_evidence[
                        i + 1
                    ]
                    all_evidence = [
                        (page_name, evidence_line),
                        (random_page_name, random_evidence_line),
                    ]
                    # formatted_evidence = "{} -- {}</s></s>{} -- {}".format(page_name, evidence_line, random_page_name, random_evidence_line)
                else:
                    all_evidence = [(page_name, evidence_line)]
                    "{} -- {}".format(page_name, evidence_line)
                sample = (claim.claim, all_evidence)
                self.data.append(sample)
                self.data_labels.append(1)  # nei

        dist_dict = {0: 0, 1: 0, 2: 0}
        for label in self.data_labels:
            dist_dict[label] += 1
        print(dist_dict)

    # def __initdataset__(self): ### OG method
    #     conn = sqlite3.connect(self.db_path)
    #     curs = conn.cursor()
    #     query = "SELECT t.page_name, l.line FROM lines AS l JOIN texts AS t ON l.page_id = t.page_id WHERE l.page_id=? and l.line_num=?"
    #     # Initializing the dataset in this way should happen outside of worker, because work gets done multiple times.
    #     for i, claim in enumerate(tqdm(self.claims, total=len(self.claims), desc="Initializing Dataset")):
    #         gold_claim_evidences = []
    #         if claim.label != "NOT ENOUGH INFO":
    #             gold_claim_evidences = [(evidence.page_id, evidence.line_number) for evidence_group in claim.evidence for evidence in evidence_group if len(evidence_group) == 1 ]
    #         pred_claim_evidences = self.topk_sentences[i,:,:2] # ignore retreival scores
    #         pred_claim_evidences = [tuple([int(xx) for xx in list(x)]) for x in pred_claim_evidences]
    #         pred_claim_evidences = [pred_evidence for pred_evidence in pred_claim_evidences if pred_evidence not in gold_claim_evidences]
    #         gold_claim_evidences = set(gold_claim_evidences)
    #         gold_evidence = [curs.execute(query, claim_evidence).fetchone() for claim_evidence in gold_claim_evidences] if gold_claim_evidences else []
    #         pred_evidence = [curs.execute(query, claim_evidence).fetchone() for claim_evidence in pred_claim_evidences]
    #         for page_name, evidence_line in gold_evidence:
    #             if evidence_line is not None:
    #                 evidence_line = untokenize(evidence_line)
    #                 if self.claim_second:
    #                     sample = (page_name, evidence_line, claim.claim)
    #                     # sample = (evidence_line, claim.claim)
    #                 if claim.get_label_num() == 0 and self.over_sample:
    #                     self.data.append(sample)
    #                     self.data_labels.append(claim.get_label_num())
    #                     self.data.append(sample)
    #                     self.data_labels.append(claim.get_label_num())
    #                 self.data.append(sample)
    #                 self.data_labels.append(claim.get_label_num())
    #         if not gold_evidence and claim.label == "NOT ENOUGH INFO":
    #             for page_name, evidence_line in pred_evidence:
    #                 if evidence_line is not None:
    #                     evidence_line = untokenize(evidence_line)
    #                     if self.claim_second:
    #                         sample = (page_name, evidence_line, claim.claim)
    #                         # sample = (evidence_line, claim.claim)
    #                     self.data.append(sample)
    #                     self.data_labels.append(1)
    #                     if self.over_sample:
    #                         self.data.append(sample)
    #                         self.data_labels.append(1)

    #     dist_dict = {0: 0, 1: 0, 2:0}
    #     for label in self.data_labels:
    #         dist_dict[label] += 1
    #     print(dist_dict)

    def __initdatasetv1__(self):  ### OG method
        conn = sqlite3.connect(self.db_path)
        curs = conn.cursor()
        query = "SELECT t.page_name, l.line FROM lines AS l JOIN texts AS t ON l.page_id = t.page_id WHERE l.page_id=? and l.line_num=?"
        # Initializing the dataset in this way should happen outside of worker, because work gets done multiple times.
        for i, claim in enumerate(
            tqdm(self.claims, total=len(self.claims), desc="Initializing Dataset")
        ):
            gold_claim_evidences = []
            if claim.label != "NOT ENOUGH INFO":
                gold_claim_evidences = [
                    (evidence.page_id, evidence.line_number)
                    for evidence_group in claim.evidence
                    for evidence in evidence_group
                    if len(evidence_group) == 1
                ]
            pred_claim_evidences = self.topk_sentences[
                i, :, :2
            ]  # ignore retreival scores
            pred_claim_evidences = [
                tuple([int(xx) for xx in list(x)]) for x in pred_claim_evidences
            ]
            pred_claim_evidences = [
                pred_evidence
                for pred_evidence in pred_claim_evidences
                if pred_evidence not in gold_claim_evidences
            ]
            gold_claim_evidences = set(gold_claim_evidences)
            gold_evidence = (
                [
                    curs.execute(query, claim_evidence).fetchone()
                    for claim_evidence in gold_claim_evidences
                ]
                if gold_claim_evidences
                else []
            )
            pred_evidence = [
                curs.execute(query, claim_evidence).fetchone()
                for claim_evidence in pred_claim_evidences
            ]
            for page_name, evidence_line in gold_evidence:
                if evidence_line is not None:
                    evidence_line = untokenize(evidence_line)
                    sample = (claim.claim, [(page_name, evidence_line)])
                    # sample = (evidence_line, claim.claim)
                    if claim.get_label_num() == 0 and self.over_sample:
                        self.data.append(sample)
                        self.data_labels.append(claim.get_label_num())
                        self.data.append(sample)
                        self.data_labels.append(claim.get_label_num())
                    self.data.append(sample)
                    self.data_labels.append(claim.get_label_num())
            if not gold_evidence and claim.label == "NOT ENOUGH INFO":
                for page_name, evidence_line in pred_evidence:
                    if evidence_line is not None:
                        evidence_line = untokenize(evidence_line)
                        sample = (claim.claim, [(page_name, evidence_line)])
                        # sample = (evidence_line, claim.claim)
                        self.data.append(sample)
                        self.data_labels.append(1)
                        if self.over_sample:
                            self.data.append(sample)
                            self.data_labels.append(1)

        dist_dict = {0: 0, 1: 0, 2: 0}
        for label in self.data_labels:
            dist_dict[label] += 1
        print(dist_dict)

    def __initdataset__(self):  ### concat approach
        conn = sqlite3.connect(self.db_path)
        curs = conn.cursor()
        query = "SELECT t.page_name, l.line FROM lines AS l JOIN texts AS t ON l.page_id = t.page_id WHERE l.page_id=? and l.line_num=?"
        # Initializing the dataset in this way should happen outside of worker, because work gets done multiple times.
        # for i, claim in enumerate(tqdm(self.claims, total=len(self.claims), desc="Initializing Dataset")):
        for i, claim in enumerate(self.claims):
            gold_claim_evidences = set()
            if claim.label != "NOT ENOUGH INFO":
                gold_claim_evidences = [
                    (evidence.page_id, evidence.line_number)
                    for evidence_group in claim.evidence
                    for evidence in evidence_group
                    if len(evidence_group) < 3
                ]
                gold_claim_evidences = set(gold_claim_evidences)
            pred_claim_evidences = self.topk_sentences[i]  # ignore retreival scores
            pred_claim_evidences = pred_claim_evidences[
                pred_claim_evidences[:, 2] >= 0.1
            ]
            pred_idx_order = np.argsort(-pred_claim_evidences[:, 2])
            pred_claim_evidences = pred_claim_evidences[pred_idx_order]
            pred_claim_evidences = [
                tuple([int(xx) for xx in list(x)]) for x in pred_claim_evidences[:, :2]
            ]
            # print(len(pred_claim_evidences))
            minus_pred_claim_evidences = [
                pred_evidence
                for pred_evidence in pred_claim_evidences
                if pred_evidence not in gold_claim_evidences
            ]
            # gold_evidence = [curs.execute(query, claim_evidence).fetchone() for claim_evidence in gold_claim_evidences] if gold_claim_evidences else []
            # print(len(pred_claim_evidences))
            # print()
            pred_evidence = [
                curs.execute(query, claim_evidence).fetchone()
                for claim_evidence in pred_claim_evidences
            ]

            if claim.label != "NOT ENOUGH INFO" and self.train and False:
                used_evidence = set()
                # for evidence_gr in claim.evidence[:3]:
                #     unique_evidence.add(evidence)
                for evidence_group in claim.evidence:
                    if len(used_evidence) == 5:
                        break
                    all_evidence = []
                    if len(evidence_group) == 1:
                        evidence = evidence_group[0]
                        if (evidence.page_id, evidence.line_number) in used_evidence:
                            continue
                        used_evidence.add((evidence.page_id, evidence.line_number))
                        evidence_title, evidence_line = get_title_line(
                            conn, (evidence.page_id, evidence.line_number)
                        )
                        all_evidence = [(evidence_title, evidence_line)]
                        # formatted_evidence = "{} -- {}".format(evidence_title, evidence_line)
                    elif len(evidence_group) == 2:
                        first_evidence, second_evidence = evidence_group
                        first_title, first_line = get_title_line(
                            conn, (first_evidence.page_id, first_evidence.line_number)
                        )
                        second_title, second_line = get_title_line(
                            conn, (second_evidence.page_id, second_evidence.line_number)
                        )
                        all_evidence = [
                            (first_title, first_line),
                            (second_title, second_line),
                        ]
                        # formatted_evidence = "{} -- {} </s></s> {} -- {}".format(first_title, first_line, second_title, second_line)
                    else:
                        continue
                    for (pred_title, pred_line) in pred_evidence:
                        all_evidence.append((pred_title, pred_line))
                    # print(all_evidence)
                    formatted_evidence = " </s></s> ".join(
                        f"{page_name} -- {evidence_line}"
                        for page_name, evidence_line in all_evidence
                    )
                    sample = (claim.claim, formatted_evidence)
                    # print(claim.label)
                    # print(sample)
                    # print()
                    self.data.append(sample)
                    self.data_labels.append(claim.get_label_num())
            else:
                # print(pred_evidence)
                # if len(minus_pred_claim_evidences) == len(pred_evidence) and claim.get_label_num() != 1 and self.train:
                #     continue
                # formatted_evidence = " </s></s> ".join([f"{page_name} -- {evidence_line}" for page_name, evidence_line in pred_evidence])
                # sample = (claim.claim, formatted_evidence)
                sample = (claim.claim, pred_evidence)
                self.data.append(sample)
                self.data_labels.append(claim.get_label_num())
                # print(claim.label, self.train)
                # if claim.label == "NOT ENOUGH INFO" and self.train:
                #     # print("adding")
                #     self.data.append(sample)
                #     self.data_labels.append(claim.get_label_num())
                # if claim.label == "REFUTES" and self.train:
                #     self.data.append(sample)
                #     self.data_labels.append(claim.get_label_num())
                #     self.data.append(sample)
                #     self.data_labels.append(claim.get_label_num())
                # if claim.get_label_num() != 2 and self.train:
                #     self.data.append(sample)
                #     self.data_labels.append(claim.get_label_num())
                #     self.data.append(sample)
                #     self.data_labels.append(claim.get_label_num())
                #
                # print(claim.label)
                # print(sample)
                # print()
            # num_nei_samples = 1 if claim.label != "NOT ENOUGH INFO" else 5
            # num_nei_samples = min(len(pred_evidence), num_nei_samples)
            # sampled_pred_evidence = random.sample(pred_evidence, k=num_nei_samples)

            # for i, (page_name, evidence_line) in enumerate(sampled_pred_evidence):
            #     if random.random() < 0.1 and len(sampled_pred_evidence) > 1 and i < len(sampled_pred_evidence) - 1:
            #         random_page_name, random_evidence_line = sampled_pred_evidence[i+1]
            #         formatted_evidence = "{} -- {} </s></s> {} -- {}".format(page_name, evidence_line, random_page_name, random_evidence_line)
            #     else:
            #         formatted_evidence = "{} -- {}".format(page_name, evidence_line)
            #     sample = (formatted_evidence, claim.claim)
            #     self.data.append(sample)
            #     self.data_labels.append(1) # nei

        dist_dict = {0: 0, 1: 0, 2: 0}
        for label in self.data_labels:
            dist_dict[label] += 1
        print(dist_dict)

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx): # more principled approach
    #     evidence, claim = self.data[idx]
    #     label = self.data_labels[idx]
    #     return (evidence, claim), label

    # def __getitem__(self, idx): # og approach
    #     if idx in self.processed_dict:
    #         num_times_processed = self.processed_dict[idx]
    #         self.processed_dict[idx] += 1
    #     else:
    #         num_times_processed = 0
    #         self.processed_dict[idx] = 0
    #     page_name, evidence_line, claim = self.data[idx]
    #     if hash(self.data[idx]) % 2 == num_times_processed % 2 or self.static_order:
    #         sample = (page_name + " -- " + evidence_line, claim)
    #     else:
    #         sample = (claim, page_name + " -- " + evidence_line)
    #     label = self.data_labels[idx]
    #     return sample, label

    def __getitem__(self, idx):  # concat appoach
        claim, pred_evidence = self.data[idx]
        pred_evidence = (
            pred_evidence.copy()
        )  # don't want to shuffle original copy, otherwise next epoch the evidence will be out of order
        if (
            self.shuffle_evidence_p
            and self.train
            and random.random() <= self.shuffle_evidence_p
        ):
            random.shuffle(pred_evidence)
        formatted_evidence = " </s></s> ".join(
            [
                f"{page_name} -- {evidence_line}"
                for page_name, evidence_line in pred_evidence
            ]
        )
        return (formatted_evidence, claim), self.data_labels[idx]
