import argparse
import json
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from src.data.utils import DataSample, dump_pickle, untokenize
from src.models import RoBERTa
from src.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


def get_title2id_map(curs):
    query = "SELECT og_page_name, page_id FROM texts"
    results = curs.execute(query).fetchall()
    return dict(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default=RAW_DATA_DIR)
    parser.add_argument("--processed_data_dir", default=PROCESSED_DATA_DIR)
    # Leave `db_path` default blank, as we don't build a DB for this set, so we want to reuse a different one
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--datasets", default=["dev", "test"], nargs="+")
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--num_labels", type=int)
    args = parser.parse_args()
    return args


def main():
    """
    Take KGAT outputs to produce necessary files for performing inference from their retrieval system.
    """
    args = parse_args()
    conn = sqlite3.connect(args.db_path)
    curs = conn.cursor()
    title2id_map = get_title2id_map(curs)

    model = RoBERTa(args.model_type, args.num_labels)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt["state_dict"])
    model.cuda()
    model.eval()
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_type)

    miss_count = 0
    count = 0
    for dataset in args.datasets:
        dataset_data = []
        # `bert_dev.json` and `bert_test.json` are from the KGAT FEVER repo.
        with open(args.raw_data_dir / f"bert_{dataset}.json") as f:
            for line in f:
                dataset_data.append(json.loads(line))

        claims = []
        topk_sentences = []
        for sample in tqdm(dataset_data):
            claims.append(
                DataSample(
                    **{
                        "id": sample["id"],
                        "claim": sample["claim"],
                        "label": sample["label"] if "label" in sample else None,
                    }
                )
            )
            page_ids_and_line_nums = []
            titles_and_lines = []
            for evidence in sample["evidence"]:
                count += 1
                title, line_num, _, _ = evidence
                try:
                    page_id = title2id_map[title]
                except:
                    page_id = title2id_map[unicodedata.normalize("NFKD", title)]
                    miss_count += 1
                query = "SELECT DISTINCT l.page_id, l.line_num, l.line FROM lines as l JOIN texts as t ON l.page_id = t.page_id WHERE t.page_id = ? AND l.line_num = ?"
                page_id, line_num, line = result = curs.execute(
                    query, (page_id, line_num)
                ).fetchone()
                page_ids_and_line_nums.append((page_id, line_num))
                titles_and_lines.append(
                    (untokenize(title, replace_underscore=True), line)
                )
            topk_evidence = []
            if len(titles_and_lines) == 0:
                topk_sentences.append(np.array([[np.nan] * 3] * 5))
                print("Skipping from lack of sentences")
                continue
            with autocast(dtype=torch.bfloat16, device_type="cuda"):
                with torch.no_grad():
                    batch_inputs = [
                        ((title + " -- " + line), sample["claim"])
                        for title, line in titles_and_lines
                    ]
                    batch_input_ids = tokenizer(
                        batch_inputs,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=256,
                    )
                    batch_input_ids["input_ids"] = batch_input_ids["input_ids"].cuda()
                    batch_input_ids["attention_mask"] = batch_input_ids[
                        "attention_mask"
                    ].cuda()
                    preds = model(batch_input_ids).logits.float().detach().cpu()
            softmax_scores = F.softmax(preds, dim=1)
            if args.num_labels == 3:
                score_criteria = 1 - softmax_scores[:, 1]
            else:
                score_criteria = softmax_scores[:, 1]
            args_sort = torch.argsort(score_criteria, descending=True)
            topk = args_sort[:5]
            for top in topk:
                page_id, line_num = page_ids_and_line_nums[top]
                # print((doc_id, line_num, score_criteria[top]))
                topk_evidence.append(
                    np.array([page_id, line_num, float(score_criteria[top])])
                )
            if len(topk_evidence) < 5:
                topk_sentences.append(
                    np.array(
                        topk_evidence
                        + ([[np.nan, np.nan, 0]] * (5 - len(topk_evidence)))
                    )
                )
            else:
                topk_sentences.append(np.array(topk_evidence))
            assert (
                len(topk_sentences[-1]) == 5
            ), f"Length of topk_sentences is {len(topk_evidence)}"
        np.save(
            args.processed_data_dir / dataset / "sentence_scores.npy",
            np.array(topk_sentences),
        )
        dump_pickle(claims, args.processed_data_dir / dataset / "claims.pkl")


if __name__ == "__main__":
    main()
