"""
Code for parsing FEVER dataset jsonl files into relevant pickle files.
"""
import argparse
import json
import random
import sqlite3
from pathlib import Path

from src.data.utils import DataSample, Evidence, dump_pickle, untokenize
from src.paths import DB_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR

CLAIMS_FILENAME = "claims.pkl"


def add_page_id(evidence, title_id_dict):
    if evidence.untok_page_title:
        page_id = title_id_dict[evidence.untok_page_title]
        evidence.page_id = page_id
    return evidence


def parse_dataset(dataset_path, title_id_dict):
    dataset = []
    with open(dataset_path) as jsonl_file:
        for line in jsonl_file:
            json_line = json.loads(line)
            json_line["claim"] = untokenize(
                json_line["claim"]
            )  # this is probably unnecessary
            if "evidence" in json_line:
                evidence = []
                for evidence_group in json_line["evidence"]:
                    evidence.append(
                        [
                            add_page_id(Evidence(*evidence), title_id_dict)
                            for evidence in evidence_group
                        ]
                    )
                json_line["evidence"] = evidence
            data_sample = DataSample(**json_line)
            dataset.append(data_sample)
    return dataset


def split_train_dataset(train_dataset, valid_size=10_000):
    random_idxs = list(range(len(train_dataset)))
    random.Random(0).shuffle(random_idxs)
    train_idxs = random_idxs[:-valid_size]
    valid_idxs = random_idxs[-valid_size:]
    new_train_dataset = [train_dataset[idx] for idx in train_idxs]
    valid_dataset = [train_dataset[idx] for idx in valid_idxs]
    return new_train_dataset, valid_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--train_dir", type=Path, default=PROCESSED_DATA_DIR / "train")
    parser.add_argument("--valid_dir", type=Path, default=PROCESSED_DATA_DIR / "valid")
    parser.add_argument("--dev_dir", type=Path, default=PROCESSED_DATA_DIR / "dev")
    parser.add_argument("--test_dir", type=Path, default=PROCESSED_DATA_DIR / "test")
    parser.add_argument("--db_path", type=Path, default=DB_PATH)
    args = parser.parse_args()
    filenames = ["train.jsonl", "shared_task_dev.jsonl", "shared_task_test.jsonl"]
    dest_dirs = [(args.train_dir, args.valid_dir), args.dev_dir, args.test_dir]
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()
    title_id_dict = dict(
        curs.execute("SELECT page_name, page_id FROM texts").fetchall()
    )
    for filename, dest_dir in zip(filenames, dest_dirs):
        dataset = parse_dataset(args.raw_data_dir / filename, title_id_dict)
        if filename == "train.jsonl":
            # Realisitically I should probably revert this, as I stopped using a train split validation set
            # and instead just used the dev set as validation, but leaving as is for now.
            train_dir, valid_dir = dest_dir
            train_dataset, valid_dataset = split_train_dataset(dataset)
            dump_pickle(train_dataset, train_dir / CLAIMS_FILENAME)
            dump_pickle(valid_dataset, valid_dir / CLAIMS_FILENAME)
        else:
            dump_pickle(dataset, dest_dir / CLAIMS_FILENAME)
