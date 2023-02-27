import argparse
import json
import sqlite3
from pathlib import Path

from src.data.parse_dataset import split_train_dataset
from src.data.utils import DataSample, Evidence, dump_pickle
from src.paths import DB_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR


def create_database(db_path):
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()
    curs.execute(
        "CREATE TABLE texts(page_id INTEGER, page_name TEXT, og_page_name TEXT, text TEXT)"
    )
    curs.execute(
        "CREATE TABLE lines(page_id INTEGER, line_num INTEGER, line TEXT, line_extra TEXT)"
    )
    curs.execute("CREATE TABLE idmap(page_id INTEGER, og_page_id INTEGER)")
    curs.execute("CREATE INDEX page_id_index ON texts(page_id)")
    curs.execute("CREATE INDEX page_id_line_num_index ON lines(page_id, line_num)")
    curs.execute("CREATE INDEX page_id_index_2 ON lines(page_id)")
    curs.execute("CREATE INDEX line_num_index ON lines(line_num)")
    conn.close()
    print("Created", db_path)


def process_corpus(corpus, db_path):
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()
    text_rows = []
    line_rows = []
    idmap_rows = []
    for i, document in enumerate(corpus):
        text_rows.append(
            (i, document["title"], document["title"], ". ".join(document["abstract"]))
        )
        idmap_rows.append((i, document["doc_id"]))
        for sentence_id, sentence in enumerate(document["abstract"]):
            line_rows.append((i, sentence_id, sentence, ""))
    curs.executemany(
        "INSERT INTO texts(page_id, page_name, og_page_name, text) VALUES (?, ?, ?, ?)",
        text_rows,
    )
    curs.executemany(
        "INSERT INTO lines(page_id, line_num, line, line_extra) VALUES (?, ?, ?, ?)",
        line_rows,
    )
    curs.executemany("INSERT INTO idmap(page_id, og_page_id) VALUES (?, ?)", idmap_rows)
    conn.commit()


def get_idmap(db_file, reverse=False):
    conn = sqlite3.connect(db_file)
    curs = conn.cursor()
    results = curs.execute("SELECT * FROM idmap")
    results = results.fetchall()
    idmap = {}
    for result in results:
        if reverse:
            idmap[result[1]] = result[0]
        else:
            idmap[result[0]] = result[1]
    return idmap


def create_claim_files(jsonl_path, idmap):
    dataset = []
    with open(jsonl_path) as f:
        for line in f:
            json_line = json.loads(line)
            all_evidence = []
            label = "NOT ENOUGH INFO"
            for page_id, evidence_collections in json_line.get("evidence", {}).items():
                for evidence_collection in evidence_collections:
                    evidence_group = []
                    label = (
                        "SUPPORTS"
                        if evidence_collection["label"] == "SUPPORT"
                        else "REFUTES"
                    )
                    for line_number in evidence_collection["sentences"]:
                        evidence_dict = {
                            "page_id": idmap[int(page_id)],
                            "line_number": int(line_number),
                        }
                        evidence = Evidence(**evidence_dict)
                        evidence_group.append(evidence)
                    all_evidence.append(evidence_group)
            json_line["label"] = label
            json_line["evidence"] = all_evidence
            dataset.append(DataSample(**json_line))
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed_data_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--db_path", type=Path, default=DB_PATH)
    args = parser.parse_args()
    corpus = []
    with open(args.raw_data_dir / "corpus.jsonl") as f:
        for line in f:
            corpus.append(json.loads(line))
    create_database(args.db_path)
    process_corpus(corpus, args.db_path)
    idmap = get_idmap(args.db_path, reverse=True)
    for dataset_tag in ["train", "dev", "test"]:
        claims_filename = f"claims_{dataset_tag}.jsonl"
        dataset = create_claim_files(args.raw_data_dir / claims_filename, idmap)
        if dataset_tag == "train":
            train_dir = args.processed_data_dir / "train"
            valid_dir = args.processed_data_dir / "valid"
            train_dataset, valid_dataset = split_train_dataset(dataset, valid_size=100)
            dump_pickle(train_dataset, train_dir / "claims.pkl")
            dump_pickle(valid_dataset, valid_dir / "claims.pkl")
        else:
            dest_dir = args.processed_data_dir / dataset_tag
            dump_pickle(dataset, dest_dir / "claims.pkl")
