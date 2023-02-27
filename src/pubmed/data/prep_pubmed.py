import argparse
import json
import random
import sqlite3
from pathlib import Path

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


def create_claim_files(dataset_split, idmap):
    dataset = []
    for sample in dataset_split:
        initial_evidence = sample["evidence"]
        evidence = {
            "page_id": idmap[int(initial_evidence["page_id"])],
            "line_number": -1,
        }
        sample["evidence"] = [[Evidence(**evidence)]]
        dataset.append(DataSample(**sample))
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed_data_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--db_path", type=Path, default=DB_PATH)
    args = parser.parse_args()
    corpus = []
    with open(args.raw_data_dir / "ori_pqaa.json") as f:
        full_dataset = json.load(f)
        dataset = []
        for key, sample in full_dataset.items():
            abstract = " ".join(sample["CONTEXTS"]).split(". ")
            title = ""
            corpus.append({"title": "", "abstract": abstract, "doc_id": int(key)})
            sample_claim = " ".join(sample["QUESTION"][:-1].split(" ")[1:]) + "."
            label = "SUPPORTS" if sample["final_decision"] == "yes" else "REFUTES"
            evidence = {"page_id": key}
            dataset.append(
                {"claim": sample_claim, "label": label, "evidence": evidence}
            )
    create_database(args.db_path)
    process_corpus(corpus, args.db_path)
    idmap = get_idmap(args.db_path, reverse=True)
    random_seed = random.Random(0)
    random_idxs = list(range(len(dataset)))
    random_seed.shuffle(random_idxs)
    train_dataset = [dataset[idx] for idx in random_idxs[:-20000]]
    valid_dataset = [dataset[idx] for idx in random_idxs[-20000:-10000]]
    dev_dataset = [dataset[idx] for idx in random_idxs[-10000:]]
    dataset_tuples = zip(
        ["train", "valid", "dev"], [train_dataset, valid_dataset, dev_dataset]
    )
    for dataset_name, dataset_split in dataset_tuples:
        claim_dataset = create_claim_files(dataset_split, idmap)
        dest_dir = args.processed_data_dir / dataset_name
        dump_pickle(claim_dataset, dest_dir / "claims_no_evidence.pkl")
