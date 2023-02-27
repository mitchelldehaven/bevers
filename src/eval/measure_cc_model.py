"""
Script for generating claim scores to NumPy files.
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.utils import load_pickle, untokenize
from src.models import RoBERTa
from src.paths import DB_PATH, PROCESSED_DATA_DIR


def parse_args():
    """
    Parse arguments.
    """
    prediction_types = ["singleton", "concat", "mixed"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["dev", "test"])
    parser.add_argument("--combined_sentence_scores", action="store_true")
    parser.add_argument("--model_type", default="roberta-large")
    parser.add_argument("--processed_data_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--db_path", type=Path, default=DB_PATH)
    parser.add_argument("--prediction_type", choices=prediction_types, default="mixed")
    args = parser.parse_args()
    return args


def get_output_size(args):
    """
    Return the output size of the prediction matix passed on prediction type.
    """
    if args.prediction_type == "singleton":
        return 5
    if args.prediction_type == "mixed":
        return 6
    return 1


def get_inputs(single_sentences, concated_sentence, args):
    """
    Return inputs based on prediction type.
    """
    if args.prediction_type == "singleton":
        return single_sentences
    if args.prediction_type == "mixed":
        return single_sentences + concated_sentence
    return concated_sentence


def write_claim_scores(model, args):
    """
    For each dataset in args.datasets:
        - Load top-5 sentences.
        - Form top-5 sentences into correct input format.
        - Generate prediction scores based on input format.
        - Write predictions to correct output file.
    """
    conn = sqlite3.connect(args.db_path)
    curs = conn.cursor()
    query = (
        "SELECT t.page_name, l.line FROM lines AS l JOIN texts AS t "
        "ON l.page_id = t.page_id WHERE l.page_id=? and l.line_num=?"
    )
    output_size = get_output_size(args)
    for dataset in args.datasets:
        all_top5_inputs = []
        all_top5_retrieval_scores = []
        all_labels = []
        all_top5_ids = []
        claims = load_pickle(args.processed_data_dir / dataset / "claims.pkl")
        sentence_filename = (
            "combined_sentence_scores.npy"
            if args.combined_sentence_scores
            else "sentence_scores.npy"
        )
        top5_sentences = np.load(args.processed_data_dir / dataset / sentence_filename)
        for claim, top5_ids in zip(claims, top5_sentences):
            ids = [
                tuple(int(i) for i in top5)
                for top5 in top5_ids[:, :-1]
                if not np.isnan(top5[0])
            ]
            top5_retrieval_scores = np.array(
                [score for score in top5_ids[:, -1] if not np.isnan(score)]
                + [np.nan] * 5
            )[:5]
            mean_top5_retrieval_scores = [top5_retrieval_scores.mean()]
            if args.prediction_type == "mixed":
                top5_retrieval_scores = (
                    top5_retrieval_scores.tolist() + mean_top5_retrieval_scores
                )
            elif args.prediction_type == "concat":
                top5_retrieval_scores = mean_top5_retrieval_scores
            else:
                top5_retrieval_scores = top5_retrieval_scores.tolist()
            all_top5_retrieval_scores.append(top5_retrieval_scores)
            all_labels.append(claim.get_label_num())
            all_top5_ids.append(ids)
            evidence = [curs.execute(query, i).fetchone() for i in ids]
            concat_inputs = " </s></s> ".join(
                [
                    page_name + " -- " + untokenize(sentence)
                    for j, (page_name, sentence) in enumerate(evidence)
                ]
            )
            single_sentences = [
                (page_name + " -- " + untokenize(sentence), claim.claim)
                for page_name, sentence in evidence
            ]
            inputs = get_inputs(single_sentences, [(concat_inputs, claim.claim)], args)
            tokenized_inputs = model.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            all_top5_inputs.append(tokenized_inputs)

        preds = []
        softmax_scores = []
        softmax = torch.nn.Softmax(dim=1)
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            with torch.no_grad():
                for input_ids in tqdm(all_top5_inputs):
                    input_ids["input_ids"] = input_ids["input_ids"].cuda()
                    input_ids["attention_mask"] = input_ids["attention_mask"].cuda()
                    pred = model(input_ids)
                    softmax_preds = softmax(pred.logits)
                    label_pred = (
                        torch.argmax(torch.mean(softmax_preds, dim=0), dim=0)
                        .detach()
                        .cpu()
                        .item()
                    )
                    softmax_score = softmax_preds.detach().cpu().numpy()
                    nan_padded = np.ones((output_size, 3))
                    nan_padded[nan_padded == 1] = np.nan
                    if softmax_score.shape[0] >= 1:
                        nan_padded[-softmax_score.shape[0] :] = softmax_score
                    softmax_scores.append(nan_padded)
                    preds.append(label_pred)
                    del input_ids
        data = np.concatenate(
            (
                np.array(softmax_scores),
                np.array(all_top5_retrieval_scores).reshape(-1, output_size, 1),
            ),
            axis=2,
        )
        claim_scores_filename = (
            "combined_claim_scores.npy"
            if args.combined_sentence_scores
            else "claim_scores.npy"
        )
        np.save(args.processed_data_dir / dataset / claim_scores_filename, data)


def main():
    args = parse_args()
    model = RoBERTa(args.model_type, num_labels=3)
    ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"))["state_dict"]
    if "roberta.loss_fct.weight" in ckpt:
        del ckpt["roberta.loss_fct.weight"]
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()
    write_claim_scores(model, args)


if __name__ == "__main__":
    main()
