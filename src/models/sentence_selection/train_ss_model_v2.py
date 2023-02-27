import argparse
import math
from functools import partial
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from torch.utils.data import ConcatDataset, DataLoader
from transformers import RobertaTokenizerFast

from src.data.utils import load_pickle
from src.models import RoBERTa
from src.models.sentence_selection.dataset import SentenceDatasetRoBERTa, collate_fn
from src.paths import MODELS_DIR, PROCESSED_DATA_DIR

print(PROCESSED_DATA_DIR)


def parse_ss_args():
    parser = argparse.ArgumentParser(
        description="Generating TF-IDF features for evidence and claims"
    )
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--ss_dataset_tag", type=str, default="k_10_random_expanded")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint_steps", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulations", type=int, default=1)
    parser.add_argument("--tokenizer_max_length", type=int, default=128)
    parser.add_argument("--model_type", type=str, default="roberta-large")
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--valid_interval", type=float, default=0.1)
    # parser.add_argument("--ckpt", type=Path)
    args = parser.parse_args()
    return args


def train(args):
    binary_labels = True if args.num_labels == 2 else False
    lr_callback = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_accuracy",
        dirpath=MODELS_DIR / "sentence_selection",
        filename=f"ss_{args.model_type}_binary_{binary_labels}"
        + "_{epoch:02d}-{valid_accuracy:.5f}",
        save_top_k=1,
        mode="max",
        verbose=True,
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_type)
    sentence_selection_dir = (
        args.processed_dir / "sentence_selection" / args.ss_dataset_tag
    )
    train_dataset_file = sentence_selection_dir / "train.csv"
    train_claims = load_pickle(args.processed_dir / "train" / "claims.pkl")
    train_csv_dataset = pd.read_csv(train_dataset_file, index_col=0).values
    valid_dataset_file = sentence_selection_dir / "valid.csv"
    valid_claims = load_pickle(args.processed_dir / "valid" / "claims.pkl")
    valid_csv_dataset = pd.read_csv(valid_dataset_file, index_col=0).values
    dev_dataset_file = sentence_selection_dir / "dev.csv"
    dev_claims = load_pickle(args.processed_dir / "dev" / "claims.pkl")
    dev_csv_dataset = pd.read_csv(dev_dataset_file, index_col=0).values
    # train_dataset = SentenceDatasetRoBERTa(train_claims, train_csv_dataset, tokenizer, binary_label=binary_labels, claim_second=True)
    # valid_dataset = SentenceDatasetRoBERTa(valid_claims, valid_csv_dataset, tokenizer, binary_label=binary_labels, claim_second=True)
    train_dataset = SentenceDatasetRoBERTa(
        train_claims,
        train_csv_dataset,
        tokenizer,
        binary_label=binary_labels,
        claim_second=True,
    )
    valid_dataset = SentenceDatasetRoBERTa(
        valid_claims,
        valid_csv_dataset,
        tokenizer,
        binary_label=binary_labels,
        claim_second=True,
    )
    train_dataset = ConcatDataset([train_dataset, valid_dataset])
    dev_dataset = SentenceDatasetRoBERTa(
        dev_claims,
        dev_csv_dataset,
        tokenizer,
        binary_label=binary_labels,
        claim_second=True,
    )
    partial_collate_fn = partial(
        collate_fn, tokenizer=tokenizer, max_length=args.tokenizer_max_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=partial_collate_fn,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        dev_dataset,
        batch_size=2 * args.batch_size,
        num_workers=4,
        collate_fn=partial_collate_fn,
    )
    steps_per_epoch = math.ceil(
        (len(train_dataset) / args.batch_size) / args.gradient_accumulations
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        ss_roberta = RoBERTa(
            args.model_type,
            args.num_labels,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
        )
        ss_roberta.load_state_dict(ckpt["state_dict"])
    else:
        ss_roberta = RoBERTa(
            args.model_type,
            args.num_labels,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
        )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        default_root_dir="checkpoints",
        precision="bf16",
        callbacks=[lr_callback, checkpoint_callback],
        accumulate_grad_batches=args.gradient_accumulations,
        val_check_interval=args.valid_interval,
    )
    print(trainer)
    trainer.fit(
        ss_roberta, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )


if __name__ == "__main__":
    args = parse_ss_args()
    train(args)

#     {
#   "abstract_label_only_precision": 0.815,
#   "abstract_label_only_recall": 0.7799043062200957,
#   "abstract_label_only_f1": 0.7970660146699265,
#   "abstract_rationalized_precision": 0.775,
#   "abstract_rationalized_recall": 0.7416267942583732,
#   "abstract_rationalized_f1": 0.7579462102689486,
#   "sentence_selection_precision": 0.6274509803921569,
#   "sentence_selection_recall": 0.6994535519125683,
#   "sentence_selection_f1": 0.661498708010336,
#   "sentence_label_precision": 0.6274509803921569,
#   "sentence_label_recall": 0.6994535519125683,
#   "sentence_label_f1": 0.661498708010336
# }
