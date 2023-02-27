import argparse
import math
from functools import partial
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer

from src.data.utils import load_pickle
from src.models import RoBERTa
from src.models.claim_classification.dataset import (
    ClaimClassificationDataset,
    collate_fn,
)
from src.paths import DB_PATH, MODELS_DIR, PROCESSED_DATA_DIR


def parse_cc_args():
    parser = argparse.ArgumentParser(
        description="Generating TF-IDF features for evidence and claims"
    )
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--db_file", type=Path, default=DB_PATH)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulations", type=int, default=1)
    parser.add_argument("--tokenizer_max_length", type=int, default=368)
    parser.add_argument("--model_type", type=str, default="roberta-large-mnli")
    parser.add_argument("--shuffle_evidence_p", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sentence_scores_files", default="combined_sentence_scores.npy")
    parser.add_argument("--valid_interval", type=float, default=0.25)
    args = parser.parse_args()
    return args


def train(args):
    lr_callback = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_accuracy",
        dirpath=MODELS_DIR / "claim_classification",
        filename=f'cc_{args.model_type.replace("/", "-")}_concat_'
        + "{epoch:02d}-{valid_accuracy:.5f}",
        save_top_k=1,
        mode="max",
        verbose=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, add_prefix_space=True)
    train_claims = load_pickle(args.processed_dir / "train" / "claims.pkl")
    train_topk_sentences = np.load(
        args.processed_dir / "train" / args.sentence_scores_files
    )
    valid_claims = load_pickle(args.processed_dir / "valid" / "claims.pkl")
    valid_topk_sentences = np.load(
        args.processed_dir / "valid" / args.sentence_scores_files
    )
    dev_claims = load_pickle(args.processed_dir / "dev" / "claims.pkl")
    dev_topk_sentences = np.load(
        args.processed_dir / "dev" / args.sentence_scores_files
    )
    train_dataset = ClaimClassificationDataset(
        train_claims,
        train_topk_sentences,
        args.db_file,
        claim_second=True,
        train=True,
        static_order=True,
        over_sample=False,
        shuffle_evidence_p=args.shuffle_evidence_p,
    )
    valid_dataset = ClaimClassificationDataset(
        valid_claims,
        valid_topk_sentences,
        args.db_file,
        claim_second=True,
        train=True,
        static_order=True,
        over_sample=False,
        shuffle_evidence_p=args.shuffle_evidence_p,
    )
    dev_dataset = ClaimClassificationDataset(
        dev_claims,
        dev_topk_sentences,
        args.db_file,
        claim_second=True,
        train=False,
        static_order=True,
        over_sample=False,
        shuffle_evidence_p=args.shuffle_evidence_p,
    )
    train_valid_dataset = ConcatDataset([train_dataset, valid_dataset])
    partial_collate_fn = partial(
        collate_fn, tokenizer=tokenizer, max_length=args.tokenizer_max_length
    )
    train_dataloader = DataLoader(
        train_valid_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=partial_collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        dev_dataset,
        batch_size=2 * args.batch_size,
        num_workers=2,
        collate_fn=partial_collate_fn,
        pin_memory=True,
    )
    steps_per_epoch = math.ceil(
        (len(train_valid_dataset) / args.batch_size) / args.gradient_accumulations
    )
    class_weights = compute_class_weight(
        "balanced", np.array([0, 1, 2]), train_dataset.data_labels
    )
    loss_fct_params = {
        "label_smoothing": 0.1,
        "weight": torch.tensor(class_weights).float(),
    }
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        cc_roberta = RoBERTa(
            model_type=args.model_type,
            num_labels=3,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            loss_fct_params=loss_fct_params,
            lr=args.lr,
        )
        cc_roberta.expand_embeddings()
        cc_roberta.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        cc_roberta = RoBERTa(
            model_type=args.model_type,
            num_labels=3,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            loss_fct_params=loss_fct_params,
            lr=args.lr,
        )
        cc_roberta.expand_embeddings()
    cc_roberta.train()
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        default_root_dir="checkpoints",
        precision="bf16",
        callbacks=[lr_callback, checkpoint_callback],
        accumulate_grad_batches=args.gradient_accumulations,
        val_check_interval=args.valid_interval,
    )
    trainer.fit(
        cc_roberta, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )


if __name__ == "__main__":
    args = parse_cc_args()
    train(args)
