import argparse
import math
import os
from functools import partial
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import ConcatDataset, DataLoader
from transformers import RobertaTokenizerFast

from src.data.utils import load_pickle
from src.models import RoBERTa
from src.models.claim_classification.dataset import (
    ClaimClassificationDataset,
    collate_fn,
)
from src.paths import DB_PATH, PROCESSED_DATA_DIR

# CONFIG = {
#     "lr": tune.grid_search([3e-6, 7e-6, 1e-5, 2.5e-5, 5e-5]),
#     "label_smoothing": tune.grid_search([0.0, 0.1, 0.2]),
# }


CONFIG = {
    "lr": tune.grid_search([3e-6]),
    "label_smoothing": tune.grid_search([0.1, 0.2]),
    "model_type": tune.grid_search(["roberta-large-mnli", "roberta-large"]),
}


def parse_cc_args():
    parser = argparse.ArgumentParser(
        description="Generating TF-IDF features for evidence and claims"
    )
    parser.add_argument("--processed_dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--db_file", type=Path, default=DB_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulations", type=int, default=1)
    parser.add_argument("--tokenizer_max_length", type=int, default=368)
    parser.add_argument("--model_type", type=str, default="roberta-large-mnli")
    args = parser.parse_args()
    return args


args = parse_cc_args()


def train(config):
    lr_callback = LearningRateMonitor(logging_interval="step")
    with tune.checkpoint_dir(step=0) as checkpoint_dir:
        ckpt_dir = str(os.path.join(checkpoint_dir, ""))
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_accuracy",
        dirpath=ckpt_dir,
        filename="best_checkpoint",
        save_top_k=1,
        mode="max",
        verbose=True,
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(
        args.model_type, add_prefix_space=True
    )
    train_claims = load_pickle(args.processed_dir / "train" / "claims.pkl")
    train_topk_sentences = np.load(
        args.processed_dir / "train" / "combined_sentence_scores.npy"
    )
    valid_claims = load_pickle(args.processed_dir / "valid" / "claims.pkl")
    valid_topk_sentences = np.load(
        args.processed_dir / "valid" / "combined_sentence_scores.npy"
    )
    dev_claims = load_pickle(args.processed_dir / "dev" / "claims.pkl")
    dev_topk_sentences = np.load(args.processed_dir / "dev" / "sentence_scores.npy")
    train_dataset = ClaimClassificationDataset(
        train_claims,
        train_topk_sentences,
        args.db_file,
        claim_second=True,
        train=True,
        static_order=True,
        over_sample=True,
    )
    valid_dataset = ClaimClassificationDataset(
        valid_claims,
        valid_topk_sentences,
        args.db_file,
        claim_second=True,
        train=True,
        static_order=True,
        over_sample=True,
    )
    dev_train_dataset = ClaimClassificationDataset(
        dev_claims,
        dev_topk_sentences,
        args.db_file,
        claim_second=True,
        train=True,
        static_order=True,
        over_sample=False,
    )
    dev_dataset = ClaimClassificationDataset(
        dev_claims,
        dev_topk_sentences,
        args.db_file,
        claim_second=True,
        train=False,
        static_order=True,
        over_sample=False,
    )
    train_valid_dataset = ConcatDataset([train_dataset, valid_dataset])
    partial_collate_fn = partial(
        collate_fn, tokenizer=tokenizer, max_length=args.tokenizer_max_length
    )
    train_dataloader = DataLoader(
        train_valid_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=partial_collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        dev_dataset,
        batch_size=2 * args.batch_size,
        num_workers=4,
        collate_fn=partial_collate_fn,
        pin_memory=True,
    )
    steps_per_epoch = math.ceil(
        (len(train_valid_dataset) / args.batch_size) / args.gradient_accumulations
    )
    metrics = {"loss": "valid_loss", "acc": "valid_accuracy"}
    tune_callback = TuneReportCallback(metrics, on="validation_end")
    class_weights = compute_class_weight(
        "balanced", np.array([0, 1, 2]), train_dataset.data_labels
    )
    loss_fct_params = {
        "label_smoothing": config["label_smoothing"],
        "weight": torch.tensor(class_weights).float(),
    }
    cc_roberta = RoBERTa(
        config["model_type"],
        num_labels=3,
        lr=config["lr"],
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        loss_fct_params=loss_fct_params,
    )
    cc_roberta.train()
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        default_root_dir="checkpoints",
        precision="bf16",
        callbacks=[lr_callback, tune_callback, checkpoint_callback],
        accumulate_grad_batches=args.gradient_accumulations,
        val_check_interval=0.25,
        enable_progress_bar=False,
    )
    trainer.fit(
        cc_roberta, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )


if __name__ == "__main__":
    reporter = CLIReporter(max_report_frequency=3600)
    analysis = tune.run(
        train,
        metric="acc",
        mode="max",
        config=CONFIG,
        num_samples=2,
        name="tune_cc_model_v2",
        resources_per_trial={"gpu": 1, "cpu": 12},
        progress_reporter=reporter,
        local_dir="models/ray_tune_results",
    )
    print()
    print(analysis.get_best_config(scope="all"))


# == Status ==
# Current time: 2022-06-06 11:35:41 (running for 1 days, 23:36:50.88)
# Memory usage on this node: 17.9/31.2 GiB
# Using FIFO scheduling algorithm.
# Resources requested: 0/16 CPUs, 0/1 GPUs, 0.0/15.39 GiB heap, 0.0/7.7 GiB objects (0.0/1.0 accelerator_type:G)
# Current best trial: fd2c9_00002 with acc=0.8082308173179626 and parameters={'lr': 3e-06, 'label_smoothing': 0.2}
# Result logdir: /home/mitchell/repos/fever/models/ray_tune_results/tune_cc_model
# Number of trials: 15/15 (15 TERMINATED)
# +-------------------+------------+----------------------+-------------------+---------+--------+------------------+----------+----------+
# | Trial name        | status     | loc                  |   label_smoothing |      lr |   iter |   total time (s) |     loss |      acc |
# |-------------------+------------+----------------------+-------------------+---------+--------+------------------+----------+----------|
# | train_fd2c9_00000 | TERMINATED | 192.168.1.217:719226 |               0   | 3e-06   |     12 |          11533.8 | 0.597046 | 0.806531 |
# | train_fd2c9_00001 | TERMINATED | 192.168.1.217:734957 |               0.1 | 3e-06   |     12 |          11438.4 | 0.668118 | 0.806931 |
# | train_fd2c9_00002 | TERMINATED | 192.168.1.217:755705 |               0.2 | 3e-06   |     12 |          11387.3 | 0.746713 | 0.808231 |
# | train_fd2c9_00003 | TERMINATED | 192.168.1.217:776052 |               0   | 7e-06   |     12 |          11391.5 | 0.622535 | 0.80393  |
# | train_fd2c9_00004 | TERMINATED | 192.168.1.217:798167 |               0.1 | 7e-06   |     12 |          11398.9 | 0.684098 | 0.80298  |
# | train_fd2c9_00005 | TERMINATED | 192.168.1.217:821168 |               0.2 | 7e-06   |     12 |          11409   | 0.757269 | 0.80453  |
# | train_fd2c9_00006 | TERMINATED | 192.168.1.217:843660 |               0   | 1e-05   |     12 |          11515.3 | 0.648079 | 0.80103  |
# | train_fd2c9_00007 | TERMINATED | 192.168.1.217:866638 |               0.1 | 1e-05   |     12 |          11525.7 | 0.689079 | 0.80388  |
# | train_fd2c9_00008 | TERMINATED | 192.168.1.217:886807 |               0.2 | 1e-05   |     12 |          11447   | 0.756094 | 0.80398  |
# | train_fd2c9_00009 | TERMINATED | 192.168.1.217:899140 |               0   | 2.5e-05 |     12 |          11444.3 | 0.621019 | 0.80198  |
# | train_fd2c9_00010 | TERMINATED | 192.168.1.217:911724 |               0.1 | 2.5e-05 |     12 |          11352.1 | 1.47565  | 0.333333 |
# | train_fd2c9_00011 | TERMINATED | 192.168.1.217:945288 |               0.2 | 2.5e-05 |     12 |          11373.5 | 0.757528 | 0.80148  |
# | train_fd2c9_00012 | TERMINATED | 192.168.1.217:954090 |               0   | 5e-05   |     12 |          11329.7 | 1.89087  | 0.333333 |
# | train_fd2c9_00013 | TERMINATED | 192.168.1.217:962762 |               0.1 | 5e-05   |     12 |          11319.9 | 1.09474  | 0.333333 |
# | train_fd2c9_00014 | TERMINATED | 192.168.1.217:972428 |               0.2 | 5e-05   |     12 |          11454.9 | 1.08355  | 0.333333 |
# +-------------------+------------+----------------------+-------------------+---------+--------+------------------+----------+----------+
