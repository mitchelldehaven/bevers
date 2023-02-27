import argparse
import math
import os
from functools import partial
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from src.data.utils import load_pickle
from src.models import RoBERTa
from src.models.sentence_selection.dataset import SentenceDatasetRoBERTa, collate_fn
from src.paths import PROCESSED_DATA_DIR

CONFIG = {
    "lr": tune.grid_search([3e-6, 7e-6, 1e-5, 5e-5]),
    "label_smoothing": tune.grid_search([0.0, 0.1, 0.2]),
}


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
    args = parser.parse_args()
    return args


args = parse_ss_args()


def train(tune_config):
    binary_labels = True if args.num_labels == 2 else False
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
    metrics = {"loss": "valid_loss", "acc": "valid_accuracy"}
    tune_callback = TuneReportCheckpointCallback(metrics, on="validation_end")
    loss_fct_params = {"label_smoothing": tune_config["label_smoothing"]}
    if args.checkpoint:
        ss_roberta = RoBERTa(
            args.model_type,
            args.num_labels,
            loss_fct_params=loss_fct_params,
            lr=tune_config["lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
        ).load_from_checkpoint(args.checkpoint)
    else:
        ss_roberta = RoBERTa(
            args.model_type,
            args.num_labels,
            loss_fct_params=loss_fct_params,
            lr=tune_config["lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
        )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        precision="bf16",
        callbacks=[lr_callback, tune_callback, checkpoint_callback],
        accumulate_grad_batches=args.gradient_accumulations,
        val_check_interval=0.1,
        enable_progress_bar=False,
    )
    trainer.fit(
        ss_roberta, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )


if __name__ == "__main__":
    reporter = CLIReporter(max_report_frequency=3600)
    analysis = tune.run(
        train,
        metric="acc",
        mode="max",
        config=CONFIG,
        # num_samples=12,
        name="tune_ss_model",
        resources_per_trial={"gpu": 1, "cpu": 12},
        progress_reporter=reporter,
        local_dir="models/ray_tune_results",
    )

# == Status ==
# Current time: 2022-06-02 19:02:12 (running for 2 days, 08:09:17.52)
# Memory usage on this node: 16.6/31.2 GiB
# Using FIFO scheduling algorithm.
# Resources requested: 0/16 CPUs, 0/1 GPUs, 0.0/16.46 GiB heap, 0.0/8.23 GiB objects (0.0/1.0 accelerator_type:G)
# Current best trial: 1e1a3_00001 with acc=0.9628247618675232 and parameters={'lr': 3e-06, 'label_smoothing': 0.1}
# Result logdir: /home/mitchell/repos/fever/models/ray_tune_results/tune_ss_model
# Number of trials: 12/12 (12 TERMINATED)
# +-------------------+------------+-----------------------+-------------------+-------+--------+------------------+----------+----------+
# | Trial name        | status     | loc                   |   label_smoothing |    lr |   iter |   total time (s) |     loss |      acc |
# |-------------------+------------+-----------------------+-------------------+-------+--------+------------------+----------+----------|
# | train_1e1a3_00000 | TERMINATED | 192.168.1.217:1631401 |               0   | 3e-06 |     10 |          16822   | 0.111945 | 0.962469 |
# | train_1e1a3_00001 | TERMINATED | 192.168.1.217:2094096 |               0.1 | 3e-06 |     10 |          16695.7 | 0.364411 | 0.962825 |
# | train_1e1a3_00002 | TERMINATED | 192.168.1.217:2555057 |               0.2 | 3e-06 |     10 |          16866.8 | 0.539994 | 0.962547 |
# | train_1e1a3_00003 | TERMINATED | 192.168.1.217:3019692 |               0   | 7e-06 |     10 |          16814.6 | 0.126722 | 0.962249 |
# | train_1e1a3_00004 | TERMINATED | 192.168.1.217:3480664 |               0.1 | 7e-06 |     10 |          16853.5 | 0.368462 | 0.962453 |
# | train_1e1a3_00005 | TERMINATED | 192.168.1.217:3943706 |               0.2 | 7e-06 |     10 |          16844.8 | 0.54252  | 0.962367 |
# | train_1e1a3_00006 | TERMINATED | 192.168.1.217:212554  |               0   | 1e-05 |     10 |          16873.5 | 0.129443 | 0.961118 |
# | train_1e1a3_00007 | TERMINATED | 192.168.1.217:675778  |               0.1 | 1e-05 |     10 |          16916   | 0.369776 | 0.96191  |
# | train_1e1a3_00008 | TERMINATED | 192.168.1.217:1138828 |               0.2 | 1e-05 |     10 |          16911.6 | 0.543254 | 0.96151  |
# | train_1e1a3_00009 | TERMINATED | 192.168.1.217:1601137 |               0   | 5e-05 |     10 |          16857.7 | 0.59736  | 0.816598 |
# | train_1e1a3_00010 | TERMINATED | 192.168.1.217:2062815 |               0.1 | 5e-05 |     10 |          16839.6 | 0.698882 | 0.816598 |
# | train_1e1a3_00011 | TERMINATED | 192.168.1.217:2523828 |               0.2 | 5e-05 |     10 |          16787   | 0.784374 | 0.816598 |
# +-------------------+------------+-----------------------+-------------------+-------+--------+------------------+----------+----------+
