import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import ConcatDataset, DataLoader

from src.data.utils import load_pickle
from src.models.aggregator.dataset import AggregatorDataset
from src.models.aggregator.linear_model import LinearModel
from src.paths import PROCESSED_DATA_DIR


def filter_data(dataset, labels, weighted_score_threshold=0.4):
    filtered_samples = []
    filtered_labels = []
    for sample, label in zip(dataset, labels):
        if label == 1:
            filtered_samples.append(sample)
            filtered_labels.append(label)
        else:
            weighted_scores = sample[:, :-1] * sample[:, -1].reshape((-1, 1))
            weighted_label_score = weighted_scores[:, label].max()
            if weighted_label_score >= weighted_score_threshold:
                filtered_samples.append(sample)
                filtered_labels.append(label)
    return np.array(filtered_samples), np.array(filtered_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", default="train")
    parser.add_argument("--valid_set", default="dev")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train_claims = load_pickle(PROCESSED_DATA_DIR / "train" / "claims.pkl")
    train_data = np.load(PROCESSED_DATA_DIR / "train" / "claim_scores.npy").astype(
        np.float32
    )
    train_sentences = np.load(
        PROCESSED_DATA_DIR / "train" / "sentence_scores.npy"
    ).astype(np.float32)
    train_labels = np.array([claim.get_label_num() for claim in train_claims])
    # train_data, train_labels = filter_data(train_data, train_labels)

    valid_claims = load_pickle(PROCESSED_DATA_DIR / "valid" / "claims.pkl")
    valid_data = np.load(PROCESSED_DATA_DIR / "valid" / "claim_scores.npy").astype(
        np.float32
    )
    valid_sentences = np.load(
        PROCESSED_DATA_DIR / "valid" / "sentence_scores.npy"
    ).astype(np.float32)
    valid_labels = np.array([claim.get_label_num() for claim in valid_claims])

    dev_claims = load_pickle(PROCESSED_DATA_DIR / "dev" / "claims.pkl")
    dev_data = np.load(PROCESSED_DATA_DIR / "dev" / "claim_scores.npy").astype(
        np.float32
    )
    dev_sentences = np.load(PROCESSED_DATA_DIR / "dev" / "sentence_scores.npy").astype(
        np.float32
    )
    dev_labels = np.array([claim.get_label_num() for claim in dev_claims])

    # print("Length before filtering:", len(train_data) + len(valid_data))
    # filtered_train, filtered_train_labels = filter_not_strictly_correct(train_claims, train_data, train_sentences)
    # filtered_valid, filtered_valid_labels = filter_not_strictly_correct(valid_claims, valid_data, valid_sentences)
    # print("Length after filtering:", len(filtered_train) + len(filtered_valid))
    # filtered_train_dataset = AggregatorDataset(filtered_train, filtered_train_labels)
    # filtered_valid_dataset = AggregatorDataset(filtered_valid, filtered_valid_labels)
    train_dataset = AggregatorDataset(train_data, train_labels)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True
    )
    valid_dataset = AggregatorDataset(valid_data, valid_labels)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=4
    )
    dev_dataset = AggregatorDataset(dev_data, dev_labels)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False
    )

    combined_dataset = ConcatDataset([train_dataset, valid_dataset])
    train_dataloader = DataLoader(
        combined_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True
    )

    class_weights = compute_class_weight("balanced", np.array([0, 1, 2]), train_labels)
    # class_weights = class_weights + np.array([0.2, 0.0, 0.2])
    print(class_weights)
    model = LinearModel(
        hidden_dim=1028,
        dropout_p=0.1,
        class_weights=torch.tensor(class_weights).float(),
    )
    trainer = pl.Trainer(
        gpus=1, max_epochs=args.epochs, default_root_dir="models/agg_checkpoints"
    )
    trainer.fit(
        model, train_dataloader=train_dataloader, val_dataloaders=dev_dataloader
    )
