# pylint: disable=W0223
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RoBERTa(pl.LightningModule):
    """
    Class for training RoBERTa via Pytorch Lightning.
    """

    def __init__(
        self,
        model_type,
        num_labels,
        tokenizer=None,
        steps_per_epoch=None,
        epochs=None,
        lr=3e-6,
        loss_fct_params={},
    ):
        super().__init__()
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            model_type, num_labels=num_labels
        )
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_type, add_prefix_space=True
            )
        else:
            self.tokenizer = tokenizer
        self.num_labels = num_labels
        if loss_fct_params:
            self.roberta.set_loss_fct(torch.nn.CrossEntropyLoss(**loss_fct_params))
        else:
            self.roberta.set_loss_fct(torch.nn.CrossEntropyLoss())
        self.roberta.config.hidden_dropout_prob = 0.1
        self.roberta.config.attention_probs_drop_prob = 0.1
        self.roberta.config.classifier_dropout = 0.1
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

    def expand_embeddings(self):
        if (
            len(self.tokenizer)
            != self.roberta.roberta.embeddings.word_embeddings.weight.shape[0]
        ):
            print("Expanding embedding size")
            self.roberta.resize_token_embeddings(len(self.tokenizer))

    def forward(self, inputs, labels=None):  # pylint: disable=arguments-differ
        output = self.roberta(**inputs, labels=labels)
        return output

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        x_batch, y_batch = batch
        output = self.forward(x_batch, y_batch)
        y_hat = torch.argmax(output.logits, dim=1)
        loss = output.loss
        accuracy = torch.sum(y_batch == y_hat).item() / (len(y_batch) * 1.0)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        x_batch, y_batch = batch
        output = self.forward(x_batch, y_batch)
        y_hat = torch.argmax(output.logits, dim=1)
        loss = output.loss
        accuracy = torch.sum(y_batch == y_hat).item() / (len(y_batch) * 1.0)
        self.log("valid_loss", loss.detach())
        self.log("valid_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.lr,
            pct_start=0.05,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            anneal_strategy="linear",
        )
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]
