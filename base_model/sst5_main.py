import argparse
import os
import random
import sys

import json
import torch
from torch.optim import AdamW
import torch.nn as nn
from transformers import (
    BertModel,
    ElectraModel,
    get_linear_schedule_with_warmup,
    RobertaModel,
)
from sst5_data import load_dataloaders
import transformers

transformers.logging.set_verbosity_error()


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


class Logger:
    def __init__(self, path):
        self.log_path = path

    def log(self, string, newline=True):
        with open(self.log_path, "a") as f:
            f.write(string)
            if newline:
                f.write("\n")

        sys.stdout.write(string)
        if newline:
            sys.stdout.write("\n")
        sys.stdout.flush()


class binary_classifier(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.linear_layer = nn.Linear(self.hidden_size, 5, bias=True)
        self.dropout = nn.Dropout(self.args.dropout_rate)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.args.model_type == "bert":
            embeddings = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )[0].mean(dim=1)
        else:
            embeddings = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )[0].mean(dim=1)
        embeddings = self.dropout(embeddings)
        logits = self.linear_layer(embeddings)
        return logits

    def compute_loss(self, logits, labels):
        loss_value = self.loss(logits, labels)
        return loss_value

    def predict(self, logits):
        assert logits.dim() == 2
        return torch.argmax(logits, dim=1)


def main(args, logger):
    train_loader, dev_loader, test_loader = load_dataloaders(args)
    if args.model_type == "bert":
        encoder = BertModel.from_pretrained("bert-large-uncased")
    elif args.model_type == "electra":
        encoder = ElectraModel.from_pretrained("google/electra-large-discriminator")
    else:
        encoder = RobertaModel.from_pretrained("roberta-large")
    model = binary_classifier(args, encoder).cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * (args.epochs * 5126 / args.batch_size)),
        num_training_steps=(args.epochs * 5126 / args.batch_size),
    )

    num_steps = 0
    loss_value = 0
    best_accuracy = 0
    model.zero_grad()
    for e in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            token_type_ids = batch[2].cuda()
            labels = batch[3].cuda()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = model.compute_loss(logits, labels)
            loss_value += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            num_steps += 1

            if num_steps % args.logging_steps == 0:
                logger.log(
                    "loss value from step {} to step {} is {}".format(
                        num_steps - args.logging_steps, num_steps, loss_value
                    )
                )
                loss_value = 0
        model.eval()
        dev_accuracy = evaluate(model, dev_loader, "dev")
        logger.log("dev accuracy of epoch {} is {}".format(e, dev_accuracy))
        if dev_accuracy > best_accuracy:
            logger.log(
                "accuracy increases from {} -----> {}".format(
                    best_accuracy, dev_accuracy
                )
            )
            torch.save(
                model.state_dict(),
                "../trained_models/SST-5_{}_meanpool.pt".format(args.model_type),
            )
            best_accuracy = dev_accuracy
            test_accuracy = evaluate(model, test_loader, "test")
            logger.log("test accuracy of epoch {} is {}".format(e, test_accuracy))


def evaluate(model, dev_loader, mode):
    predictions = []
    correct = 0
    total = 0
    softmax_fc = nn.Softmax(dim=-1)
    probs = []
    for batch in dev_loader:
        input_ids = batch[0].cuda()
        attention_mask = batch[1].cuda()
        token_type_ids = batch[2].cuda()
        labels = batch[3].cuda()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        prediction = model.predict(logits).tolist()
        prob = softmax_fc(logits).tolist()
        predictions += prediction
        probs += prob
        for p, l in zip(prediction, labels):
            total += 1
            if p == l:
                correct += 1
    with open("SST-5_{}_probs_{}.json".format(mode, args.model_type), "w") as f:
        json.dump(probs, f)
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="../glue_data/SST-5")
    parser.add_argument(
        "--model_type", type=str, default="bert", help="roberta, bert, electra"
    )

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()
    args.log_dir = "SST-5_{}.log".format(args.model_type)

    logger = Logger(args.log_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args)
    main(args, logger)
