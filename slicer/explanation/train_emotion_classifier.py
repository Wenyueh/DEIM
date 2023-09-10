import argparse
import csv
import os
import random
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer


def set_seed(args):
    random.seed(args.seed)


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


def load_data(args):
    data = []
    with open(args.data_dir + "/emobank.csv", "r", newline="\n") as f:
        reader = csv.reader(f, delimiter=",")
        for l in reader:
            data.append(l)
    data = data[1:]
    # three way classification: neutral 0, positive 1, negative 2
    new_data = []
    for d in data:
        if d[2] == "3.0":
            d[2] = 0
        elif float(d[2]) > 3.0:
            d[2] = 1
        else:
            d[2] = 2
        if d[3] == "3.0":
            d[3] = 0
        elif float(d[3]) > 3.0:
            d[3] = 1
        else:
            d[3] = 2
        if d[4] == "3.0":
            d[4] = 0
        elif float(d[4]) > 3.0:
            d[4] = 1
        else:
            d[4] = 2
        new_data.append(d)
    data = new_data
    # 8062 train
    train = [d for d in data if d[1] == "train"]
    # 1000 dev
    dev = [d for d in data if d[1] == "dev"]
    # 1000 test
    test = [d for d in data if d[1] == "test"]

    valency_train = [[d[-1], d[2]] for d in train]
    valency_dev = [[d[-1], d[2]] for d in dev]
    valency_test = [[d[-1], d[2]] for d in test]

    arousal_train = [[d[-1], d[3]] for d in train]
    arousal_dev = [[d[-1], d[3]] for d in dev]
    arousal_test = [[d[-1], d[3]] for d in test]

    dominance_train = [[d[-1], d[4]] for d in train]
    dominance_dev = [[d[-1], d[4]] for d in dev]
    dominance_test = [[d[-1], d[4]] for d in test]

    return (
        valency_train,
        valency_dev,
        valency_test,
        arousal_train,
        arousal_dev,
        arousal_test,
        dominance_train,
        dominance_dev,
        dominance_test,
    )


class InputDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = 64

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        word = data[0]
        label = data[1]
        encoding = self.tokenizer(
            word,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        input_ids = encoding["input_ids"]
        attn_mask = encoding["attention_mask"]
        label = torch.tensor(label)

        return input_ids.squeeze(), attn_mask.squeeze(), label


def main(args, logger):
    (
        valency_train,
        valency_dev,
        valency_test,
        arousal_train,
        arousal_dev,
        arousal_test,
        dominance_train,
        dominance_dev,
        dominance_test,
    ) = load_data(args)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.model_type == "valency":
        traindata = InputDataset(valency_train, tokenizer)
        train_loader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
        devdata = InputDataset(valency_dev, tokenizer)
        dev_loader = DataLoader(devdata, batch_size=args.batch_size, shuffle=False)
        testdata = InputDataset(valency_test, tokenizer)
        test_loader = DataLoader(testdata, batch_size=args.batch_size, shuffle=False)
    if args.model_type == "arousal":
        traindata = InputDataset(arousal_train, tokenizer)
        train_loader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
        devdata = InputDataset(arousal_dev, tokenizer)
        dev_loader = DataLoader(devdata, batch_size=args.batch_size, shuffle=False)
        testdata = InputDataset(arousal_test, tokenizer)
        test_loader = DataLoader(testdata, batch_size=args.batch_size, shuffle=False)
    if args.model_type == "dominance":
        traindata = InputDataset(dominance_train, tokenizer)
        train_loader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
        devdata = InputDataset(dominance_dev, tokenizer)
        dev_loader = DataLoader(devdata, batch_size=args.batch_size, shuffle=False)
        testdata = InputDataset(dominance_test, tokenizer)
        test_loader = DataLoader(testdata, batch_size=args.batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    ).cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    loss_value = 0
    num_step = 0
    for e in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch[0].cuda()
            attn_mask = batch[1].cuda()
            label = batch[2].cuda()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=label)
            loss = outputs[0]
            loss_value += loss.item()
            loss.backward()
            optimizer.step()
            num_step += 1
            if (num_step + 1) % args.logging_step == 0:
                print("loss value after {} steps is {}".format(num_step, loss_value))
                loss_value = 0

        model.eval()
        correct = 0
        for batch in test_loader:
            input_ids = batch[0].cuda()
            attn_mask = batch[1].cuda()
            label = batch[2].tolist()
            logits = model(input_ids, attention_mask=attn_mask).logits
            prediction = torch.argmax(logits, dim=-1).tolist()
            for p, l in zip(prediction, label):
                if p == l:
                    correct += 1
        acc = correct / len(testdata)

        logger.log("accuracy after epoch {} is {}".format(e, acc))

        torch.save(model, args.model_type + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.log_dir = args.model_type + ".log"

    set_seed(args)
    logger = Logger(args.log_dir)

    main(args, logger)
