import argparse
import csv

import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, RobertaTokenizer, ElectraTokenizer


def collect_tsv_data(args):
    if args.task in ["jigsaw_gender", "jigsaw_racial", "jigsaw_religion"]:
        train_dir = "../myglue/glue_data/toxicity/{}_train.json".format(
            args.task
        ).replace("jigsaw_", "")
        dev_dir = "../myglue/glue_data/toxicity/{}_dev.json".format(args.task).replace(
            "jigsaw_", ""
        )
        test_dir = "../myglue/glue_data/toxicity/{}_test.json".format(
            args.task
        ).replace("jigsaw_", "")

        with open(train_dir, "r") as f:
            train = json.load(f)
        with open(dev_dir, "r") as f:
            dev = json.load(f)
        with open(test_dir, "r") as f:
            test = json.load(f)

        train = [
            [train["text"][i], train["label"][i]] for i in range(len(train["text"]))
        ]
        dev = [[dev["text"][i], dev["label"][i]] for i in range(len(dev["text"]))]
        test = [[test["text"][i], test["label"][i]] for i in range(len(test["text"]))]

        return train, dev, test

    if args.task == "MRPC":
        train_dir = args.data_dir + "train.txt"
        dev_dir = args.data_dir + "dev.txt"
        test_dir = args.data_dir + "test.tsv"

        with open(train_dir, "r") as f:
            train = f.read().split("\n")
        with open(dev_dir, "r") as f:
            dev = f.read().split("\n")
        test = []
        with open(test_dir, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\n")
            for l in reader:
                test.append(l)
    else:
        train_dir = args.data_dir + "train.tsv"
        dev_dir = args.data_dir + "dev.tsv"
        test_dir = args.data_dir + "test.tsv"

        train = []
        with open(train_dir, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\n")
            for l in reader:
                train.append(l)

        dev = []
        with open(dev_dir, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\n")
            for l in reader:
                dev.append(l)

        test = []
        with open(test_dir, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\n")
            for l in reader:
                test.append(l)

        if args.toy:
            train = train[:10]

    if args.task == "RTE":  # natural language inference
        train.remove(train[0])
        train = [
            (
                d[0].split("\t")[-3].strip(),
                d[0].split("\t")[-2].strip(),
                1 if d[0].split("\t")[-1] == "entailment" else 0,
            )
            for d in train
        ]
        dev.remove(dev[0])
        dev = [
            (
                d[0].split("\t")[-3].strip(),
                d[0].split("\t")[-2].strip(),
                1 if d[0].split("\t")[-1] == "entailment" else 0,
            )
            for d in dev
        ]
        test.remove(test[0])
        test = [
            (d[0].split("\t")[-2].strip(), d[0].split("\t")[-1].strip(),) for d in test
        ]
    elif args.task == "WNLI":  # natural language inference
        train.remove(train[0])
        train = [
            (
                d[0].split("\t")[-3].strip(),
                d[0].split("\t")[-2].strip(),
                int(d[0].split("\t")[-1]),
            )
            for d in train
        ]
        dev.remove(dev[0])
        dev = [
            (
                d[0].split("\t")[-3].strip(),
                d[0].split("\t")[-2].strip(),
                int(d[0].split("\t")[-1]),
            )
            for d in dev
        ]
        test.remove(test[0])
        test = [
            (d[0].split("\t")[-2].strip(), d[0].split("\t")[-1].strip(),) for d in test
        ]
    elif args.task == "MRPC":  # paraphrase
        train.remove(train[0])
        train.remove(train[-1])
        train = [
            (
                d.split("\t")[-2].strip(),
                d.split("\t")[-1].strip(),
                int(d.split("\t")[0]),
            )
            for d in train
        ]
        dev.remove(dev[0])
        dev.remove(dev[-1])
        dev = [
            (
                d.split("\t")[-2].strip(),
                d.split("\t")[-1].strip(),
                int(d.split("\t")[0]),
            )
            for d in dev
        ]
        test.remove(test[0])
        test = [
            (d[0].split("\t")[-2].strip(), d[0].split("\t")[-1].strip(),) for d in test
        ]
    elif args.task == "MNLI":  # 3-labels, natural language inference
        # entailment=1
        # neural=0
        # contradiction=2
        train.remove(train[0])
        train = [
            (
                d[0].split("\t")[-4].strip(),
                d[0].split("\t")[-3].strip(),
                1
                if d[0].split("\t")[-1] == "entailment"
                else 0
                if d[0].split("\t")[-1] == "neutral"
                else 2,
            )
            for d in train
        ]
        dev.remove(dev[0])
        dev = [
            (
                d[0].split("\t")[-8].strip(),
                d[0].split("\t")[-7].strip(),
                1
                if d[0].split("\t")[-1] == "entailment"
                else 0
                if d[0].split("\t")[-1] == "neutral"
                else 2,
            )
            for d in dev
        ]
        test.remove(test[0])
        test = [
            (d[0].split("\t")[-2].strip(), d[0].split("\t")[-1].strip(),) for d in test
        ]
    elif args.task == "SST-2":  # sentiment
        train.remove(train[0])
        train = [(d[0].split("\t")[0].strip(), int(d[0].split("\t")[1])) for d in train]
        dev.remove(dev[0])
        dev = [(d[0].split("\t")[0].strip(), int(d[0].split("\t")[1])) for d in dev]
        test.remove(test[0])
        test = [(d[0].split("\t")[1].strip(), int(d[0].split("\t")[0])) for d in test]
    elif args.task == "QNLI":  # inference
        train.remove(train[0])
        train = [
            (
                d[0].split("\t")[1].strip(),
                d[0].split("\t")[2].strip(),
                1 if d[0].split("\t")[3] == "entailment" else 0,
            )
            for d in train
        ]
        dev.remove(dev[0])
        dev = [
            (
                d[0].split("\t")[1].strip(),
                d[0].split("\t")[2].strip(),
                1 if d[0].split("\t")[3] == "entailment" else 0,
            )
            for d in dev
        ]
        test.remove(test[0])
        test = [
            (d[0].split("\t")[0].strip(), d[0].split("\t")[1].strip(),) for d in test
        ]
    elif args.task == "QQP":  # paraphrase
        train.remove(train[0])
        train = [
            (
                d[0].split("\t")[3].strip(),
                d[0].split("\t")[4].strip(),
                int(d[0].split("\t")[-1]),
            )
            for d in train
        ]
        dev.remove(dev[0])
        dev = [
            (
                d[0].split("\t")[3].strip(),
                d[0].split("\t")[4].strip(),
                int(d[0].split("\t")[-1]),
            )
            for d in dev
        ]
        test.remove(test[0])
        test = [
            (d[0].split("\t")[0].strip(), d[0].split("\t")[1].strip(),) for d in test
        ]
    elif args.task == "CoLA":
        train = [[d[0].split("\t")[3], int(d[0].split("\t")[1])] for d in train]
        dev = [[d[0].split("\t")[3], int(d[0].split("\t")[1])] for d in dev]

    length = len(train)
    switched = int(length / 2)
    half_train = train[:switched]
    test = dev
    dev = train[switched:]
    train = half_train

    return train, dev, test


class OneSentInputData(Dataset):
    def __init__(self, args, data, tokenizer, max_len):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datapoint = self.data[idx]
        text = datapoint[0]
        label = datapoint[1]
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokenized_text["input_ids"].squeeze()
        attention_mask = tokenized_text["attention_mask"].squeeze()
        if self.args.model_type == "bert":
            token_type_ids = tokenized_text["token_type_ids"].squeeze()
        else:
            token_type_ids = torch.ones_like(attention_mask)
        return input_ids, attention_mask, token_type_ids, label


class TwoSentInputData(Dataset):
    def __init__(self, args, data, tokenizer, max_len):
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datapoint = self.data[idx]
        textone = datapoint[0]
        texttwo = datapoint[1]
        label = datapoint[2]
        tokenized_text = self.tokenizer(
            textone,
            texttwo,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokenized_text["input_ids"].squeeze()
        attention_mask = tokenized_text["attention_mask"].squeeze()
        if self.args.model_type == "bert":
            token_type_ids = tokenized_text["token_type_ids"].squeeze()
        else:
            token_type_ids = torch.ones_like(attention_mask)
        return input_ids, attention_mask, token_type_ids, label


def load_dataloaders(args):
    train, dev, test = collect_tsv_data(args)
    print(train[0])
    print(dev[0])
    print(test[0])
    if args.model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif args.model_type == "electra":
        tokenizer = ElectraTokenizer.from_pretrained(
            "google/electra-large-discriminator"
        )

    if args.task == "SST-2" or args.task == "CoLA":
        train_dataset = OneSentInputData(args, train, tokenizer, 64)
        dev_dataset = OneSentInputData(args, dev, tokenizer, 64)
        test_dataset = OneSentInputData(args, test, tokenizer, 64)
        print(len(train_dataset))
        print(len(dev_dataset))
        print(len(test_dataset))
    elif args.task in ["jigsaw_gender", "jigsaw_racial", "jigsaw_religion"]:
        train_dataset = OneSentInputData(args, train, tokenizer, 256)
        dev_dataset = OneSentInputData(args, dev, tokenizer, 256)
        test_dataset = OneSentInputData(args, test, tokenizer, 256)
        print(len(train_dataset))
        print(len(dev_dataset))
        print(len(test_dataset))
    else:
        train_dataset = TwoSentInputData(args, train, tokenizer, 64)
        dev_dataset = TwoSentInputData(args, dev, tokenizer, 64)
        test_dataset = TwoSentInputData(args, test, tokenizer, 64)
        print(len(train_dataset))
        print(len(dev_dataset))
        print(len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="MNLI")
    parser.add_argument("--data_dir", type=str, default="glue_data/")
    parser.add_argument("--toy", action="store_true")

    args = parser.parse_args()
    args.data_dir += args.task + "/"

    train, dev, test = collect_tsv_data(args)
    print(train[0])
    print(dev[0])
    print(test[0])
    # print(len(train))
    # print(len(dev))
    # print(len(test))

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # inputdata = TwoSentInputData(dev, tokenizer, 128)

    # dataloader = DataLoader(inputdata, batch_size=2, shuffle=False)
