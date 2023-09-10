import csv
import random

import torch
from torch.utils.data import DataLoader, Dataset
import json


def load_data(args):
    if args.task in ["jigsaw_gender", "jigsaw_racial", "jigsaw_religion"]:
        train_dir = "../../myglue/glue_data/toxicity/{}_train.json".format(
            args.task
        ).replace("jigsaw_", "")
        dev_dir = "../../myglue/glue_data/toxicity/{}_dev.json".format(
            args.task
        ).replace("jigsaw_", "")
        test_dir = "../../myglue/glue_data/toxicity/{}_test.json".format(
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

        random.shuffle(train)
        length = len(train)
        extra_training_prop = int(length * args.initial_prop)
        extra_training_data = train[extra_training_prop:]
        initial_train = train[:extra_training_prop]
        train = {
            "sentence": [d[0] for d in initial_train],
            "label": [d[1] for d in initial_train],
        }
        extra_training_data = {
            "sentence": [d[0] for d in extra_training_data],
            "label": [d[1] for d in extra_training_data],
        }
        dev_data = {
            "sentence": [d[0] for d in dev],
            "label": [d[1] for d in dev],
        }
        test = {
            "sentence": [d[0] for d in test],
            "label": [d[1] for d in test],
        }

        return train, extra_training_data, dev_data, test

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

    random.shuffle(train)

    length = len(train)
    half_length = int(length / 2)
    half_train = train[:half_length]
    test = dev
    dev = train[half_length:]
    train = half_train
    length = len(train)
    extra_training_prop = int(length * args.initial_prop)
    extra_training_data = train[extra_training_prop:]
    initial_train = train[:extra_training_prop]

    if args.task == "SST-2":  # sentiment
        initial_train.remove(initial_train[0])
        train = {
            "sentence": [d[0].split("\t")[0].strip() for d in initial_train],
            "label": [int(d[0].split("\t")[1]) for d in initial_train],
        }

        extra_training_data_set = {"sentence": [], "label": []}
        extra_training_data.remove(["sentence\tlabel"])
        for d in extra_training_data:
            extra_training_data_set["sentence"].append(d[0].split("\t")[0].strip())
            extra_training_data_set["label"].append(int(d[0].split("\t")[1]))
        extra_training_data = extra_training_data_set

        dev_data = {
            "sentence": [d[0].split("\t")[0].strip() for d in dev],
            "label": [int(d[0].split("\t")[1]) for d in dev],
        }

        test.remove(test[0])
        test = {
            "sentence": [d[0].split("\t")[0].strip() for d in test],
            "label": [int(d[0].split("\t")[1]) for d in test],
        }
    elif args.task == "QNLI":
        initial_train.remove(initial_train[0])
        train = {
            "sentence": [
                (d[0].split("\t")[1].strip(), d[0].split("\t")[2].strip(),)
                for d in initial_train
            ],
            "label": [
                1 if d[0].split("\t")[3] == "entailment" else 0 for d in initial_train
            ],
        }
        extra_training_data = {
            "sentence": [
                (d[0].split("\t")[1].strip(), d[0].split("\t")[2].strip(),)
                for d in extra_training_data
            ],
            "label": [
                1 if d[0].split("\t")[3] == "entailment" else 0
                for d in extra_training_data
            ],
        }
        dev_data = {
            "sentence": [
                (d[0].split("\t")[1].strip(), d[0].split("\t")[2].strip(),) for d in dev
            ],
            "label": [1 if d[0].split("\t")[3] == "entailment" else 0 for d in dev],
        }
        test.remove(test[0])
        test = {
            "sentence": [
                (d[0].split("\t")[1].strip(), d[0].split("\t")[2].strip(),)
                for d in test
            ],
            "label": [1 if d[0].split("\t")[3] == "entailment" else 0 for d in test],
        }
    elif args.task == "QQP":  # paraphrase
        train.remove(train[0])
        train = {
            "sentence": [
                (d[0].split("\t")[3].strip(), d[0].split("\t")[4].strip(),)
                for d in initial_train
            ],
            "label": [int(d[0].split("\t")[-1]) for d in initial_train],
        }
        extra_training_data = {
            "sentence": [
                (d[0].split("\t")[3].strip(), d[0].split("\t")[4].strip(),)
                for d in extra_training_data
            ],
            "label": [int(d[0].split("\t")[-1]) for d in extra_training_data],
        }
        dev_data = {
            "sentence": [
                (d[0].split("\t")[3].strip(), d[0].split("\t")[4].strip(),) for d in dev
            ],
            "label": [int(d[0].split("\t")[-1]) for d in dev],
        }
        test.remove(test[0])
        test = {
            "sentence": [
                (d[0].split("\t")[3].strip(), d[0].split("\t")[4].strip(),)
                for d in test
            ],
            "label": [int(d[0].split("\t")[-1]) for d in test],
        }
    elif args.task == "CoLA":
        train = {
            "sentence": [d[0].split("\t")[3] for d in initial_train],
            "label": [int(d[0].split("\t")[1]) for d in initial_train],
        }
        extra_training_data = {
            "sentence": [d[0].split("\t")[3] for d in extra_training_data],
            "label": [int(d[0].split("\t")[1]) for d in extra_training_data],
        }
        dev_data = {
            "sentence": [d[0].split("\t")[3] for d in dev],
            "label": [int(d[0].split("\t")[1]) for d in dev],
        }
        test = {
            "sentence": [d[0].split("\t")[3] for d in test],
            "label": [int(d[0].split("\t")[1]) for d in test],
        }

    return train, extra_training_data, dev_data, test


class OneSentInputData(Dataset):
    def __init__(self, data, tokenizer, max_len):
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
        token_type_ids = tokenized_text["token_type_ids"].squeeze()

        label = torch.tensor(label)

        return input_ids, attention_mask, token_type_ids, label


class TwoSentInputData(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datapoint = self.data[idx]
        textone = datapoint[0][0]
        texttwo = datapoint[0][1]
        label = datapoint[1]
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
        token_type_ids = tokenized_text["token_type_ids"].squeeze()

        label = torch.tensor(label)

        return input_ids, attention_mask, token_type_ids, label


def generate_dataloader(args, tokenizer, data, test):
    if args.task in ["CoLA", "SST-2"]:
        if isinstance(data, dict):
            data = [
                [data["sentence"][i], data["label"][i]]
                for i in range(len(data["sentence"]))
            ]
        dataset = OneSentInputData(data, tokenizer, 64)
    elif "jigsaw" in args.task:
        if isinstance(data, dict):
            data = [
                [data["sentence"][i], data["label"][i]]
                for i in range(len(data["sentence"]))
            ]
        dataset = OneSentInputData(data, tokenizer, 128)
    else:
        if isinstance(data, dict):
            data = [
                [data["sentence"][i], data["label"][i]]
                for i in range(len(data["sentence"]))
            ]
        dataset = TwoSentInputData(data, tokenizer, 128)

    if not test:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataloader
