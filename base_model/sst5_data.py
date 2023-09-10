import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers import RobertaTokenizer, ElectraTokenizer


def load_data(args):
    with open(args.data_dir + "/train.txt", "r") as f:
        train = f.read()
    with open(args.data_dir + "/dev.txt", "r") as f:
        dev = f.read()
    with open(args.data_dir + "/test.txt", "r") as f:
        test = f.read()

    train = train.split("\n")[:-1]
    dev = dev.split("\n")[:-1]
    test = test.split("\n")[:-1]

    train = [[d.split("\t")[1], int(d.split("\t")[0])] for d in train]
    switch = int(len(train) * (3 / 5))
    dev = [[d.split("\t")[1], int(d.split("\t")[0])] for d in dev] + train[switch:]
    train = train[:switch]
    test = [[d.split("\t")[1], int(d.split("\t")[0])] for d in test]

    print("length of train dataset is {}".format(len(train)))
    print("length of dev dataset is {}".format(len(dev)))
    print("length of test dataset is {}".format(len(test)))

    return train, dev, test


class InputDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.args = args
        self.label = [d[1] for d in data]
        self.sentences = [d[0] for d in data]
        self.tokenizer = tokenizer
        self.max_len = 64
        tokenized_data = self.tokenizer(
            self.sentences,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        if self.args.model_type == "bert":
            self.token_type_ids = tokenized_data["token_type_ids"]
        else:
            self.token_type_ids = torch.ones_like(tokenized_data["attention_mask"])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx]),
            torch.tensor(self.attention_mask[idx]),
            torch.tensor(self.token_type_ids[idx]),
            torch.tensor(self.label[idx]),
        )


def load_dataloaders(args):
    train, dev, test = load_data(args)
    if args.model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.model_type == "electra":
        tokenizer = ElectraTokenizer.from_pretrained(
            "google/electra-large-discriminator"
        )
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    train_dataset = InputDataset(args, train, tokenizer)
    dev_dataset = InputDataset(args, dev, tokenizer)
    test_dataset = InputDataset(args, test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="sst5")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    train_loader, dev_loader, test_loader = load_dataloaders(args)

    for batch in train_loader:
        print(batch[0])
