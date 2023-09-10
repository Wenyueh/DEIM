import argparse
import json
import random
import sys
import os

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
    with open(args.data_dir + "/" + args.model_type+'.json', "r") as f:
        data = json.load(f)

    sent = list(set([d["context"] for d in data["examples"]]))

    with open(args.data_dir + "/" + "msr_paraphrase_train.txt", "r") as f:
        data = f.read()
    neg_sent = [d.split("\t")[3] for d in data.split('\n')[1:-1]]
    neg_sent = random.sample(neg_sent, k=min(len(neg_sent), len(sent)))

    data = []
    for s in sent:
        data.append((s, 1))
    for s in neg_sent:
        data.append((s, 0))

    random.shuffle(data)
    train = data[:-50]
    test = data[-50:]

    return train, test


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
    train, test = load_data(args)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    traindata = InputDataset(train, tokenizer)
    train_loader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True)
    testdata = InputDataset(test, tokenizer)
    test_loader = DataLoader(testdata, batch_size=args.batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    loss_value = 0
    num_step =0
    for e in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch[0].cuda()
            attn_mask = batch[1].cuda()
            label = batch[2].cuda()
            outputs = model(input_ids = input_ids, attention_mask=attn_mask, labels=label)
            loss = outputs[0]
            loss_value += loss.item()
            loss.backward()
            optimizer.step()
            num_step +=1
            if (num_step+1) % args.logging_step == 0:
                print('loss value after {} steps is {}'.format(num_step, loss_value))
                loss_value =0

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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--logging_step', type=int, default=10)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.log_dir = args.model_type + ".log"

    set_seed(args)
    logger = Logger(args.log_dir)

    main(args, logger)
