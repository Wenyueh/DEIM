import argparse
import math
import os
import random
import sys
import json
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.models.bert.tokenization_bert import BertTokenizer

from active_learning_data import generate_dataloader, load_data
import transformers

transformers.logging.set_verbosity_error()


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        self.encoder.config.output_hidden_states = True
        self.linear_layer = nn.Linear(self.hidden_size, 2, bias=True)
        self.dropout = nn.Dropout(self.args.dropout_rate)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        embeddings = out[0].mean(dim=1)
        embeddings = self.dropout(embeddings)
        logits = self.linear_layer(embeddings)

        all_hidden_states = out[2]
        return logits, embeddings, all_hidden_states

    def compute_loss(self, logits, labels):
        loss_value = self.loss(logits, labels)
        return loss_value

    def predict(self, logits):
        assert logits.dim() == 2
        return torch.argmax(logits, dim=1)


def compute_embeddings(args, classifier, data_loader):
    embeddings = []
    for batch in data_loader:
        input_ids = batch[0].cuda()
        attn_mask = batch[1].cuda()
        segment_ids = batch[2].cuda()
        b = input_ids.size(0)
        _, hidden_states, all_hidden_states = classifier(
            input_ids, attn_mask, segment_ids
        )
        for i in range(b):
            one_attn_mask = attn_mask.tolist()[i]
            # remove padding embeddings
            if 0 in one_attn_mask:
                padding_position = one_attn_mask.index(0)
            else:
                padding_position = len(one_attn_mask)
            one_hidden_states = (
                all_hidden_states[args.layer_n][i][:padding_position, :]
                .mean(dim=0)
                .tolist()
            )
            embeddings.append(one_hidden_states)

    return embeddings


def main(args, logger):

    logger.log("load data")
    train, extra_train_data, dev_data, test = load_data(args)
    logger.log("initial training datapoints number {}".format(len(train["sentence"])))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader = generate_dataloader(args, tokenizer, train, test=False)
    test_loader = generate_dataloader(args, tokenizer, test, test=True)

    logger.log("load model")
    encoder = AutoModel.from_pretrained("bert-large-uncased")

    total_loop_number = math.ceil(
        len(extra_train_data["sentence"]) / args.extra_data_number
    )
    accuracy = []
    number_of_datapoints = []
    for loop_number in range(total_loop_number):
        classifier = binary_classifier(args, encoder).cuda()
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
        logger.log(
            "train model on {} loop using {} datapoints".format(
                loop_number, len(train["sentence"])
            )
        )
        if len(train["sentence"]) > 16000:
            break
        classifier = train_model(args, logger, classifier, train_loader, optimizer)

        logger.log("evaluate test data")
        test_accuracy, test_probs = evaluate(classifier, test_loader)
        test["probs"] = test_probs

        logger.log(
            "after {} loops, test accuracy is {}".format(loop_number, test_accuracy)
        )
        accuracy.append(test_accuracy)
        number_of_datapoints.append(len(train["sentence"]))

        selected_train_indices = random.sample(
            list(range(len(extra_train_data["sentence"]))),
            k=min(args.extra_data_number, len(extra_train_data["sentence"])),
        )

        train["sentence"] += [
            extra_train_data["sentence"][i] for i in selected_train_indices
        ]
        train["label"] += [extra_train_data["label"][i] for i in selected_train_indices]

        train_loader = generate_dataloader(args, tokenizer, train, False)

        extra_train_data["sentence"] = [
            extra_train_data["sentence"][i]
            for i in range(len(extra_train_data["sentence"]))
            if i not in selected_train_indices
        ]
        extra_train_data["label"] = [
            extra_train_data["label"][i]
            for i in range(len(extra_train_data["label"]))
            if i not in selected_train_indices
        ]
    return number_of_datapoints, accuracy


def train_model(args, logger, classifier, dataloader, optimizer):
    logger.log("start training")
    classifier.zero_grad()
    num_steps = 0
    for epoch in range(args.epochs):
        logger.log("training on epoch {}".format(epoch))
        loss_value = 0
        classifier.train()
        for batch in dataloader:
            input_ids = batch[0].cuda()
            attn_mask = batch[1].cuda()
            token_type_ids = batch[2].cuda()
            labels = batch[3].cuda()

            logits, _, _ = classifier(
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids,
            )

            loss = classifier.compute_loss(logits, labels)
            loss_value += loss.item()

            loss.backward()
            num_steps += 1

            if (num_steps + 1) % args.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.clip)
                optimizer.step()
                classifier.zero_grad()

            if (num_steps + 1) % args.logging_steps == 0:
                logger.log(
                    "loss value for steps {} to {} is {}".format(
                        num_steps - args.logging_steps, num_steps, loss_value
                    )
                )
                loss_value = 0

    return classifier


def evaluate(model, data_loader):
    model.eval()
    predictions = []
    probs = []
    num_correct = 0
    num_total = 0
    softmax_fc = nn.Softmax(dim=1)
    for batch in data_loader:
        input_ids = batch[0].cuda()
        attn_mask = batch[1].cuda()
        token_type_ids = batch[2].cuda()
        labels = batch[3].tolist()

        logits, _, _ = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        # save results
        prediction = model.predict(logits).tolist()
        predictions += prediction
        prob = softmax_fc(logits).tolist()
        probs += prob
        for p, l in zip(prediction, labels):
            if p == l:
                num_correct += 1
        num_total += batch[0].size(0)
    accuracy = num_correct / num_total
    return accuracy, probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model training part
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="jigsaw_racial")
    parser.add_argument("--data_dir", type=str, default="../../myglue/glue_data/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--accumulate_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    # data generation part
    parser.add_argument("--initial_prop", type=float, default=0.15)
    parser.add_argument("--extra_data_number", type=int, default=500)
    parser.add_argument("--error_threshold", type=float, default=0.5)
    parser.add_argument("--extra_train_selection_threshold", type=float, default=0.9)
    # domino part
    parser.add_argument("--layer_n", type=int, default=12)
    parser.add_argument("--slice_n", type=int, default=128)
    parser.add_argument("--emb_log_likelihood_weight", type=float, default=0.15)
    parser.add_argument("--distance_log_likelihood_weight", type=float, default=0.1)
    parser.add_argument("--y_hat_log_likelihood_weight", type=float, default=1)
    parser.add_argument("--method", type=str, default="likelihood")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    args.log_path = "log/random_learning_" + args.task + "_{}.log".format(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if "jigsaw" not in args.task:
        args.data_dir = args.data_dir + args.task + "/"

    logger = Logger(args.log_path)

    set_seed(args)
    number_of_datapoints, accuracy = main(args, logger)
    with open(
        "log/random_learning_" + args.task + "_{}.json".format(args.seed), "w"
    ) as f:
        json.dump([number_of_datapoints, accuracy], f)
