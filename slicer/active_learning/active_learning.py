import argparse
import math
import os
import random
import sys

# setting path
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
from domino import (
    DominoErrorSlicer,
    DistanceDominoSlicer,
    DominoSlicerGaussianYhat,
    DominoSlicerGaussianYhatNT,
)
from transformers import BertModel, BertTokenizer
import transformers

transformers.logging.set_verbosity_error()
# from transformers.models.bert.tokenization_bert import BertTokenizer

from active_learning_data import generate_dataloader, load_data
from MML_domino import train_MML_domino_pipeline
import json


def grouping(n, data):
    gro = []
    for i in range(len(data["sentence"])):
        if data["predicted_group"][i] == n:
            gro.append(
                [
                    data["sentence"][i],
                    data["label"][i],
                    data["probs"][i],
                    data["predicted_group"][i],
                ]
            )
    return gro


def compute_acc(gro):
    acc = []
    for g in gro:
        acc.append(np.argmax(g[2]) == g[1])
    return np.mean(acc)


def compute_error_slices(logger, args, data):
    error_slices = []
    for i in range(args.slice_n):
        gro = grouping(i, data)
        if len(gro) > 0:
            acc = compute_acc(gro)
        else:
            acc = 100
        error_slices.append((i, acc, len(gro)))
    error_slices = sorted(error_slices, key=lambda x: x[1])
    slice_number = [error_slice[0] for i, error_slice in enumerate(error_slices)]

    return error_slices, slice_number


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
    train, extra_train_data, dev, test = load_data(args)
    logger.log("initial training datapoints number {}".format(len(train["sentence"])))
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    train_loader = generate_dataloader(args, tokenizer, train, False)
    dev_loader = generate_dataloader(args, tokenizer, dev, True)
    test_loader = generate_dataloader(args, tokenizer, test, True)

    logger.log("load model")

    total_loop_number = math.ceil(
        len(extra_train_data["sentence"]) / args.extra_data_number
    )

    number_of_datapoints = []
    accuracy = []
    for loop_number in range(total_loop_number + 1):
        encoder = BertModel.from_pretrained("bert-large-uncased")
        classifier = binary_classifier(args, encoder).cuda()
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
        logger.log(
            "train model on {} loop using {} datapoints, {} many extra training datapoints can be selected".format(
                loop_number, len(train["sentence"]), len(extra_train_data["sentence"])
            )
        )
        classifier = train_model(args, logger, classifier, train_loader, optimizer)

        logger.log("evaluate test data")
        test_accuracy, test_probs = evaluate(classifier, test_loader)
        logger.log("compute validation data probability")
        _, dev_probs = evaluate(classifier, dev_loader)
        dev["probs"] = dev_probs

        logger.log(
            "after {} loops, test accuracy is {}".format(loop_number, test_accuracy)
        )
        number_of_datapoints.append(len(train["sentence"]))
        accuracy.append(test_accuracy)

        # when fininshed the last loop, do not keep computing extra datapoints
        # if loop_number == total_loop_number:
        #    break
        if len(train["sentence"]) > 16000:
            break

        selected_train = select_error_slice_train_data(
            args, logger, tokenizer, classifier, dev, extra_train_data
        )

        logger.log(
            "{} more training data are selected".format(len(selected_train["sentence"]))
        )

        train["sentence"] += selected_train["sentence"]
        train["label"] += selected_train["label"]

        train_loader = generate_dataloader(args, tokenizer, train, False)

        extra_train_data_indices = [
            i
            for i in range(len(extra_train_data["sentence"]))
            if extra_train_data["sentence"][i] not in selected_train["sentence"]
        ]
        extra_train_data["sentence"] = [
            extra_train_data["sentence"][i] for i in extra_train_data_indices
        ]
        extra_train_data["label"] = [
            extra_train_data["label"][i] for i in extra_train_data_indices
        ]

        with open(args.log_path.replace(".log", ".json"), "w") as f:
            json.dump([number_of_datapoints, accuracy], f)

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


def select_error_slice_train_data(
    args, logger, tokenizer, classifier, dev, extra_train_data
):
    #################### fit domino on test data ####################
    logger.log("compute validation embedding")
    dev_loader = generate_dataloader(args, tokenizer, dev, True)
    dev_embeddings = compute_embeddings(args, classifier, dev_loader)
    dev["embeddings"] = dev_embeddings

    logger.log("fit domino on validation dataset")
    domino = DistanceDominoSlicer(
        emb_log_likelihood_weight=args.emb_log_likelihood_weight,
        distance_log_likelihood_weight=args.distance_log_likelihood_weight,
        y_hat_log_likelihood_weight=args.y_hat_log_likelihood_weight,
        n_mixture_components=args.n_components,
        n_slices=args.slice_n,
        n_pca_components=args.pca_n_components,
        test=False,
        init_params="kmeans",
    )

    domino.fit(data=dev, embeddings="embeddings", targets="label", pred_probs="probs")
    dev["log_domino_slices"], dev["resp"] = domino.predict_proba(
        data=dev,
        embeddings="embeddings",
        targets="label",
        pred_probs="probs",
        test=False,
    )
    slice_prediction = dev["resp"][:]
    slice_group = np.argmax(slice_prediction, axis=1).tolist()
    dev["predicted_group"] = slice_group
    dev["resp"] = dev["resp"].tolist()
    dev["log_domino_slices"] = dev["log_domino_slices"].tolist()

    #################### compute domino on extra training data ####################
    logger.log("generate domino input dataset of extra training data")
    extra_train_loader = generate_dataloader(args, tokenizer, extra_train_data, False)
    extra_train_embeddings = compute_embeddings(args, classifier, extra_train_loader)
    extra_train_data["embeddings"] = extra_train_embeddings

    # add prediction result and prediction probability score here
    _, probs = evaluate(classifier, extra_train_loader)
    extra_train_data["probs"] = probs
    logger.log(
        "compute domino membership for {} extra training datapoints".format(len(probs))
    )
    (
        extra_train_data["log_domino_slices"],
        extra_train_data["resp"],
    ) = domino.predict_proba(
        data=extra_train_data,
        embeddings="embeddings",
        targets="label",
        pred_probs="probs",
        test=True,
    )
    slice_prediction = extra_train_data["resp"][:]
    slice_group = np.argmax(slice_prediction, axis=1).tolist()
    extra_train_data["predicted_group"] = slice_group
    extra_train_data["resp"] = extra_train_data["resp"].tolist()
    extra_train_data["log_domino_slices"] = extra_train_data[
        "log_domino_slices"
    ].tolist()

    logger.log(
        "select error datapoints based on fitted domino with data balance controlling"
    )
    selected_extra_train_data = filter_extra_train_data_likelihood(
        args, logger, dev, extra_train_data
    )

    return selected_extra_train_data


def filter_extra_train_data_likelihood(args, logger, dev, extra_train_data):
    error_slices, slice_number = compute_error_slices(logger, args, dev)
    error_threshold = 0
    for e in error_slices:
        if e[1] < args.error_threshold:
            error_threshold += 1
        else:
            break
    error_slice_number = [slice_number[i] for i in range(error_threshold)]
    extra_train_data_data_error_likelihood = []
    error_i = []
    for i in range(len(extra_train_data["sentence"])):
        if extra_train_data["predicted_group"][i] in error_slice_number:
            likelihood = np.sum(
                [extra_train_data["resp"][j] for j in error_slice_number]
            )
            extra_train_data_data_error_likelihood.append(
                [
                    extra_train_data["sentence"][i],
                    extra_train_data["label"][i],
                    extra_train_data["probs"][i],
                    likelihood,
                ]
            )
            error_i.append(i)
    extra_train_data_data_error_likelihood = sorted(
        extra_train_data_data_error_likelihood, key=lambda x: x[-1], reverse=True
    )

    selected_extra_train_data = {
        "sentence": [
            extra_train_data_data_error_likelihood[i][0]
            for i in range(len(extra_train_data_data_error_likelihood))
        ],
        "label": [
            extra_train_data_data_error_likelihood[i][1]
            for i in range(len(extra_train_data_data_error_likelihood))
        ],
    }

    return selected_extra_train_data


"""
def filter_extra_train_data_likelihood(
    args, logger, dev, extra_train_data, slice_label
):
    error_slices, error_slice_number = compute_error_slices(logger, args, dev)
    error_number_threshold = 0
    for e in error_slices:
        if e[1] < args.error_threshold:
            error_number_threshold += 1
        else:
            break
    error_slice_number = [error_slice_number[i] for i in range(error_number_threshold)]
    test_data_error_likelihood = []
    for i in range(len(extra_train_data["sentence"])):
        domino_slice_probs = extra_train_data["domino_slices"][i]
        predicted_slice_group = np.argmax(domino_slice_probs)
        likelihood = np.sum([domino_slice_probs[j] for j in error_slice_number])
        test_data_error_likelihood.append(
            [
                extra_train_data["sentence"][i],
                slice_label[predicted_slice_group],
                extra_train_data['label'][i],
                likelihood,
            ]
        )
    test_data_error_likelihood = sorted(
        test_data_error_likelihood, key=lambda x: x[-1], reverse=True
    )

    ##### selec datapoints with large error likelihood #####
    selected_extra_train_data = {"sentence": [], "label": []}
    number_one = 0
    number_zero = 0
    current_one = 0
    current_zero = 0
    total_zero = np.sum(
        [
            test_data_error_likelihood[i][1] == 0
            for i in range(len(test_data_error_likelihood))
        ]
    )
    total_one = np.sum(
        [
            test_data_error_likelihood[i][1] == 1
            for i in range(len(test_data_error_likelihood))
        ]
    )
    logger.log(
        "in total {} datapoints are guessed to be one, {} datapoints are guessed to be zero".format(
            total_one, total_zero
        )
    )
    if (
        total_zero > args.extra_data_number / 2
        and total_one > args.extra_data_number / 2
    ) or (
        total_one < args.extra_data_number / 2
        and total_zero < args.extra_data_number / 2
    ):
        number_one = args.extra_data_number / 2
        number_zero = args.extra_data_number / 2
    elif (
        total_one > args.extra_data_number / 2
        and total_zero < args.extra_data_number / 2
    ):
        number_one = min(
            args.extra_data_number / 2 - total_zero, total_zero + total_one
        )
        number_zero = total_zero
    elif (
        total_zero > args.extra_data_number / 2
        and total_one < args.extra_data_number / 2
    ):
        number_zero = min(
            args.extra_data_number / 2 - total_one, total_one + total_zero
        )
        number_one = total_one
    for i in range(len(test_data_error_likelihood)):
        if test_data_error_likelihood[i][1] == 1:
            if current_one < number_one:
                selected_extra_train_data["sentence"].append(
                    test_data_error_likelihood[i][0]
                )
                selected_extra_train_data["label"].append(
                    test_data_error_likelihood[i][2]
                )
                current_one += 1
        if test_data_error_likelihood[i][1] == 0:
            if current_zero < number_zero:
                selected_extra_train_data["sentence"].append(
                    test_data_error_likelihood[i][0]
                )
                selected_extra_train_data["label"].append(
                    test_data_error_likelihood[i][2]
                )
                current_zero += 1
        if current_zero >= number_zero and current_one >= number_one:
            break

    return selected_extra_train_data
"""


def compute_slice_label(args, test):
    length_of_test = len(test["embeddings"])

    def one_slice_label(test, i):
        return [
            test["label"][j]
            for j in range(length_of_test)
            if test["predicted_group"][j] == i
        ]

    def find_majority_label(labels):
        one = labels.count(1)
        zero = labels.count(0)
        if one > zero:
            return 1
        else:
            return 0

    slice_label = {}
    for i in range(args.slice_n):
        labels = one_slice_label(test, i)
        slice_label[i] = find_majority_label(labels)
    return slice_label


"""
def select_extra_train_data_confidence(
    args, logger, tokenizer, classifier, test, extra_train_data
):
    error_slices, error_slice_number = compute_error_slices(logger, args, test)
    error_number_threshold = 0
    for e in error_slices:
        if e[1] < args.error_threshold:
            error_number_threshold += 1
        else:
            break
    error_slice_number = [error_slice_number[i] for i in range(error_number_threshold)]
    extra_train_data["pred_slice"] = [
        np.argmax(extra_train_data["domino_slices"][i])
        for i in range(len(extra_train_data["domino_slices"]))
    ]
    error_slice_data = [
        [
            extra_train_data["sentence"][i],
            extra_train_data["label"][i],
            extra_train_data["probs"][i],
        ]
        for i in range(len(extra_train_data["sentence"]))
        if extra_train_data["pred_slice"][i] in error_slice_number
    ]
    if len(extra_train_data) > args.extra_data_number:
        confidences = []
        logger.log("compute confidence score and then do selection")
        for i in range(len(extra_train_data)):
            zero, one = error_slice_data[i][-1]
            if zero > 0.5:
                confidence = zero - 0.5
            else:
                confidence = one - 0.5
            confidences.append(confidence)
        low_confidence_indices = np.argsort(confidences)
        ##### selec datapoints with large error likelihood #####
        selected_extra_train_data = {"sentence": [], "label": []}
        number_one = 0
        number_zero = 0
        for i in low_confidence_indices[:args.extra_data_number]:
            selected_extra_train_data["sentence"].append(extra_train_data[i][0])
            selected_extra_train_data["label"].append(extra_train_data[i][1])
        return selected_extra_train_data
    else:
        logger.log("output all datapoints")
        return extra_train_data
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # setting
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="QNLI")

    # parameters for domino
    parser.add_argument("--layer_n", type=int, default=12)
    parser.add_argument("--pca_n_components", type=int, default=128)
    parser.add_argument("--slice_n", type=int, default=128)
    parser.add_argument("--n_components", type=int, default=128)
    parser.add_argument("--y_hat_log_likelihood_weight", type=float, default=1)
    parser.add_argument("--distance_log_likelihood_weight", type=float, default=0.1)
    parser.add_argument("--emb_log_likelihood_weight", type=float, default=0.15)

    # base model training part
    parser.add_argument("--data_dir", type=str, default="../../myglue/glue_data/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--accumulate_steps", type=int, default=2)

    # data generation part
    parser.add_argument("--initial_prop", type=float, default=0.01)
    parser.add_argument("--extra_data_number", type=int, default=500)
    parser.add_argument("--error_threshold", type=float, default=0.5)
    parser.add_argument(
        "--selection_method",
        type=str,
        default="likelihood",
        help="likelihood or likelihood_confidence",
    )
    # domino part

    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    args.log_path = "log/sdm_" + args.task + "_{}.log".format(args.seed)
    if "jigsaw" not in args.task:
        args.data_dir = args.data_dir + args.task + "/"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger = Logger(args.log_path)

    set_seed(args)

    number_of_datapoints, accuracy = main(args, logger)
    with open(args.log_path.replace(".log", ".json"), "w") as f:
        json.dump([number_of_datapoints, accuracy], f)
