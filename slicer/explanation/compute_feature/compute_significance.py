from scipy.stats import ttest_ind
import argparse
import json
import os
import random
import sys

import numpy as np


def set_seed(args):
    np.random.seed(args.seed)
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


def grouping_idx(n, data):
    gro = []
    for i in range(len(data["sentence"])):
        if data["predicted_group"][i] == n:
            if len(data["sentence"][i]) == 2:
                gro.append(
                    [
                        i,
                        data["sentence"][i][1],
                        data["label"][i],
                        data["probs"][i],
                        data["resp"][i][n],
                    ]
                )
            else:
                gro.append(
                    [
                        i,
                        data["sentence"][i],
                        data["label"][i],
                        data["probs"][i],
                        data["resp"][i][n],
                    ]
                )
    return gro


def alldata_idx(n, data):
    others = []
    for i in range(len(data["sentence"])):
        if len(data["sentence"][i]) == 2:
            others.append(
                [
                    i,
                    data["sentence"][i][1],
                    data["label"][i],
                    data["probs"][i],
                    data["resp"][i][n],
                ]
            )
        else:
            others.append(
                [
                    i,
                    data["sentence"][i],
                    data["label"][i],
                    data["probs"][i],
                    data["resp"][i][n],
                ]
            )
    return others


def compute_acc(gro):
    acc = []
    for g in gro:
        acc.append(np.argmax(g[3]) == g[2])
    return np.mean(acc)


def significance_test(args, logger, feature_data, grouping_data):
    if args.task in ["QQP", "MNLI"]:
        small_feature_data = {}
        for k, v in feature_data.items():
            small_feature_data[k] = v[:10000]
        small_grouping_data = {}
        for k, v in grouping_data.items():
            small_grouping_data[k] = v[:10000]
        feature_data = small_feature_data
        grouping_data = small_grouping_data

    all_data_indices = list(range(len(feature_data["sentence"])))

    for f in [
        "length",
        "reflexive",
        "negation",
        "frequency",
        "how",
        "why",
        "multi_preposition",
        "NP_sub",
        "echo_question",
        # "tree_depth",
        "extra_to",
        "multi_modal",
        "quantifier",
        "voice",
        "tense",
        "aspect",
        "comparison",
        "foreign_word",
        "long_distance",
    ]:
        if feature_data[f] == []:
            continue
        logger.log("compute feature {}".format(f))
        feature_indices = []
        target_distributions = []
        for n in range(args.slice_n):
            target_slice = grouping_idx(n, grouping_data)
            target_accuracy = compute_acc(target_slice)
            if target_accuracy < args.error_threshold:
                target_indices = [t[0] for t in target_slice]
                others_indices = random.sample(
                    [i for i in all_data_indices if i not in target_indices],
                    k=max(len(target_slice), 100),
                )
                target_distribution = return_distribution(
                    feature_data, target_indices, f
                )
                others_distribution = return_distribution(
                    feature_data, others_indices, f
                )
                pvalue = ttest_ind(target_distribution, others_distribution).pvalue
                target_mean = np.mean(target_distribution)
                all_mean = np.mean(others_distribution)
                if pvalue < args.significance_threshold and target_mean > all_mean:
                    feature_indices += target_indices
                    target_distributions += target_distribution
        logger.log(
            "for feature {}, there are {} relevant datapoints".format(
                f, len(feature_indices)
            )
        )
        if not feature_indices:
            logger.log("no slice featured with linguistic {} found".format(f))
            logger.log("****")
        else:
            other_indices = random.sample(
                [i for i in all_data_indices if i not in feature_indices],
                k=max(len(feature_indices), 100),
            )
            all_indices = feature_indices + other_indices
            other_distributions = return_distribution(feature_data, other_indices, f)
            all_distributions = target_distributions + other_distributions
            target_prediction_result = [
                int(label == np.argmax(prediction))
                for i, (label, prediction) in enumerate(
                    zip(grouping_data["label"], grouping_data["probs"])
                )
                if i in other_indices
            ]
            slice_accuracy = target_prediction_result.count(0) / len(
                target_prediction_result
            )
            all_prediction_result = [
                int(label == np.argmax(prediction))
                for i, (label, prediction) in enumerate(
                    zip(grouping_data["label"], grouping_data["probs"])
                )
                if i in all_indices
            ]
            homogeneity = compute_homogeneity(
                target_distributions, all_distributions, target_prediction_result
            )
            completeness = compute_completeness(
                target_distributions, all_distributions, all_prediction_result
            )
            if (homogeneity + completeness) != 0:
                vscore = (2 * homogeneity * completeness) / (homogeneity + completeness)
            else:
                vscore = 0

            logger.log(
                "for feature {}, accuracy of error slice is {}, datapoints have V-score {}, homogeneity {}, completeness {}".format(
                    f, slice_accuracy, vscore, homogeneity, completeness
                )
            )
            logger.log("****")


def return_distribution(feature_data, indices, f):
    dist = []
    all_feature_result = feature_data[f]
    for i in indices:
        dist.append(all_feature_result[i])

    return dist


def compute_homogeneity(
    target_distributions, all_distributions, target_prediction_result
):
    all_mean = np.mean(all_distributions)
    all_std = np.std(all_distributions)

    correct = 0
    for i, t in enumerate(target_distributions):
        if (
            t > min(all_mean + all_std, max(target_distributions))
            and target_prediction_result[i] == 0
        ):
            correct += 1

    if len(target_distributions) != 0:
        homogeneity = correct / len(target_distributions)
    else:
        return 0

    return homogeneity


def compute_completeness(
    target_distributions, all_distributions, all_prediction_result
):
    all_mean = np.mean(all_distributions)
    all_std = np.std(all_distributions)

    found = 0
    all_correct = 0
    for i, t in enumerate(all_distributions):
        if (
            t > min(all_mean + all_std, max(target_distributions))
            and all_prediction_result[i] == 0
        ):
            if i < len(target_distributions):
                found += 1
            all_correct += 1

    if all_correct != 0:
        completeness = found / all_correct
    else:
        completeness = 1

    return completeness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="CoLA")
    parser.add_argument("--layer_n", type=int, default=12)
    parser.add_argument("--slice_n", type=int, default=128)
    parser.add_argument(
        "--error_threshold", type=float, default=0.5, help="accuracy of slice score"
    )
    parser.add_argument(
        "--significance_threshold",
        type=float,
        default=0.05,
        help="pvalue of significance test; if computed pvalue is smaller than the threshold, we reject the null hypothesis",
    )
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--feature_data_path", type=str)
    parser.add_argument("--grouping_data_path", type=str)
    parser.add_argument("--model_type", type=str, default="SDM")

    args = parser.parse_args()

    args.log_path = "significance_log/{}_{}.log".format(args.task, args.model_type)
    args.feature_data_path = "featured_data/to_dev_{}.json".format(args.task)
    args.grouping_data_path = "../../feature_result/to_dev_bert_{}".format(args.task)

    if args.model_type == "SDM":
        args.grouping_data_path += ".json"
    elif args.model_type == "domino":
        args.grouping_data_path = "../../baseline_result/to_dev_bert_{}.json".format(
            args.task
        )
    elif args.model_type == "embedding":
        args.grouping_data_path += "_embedding.json"
    elif args.model_type == "confidence":
        args.grouping_data_path += "_confidence.json"

    if args.task == "MNLI":
        args.error_threshold = 0.4
    if args.task == "SST-5":
        args.error_threshold = 0.3

    set_seed(args)
    logger = Logger(args.log_path)

    logger.log("logging data ...")
    with open(args.feature_data_path, "r") as f:
        feature_data = json.load(f)

    with open(args.grouping_data_path, "r") as f:
        grouping_data = json.load(f)

    logger.log("data loaded")

    significance_test(args, logger, feature_data, grouping_data)
