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


def grouping(n, data):
    gro = []
    for i in range(len(data["sentence"])):
        if data["predicted_group"][i] == n:
            gro.append(
                [
                    data["sentence"][i],
                    data["label"][i],
                    data["probs"][i],
                    data["domino_slices"][i][n],
                ]
            )
    return gro


def compute_acc(gro):
    acc = []
    for g in gro:
        acc.append(np.argmax(g[2]) == g[1])
    return np.mean(acc)


def expected_acc(slice_number, data):
    expected_accuracy = 0
    expected_count = 0
    for i in range(len(data)):
        slice_prob = data["domino_slices"][i][slice_number] / np.sum(
            data["domino_slices"][i]
        )
        expected_count += slice_prob
        if data["label"][i] == np.argmax(data["probs"][i]):
            expected_accuracy += slice_prob
    # print('for slice number {}, expected count is {}, acc sum is {}'.format(slice_number, expected_count, expected_acc\
    # uracy))
    return expected_accuracy / expected_count


def compute_error_slices(logger, args, data):
    error_slices = []
    error_slice_number = []
    for i in range(args.slice_n):
        gro = grouping(i, data)
        if args.method in ["likelihood", "flipping"]:
            if len(gro) > 0:
                acc = compute_acc(gro)
            else:
                acc = 100
        else:
            if len(gro) > 0:
                acc = expected_acc(i, data)
            else:
                acc = 100
        error_slices.append((i, acc, len(gro)))
    error_slices = sorted(error_slices, key=lambda x: x[1])
    '''
    for e in error_slices:
        logger.log(
            "slice {} with {} datapoints has accuracy {}".format(e[0], e[2], e[1])
        )
    '''
    error_slice_number = [error_slice[0] for i, error_slice in enumerate(error_slices)]

    return error_slices, error_slice_number


def filtered_test_data(slice_number, test, threshold):
    total = 0
    correct = 0
    for i in range(len(test["sentence"])):
        group_n = test["predicted_group"][i]
        if group_n in slice_number[:threshold]:
            continue
        else:
            total += 1
            if test["label"][i] == np.argmax(test["probs"][i]):
                correct += 1
    return total, correct / total


def filtered_test_data_expectation(error_slice, test):
    error_slice_number = [error_slice[i][0] for i in range(len(error_slice))]
    error_slice_accuracy = [error_slice[i][1] for i in range(len(error_slice))]
    test_data_error_expectation = []
    for i in range(len(test["sentence"])):
        domino_slice_probs = test["domino_slices"][i]
        domino_slice_probs = [
            domino_slice_probs[i] / sum(domino_slice_probs)
            for i in range(len(domino_slice_probs))
        ]
        expectation = np.sum(
            [
                domino_slice_probs[j] * error_slice_accuracy[pos]
                for pos, j in enumerate(error_slice_number)
            ]
        )
        test_data_error_expectation.append(
            [test["sentence"][i], test["label"][i], test["probs"][i], expectation]
        )
    test_data_error_expectation = sorted(
        test_data_error_expectation, key=lambda x: x[-1], reverse=False
    )
    recalls = []
    accs = []
    for i in range(len(test_data_error_expectation) - 20):
        left = test_data_error_expectation[i:]
        correct = 0
        for l in left:
            if np.argmax(l[2]) == l[1]:
                correct += 1
        acc = correct / len(left)
        recalls.append(len(left) / len(test_data_error_expectation))
        accs.append(acc)
    return recalls, accs


def filtered_test_data_likelihood(args, slice_number, test):
    error_threshold = 0
    for e in error_slices:
        if e[1] < args.error_threshold:
            error_threshold += 1
        else:
            break
    error_slice_number = [slice_number[i] for i in range(error_threshold)]
    test_data_error_likelihood = []
    for i in range(len(test["sentence"])):
        domino_slice_probs = test["domino_slices"][i]
        likelihood = np.sum([domino_slice_probs[j] for j in error_slice_number])
        test_data_error_likelihood.append(
            [test["sentence"][i], test["label"][i], test["probs"][i], likelihood]
        )
    test_data_error_likelihood = sorted(
        test_data_error_likelihood, key=lambda x: x[-1], reverse=True
    )
    recalls = []
    accs = []
    for i in range(len(test_data_error_likelihood) - 20):
        left = test_data_error_likelihood[i:]
        correct = 0
        for l in left:
            if np.argmax(l[2]) == l[1]:
                correct += 1
        acc = correct / len(left)
        recalls.append(len(left) / len(test_data_error_likelihood))
        accs.append(acc)
    return recalls, accs


def flipping_test_data_likelihood(args, slice_number, test):
    error_threshold = 0
    for e in error_slices:
        if e[1] < args.error_threshold:
            error_threshold += 1
        else:
            break
    error_slice_number = [slice_number[i] for i in range(error_threshold)]
    test_data_error_likelihood = []
    for i in range(len(test["sentence"])):
        domino_slice_probs = test["domino_slices"][i]
        likelihood = np.sum([domino_slice_probs[j] for j in error_slice_number])
        test_data_error_likelihood.append(
            [
                test["sentence"][i],
                test["label"][i],
                test["probs"][i],
                np.argmax(test["domino_slices"][i]),
                likelihood,
            ]
        )
    test_data_error_likelihood = sorted(
        test_data_error_likelihood, key=lambda x: x[-1], reverse=True
    )
    accs = []
    for i in range(len(test_data_error_likelihood) - 20):
        if test_data_error_likelihood[i][-2] in error_slice_number:
            if test_data_error_likelihood[i][-1] > args.likelihood_threshold:
                test_data_error_likelihood[i][2].reverse()
        correct = 0
        for l in test_data_error_likelihood:
            if np.argmax(l[2]) == l[1]:
                correct += 1
        acc = correct / len(test_data_error_likelihood)
        accs.append(acc)
    likelihood = [
        test_data_error_likelihood[i][-1]
        for i in range(len(test_data_error_likelihood))
    ]
    return likelihood, accs

def compute_peak(recalls, accs):
    recalls = [1-a for a in recalls]
    best_recall = 0
    for recall, acc in zip(recalls, accs):
        if acc > 0.98 and recall < 0.5:
            best_recall = recall
            best_acc = acc
            break
    if best_recall != 0:
        broken_acc = []
        for recall, acc in zip(recalls, accs):
            if recall > best_recall:
                if acc < 0.98:
                    broken_acc.append((recall, acc))
        print('domino functions. best recall is {} with acc {}, acc drops for {} points'.format(best_recall, best_acc, len(broken_acc)))
        return best_recall
    if best_recall == 0:
        highest_acc = np.max(accs)
        highest_recall = recalls[np.argmax(accs)]
        broken_acc = []
        for recall, acc in zip(recalls, accs):
            if recall > highest_recall:
                if acc < highest_acc:
                    broken_acc.append((recall, acc))
        print('domino does not function. best recall is {} with acc {}, acc drops for {} points'.format(highest_recall, highest_acc, len(broken_acc)))
        return highest_recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--task", type=str, default="SST-2")
    parser.add_argument("--domino_type", type=str, default="MML")
    parser.add_argument("--layer_n", type=int, default=0)
    parser.add_argument("--slice_n", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="../myglue/glue_data/")
    parser.add_argument("--prob_dir", type=str, default="../myglue/")
    parser.add_argument("--prediction_dir", type=str, default="../myglue/")
    parser.add_argument(
        "--error_threshold", type=float, default=0.5, help="accuracy of slice score"
    )
    parser.add_argument(
        "--likelihood_threshold",
        type=float,
        default=0.9,
        help="threhsold for flipping, likelihood under the threshold, no flipping",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="likelihood",
        help="expectation, likelihood, or flipping",
    )

    parser.add_argument("--pca_n_components", type=int, default=128)
    parser.add_argument("--non_embedding_lr", type=float, default=5e-3)
    parser.add_argument("--embedding_lr", type=float, default=10.0)
    parser.add_argument("--non_embedding_clip", type=float, default=1)
    parser.add_argument("--embedding_clip", type=float, default=0.1)
    parser.add_argument("--noise_channel", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_prop", type=float, default=0.2)
    
    parser.add_argument("--y_log_likelihood_weight", type=int, default=40)
    parser.add_argument("--y_hat_log_likelihood_weight", type=int, default=40)
    parser.add_argument('--mu_weight', type=float, default=0.0)
    parser.add_argument('--mu_kl_weight', type=float, default=0)
    parser.add_argument('--var_weight', type=float, default=0.1)

    args = parser.parse_args()

    set_seed(args)

    args.data_dir += args.task + "/"
    args.prob_dir += args.task + "_dev_probs.json"
    args.prediction_dir += args.task + "_dev_prediction.json"
    args.log_path = args.task + ".log"

    logger = Logger(args.log_path)
    logger.log(str(args))

    domino_validation_dir = "domino_result/domino_{}_dev_{}_{}_{}_{}_{}_{}_{}_{}.json".format(
            args.domino_type,
            args.y_log_likelihood_weight,
            args.y_hat_log_likelihood_weight,
            args.embedding_lr,
            args.non_embedding_lr,
            args.warmup_prop,
            args.noise_channel,
            args.mu_weight,
            args.var_weight,
        )

    assert os.path.isfile(domino_validation_dir)
    with open(domino_validation_dir, "r") as f:
        dev = json.load(f)
    error_slices, error_slice_number = compute_error_slices(logger, args, dev)

    domino_test_dir = "domino_result/domino_{}_test_{}_{}_{}_{}_{}_{}_{}_{}.json".format(
            args.domino_type,
            args.y_log_likelihood_weight,
            args.y_hat_log_likelihood_weight,
            args.embedding_lr,
            args.non_embedding_lr,
            args.warmup_prop,
            args.noise_channel,
            args.mu_weight,
            args.var_weight
        )
    assert os.path.isfile(domino_test_dir)
    with open(domino_test_dir, "r") as f:
        test = json.load(f)

    if args.method == "expectation":
        recalls, accs = filtered_test_data_expectation(error_slices, test)
        compute_peak(recalls, accs)
    elif args.method == "likelihood":
        recalls, accs = filtered_test_data_likelihood(args, error_slice_number, test)
        compute_peak(recalls, accs)
    elif args.method == "flipping":
        likelihood, accs = flipping_test_data_likelihood(args, error_slice_number, test)
        compute_peak(likelihood, accs)

    recall_acc_save_dir = "evaluation_result/domino_{}_test_{}_{}_{}_{}_{}_{}_{}_{}.json".format(
            args.domino_type,
            args.y_log_likelihood_weight,
            args.y_hat_log_likelihood_weight,
            args.embedding_lr,
            args.non_embedding_lr,
            args.warmup_prop,
            args.noise_channel,
            args.mu_weight,
            args.var_weight
        )

    if args.method in ["expectation", "likelihood"]:
        with open(recall_acc_save_dir, "w") as f:
            json.dump([recalls, accs], f)
    else:
        with open(recall_acc_save_dir, "w") as f:
            json.dump([likelihood, accs], f)
