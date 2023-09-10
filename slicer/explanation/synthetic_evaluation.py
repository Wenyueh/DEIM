import argparse
import json
import os
import random
import sys

# setting path
sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn
from domino import DominoSlicer, DistanceDominoSlicer
from nltk.parse.corenlp import CoreNLPParser
from scipy.stats import ttest_ind
import stanza

# from torch.utils.data.dataloader import DataLoader
from transformers import AutoModel, BertTokenizer

from explanation.parser_method import (
    compute_aspect,
    compute_comparison,
    compute_echo_question,
    compute_foreign_word,
    compute_multiple_modal,
    compute_multiple_prep,
    compute_NP_sub,
    compute_quantifier,
    compute_tense,
    compute_to_after_modal,
    compute_tree_depth,
    compute_voice,
    compute_long_distance_dependency,
)
from explanation.surface_method import (
    compute_frequency,
    compute_length,
    compute_negation,
    compute_reflexive,
)
from explanation.pragmatic_method import (
    # classifier_identification,
    gender_identification,
    compute_female_dist,
    compute_male_dist,
    compute_atheist_dist,
    compute_buddhist_dist,
    compute_christian_dist,
    compute_hindu_dist,
    compute_jewish_dist,
    compute_muslim_dist,
    compute_asian_dist,
    compute_white_dist,
    compute_black_dist,
    compute_latino_dist,
    # ner_number_identification,
)
from sklearn.feature_extraction.text import TfidfVectorizer


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
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


def select_synthetic_datapoints_jigsaw(args, slice_data, orig_data):
    target_dataset = []
    random_dataset = []
    all_dist = [float(a) if a != "" else 0 for a in orig_data[args.feature][:10000]]
    avg = np.mean(all_dist)
    std = np.std(all_dist)
    for i, d in enumerate(slice_data[:10000]):
        if np.argmax(d[3]) != d[1]:
            value = (
                float(orig_data[args.feature][i]) if orig_data[args.feature][i] else 0
            )
            if value > avg + std:
                target_dataset.append(i)
        if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
            value = (
                float(orig_data[args.feature][i]) if orig_data[args.feature][i] else 0
            )
            if value < avg:
                random_dataset.append(i)
    synthetic_dataset = (
        random.sample(random_dataset, k=int(len(random_dataset) / 2)) + target_dataset
    )
    assert len(synthetic_dataset) > 128
    random.shuffle(synthetic_dataset)

    return target_dataset, synthetic_dataset


def select_synthetic_datapoints(args, data):
    data = data[:10000]
    if args.feature == "length":
        _, others_dist = compute_length([], data)
        all_dist = others_dist
        avg = np.mean(all_dist)
        std = np.std(all_dist)
        target_dataset = []
        random_dataset = []
        for i, d in enumerate(data):
            if np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_length([d], [])
                if target_dist[0] > avg + std:
                    target_dataset.append(i)
            if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_length([d], [])
                if target_dist[0] < avg:
                    random_dataset.append(i)
        synthetic_dataset = (
            random.sample(random_dataset, k=int(len(random_dataset) / 2))
            + target_dataset
        )
        assert len(synthetic_dataset) > 128
        random.shuffle(synthetic_dataset)
    elif args.feature == "short_length":
        _, others_dist = compute_length([], data)
        all_dist = others_dist
        avg = np.mean(all_dist)
        std = np.std(all_dist)
        target_dataset = []
        random_dataset = []
        for i, d in enumerate(data):
            if np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_length([d], [])
                if target_dist[0] < avg - std:
                    target_dataset.append(i)
            if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_length([d], [])
                if target_dist[0] > avg:
                    random_dataset.append(i)
        synthetic_dataset = (
            random.sample(random_dataset, k=int(len(random_dataset) / 2))
            + target_dataset
        )
        assert len(synthetic_dataset) > 128
        random.shuffle(synthetic_dataset)
    elif args.feature == "negation":
        _, others_dist = compute_negation([], data)
        all_dist = others_dist
        avg = np.mean(all_dist)
        std = np.std(all_dist)
        target_dataset = []
        random_dataset = []
        for i, d in enumerate(data):
            if np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_negation([d], [])
                if target_dist[0] > avg + std:
                    target_dataset.append(i)
            if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_negation([d], [])
                if target_dist[0] < avg:
                    random_dataset.append(i)
        synthetic_dataset = (
            random.sample(random_dataset, k=int(len(random_dataset) / 2))
            + target_dataset
        )
        random.shuffle(synthetic_dataset)
    elif args.feature == "reflexive":
        _, others_dist = compute_reflexive([], data)
        all_dist = others_dist
        avg = np.mean(all_dist)
        std = np.std(all_dist)
        target_dataset = []
        random_dataset = []
        for i, d in enumerate(data):
            if np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_reflexive([d], [])
                if target_dist[0] > avg + std:
                    target_dataset.append(i)
            if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_reflexive([d], [])
                if target_dist[0] < avg:
                    random_dataset.append(i)
        synthetic_dataset = (
            random.sample(random_dataset, k=int(len(random_dataset) / 2))
            + target_dataset
        )
        random.shuffle(synthetic_dataset)
    elif args.feature == "frequency":
        _, others_dist = compute_frequency([], data, False)
        all_dist = others_dist
        avg = np.mean(all_dist)
        std = np.std(all_dist)
        target_dataset = []
        random_dataset = []
        for i, d in enumerate(data):
            if np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_frequency([d], data, True)
                if target_dist[0] > avg + std:
                    target_dataset.append(i)
            if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                target_dist, _ = compute_frequency([d], data, True)
                if target_dist[0] < avg:
                    random_dataset.append(i)
        synthetic_dataset = (
            random.sample(random_dataset, k=int(len(random_dataset) / 2))
            + target_dataset
        )
        random.shuffle(synthetic_dataset)
    else:
        if args.feature == "multi_preposition":
            parser = CoreNLPParser()
            _, others_dist = compute_multiple_prep(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_multiple_prep(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_multiple_prep(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "NP_sub":
            parser = CoreNLPParser()
            _, others_dist = compute_NP_sub(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_NP_sub(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_NP_sub(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "echo_question":
            parser = CoreNLPParser()
            _, others_dist = compute_echo_question(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_echo_question(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_echo_question(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "tree_depth":
            parser = CoreNLPParser()
            _, others_dist = compute_tree_depth(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_tree_depth(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_tree_depth(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "extra_to":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_to_after_modal(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_to_after_modal(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_to_after_modal(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "multi_modal":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_multiple_modal(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_multiple_modal(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_multiple_modal(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "quantifier":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_quantifier(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_quantifier(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_quantifier(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "voice":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_voice(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_voice(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_voice(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "tense":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_tense(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_tense(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_tense(parser, [d], [])
                    if target_dist[0] < avg or np.argmax(d[3]) != d[1]:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "aspect":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_aspect(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_aspect(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_aspect(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "comparison":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_comparison(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_comparison(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_comparison(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "foreign_word":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_foreign_word(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_foreign_word(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_foreign_word(parser, [d], [])
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)
        elif args.feature == "long_distance":
            stanza_parser = stanza.Pipeline(
                lang="en", processors="tokenize,mwt,pos,lemma,depparse"
            )
            _, others_dist = compute_long_distance_dependency(stanza_parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            random_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_long_distance_dependency(
                        stanza_parser, [d], []
                    )
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
                if np.argmax(d[3]) == d[1] or np.argmax(d[3]) != d[1]:
                    target_dist, _ = compute_long_distance_dependency(
                        stanza_parser, [d], []
                    )
                    if target_dist[0] < avg:
                        random_dataset.append(i)
            synthetic_dataset = (
                random.sample(random_dataset, k=int(len(random_dataset) / 2))
                + target_dataset
            )
            random.shuffle(synthetic_dataset)

    return target_dataset, synthetic_dataset


def verify_slice(args, logger, synthetic, target_dataset):
    def g(n, data):
        gro = []
        for i in range(len(data["sentence"])):
            if data["predicted_group"][i] == n:
                gro.append(
                    [data["sentence"][i], data["label"][i], data["probs"][i],]
                )
        return gro

    def compute_acc(gro):
        acc = []
        for g in gro:
            acc.append(np.argmax(g[2]) == g[1])
        return np.mean(acc)

    target_slice = []
    others = []
    for i in range(args.slice_n):
        gro = g(i, synthetic)
        # feature = slice_topic(gro, synthetic)
        # logger.log("topic word of slice {} is {} based on TF-IDF".format(feature))
        acc = compute_acc(gro)
        if acc < 0.5:
            target_slice += gro
        else:
            others += gro

    target_slice_acc = compute_acc(target_slice)
    others_acc = compute_acc(others)
    logger.log(
        "after mergning, error slice has {} datapoints, accuracy is {}; default slice has {} datapoints, accuracy {}".format(
            len(target_slice), target_slice_acc, len(others), others_acc
        )
    )

    precision, recall, f1 = compute_recovery_rate(target_slice, target_dataset)
    logger.log(
        "target datapoints recovery: precision {}, recall {}, f1 {}".format(
            precision, recall, f1
        )
    )

    parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
    if args.feature == "length":
        target_dist, others_dist = compute_length(target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log("pvalue for error slice is {}".format(pvalue))
    if args.feature == "reflexive":
        target_dist, others_dist = compute_reflexive(target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log("pvalue for error slice is {}".format(pvalue))
    if args.feature == "negation":
        target_dist, others_dist = compute_negation(target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log("pvalue for error slice is {}".format(pvalue))
    if args.feature == "comparison":
        target_dist, others_dist = compute_comparison(parser, target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log("pvalue for error slice is {}".format(pvalue))
    if args.feature == "NP_sub":
        target_dist, others_dist = compute_NP_sub(parser, target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log("pvalue for error slice is {}".format(pvalue))
    if args.feature == "multi_preposition":
        target_dist, others_dist = compute_multiple_prep(parser, target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log("pvalue for error slice is {}".format(pvalue))
    if args.feature == "tree_depth":
        target_dist, others_dist = compute_tree_depth(parser, target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log("pvalue for error slice is {}".format(pvalue))
    if args.feature == "quantifier":
        target_dist, others_dist = compute_quantifier(parser, target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log("pvalue for error slice is {}".format(pvalue))
    if args.feature == "long_distance":
        stanza_parser = stanza.Pipeline(
            lang="en", processors="tokenize,mwt,pos,lemma,depparse"
        )
        target_dist, others_dist = compute_long_distance_dependency(
            stanza_parser, target_slice, others
        )
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log("pvalue for error slice is {}".format(pvalue))

    return target_slice, others


def slice_topic(target_slice, all_slices):
    target_stc = " ".join(target_slice["sentence"])
    all_stc = " ".join(all_slices["sentence"])
    corpus = [target_stc, all_stc]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    features = vectorizer.get_feature_names_out()
    feature = features[np.argmax(X[0, :])]

    return feature


def compute_recovery_rate(error_slice, target_dataset):
    correct = 0
    for i in range(len(error_slice)):
        if error_slice[i][0] in [d[0] for d in target_dataset]:
            correct += 1
    incorrect = len(error_slice) - correct
    precision = correct / len(error_slice)
    recall = correct / len(target_dataset)
    F1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, F1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--task", type=str, default="CoLA")
    parser.add_argument("--feature", type=str, default="length")
    parser.add_argument("--layer_n", type=int, default=12)
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--pca_n_components", type=int, default=128)
    parser.add_argument("--slice_n", type=int, default=10)
    parser.add_argument("--prob_dir", type=str, default="../myglue/")
    parser.add_argument("--toy", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--gpu", type=str, default="4")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--emb_log_likelihood_weight", type=float, default=0.15)
    parser.add_argument("--distance_log_likelihood_weight", type=float, default=1)
    parser.add_argument("--y_log_likelihood_weight", type=float, default=1)
    parser.add_argument("--y_hat_log_likelihood_weight", type=float, default=1)

    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--model_type", type=str, default="bert")

    args = parser.parse_args()

    set_seed(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.log_path = "synthetic_datasets/SST-5/" + args.task + ".log"
    logger = Logger(args.log_path)
    logger.log(str(args))

    #################### use dev to do the synthetic experiment ####################
    with open(
        "../" + args.task + "_dev" + "_" + str(args.layer_n) + "_128.json", "r"
    ) as f:
        dev = json.load(f)

    if args.task in ["QNLI", "QQP", "MNLI"]:
        dev = [
            [
                dev["sentence"][i][1],
                dev["label"][i],
                dev["embeddings"][i],
                dev["probs"][i],
            ]
            for i in range(len(dev["sentence"]))
        ]
    else:
        dev = [
            [dev["sentence"][i], dev["label"][i], dev["embeddings"][i], dev["probs"][i]]
            for i in range(len(dev["sentence"]))
        ]
    synthetic_data_dir = "synthetic_datasets/{}_target_dataset.json".format(
        args.feature
    )
    target_data_dir = "synthetic_datasets/{}_synthetic_dataset.json".format(
        args.feature
    )
    if os.path.isfile(synthetic_data_dir) and os.path.isfile(target_data_dir):
        with open(
            "synthetic_datasets/{}_target_dataset.json".format(args.feature), "r",
        ) as f:
            target_indices = json.load(f)
        with open(
            "synthetic_datasets/{}_synthetic_dataset.json".format(args.feature), "r",
        ) as f:
            synthetic_indices = json.load(f)
    else:
        if "jigsaw" not in args.task:
            # return indices
            target_indices, synthetic_indices = select_synthetic_datapoints(args, dev)
            with open(
                "synthetic_datasets/{}_target_dataset.json".format(args.feature), "w",
            ) as f:
                json.dump(target_indices, f)
            with open(
                "synthetic_datasets/{}_synthetic_dataset.json".format(args.feature),
                "w",
            ) as f:
                json.dump(synthetic_indices, f)
        else:
            jigsaw_type = args.task.replace("jigsaw_", "")
            with open(
                "../../myglue/glue_data/toxicity/{}_dev.json".format(jigsaw_type)
            ) as f:
                orig_data = json.load(f)
            target_indices, synthetic_indices = select_synthetic_datapoints_jigsaw(
                args, dev, orig_data
            )
            with open(
                "synthetic_datasets/{}_target_dataset.json".format(args.feature), "w",
            ) as f:
                json.dump(target_indices, f)
            with open(
                "synthetic_datasets/{}_synthetic_dataset.json".format(args.feature),
                "w",
            ) as f:
                json.dump(synthetic_indices, f)
    logger.log("created synthetic datasets saved")

    #################### build dev datapanel ####################
    synthetic = {}
    synthetic["sentence"] = [dev[i][0] for i in synthetic_indices]
    synthetic["label"] = [dev[i][1] for i in synthetic_indices]
    synthetic["embeddings"] = [dev[i][2] for i in synthetic_indices]
    synthetic["probs"] = [dev[i][3] for i in synthetic_indices]

    logger.log("there are {} synthetic datapoints".format(len(synthetic["sentence"])))
    logger.log("fit domino on validation dataset")

    ### EM labels
    if args.baseline:
        domino = DominoSlicer(
            emb_log_likelihood_weight=args.emb_log_likelihood_weight,
            y_log_likelihood_weight=args.y_log_likelihood_weight,
            y_hat_log_likelihood_weight=args.y_hat_log_likelihood_weight,
            n_mixture_components=min(args.n_components, len(synthetic["sentence"])),
            n_slices=args.slice_n,
            n_pca_components=args.pca_n_components,
            test=False,
            init_params="confusion",
        )
    else:
        domino = DistanceDominoSlicer(
            emb_log_likelihood_weight=args.emb_log_likelihood_weight,
            distance_log_likelihood_weight=args.distance_log_likelihood_weight,
            y_hat_log_likelihood_weight=args.y_hat_log_likelihood_weight,
            n_mixture_components=min(args.n_components, len(synthetic["sentence"])),
            n_slices=args.slice_n,
            n_pca_components=args.pca_n_components,
            test=False,
            init_params="confusion",
        )

    domino.fit(
        data=synthetic, embeddings="embeddings", targets="label", pred_probs="probs"
    )

    synthetic["log_domino_slices"], synthetic["resp"] = domino.predict_proba(
        data=synthetic, embeddings="embeddings", targets="label", pred_probs="probs"
    )

    slice_prediction = synthetic["resp"][:]
    slice_group = np.argmax(slice_prediction, axis=1).tolist()
    synthetic["predicted_group"] = slice_group
    synthetic["resp"] = synthetic["resp"].tolist()
    synthetic["log_domino_slices"] = synthetic["log_domino_slices"].tolist()

    #################### save synthetic result  ####################
    save_synthetic_dir = (
        "synthetic_datasets/"
        + args.task
        + "_"
        + args.model_type
        + "_"
        + args.feature
        + "_"
        + str(args.baseline)
        + ".json"
    )

    target_dataset = []
    for i in target_indices:
        # sentence, label, probs
        target_dataset.append([dev[i][0], dev[i][1], dev[i][3]])

    target_slice, others = verify_slice(args, logger, synthetic, target_dataset)

    # compute_recovery_rate(target_slice, target_dataset)

    with open(save_synthetic_dir, "w",) as f:
        json.dump([target_slice, others], f)
