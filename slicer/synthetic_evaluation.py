import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from data_process import OneSentInputData, TwoSentInputData
from domino import DominoSlicer
from nltk.parse.corenlp import CoreNLPParser
from scipy.stats import ttest_ind
# from torch.utils.data.dataloader import DataLoader
from transformers import AutoModel, BertTokenizer

from domino_slicer import collect_tsv_data
from explanation.parser_method import (compute_aspect, compute_comparison,
                                       compute_echo_question,
                                       compute_foreign_word,
                                       compute_multiple_modal,
                                       compute_multiple_prep, compute_NP_sub,
                                       compute_quantifier, compute_tense,
                                       compute_to_after_modal,
                                       compute_tree_depth, compute_voice)
from explanation.surface_method import (compute_frequency, compute_length,
                                        compute_negation, compute_reflexive)
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


def select_sythetic_datapoints(args, data):
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
        synthetic_dataset = random.sample(random_dataset, k=int(len(random_dataset)/2)) + target_dataset
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
            if np.argmax(d[3]) == d[1]:
                target_dist, _ = compute_negation([d], [])
                if target_dist[0] < avg:
                    random_dataset.append(i)
        synthetic_dataset = random.sample(random_dataset, k=int(len(random_dataset)/2)) + target_dataset
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
            if np.argmax(d[3]) == d[1]:
                target_dist, _ = compute_reflexive([d], [])
                if target_dist[0] < avg:
                    random_dataset.append(i)
        synthetic_dataset = random.sample(random_dataset, k=int(len(random_dataset)/2)) + target_dataset
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
            if np.argmax(d[3]) == d[1]:
                target_dist, _ = compute_frequency([d], data, True)
                if target_dist[0] < avg:
                    random_dataset.append(i)
        synthetic_dataset = random.sample(random_dataset, k=int(len(random_dataset)/2)) + target_dataset
        random.shuffle(synthetic_dataset)
    else:
        if args.feature == "multi_preposition":
            parser = CoreNLPParser()
            _, others_dist = compute_multiple_prep(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_multiple_prep(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "NP_sub":
            parser = CoreNLPParser()
            _, others_dist = compute_NP_sub(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_NP_sub(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "echo_question":
            parser = CoreNLPParser()
            _, others_dist = compute_echo_question(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_echo_question(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "tree_depth":
            parser = CoreNLPParser()
            _, others_dist = compute_tree_depth(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_tree_depth(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "extra_to":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_to_after_modal(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_to_after_modal(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "multi_modal":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_multiple_modal(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_multiple_modal(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(data, k=len(target_dataset))
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "quantifier":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_quantifier(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_quantifier(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "voice":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_voice(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_voice(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "tense":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_tense(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_tense(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "aspect":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_aspect(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_aspect(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "comparison":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_comparison(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_comparison(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)
        elif args.feature == "foreign_word":
            parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
            _, others_dist = compute_foreign_word(parser, [], data)
            all_dist = others_dist
            avg = np.mean(all_dist)
            std = np.std(all_dist)
            target_dataset = []
            for i, d in enumerate(data):
                if np.argmax(d[2]) != d[1]:
                    target_dist, _ = compute_foreign_word(parser, [d], [])
                    if target_dist[0] > avg + std:
                        target_dataset.append(i)
            random_dataset = random.sample(
                list(range(len(data))), k=len(target_dataset)
            )
            synthetic_dataset = random_dataset + target_dataset
            random.shuffle(synthetic_dataset)

    return target_dataset, synthetic_dataset


def verify_slice(args, logger, synthetic):
    def g(n, data):
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

    target_slice = []
    others = []
    for i in range(args.slice_n):
        gro = g(i, synthetic)
        feature = slice_topic(gro, synthetic)
        logger.log('topic word of slice {} is {} based on TF-IDF'.format(feature))
        acc = compute_acc(gro)
        if acc < 0.5:
            target_slice += gro
        else:
            others += gro
        logger.log("number of datapoints of slice {} is {}, accuracy is {}".format(i, len(gro), acc))

    target_slice_acc = compute_acc(target_slice)
    others_acc = compute_acc(others)
    logger.log('after mergning, error slice has {} datapoints, accuracy is {}; default slice has {} datapoints, accuracy {}'.format(len(target_slice), target_slice_acc, len(others), others_acc))

    precision, recall, f1 = compute_recovery_rate(error_slice, target_dataset)
    logger.log('target datapoints recovery: precision {}, recall {}, f1 {}'.format(precision, recall, f1))

    if args.feature == "length":
        target_dist, others_dist = compute_length(target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log('pvalue for slice {} is {}'.format(i,pvalue))
    if args.feature == "reflexive":
        target_dist, others_dist = compute_reflexive(target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log('pvalue for slice {} is {}'.format(i,pvalue))
    if args.feature == 'negation':
        target_dist, others_dist = compute_negation(target_slice, others)
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        logger.log('pvalue for slice {} is {}'.format(i,pvalue))

    return target_slice, others

def slice_topic(target_slice, all_slices):
    target_stc = ' '.join(target_slice['sentence'])
    all_stc = ' '.join(all_slices['sentence'])
    corpus = [target_stc, all_stc]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    features = vectorizer.get_feature_names_out()
    feature = features[np.argmax(X[0,:])]

    return feature

def compute_recovery_rate(error_slice, target_dataset):
    correct = 0
    for i in range(len(error_slice)):
        if error_slice[i][0] in [d[0] for d in target_dataset]:
            correct += 1
    incorrect = len(error_slice) - correct
    precision = correct/len(error_slice)
    recall = correct/len(target_dataset)
    F1 = (2*precision*recall)/(precision + recall)
    return precision, recall, F1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--task", type=str, default="CoLA")
    parser.add_argument("--feature", type=str, default="length")
    parser.add_argument("--layer_n", type=int, default=12)
    parser.add_argument("--slice_n", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="../myglue/glue_data/")
    parser.add_argument("--prob_dir", type=str, default="../myglue/")
    parser.add_argument("--prediction_dir", type=str, default="../myglue/")
    parser.add_argument("--toy", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--gpu", type=str, default="4")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--gamma_y", type=float, default=1)
    parser.add_argument("--gamma_y_hat", type=float, default=1)

    args = parser.parse_args()

    set_seed(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.save_position = "../../../../../harddisk/user/wenyuehua/"
    args.data_dir += args.task + "/"
    args.prob_dir += args.task + "_test_probs.json"
    args.prediction_dir += args.task + "_test_prediction.json"
    args.log_path = args.task + "_" + args.feature + ".log"
    args.model_dir = "../myglue/" + args.task + "_meanpool.pt"
    logger = Logger(args.log_path)
    logger.log(str(args))

    #################### use test to do the synthetic experiment ####################
    with open(args.task + "_dev" + "_" + str(args.layer_n) + "_50.json", "r") as f:
        test = json.load(f)

    test = [
        [test["sentence"][i], test["label"][i], test["embeddings"][i], test["probs"][i]]
        for i in range(len(test["sentence"]))
    ]
    # return indices
    target_datapoints, synthetic_test = select_sythetic_datapoints(args, test)

    #################### build dev datapanel ####################
    synthetic = {}
    synthetic["sentence"] = [test[i][0] for i in synthetic_test]
    synthetic["label"] = [test[i][1] for i in synthetic_test]
    synthetic["embeddings"] = [test[i][2] for i in synthetic_test]
    synthetic["probs"] = [test[i][3] for i in synthetic_test]

    logger.log("there are {} synthetic datapoints".format(len(synthetic["sentence"])))
    logger.log("fit domino on validation dataset")
    domino = DominoSlicer(
        y_log_likelihood_weight=args.gamma_y,
        y_hat_log_likelihood_weight=args.gamma_y_hat,
        n_mixture_components=128,
        n_slices=args.slice_n,
        test=False,
    )

    domino.fit(
        data=synthetic, embeddings="embeddings", targets="label", pred_probs="probs"
    )

    synthetic["domino_slices"] = domino.predict_proba(
        data=synthetic, embeddings="embeddings", targets="label", pred_probs="probs"
    )

    # numpy array shape (x, 5)
    slice_prediction = synthetic["domino_slices"][:]
    slice_group = np.argmax(slice_prediction, axis=1).tolist()
    synthetic["predicted_group"] = slice_group
    synthetic["domino_slices"] = synthetic["domino_slices"].tolist()

    #################### save synthetic result  ####################
    save_synthetic_dir = (
        args.task + "_" + str(args.layer_n) + "_" + args.feature + ".json"
    )

    target_slice, others = verify_slice(args, logger, synthetic)
    
    compute_recovery_rate(target_slice, target_datapoints)

    with open(save_synthetic_dir, "w",) as f:
        json.dump([target_slice, others], f)
