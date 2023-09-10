import argparse
import json
import os
import random
import sys
import torch

import numpy as np
import stanza
from nltk.parse.corenlp import CoreNLPParser
from scipy.stats import ttest_ind

from parser_method import (
    compute_aspect,
    compute_comparison,
    compute_echo_question,
    compute_foreign_word,
    compute_long_distance_dependency,
    compute_multiple_modal,
    compute_multiple_prep,
    compute_NP_sub,
    compute_quantifier,
    compute_tense,
    compute_to_after_modal,
    compute_tree_depth,
    compute_voice,
)
from pragmatic_method import (
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
from surface_method import (
    compute_frequency,
    compute_length,
    compute_negation,
    compute_reflexive,
    compute_how_question,
    compute_why_question,
)

import warnings

warnings.filterwarnings("ignore")


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
            if len(data["sentence"][i]) == 2:
                gro.append(
                    [
                        data["sentence"][i][1],
                        data["label"][i],
                        data["probs"][i],
                        data["resp"][i][n],
                    ]
                )
            else:
                gro.append(
                    [
                        data["sentence"][i],
                        data["label"][i],
                        data["probs"][i],
                        data["resp"][i][n],
                    ]
                )
    return gro


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


def alldata(n, data):
    others = []
    for i in range(len(data["sentence"])):
        if len(data["sentence"][i]) == 2:
            others.append(
                [
                    data["sentence"][i][1],
                    data["label"][i],
                    data["probs"][i],
                    data["resp"][i][n],
                ]
            )
        else:
            others.append(
                [
                    data["sentence"][i],
                    data["label"][i],
                    data["probs"][i],
                    data["resp"][i][n],
                ]
            )
    return others


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
        acc.append(np.argmax(g[2]) == g[1])
    return np.mean(acc)


def one_slice_explanation(args, target_slice, others, stanza_parser):
    def significant_feature(args, target_dist, others_dist):
        pvalue = ttest_ind(target_dist, others_dist).pvalue
        target_mean = np.mean(target_dist)
        all_mean = np.mean(others_dist)
        if pvalue < args.significance_threshold and target_mean > all_mean:
            return True

    parser = CoreNLPParser()
    corenlpparser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
    all_features = []
    target_dist, others_dist = compute_length(target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("length")
    target_dist, others_dist = compute_negation(target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("negation")
    target_dist, others_dist = compute_reflexive(target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("reflexive")
    target_dist, others_dist = compute_frequency(target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("frequency")
    target_dist, others_dist = compute_how_question(target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("how question")
    target_dist, others_dist = compute_why_question(target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("why question")
    target_dist, others_dist = compute_multiple_prep(parser, target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("MP")
    target_dist, others_dist = compute_NP_sub(target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("NS")
    target_dist, others_dist = compute_echo_question(target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("EQ")
    target_dist, others_dist = compute_tree_depth(target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("TD")
    target_dist, others_dist = compute_to_after_modal(
        corenlpparser, target_slice, others
    )
    if significant_feature(args, target_dist, others_dist):
        all_features.append("to after modal")
    target_dist, others_dist = compute_multiple_modal(
        corenlpparser, target_slice, others
    )
    if significant_feature(args, target_dist, others_dist):
        all_features.append("multiple modal")
    target_dist, others_dist = compute_quantifier(corenlpparser, target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("quantifier")
    target_dist, others_dist = compute_tense(corenlpparser, target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("tense")
    target_dist, others_dist = compute_aspect(corenlpparser, target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("aspect")
    target_dist, others_dist = compute_comparison(corenlpparser, target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("comparison")
    target_dist, others_dist = compute_foreign_word(corenlpparser, target_slice, others)
    if significant_feature(args, target_dist, others_dist):
        all_features.append("foreign word")
    target_dist, others_dist = compute_long_distance_dependency(
        stanza_parser, target_slice, others
    )
    if significant_feature(args, target_dist, others_dist):
        all_features.append("long distance")

    return all_features


def significance_test_jigsaw(args, logger, slice_data, orig_data):
    target_slices = []
    all_datapoints = []
    target_dists = []
    all_dists = []
    for n in range(args.slice_n):
        target_slice = grouping_idx(n, slice_data)
        target_accuracy = compute_acc(target_slice)
        if target_accuracy < args.error_threshold:
            if len(target_slice) > 500:
                target_slice = random.sample(target_slice, k=500)
            all_others = alldata_idx(n, slice_data)
            all_others = target_slice + random.sample(
                [d for d in all_others if d not in target_slice],
                k=max(len(target_slice), 50),
            )
            if args.task.replace("jigsaw_", "") == "gender":
                if args.feature == "female":
                    target_dist = compute_female_dist(target_slice, orig_data)
                    others_dist = compute_female_dist(all_others, orig_data)
                elif args.feature == "male":
                    target_dist = compute_male_dist(target_slice, orig_data)
                    others_dist = compute_male_dist(all_others, orig_data)
            elif args.task.replace("jigsaw_", "") == "religion":
                if args.feature == "atheist":
                    target_dist = compute_atheist_dist(target_slice, orig_data)
                    others_dist = compute_atheist_dist(all_others, orig_data)
                elif args.feature == "buddhist":
                    target_dist = compute_buddhist_dist(target_slice, orig_data)
                    others_dist = compute_buddhist_dist(all_others, orig_data)
                elif args.feature == "christian":
                    target_dist = compute_christian_dist(target_slice, orig_data)
                    others_dist = compute_christian_dist(all_others, orig_data)
                elif args.feature == "hindu":
                    target_dist = compute_hindu_dist(target_slice, orig_data)
                    others_dist = compute_hindu_dist(all_others, orig_data)
                elif args.feature == "jewish":
                    target_dist = compute_jewish_dist(target_slice, orig_data)
                    others_dist = compute_jewish_dist(all_others, orig_data)
                elif args.feature == "muslim":
                    target_dist = compute_muslim_dist(target_slice, orig_data)
                    others_dist = compute_muslim_dist(all_others, orig_data)
            elif args.task.replace("jigsaw_", "") == "racial":
                if args.feature == "asian":
                    target_dist = compute_asian_dist(target_slice, orig_data)
                    others_dist = compute_asian_dist(all_others, orig_data)
                elif args.feature == "white":
                    target_dist = compute_white_dist(target_slice, orig_data)
                    others_dist = compute_white_dist(all_others, orig_data)
                elif args.feature == "black":
                    target_dist = compute_black_dist(target_slice, orig_data)
                    others_dist = compute_black_dist(all_others, orig_data)
                elif args.feature == "latino":
                    target_dist = compute_latino_dist(target_slice, orig_data)
                    others_dist = compute_latino_dist(all_others, orig_data)

            pvalue = ttest_ind(target_dist, others_dist).pvalue
            if pvalue < 0.05:
                target_slices += target_slice
                all_datapoints += all_others
                target_dists += target_dist
                all_dists += others_dist

    homogeneity = compute_homogeneity(target_dists, all_dists, target_slices)
    completeness = compute_completeness(target_dists, all_dists, all_datapoints)
    vscore = (2 * homogeneity * completeness) / (homogeneity + completeness)
    logger.log(
        "Vscore = {}, homogeneity = {}, completeness = {} on feature {}".format(
            vscore, homogeneity, completeness, args.feature
        )
    )


def significance_test(args, logger, data, stanza_parser):
    feature_group = []
    homogeneities = []
    completenesses = []

    s_feature_group = []
    s_homogeneities = []
    s_completenesses = []
    for n in range(args.slice_n):
        target_slice = grouping(n, data)
        if len(target_slice) > 500:
            target_slice = random.sample(target_slice, k=500)
        target_accuracy = compute_acc(target_slice)
        if target_accuracy < args.error_threshold:
            all_others = alldata(n, data)
            others = random.sample(all_others, k=max(len(target_slice), 100))
            all_others = random.sample(all_others, k=500) + target_slice
            if args.feature in [
                "length",
                "negation",
                "reflexive",
                "frequency",
                "how",
                "why",
            ]:
                if args.feature == "length":
                    target_dist, others_dist = compute_length(target_slice, others)
                    _, all_others_dist = compute_length([], all_others)
                elif args.feature == "negation":
                    target_dist, others_dist = compute_negation(target_slice, others)
                    _, all_others_dist = compute_negation([], all_others)
                elif args.feature == "reflexive":
                    target_dist, others_dist = compute_reflexive(target_slice, others)
                    _, all_others_dist = compute_reflexive(target_slice, all_others)
                elif args.feature == "frequency":
                    target_dist, others_dist = compute_frequency(target_slice, others)
                    _, all_others_dist = compute_frequency(target_slice, all_others)
                elif args.feature == "how":
                    target_dist, others_dist = compute_how_question(
                        target_slice, others
                    )
                    _, all_others_dist = compute_how_question(target_slice, all_others)
                elif args.feature == "why":
                    target_dist, others_dist = compute_why_question(
                        target_slice, others
                    )
                    _, all_others_dist = compute_why_question(target_slice, all_others)
            elif args.feature in [
                "multi_preposition",
                "NP_sub",
                "echo_question",
                "tree_depth",
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
                if args.feature == "multi_preposition":
                    parser = CoreNLPParser()
                    target_dist, others_dist = compute_multiple_prep(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_multiple_prep(
                        parser, target_slice, all_others
                    )
                elif args.feature == "NP_sub":
                    parser = CoreNLPParser()
                    target_dist, others_dist = compute_NP_sub(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_NP_sub(
                        parser, target_slice, all_others
                    )
                elif args.feature == "echo_question":
                    parser = CoreNLPParser()
                    target_dist, others_dist = compute_echo_question(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_echo_question(
                        parser, target_slice, all_others
                    )
                elif args.feature == "tree_depth":
                    parser = CoreNLPParser()
                    target_dist, others_dist = compute_tree_depth(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_tree_depth(
                        parser, target_slice, all_others
                    )
                elif args.feature == "extra_to":
                    parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
                    target_dist, others_dist = compute_to_after_modal(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_to_after_modal(
                        parser, target_slice, all_others
                    )
                elif args.feature == "multi_modal":
                    parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
                    target_dist, others_dist = compute_multiple_modal(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_multiple_modal(
                        parser, target_slice, all_others
                    )
                elif args.feature == "quantifier":
                    parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
                    target_dist, others_dist = compute_quantifier(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_quantifier(
                        parser, target_slice, all_others
                    )
                elif args.feature == "voice":
                    parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
                    target_dist, others_dist = compute_voice(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_voice(parser, target_slice, all_others)
                elif args.feature == "tense":
                    parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
                    target_dist, others_dist = compute_tense(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_tense(parser, target_slice, all_others)
                elif args.feature == "aspect":
                    parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
                    target_dist, others_dist = compute_aspect(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_aspect(
                        parser, target_slice, all_others
                    )
                elif args.feature == "comparison":
                    parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
                    target_dist, others_dist = compute_comparison(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_comparison(
                        parser, target_slice, all_others
                    )
                elif args.feature == "foreign_word":
                    parser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
                    target_dist, others_dist = compute_foreign_word(
                        parser, target_slice, others
                    )
                    _, all_others_dist = compute_foreign_word(
                        parser, target_slice, all_others
                    )
                elif args.feature == "long_distance":
                    target_dist, others_dist = compute_long_distance_dependency(
                        stanza_parser, target_slice, others
                    )
                    _, all_others_dist = compute_long_distance_dependency(
                        stanza_parser, target_slice, all_others
                    )
            elif args.feature in [
                "gender",
                "nationality",
                "physical_appearance",
                "race_ethnicity",
                "religion",
                "ses",
                "sexual_orientation",
            ]:
                if args.feature == "gender":
                    target_dist, others_dist = gender_identification(
                        target_slice, others
                    )
                    _, all_others_dist = gender_identification(target_slice, all_others)
                else:
                    target_dist, others_dist = classifier_identification(
                        args.feature, target_slice, others
                    )
            elif args.feature in [
                "person_number",
                "org_number",
                "gpe_number",
                "money_number",
                "date_number",
                "product_number",
                "ordinal_number",
            ]:
                feature = args.feature.replace("_number", "").upper()
                target_dist, others_dist = ner_number_identification(
                    feature, target_slice, others
                )
                _, all_others_dist = ner_number_identification(
                    feature, target_slice, all_others
                )
            pvalue = ttest_ind(target_dist, others_dist).pvalue
            target_mean = np.mean(target_dist)
            all_mean = np.mean(others_dist)
            if args.feature == "length":
                if pvalue < args.significance_threshold and target_mean < all_mean:
                    homogeneity = compute_homogeneity_short_stc(
                        target_dist, others_dist
                    )
                    s_homogeneities.append(homogeneity)
                    completeness = compute_completeness_short_stc(
                        target_dist, all_others_dist, all_others
                    )
                    s_completenesses.append(completeness)
                    logger.log(
                        "homogeneity = {}, completeness = {}, of slice {} on feature short {}".format(
                            homogeneity, completeness, n, args.feature
                        )
                    )
                    s_feature_group.append(n)
                    target_mean = np.mean(target_dist)
                    target_std = np.std(target_dist)
                    all_mean = np.mean(others_dist)
                    all_std = np.std(others_dist)
                    logger.log(
                        "slice {} has accuracy {}, pvalue for {} is {} since average {}, std {}; all data average {}, std {}".format(
                            n,
                            target_accuracy,
                            args.feature,
                            pvalue,
                            target_mean,
                            target_std,
                            all_mean,
                            all_std,
                        )
                    )
            if pvalue < args.significance_threshold and target_mean > all_mean:
                homogeneity = compute_homogeneity(target_dist, others_dist)
                homogeneities.append(homogeneity)
                completeness = compute_completeness(
                    target_dist, all_others_dist, all_others
                )
                completenesses.append(completeness)
                logger.log(
                    "homogeneity = {}, completeness = {}, of slice {} on feature {}".format(
                        homogeneity, completeness, n, args.feature
                    )
                )
                feature_group.append(n)
                target_mean = np.mean(target_dist)
                target_std = np.std(target_dist)
                all_mean = np.mean(others_dist)
                all_std = np.std(others_dist)
                logger.log(
                    "slice {} has accuracy {}, pvalue for {} is {} since average {}, std {}; all data average {}, std {}".format(
                        n,
                        target_accuracy,
                        args.feature,
                        pvalue,
                        target_mean,
                        target_std,
                        all_mean,
                        all_std,
                    )
                )
    if s_feature_group and args.feature == "length":
        logger.log(
            "number of slices = {}, average homogeneity = {}, average completeness = {} for feature short {}".format(
                len(s_feature_group),
                np.mean(s_homogeneities),
                np.mean(s_completenesses),
                args.feature,
            )
        )
    if feature_group:
        logger.log(
            "number of slices = {}, average homogeneity = {}, average completeness = {} for feature {}".format(
                len(feature_group),
                np.mean(homogeneities),
                np.mean(completenesses),
                args.feature,
            )
        )
    return feature_group


def compute_homogeneity(target_dist, others_dist, target_slice):
    all_mean = np.mean(others_dist)
    all_std = np.std(others_dist)

    incorrect_other_data_indices = [
        i for i, t in enumerate(target_slice) if np.argmax(t[2]) != t[1]
    ]

    correct = 0
    for i, t in enumerate(target_dist):
        if (
            t > min(all_mean + all_std, max(target_dist))
            and i in incorrect_other_data_indices
        ):
            correct += 1

    homogeneity = correct / len(target_dist)

    return homogeneity


def compute_homogeneity_short_stc(target_dist, others_dist):
    all_mean = np.mean(others_dist)
    all_std = np.std(others_dist)

    correct = 0
    for t in target_dist:

        if t <= max(all_mean - all_std, 0):
            correct += 1

    homogeneity = correct / len(target_dist)

    return homogeneity


def compute_completeness(target_dist, others_dist, others):
    incorrect_other_data_indices = [
        i for i, t in enumerate(others) if np.argmax(t[2]) != t[1]
    ]

    all_mean = np.mean(others_dist)
    all_std = np.std(others_dist)

    correct = 0
    all_correct = 0
    for i, t in enumerate(others_dist):
        if (
            t > min(all_mean + all_std, max(target_dist))
            and i in incorrect_other_data_indices
        ):
            if i < len(target_dist):
                correct += 1
            all_correct += 1

    if all_correct == 0:
        completeness = 1
    else:
        completeness = correct / all_correct

    return completeness


def compute_completeness_short_stc(target_dist, others_dist, others):
    incorrect_other_data_indices = [
        i for i, t in enumerate(others) if np.argmax(t[2]) != t[1]
    ]
    incorrect_other_result = [others_dist[i] for i in incorrect_other_data_indices]

    all_mean = np.mean(others_dist)
    all_std = np.std(others_dist)

    correct = 0
    for t in incorrect_other_result:
        if t <= max(all_mean - all_std, 0):
            correct += 1

    completeness = len(target_dist) / (correct + len(target_dist))

    return completeness


def compute_TFIDF(target_slice, others):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--task", type=str, default="CoLA")
    parser.add_argument("--layer_n", type=int, default=12)
    parser.add_argument("--slice_n", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="../myglue/glue_data/")
    parser.add_argument(
        "--error_threshold", type=float, default=0.5, help="accuracy of slice score"
    )
    parser.add_argument(
        "--significance_threshold",
        type=float,
        default=0.05,
        help="pvalue of significance test; if computed pvalue is smaller than the threshold, we reject the null hypothesis",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="all",
        help="features to detect, such as length, negation, multi_modal, etc",
    )
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--validation_path", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    # args.log_path += "_" + str(args.layer_n) + "_" + str(args.slice_n) + ".log"
    set_seed(args)
    logger = Logger(args.log_path)

    domino_validation_dir = (
        args.validation_path
    )  # "../feature_result/to_dev_bert_{}.json".format(args.task)
    # domino_validation_dir = (
    #    "../to_dev_QNLI_confidence.json"
    # )
    assert os.path.isfile(domino_validation_dir)
    with open(domino_validation_dir, "r") as f:
        dev = json.load(f)

    if "jigsaw" not in domino_validation_dir:
        stanza_parser = stanza.Pipeline(
            lang="en", processors="tokenize,mwt,pos,lemma,depparse"
        )
        if args.feature == "all":
            if not torch.cuda.is_available():
                number_of_features = 0
                total_features = 27
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
                    "tree_depth",
                    "extra_to",
                    "multi_modal",
                    "quantifier",
                    "voice",
                    "tense",
                    "aspect",
                    "comparison",
                    "foreign_word",
                    "long_distance",
                    "gender",
                    "person_number",
                    "org_number",
                    "gpe_number",
                    "money_number",
                    "date_number",
                    "product_number",
                    "ordinal_number",
                ]:
                    args.feature = f
                    logger.log("Feature {}:".format(f))
                    feature_group = significance_test(args, logger, dev, stanza_parser)
                    if feature_group:
                        number_of_features += 1
                    logger.log("*****")
            else:
                total_features = 6
                number_of_features = 0
                for f in [
                    "nationality",
                    "physical_appearance",
                    "race_ethnicity",
                    "religion",
                    "ses",
                    "sexual_orientation",
                ]:
                    args.feature = f
                    logger.log("Feature {}:".format(f))
                    feature_group = significance_test(args, logger, dev, stanza_parser)
                    if feature_group:
                        number_of_features += 1
                    logger.log("*****")
            logger.log(
                "number of features discovered: {} out of {}".format(
                    number_of_features, total_features
                )
            )
        else:
            logger.log("Feature {}:".format(f))
            length_as_exp = significance_test(args, logger, dev, stanza_parser)

    else:
        jigsaw_type = args.task.replace("jigsaw_", "")
        with open(
            "../../myglue/glue_data/toxicity/{}_dev.json".format(jigsaw_type), "r"
        ) as f:
            orig_data = json.load(f)
        logger.log(str(orig_data.keys()))

        significance_test_jigsaw(args, logger, dev, orig_data)
