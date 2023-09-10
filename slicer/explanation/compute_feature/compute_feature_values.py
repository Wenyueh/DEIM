import argparse

from feature_functions import (
    compute_how_question,
    compute_length,
    compute_reflexive,
    compute_negation,
    compute_why_question,
    compute_multiple_prep,
    compute_NP_sub,
    compute_echo_question,
    compute_tree_depth,
    compute_to_after_modal,
    compute_multiple_modal,
    compute_quantifier,
    compute_tense,
    compute_voice,
    compute_aspect,
    compute_comparison,
    compute_long_distance_dependency,
)

import json
import os
import stanza
from nltk.parse.corenlp import CoreNLPParser
import sys
import warnings

warnings.filterwarnings("ignore")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=100)
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
    parser.add_argument(
        "--feature",
        type=str,
        default="all",
        help="features to detect, such as length, negation, multi_modal, etc",
    )
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--validation_path", type=str)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    args.validation_path = "featured_data/to_dev_{}.json".format(args.task)
    args.save_path = "featured_data/to_dev_{}.json".format(args.task)
    args.log_path = "{}.log".format(args.task)

    logger = Logger(args.log_path)

    domino_validation_dir = args.validation_path

    assert os.path.isfile(domino_validation_dir)
    with open(domino_validation_dir, "r") as f:
        dev = json.load(f)

    stanza_parser = stanza.Pipeline(
        lang="en", processors="tokenize,mwt,pos,lemma,depparse"
    )
    corenlpparser = CoreNLPParser(url="http://localhost:9000", tagtype="pos")
    parser = CoreNLPParser()
    logger.log("start computing features")
    for f in [
        # "length",
        # "reflexive",
        # "negation",
        # "frequency",
        # "how",
        # "why",
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
        dev[f] = []
        if args.task in ["MNLI", "QQP"]:
            all_sentences = dev["sentence"][:10000]
        else:
            all_sentences = dev["sentence"]
        for sentence in all_sentences:
            if args.task in ["QNLI", "MNLI", "QQP"]:
                sentence = sentence[0]
            if f == "length":
                dev[f].append(compute_length(sentence))
            if f == "negation":
                dev[f].append(compute_negation(sentence))
            if f == "reflexive":
                dev[f].append(compute_reflexive(sentence))
            if f == "how":
                dev[f].append(compute_how_question(sentence))
            if f == "why":
                dev[f].append(compute_why_question(sentence))
            if f == "multi_preposition":
                dev[f].append(compute_multiple_prep(parser, sentence))
            if f == "NP_sub":
                dev[f].append(compute_NP_sub(parser, sentence))
            if f == "echo_question":
                dev[f].append(compute_echo_question(parser, sentence))
            if f == "tree_depth":
                dev[f].append(compute_tree_depth(parser, sentence))
            if f == "extra_to":
                dev[f].append(compute_to_after_modal(corenlpparser, sentence))
            if f == "multi_modal":
                dev[f].append(compute_multiple_modal(corenlpparser, sentence))
            if f == "quantifier":
                dev[f].append(compute_quantifier(corenlpparser, sentence))
            if f == "tense":
                dev[f].append(compute_tense(corenlpparser, sentence))
            if f == "voice":
                dev[f].append(compute_voice(corenlpparser, sentence))
            if f == "aspect":
                dev[f].append(compute_aspect(corenlpparser, sentence))
            if f == "comparison":
                dev[f].append(compute_comparison(corenlpparser, sentence))
            if f == "long_distance":
                dev[f].append(compute_long_distance_dependency(stanza_parser, sentence))

        logger.log("finished {} for all sentences".format(f))
        with open(args.save_path, "w") as f:
            json.dump(dev, f)
