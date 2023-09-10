import nltk
import numpy as np


################## with dependency parser ##################
def compute_multiple_prep(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        stc_tree = next(parser.raw_parse(sentence))
        stc_substrees = stc_tree.subtrees()
        labels = [stc_substree.label() for stc_substree in stc_substrees]
        target_dist.append(labels.count("IN") / len(sentence.split()))

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        stc_tree = next(parser.raw_parse(sentence))
        stc_substrees = stc_tree.subtrees()
        labels = [stc_substree.label() for stc_substree in stc_substrees]
        all_dist.append(labels.count("IN") / len(sentence.split()))

    return target_dist, all_dist


def compute_NP_sub(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        stc_tree = next(parser.raw_parse(sentence))
        stc_substrees = stc_tree.subtrees()
        labels = [stc_substree.label() for stc_substree in stc_substrees]
        if "WHNP" in labels and ("SBAR" in labels or "SBARQ" in labels):
            target_dist.append(1)
        else:
            target_dist.append(0)

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        stc_tree = next(parser.raw_parse(sentence))
        stc_substrees = stc_tree.subtrees()
        labels = [stc_substree.label() for stc_substree in stc_substrees]
        if "WHNP" in labels and ("SBAR" in labels or "SBARQ" in labels):
            all_dist.append(1)
        else:
            all_dist.append(0)

    return target_dist, all_dist


def compute_echo_question(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        stc_tree = next(parser.raw_parse(sentence))
        stc_substrees = stc_tree.subtrees()
        labels = [stc_substree.label() for stc_substree in stc_substrees]
        if labels.count("SQ") > 1:
            target_dist.append(1)
        else:
            target_dist.append(0)

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        stc_tree = next(parser.raw_parse(sentence))
        stc_substrees = stc_tree.subtrees()
        labels = [stc_substree.label() for stc_substree in stc_substrees]
        if labels.count("SQ") > 1:
            all_dist.append(1)
        else:
            all_dist.append(0)

    return target_dist, all_dist


def compute_tree_depth(parser, target_slice, others):
    def compute_length(parse):
        if type(parse) == nltk.tree.Tree:
            deepest = 0
            breadth = len(parse)
            for i in range(breadth):
                depth = compute_length(parse[i])
                if depth > deepest:
                    deepest = depth
        else:
            deepest = 0
        return deepest + 1

    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        stc_tree = next(parser.raw_parse(sentence))
        target_dist.append(compute_length(stc_tree))

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        stc_tree = next(parser.raw_parse(sentence))
        all_dist.append(compute_length(stc_tree))

    return target_dist, all_dist


################## with tagger ##################
def compute_to_after_modal(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        has = False
        for t in range(len(labels) - 1):
            if labels[t] == "MD" and labels[t + 1] == "TO":
                print(1)
                has = True
                break
        if has:
            target_dist.append(1)
        else:
            target_dist.append(0)

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        has = False
        for t in range(len(labels) - 1):
            if labels[t] == "MD" and labels[t + 1] == "TO":
                has = True
                break
        if has:
            all_dist.append(1)
        else:
            all_dist.append(0)

    return target_dist, all_dist


def compute_multiple_modal(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        has = False
        for t in range(len(labels) - 1):
            if labels[t] == "MD" and labels[t + 1] == "MD":
                has = True
                break
        if has:
            target_dist.append(1)
        else:
            target_dist.append(0)

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        has = False
        for t in range(len(labels) - 1):
            if labels[t] == "MD" and labels[t + 1] == "MD":
                has = True
                break
        if has:
            all_dist.append(1)
        else:
            all_dist.append(0)

    return target_dist, all_dist


def compute_quantifier(parser, target_slice, others):
    quantifier_list = [
        " anything ",
        " everything ",
        " nothing ",
        " most ",
        " all ",
        " no ",
        " some ",
        " any ",
    ]

    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        has_quantififer = False
        for quantifier in quantifier_list:
            if quantifier in sentence:
                has_quantififer = True
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        if "CD" in labels:
            has_quantififer = True

        if has_quantififer:
            target_dist.append(1)
        else:
            target_dist.append(0)

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        has_quantififer = False
        for quantifier in quantifier_list:
            if quantifier in sentence:
                has_quantififer = True
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        if "CD" in labels:
            has_quantififer = True

        if has_quantififer:
            all_dist.append(1)
        else:
            all_dist.append(0)

    return target_dist, all_dist


def compute_tense(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        if "VBD" in labels:
            target_dist.append(1)
        else:
            target_dist.append(0)

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        if "VBD" in labels:
            all_dist.append(1)
        else:
            all_dist.append(0)

    return target_dist, all_dist


def compute_voice(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        has_passive_voice = False
        for i in range(len(parse) - 1):
            if parse[i][0] in ["be", "is", "was", "are", "were", "been"] and (
                parse[i + 1][1] == "VBN"
            ):
                has_passive_voice = True
                break
        for i in range(len(parse) - 2):
            if (
                parse[i][0] in ["be", "is", "was", "are", "were", "been"]
                and parse[i + 1][1] == "RB"
                and parse[i + 2][1] == "VBN"
            ):
                has_passive_voice = True
                break
        if has_passive_voice:
            target_dist.append(1)
        else:
            target_dist.append(0)

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        has_passive_voice = False
        for i in range(len(parse) - 1):
            if parse[i][0] in ["be", "is", "was", "are", "were", "been"] and (
                parse[i + 1][1] == "VBN"
            ):
                has_passive_voice = True
                break
        for i in range(len(parse) - 2):
            if (
                parse[i][0] in ["be", "is", "was", "are", "were", "been"]
                and parse[i + 1][1] == "RB"
                and parse[i + 2][1] == "VBN"
            ):
                has_passive_voice = True
                break
        if has_passive_voice:
            target_dist.append(1)
        else:
            target_dist.append(0)

    return target_dist, all_dist


def compute_aspect(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        has_complete_aspect = False
        for i in range(len(parse) - 1):
            if parse[i][0] in ["has", "have", "had"] and (parse[i + 1][1] == "VBN"):
                has_complete_aspect = True
                break
        for i in range(len(parse) - 2):
            if (
                parse[i][0] in ["has", "have", "had"]
                and parse[i + 1][1] == "RB"
                and parse[i + 2][1] == "VBN"
            ):
                has_complete_aspect = True
                break
        if has_complete_aspect:
            target_dist.append(1)
        else:
            target_dist.append(0)

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        has_complete_aspect = False
        for i in range(len(parse) - 1):
            if parse[i][0] in ["has", "have", "had"] and (parse[i + 1][1] == "VBN"):
                has_complete_aspect = True
                break
        for i in range(len(parse) - 2):
            if (
                parse[i][0] in ["has", "have", "had"]
                and parse[i + 1][1] == "RB"
                and parse[i + 2][1] == "VBN"
            ):
                has_complete_aspect = True
                break
        if has_complete_aspect:
            all_dist.append(1)
        else:
            all_dist.append(0)

    return target_dist, all_dist


def compute_comparison(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        if "RBR" in labels or "RBS" in labels or "JJR" in labels or "JJS" in labels:
            target_dist.append(1)
        else:
            target_dist.append(0)

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        if "RBR" in labels or "RBS" in labels or "JJR" in labels or "JJS" in labels:
            all_dist.append(1)
        else:
            all_dist.append(0)

    return target_dist, all_dist


def compute_foreign_word(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        target_dist.append(labels.count("FW"))

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        tokens = sentence.split()
        parse = parser.tag(tokens)
        labels = [token[1] for token in parse]
        all_dist.append(labels.count("FW"))

    return target_dist, all_dist


################## with Stanza dependency parser ##################
def compute_long_distance_dependency(parser, target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        doc = parser(sentence)
        all_dependencies = []
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos != "PUNCT":
                    all_dependencies.append(np.abs(word.id - word.head))
        target_dist.append(np.max(all_dependencies) / len(all_dependencies))

    all_dist = []
    for i in range(len(others)):
        sentence = others[i][0]
        doc = parser(sentence)
        all_dependencies = []
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos != "PUNCT":
                    all_dependencies.append(np.abs(word.id - word.head))
        all_dist.append(np.max(all_dependencies) / len(all_dependencies))

    return target_dist, all_dist
