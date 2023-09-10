import numpy as np
import nltk

################## without parser ##################
def compute_length(datapoint):
    return len(datapoint.split())


def compute_negation(datapoint):
    stc = datapoint
    if (
        "not" in stc
        or "n't " in stc
        or " no " in stc
        or " nothing" in stc
        or " nobody" in stc
        or " neither " in stc
    ):
        return 1
    else:
        return 0


def compute_reflexive(datapoint):
    reflexives = [
        "myself",
        "yourself",
        "yourselves",
        "itself",
        "herself",
        "himself",
        "themselves",
        "themself",
    ]
    stc = datapoint
    has_reflexive = False
    for ref in reflexives:
        if ref in stc:
            has_reflexive = True
            break
    if has_reflexive:
        return 1
    else:
        return 0


def compute_frequency(target_slice, others):
    all_words = [token for data in target_slice + others for token in data[0].split()]

    target_dist = []
    for i in range(len(target_slice)):
        stc = target_slice[i][0]
        tokens = stc.split()
        target_dist.append(
            np.sum([all_words.count(token) for token in tokens]) / len(tokens)
        )

    others_dist = []
    for i in range(len(others)):
        stc = others[i][0]
        tokens = stc.split()
        others_dist.append(
            np.sum([all_words.count(token) for token in tokens]) / len(tokens)
        )

    return target_dist, others_dist


def compute_how_question(datapoint):
    stc = datapoint
    if stc.startswith("How") and stc.endswith("?"):
        return 1
    else:
        return 0


def compute_why_question(datapoint):
    stc = datapoint
    if stc.startswith("Why") and stc.endswith("?"):
        return 1
    else:
        return 0


################## with parser ##################
def compute_multiple_prep(parser, datapoint):
    sentence = datapoint
    stc_tree = next(parser.raw_parse(sentence))
    stc_substrees = stc_tree.subtrees()
    labels = [stc_substree.label() for stc_substree in stc_substrees]
    return labels.count("IN") / len(sentence.split())


def compute_NP_sub(parser, datapoint):
    sentence = datapoint
    stc_tree = next(parser.raw_parse(sentence))
    stc_substrees = stc_tree.subtrees()
    labels = [stc_substree.label() for stc_substree in stc_substrees]
    if "WHNP" in labels and ("SBAR" in labels or "SBARQ" in labels):
        return 1
    else:
        return 0


def compute_echo_question(parser, datapoint):
    sentence = datapoint
    stc_tree = next(parser.raw_parse(sentence))
    stc_substrees = stc_tree.subtrees()
    labels = [stc_substree.label() for stc_substree in stc_substrees]
    if labels.count("SQ") > 1:
        return 1
    else:
        return 0


def compute_tree_depth(parser, datapoint):
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

    sentence = datapoint[0]
    stc_tree = next(parser.raw_parse(sentence))
    return compute_length(stc_tree)


################## with tagger ##################
def compute_to_after_modal(parser, datapoint):
    sentence = datapoint
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
        return 1
    else:
        return 0


def compute_multiple_modal(parser, datapoint):
    sentence = datapoint
    tokens = sentence.split()
    parse = parser.tag(tokens)
    labels = [token[1] for token in parse]
    has = False
    for t in range(len(labels) - 1):
        if labels[t] == "MD" and labels[t + 1] == "MD":
            has = True
            break
    if has:
        return 1
    else:
        return 0


def compute_quantifier(parser, datapoint):
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

    sentence = datapoint
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
        return 1
    else:
        return 0


def compute_tense(parser, datapoint):
    sentence = datapoint
    tokens = sentence.split()
    parse = parser.tag(tokens)
    labels = [token[1] for token in parse]
    if "VBD" in labels:
        return 1
    else:
        return 0


def compute_voice(parser, datapoint):
    sentence = datapoint
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
        return 1
    else:
        return 0


def compute_aspect(parser, datapoint):
    sentence = datapoint
    tokens = sentence.split()
    parse = parser.tag(tokens)
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
        return 1
    else:
        return 0


def compute_comparison(parser, datapoint):
    sentence = datapoint
    tokens = sentence.split()
    parse = parser.tag(tokens)
    labels = [token[1] for token in parse]
    if "RBR" in labels or "RBS" in labels or "JJR" in labels or "JJS" in labels:
        return 1
    else:
        return 0


def compute_foreign_word(parser, datapoint):
    sentence = datapoint
    tokens = sentence.split()
    parse = parser.tag(tokens)
    labels = [token[1] for token in parse]
    return labels.count("FW")


################## with Stanza dependency parser ##################
def compute_long_distance_dependency(parser, datapoint):
    sentence = datapoint
    doc = parser(sentence)
    all_dependencies = []
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos != "PUNCT":
                all_dependencies.append(np.abs(word.id - word.head))
    return np.max(all_dependencies) / len(all_dependencies)
