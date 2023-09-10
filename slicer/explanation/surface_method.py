import numpy as np


################## without parser ##################
def compute_length(target_slice, others):
    target_lengths = [len(d[0].split()) for d in target_slice]
    others_lengths = [len(d[0].split()) for d in others]

    return target_lengths, others_lengths


def compute_negation(target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        stc = target_slice[i][0]
        if (
            "not" in stc
            or "n't " in stc
            or " no " in stc
            or " nothing" in stc
            or " nobody" in stc
            or " neither " in stc
        ):
            target_dist.append(1)
        else:
            target_dist.append(0)

    others_dist = []
    for i in range(len(others)):
        stc = others[i][0]
        if (
            "not" in stc
            or "n't " in stc
            or " no " in stc
            or " nothing" in stc
            or " nobody" in stc
            or " neither " in stc
        ):
            others_dist.append(1)
        else:
            others_dist.append(0)

    return target_dist, others_dist


def compute_reflexive(target_slice, others):
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
    target_dist = []
    for i in range(len(target_slice)):
        stc = target_slice[i][0]
        has_reflexive = False
        for ref in reflexives:
            if ref in stc:
                has_reflexive = True
                break
        if has_reflexive:
            target_dist.append(1)
        else:
            target_dist.append(0)

    others_dist = []
    for i in range(len(others)):
        stc = others[i][0]
        has_reflexive = False
        for ref in reflexives:
            if ref in stc:
                has_reflexive = True
                break
        if has_reflexive:
            others_dist.append(1)
        else:
            others_dist.append(0)

    return target_dist, others_dist


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

def compute_how_question(target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        stc = target_slice[i][0]
        if stc.startswith('How') and stc.endswith('?'):
            target_dist.append(1)
        else:
            target_dist.append(0)

    others_dist = []
    for i in range(len(others)):
        stc = others[i][0]
        if stc.startswith('How') and stc.endswith('?'):
            others_dist.append(1)
        else:
            others_dist.append(0)

    return target_dist, others_dist


def compute_why_question(target_slice, others):
    target_dist = []
    for i in range(len(target_slice)):
        stc = target_slice[i][0]
        if stc.startswith('Why') and stc.endswith('?'):
            target_dist.append(1)
        else:
            target_dist.append(0)

    others_dist = []
    for i in range(len(others)):
        stc = others[i][0]
        if stc.startswith('Why') and stc.endswith('?'):
            others_dist.append(1)
        else:
            others_dist.append(0)

    return target_dist, others_dist
