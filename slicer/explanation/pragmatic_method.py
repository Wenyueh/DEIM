import json

import torch
from torch.utils.data import DataLoader, Dataset


class InputDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def gender_identification(target_slice, others):
    with open("gender_identification.json", "r") as f:
        gender_identification = json.load(f)
    female = gender_identification[0]
    male = gender_identification[1]

    target_dist = []
    for i in range(len(target_slice)):
        sentence = target_slice[i][0]
        gender = 0  # 0 no gender, 1 male, 2 female
        has_female = False
        has_male = False
        for f in female:
            if f in sentence:
                has_female = True
                break
        for m in male:
            if m in sentence:
                has_male = True
        if (has_female and has_male) or not (has_female or has_male):
            gender = 0
        elif has_female and not has_male:
            gender = 2
        else:
            gender = 1
        target_dist.append(gender)

    all_dist = []
    for i in range(len(others)):
        sentence = target_slice[i][0]
        gender = 0  # 0 no gender, 1 male, 2 female
        has_female = False
        has_male = False
        for f in female:
            if f in sentence:
                has_female = True
                break
        for m in male:
            if m in sentence:
                has_male = True
        if (has_female and has_male) or not (has_female or has_male):
            gender = 0
        elif has_female and not has_male:
            gender = 2
        else:
            gender = 1
        all_dist.append(gender)

    return target_dist, all_dist


def toxicity_identification(classifier, target_slice, others):
    target_dist = []
    target_sentences = [d[0] for d in target_slice]
    target_dataset = InputDataset(target_sentences)
    target_loader = DataLoader(target_dataset, batch_size=8, shuffle=False)
    for data in target_loader:
        prediction = classifier(data)
        target_dist += prediction.tolist()

    all_dist = []
    all_sentences = [d[0] for d in others]
    all_dataset = InputDataset(all_sentences)
    all_loader = DataLoader(all_dataset, batch_size=8, shuffle=False)
    for data in all_loader:
        prediction = classifier(data)
        all_dist += prediction.tolist()

    return target_dist, all_dist


def compute_female_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        female_value = orig_data["female"][idx]
        if female_value:
            female_value = float(female_value)
        else:
            female_value = 0
        dist.append(female_value)
    return dist


def compute_male_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        male_value = orig_data["male"][idx]
        if male_value:
            male_value = float(male_value)
        else:
            male_value = 0
        dist.append(male_value)
    return dist


def compute_atheist_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        atheist_value = orig_data["atheist"][idx]
        if atheist_value:
            atheist_value = float(atheist_value)
        else:
            atheist_value = 0
        dist.append(atheist_value)
    return dist


def compute_buddhist_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        buddhist_value = orig_data["buddhist"][idx]
        if buddhist_value:
            buddhist_value = float(buddhist_value)
        else:
            buddhist_value = 0
        dist.append(buddhist_value)
    return dist


def compute_christian_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        christian_value = orig_data["christian"][idx]
        if christian_value:
            christian_value = float(christian_value)
        else:
            christian_value = 0
        dist.append(christian_value)
    return dist


def compute_hindu_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        hindu_value = orig_data["hindu"][idx]
        if hindu_value:
            hindu_value = float(hindu_value)
        else:
            hindu_value = 0
        dist.append(hindu_value)
    return dist


def compute_jewish_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        jewish_value = orig_data["jewish"][idx]
        if jewish_value:
            jewish_value = float(jewish_value)
        else:
            jewish_value = 0
        dist.append(jewish_value)
    return dist


def compute_muslim_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        muslim_value = orig_data["muslim"][idx]
        if muslim_value:
            muslim_value = float(muslim_value)
        else:
            muslim_value = 0
        dist.append(muslim_value)
    return dist


def compute_asian_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        asian_value = orig_data["asian"][idx]
        if asian_value:
            asian_value = float(asian_value)
        else:
            asian_value = 0
        dist.append(asian_value)
    return dist


def compute_white_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        white_value = orig_data["white"][idx]
        if white_value:
            white_value = float(white_value)
        else:
            white_value = 0
        dist.append(white_value)
    return dist


def compute_black_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        black_value = orig_data["black"][idx]
        if black_value:
            black_value = float(black_value)
        else:
            black_value = 0
        dist.append(black_value)
    return dist


def compute_latino_dist(data, orig_data):
    dist = []
    for one_data in data:
        idx = one_data[0]
        latino_value = orig_data["latino"][idx]
        if latino_value:
            latino_value = float(latino_value)
        else:
            latino_value = 0
        dist.append(latino_value)
    return dist
