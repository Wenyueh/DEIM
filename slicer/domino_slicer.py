import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from data_process import load_dataloaders
from torch.utils.data import DataLoader
from domino import DominoSlicer, explore, DominoErrorSlicer, DominoSlicerGaussianYhat, DominoSlicerAllGaussian, DominoSlicerAllGaussianNT
from transformers import AutoModel
from sklearn.decomposition import PCA
from GG_domino_MML import DOMINO_MML, train_domino_MML, label_binarize, create_optimizer, load_loaders
from DE_domino import DOMINO_DE, train_domino_DE
from GG_domino_MML import domino_dataset
from scipy.special import logsumexp as nplogsumexp

from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.mixture import GaussianMixture


def grouping(n, data):
    gro = []
    for i in range(len(data["sentence"])):
        if data["predicted_group"][i] == n:
            gro.append(
                [
                    data["sentence"][i],
                    data["label"][i],
                    data["probs"][i],
                ]
            )
    return gro


def compute_acc(gro):
    acc = []
    for g in gro:
        acc.append(np.argmax(g[2]) == g[1])
    return np.mean(acc)

def flatten_y(y):
    normal_random_noise = np.random.normal(loc=0.0, scale=0.01, size=[len(y), len(y[0])])
    y = y + normal_random_noise
    return y


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


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


def collect_tsv_data(args):
    if args.task == 'SST-5':
        train_dir = args.data_dir + "train.txt"
        dev_dir = args.data_dir + "dev.txt"
        test_dir = args.data_dir + "test.txt"

        with open(train_dir, "r") as f:
            train = f.read()
        with open(dev_dir, "r") as f:
            dev = f.read()
        with open(test_dir, "r") as f:
            test = f.read()

        train = train.split("\n")[:-1]
        dev = dev.split("\n")[:-1]
        test = test.split("\n")[:-1]

        train = [[d.split("\t")[1], int(d.split("\t")[0])] for d in train]
        switch = int(len(train)*(3/5))
        dev = [[d.split("\t")[1], int(d.split("\t")[0])] for d in dev] + train[switch:]
        train = train[:switch]
        test = [[d.split("\t")[1], int(d.split("\t")[0])] for d in test]

        train = {'sentence':[t[0].strip() for t in train], 'label':[t[1] for t in train]}
        dev = {'sentence':[t[0].strip() for t in dev], 'label':[t[1] for t in dev]}
        test = {'sentence':[t[0].strip() for t in test], 'label':[t[1] for t in test]}

    if args.task != 'SST-5':
        train_dir = args.data_dir + "train.tsv"
        dev_dir = args.data_dir + "dev.tsv"
        test_dir = args.data_dir + "test.tsv"

        train = []
        with open(train_dir, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\n")
            for l in reader:
                train.append(l)

        dev = []
        with open(dev_dir, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\n")
            for l in reader:
                dev.append(l)

        test = []
        with open(test_dir, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter="\n")
            for l in reader:
                test.append(l)

        length = len(train)
        half_length = int(length / 2)
        half_train = train[:half_length]
        test = dev
        dev = train[half_length:]
        train = half_train

        if args.task == "SST-2":  # sentiment
            train.remove(train[0])
            train = {
                "sentence": [d[0].split("\t")[0].strip() for d in train],
                "label": [int(d[0].split("\t")[1]) for d in train],
            }

            # dev.remove(dev[0])
            dev = {
                "sentence": [d[0].split("\t")[0].strip() for d in dev],
                "label": [int(d[0].split("\t")[1]) for d in dev],
            }

            test.remove(test[0])
            test = {
                "sentence": [d[0].split("\t")[0].strip() for d in test],
                "label": [int(d[0].split("\t")[1]) for d in test],
            }
        elif args.task == "QNLI":
            train.remove(train[0])
            train = {
                "sentence": [
                    (d[0].split("\t")[1].strip(), d[0].split("\t")[2].strip(),)
                    for d in train
                ],
                "label": [1 if d[0].split("\t")[3] == "entailment" else 0 for d in train],
            }
            dev = {
                "sentence": [
                    (d[0].split("\t")[1].strip(), d[0].split("\t")[2].strip(),) for d in dev
                ],
                "label": [1 if d[0].split("\t")[3] == "entailment" else 0 for d in dev],
            }
            test.remove(test[0])
            test = {
                "sentence": [
                    (d[0].split("\t")[1].strip(), d[0].split("\t")[2].strip(),)
                    for d in test
                ],
                "label": [1 if d[0].split("\t")[3] == "entailment" else 0 for d in test],
            }
        elif args.task == "QQP":  # paraphrase
            train.remove(train[0])
            train = {
                "sentence": [
                    (d[0].split("\t")[3].strip(), d[0].split("\t")[4].strip(),)
                    for d in train
                ],
                "label": [int(d[0].split("\t")[-1]) for d in train],
            }
            dev = {
                "sentence": [
                    (d[0].split("\t")[3].strip(), d[0].split("\t")[4].strip(),) for d in dev
                ],
                "label": [int(d[0].split("\t")[-1]) for d in dev],
            }
            test.remove(test[0])
            test = {
                "sentence": [
                    (d[0].split("\t")[3].strip(), d[0].split("\t")[4].strip(),)
                    for d in test
                ],
                "label": [int(d[0].split("\t")[-1]) for d in test],
            }
        elif args.task == "CoLA":
            train = {
                "sentence": [d[0].split("\t")[3] for d in train],
                "label": [int(d[0].split("\t")[1]) for d in train],
            }
            dev = {
                "sentence": [d[0].split("\t")[3] for d in dev],
                "label": [int(d[0].split("\t")[1]) for d in dev],
            }
            test = {
                "sentence": [d[0].split("\t")[3] for d in test],
                "label": [int(d[0].split("\t")[1]) for d in test],
            }

    return train, dev, test


class binary_classifier(nn.Module):
    def __init__(self, args, encoder, num_classes):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.linear_layer = nn.Linear(self.hidden_size, num_classes, bias=True)
        self.dropout = nn.Dropout(self.args.dropout_rate)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        embeddings = out[0].mean(dim=1)
        embeddings = self.dropout(embeddings)
        logits = self.linear_layer(embeddings)

        all_hidden_states = out[2]
        return logits, embeddings, all_hidden_states

    def compute_loss(self, logits, labels):
        loss_value = self.loss(logits, labels)
        return loss_value

    def predict(self, logits):
        assert logits.dim() == 2
        return torch.argmax(logits, dim=1)


def compute_embeddings(classifier, data_loader):
    embeddings = []
    for batch in data_loader:
        input_ids = batch[0].cuda()
        attn_mask = batch[1].cuda()
        segment_ids = batch[2].cuda()
        b = input_ids.size(0)
        _, hidden_states, all_hidden_states = classifier(
            input_ids, attn_mask, segment_ids
        )
        for i in range(b):
            one_attn_mask = attn_mask.tolist()[i]
            # remove padding embeddings
            if 0 in one_attn_mask:
                padding_position = one_attn_mask.index(0)
            else:
                padding_position = len(one_attn_mask)
            one_hidden_states = (
                all_hidden_states[args.layer_n][i][:padding_position, :]
                .mean(dim=0)
                .tolist()
            )
            embeddings.append(one_hidden_states)

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--task", type=str, default="CoLA", help='SST-2, SST-5, CoLA, QNLI')
    parser.add_argument("--domino_type", type=str, default="MML", help="EM, MML, DE")
    parser.add_argument(
        "--initialization", type=str, default="neural", help="random, confusion, neural"
    )
    parser.add_argument("--layer_n", type=int, default=12)
    parser.add_argument(
        "--n_components",
        type=int,
        default=128,
        help="number of components for Gaussian Mixture",
    )
    parser.add_argument('--label_emb_size', type=int, default=10, help='use for DE model specifically')
    parser.add_argument("--slice_n", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="../myglue/glue_data/")
    parser.add_argument("--prob_dir", type=str, default="../myglue/")
    parser.add_argument("--toy", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--domino_batch_size", type=int, default=500)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--model_dir", type=str)

    parser.add_argument("--pca_n_components", type=int, default=128)
    parser.add_argument("--non_embedding_lr", type=float, default=1e-3)
    parser.add_argument("--embedding_lr", type=float, default=5e-4)
    parser.add_argument("--non_embedding_clip", type=float, default=1)
    parser.add_argument("--embedding_clip", type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=1)
    parser.add_argument("--noise_transition", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--domino_warmup_prop", type=float, default=0.2)
    parser.add_argument("--spiky_alpha_y", type=float, default=1)
    parser.add_argument("--spiky_alpha_yhat", type=float, default=1)
    parser.add_argument('--load_trained_embedding', type=bool, default=False)
    
    parser.add_argument("--y_log_likelihood_weight", type=float, default=0.3)
    parser.add_argument("--y_hat_log_likelihood_weight", type=float, default=1)
    parser.add_argument('--emb_log_likelihood_weight' ,type=float, default=0.2)
    parser.add_argument('--mu_weight', type=float, default=1)
    parser.add_argument('--mu_kl_weight', type=float, default=1)
    parser.add_argument('--var_weight', type=float, default=0.1)

    parser.add_argument('--use_em_init', type=str, choices=['resp', 'param', 'none'], default='param')

    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    set_seed(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.save_position = "../../../../../harddisk/user/wenyuehua/"
    args.data_dir += args.task + "/"
    args.prob_dir += args.task + "_dev_probs.json"
    args.log_path = args.task + "_{}_{}_{}.log".format(args.y_log_likelihood_weight, args.y_hat_log_likelihood_weight, args.emb_log_likelihood_weight)
    args.model_dir = '../myglue/'+args.task + '_meanpool.pt'
    logger = Logger(args.log_path)
    logger.log(str(args))

    # return dictionary
    train, dev, test = collect_tsv_data(args)
    logger.log("train data length is {}".format(len(train["sentence"])))
    logger.log("dev data length is {}".format(len(dev["sentence"])))
    logger.log("test data length is {}".format(len(test["sentence"])))
    train_loader, dev_loader, test_loader = load_dataloaders(args)

    #################### prepare dev data ####################
    domino_validation_dir = (
        args.task + "_dev" + "_" + str(args.layer_n) + "_128.json"
    )
    if not os.path.isfile(domino_validation_dir):        
        logger.log("load trained model")
        encoder = AutoModel.from_pretrained("bert-large-uncased").cuda()
        encoder.config.output_hidden_states = True
        if args.task == 'SST-5':
            num_classes = 5
            classifier = binary_classifier(args, encoder, num_classes).cuda()
            # accidentally save the whole model, not just state dict
            classifier = torch.load(args.model_dir)
            classifier.encoder.config.output_hidden_states = True
        else:
            num_classes = 2
            classifier = binary_classifier(args, encoder, num_classes).cuda()
            classifier.load_state_dict(torch.load(args.model_dir))
        classifier.eval()

        #################### build dev datapanel ####################
        logger.log("compute validation embedding")
        dev_embeddings = compute_embeddings(classifier, dev_loader)
        dev["embeddings"] = dev_embeddings

        # add prediction result and prediction probability score here
        logger.log("add validation probability prediction")
        with open(args.prob_dir, "r") as f:
            probs = json.load(f)
        dev["probs"] = probs

        with open(domino_validation_dir, 'w') as f:
            json.dump(dev, f)
    else:
        with open(domino_validation_dir, 'r') as f:
            dev = json.load(f)
        dev_embeddings = dev['embeddings']

    #################### prepare test data ####################
    logger.log('compute domino on test dataset')
    domino_test_dir = (
        args.task + "_test" + "_" + str(args.layer_n) + "_128.json"
    )
    if not os.path.isfile(domino_test_dir):
        logger.log("compute slice membership on test")
        args.prob_dir = args.prob_dir.replace("dev", "test")
        test_embeddings = compute_embeddings(classifier, test_loader)
        test["embeddings"] = test_embeddings

        with open(args.prob_dir, "r") as f:
            probs = json.load(f)
        test["probs"] = probs
        with open(domino_test_dir, 'w') as f:
            json.dump(test, f)
    else:
        with open(domino_test_dir, 'r') as f:
            test = json.load(f)
        test_embeddings = test['embeddings']
    logger.log('finish loading test data')

    if args.domino_type == "EM":
        domino = DominoSlicerAllGaussian(
            y_log_likelihood_weight=args.y_log_likelihood_weight,
            y_hat_log_likelihood_weight=args.y_hat_log_likelihood_weight,
            emb_log_likelihood_weight=args.emb_log_likelihood_weight,
            n_mixture_components=args.n_components,
            n_slices=args.slice_n,
            init_params='kmeans'
        )
        domino.fit(
            data=dev, embeddings="embeddings", targets="label", pred_probs="probs"
        )
    else:
        dev_embeddings = dev['embeddings']
        if args.pca_n_components is None:
            pca = None
        else:
            pca = PCA(n_components=args.pca_n_components)
            # preprocess dev and test embeddings
            pca.fit(dev_embeddings)
            dev_embeddings_np = pca.transform(X=dev_embeddings)
            test_embeddings_np = pca.transform(X=test_embeddings)
            dev_embeddings = torch.tensor(dev_embeddings_np).cuda()
            test_embeddings = torch.tensor(test_embeddings_np).cuda()

            #################### run domino ####################
            logger.log("fit domino on validation dataset")

            ### EM labels
            if args.noise_transition:
                domino_em = DominoSlicerAllGaussianNT(
                    y_log_likelihood_weight=args.y_log_likelihood_weight,
                    y_hat_log_likelihood_weight=args.y_hat_log_likelihood_weight,
                    emb_log_likelihood_weight=args.emb_log_likelihood_weight,
                    n_mixture_components=args.n_components,
                    n_slices=args.slice_n,
                    n_pca_components=args.pca_n_components,
                    init_params='kmeans'
                )
            else:
                domino_em = DominoSlicerAllGaussian(
                    y_log_likelihood_weight=args.y_log_likelihood_weight,
                    y_hat_log_likelihood_weight=args.y_hat_log_likelihood_weight,
                    emb_log_likelihood_weight=args.emb_log_likelihood_weight,
                    n_mixture_components=args.n_components,
                    n_slices=args.slice_n,
                    n_pca_components=args.pca_n_components,
                    init_params='kmeans'
                )
            em_dev_labels = domino_em.mm.fit_predict(
                X=dev_embeddings_np, y=np.array(dev["label"]), y_hat=np.array(dev["probs"])
            ).tolist()

            y, y_hat = domino_em.mm._preprocess_ys(y=np.array(dev["label"]), y_hat=np.array(dev["probs"]))
            y = flatten_y(y)
            em_likelihood, _ = domino_em.mm._estimate_log_prob_resp(
                X=dev_embeddings_np, y=y, y_hat=y_hat
            )
            print('EM loglikehood:', nplogsumexp(em_likelihood, axis=1).sum())

            em_log, em_resp = domino_em.mm.predict_proba(
                X=dev_embeddings_np, y=y, y_hat=y_hat
            )
            dev["log_domino_slices"] = em_log.tolist()
            dev["resp"] = em_resp
            dev['predicted_group'] = np.argmax(dev["resp"], axis=-1).tolist()

            domino_params = domino_em.mm._get_parameters()

            test_y, test_y_hat = domino_em.mm._preprocess_ys(y=np.array(test["label"]), y_hat=np.array(test["probs"]))
            test_y = flatten_y(test_y)
            test_em_log, test_em_resp = domino_em.mm.predict_proba(
                X=test_embeddings_np, y=test_y, y_hat=test_y_hat, test=True
            )

            test["log_domino_slices"] = test_em_log.tolist()
            test['resp'] = test_em_resp.tolist()
            slice_prediction = test["resp"][:]
            slice_group = np.argmax(slice_prediction, axis=1).tolist()
            test["predicted_group"] = slice_group

            # print dev result first 
            n = 0
            for i in range(args.n_components):
                gro = grouping(i, dev)
                if gro:
                    acc = compute_acc(gro)
                    if acc < 0.5:
                        n += 1

            logger.log('number of dev slices with acc < 0.5 is {}'.format(n))

            n = 0
            for i in range(args.n_components):
                gro = grouping(i, dev)
                if gro:
                    acc = compute_acc(gro)
                    if acc < 0.5:
                        n += len(gro)
            logger.log('number of dev datapoints in slices with acc < 0.5 is {}'.format(n))

            # print test result 
            n = 0
            for i in range(args.n_components):
                gro = grouping(i, test)
                if gro:
                    acc = compute_acc(gro)
                    if acc < 0.5:
                        n += 1

            logger.log('number of test slices with acc < 0.5 is {}'.format(n))

            n = 0
            for i in range(args.n_components):
                gro = grouping(i, test)
                if gro:
                    acc = compute_acc(gro)
                    if acc < 0.5:
                        n += len(gro)
            logger.log('number of test datapoints in slices with acc < 0.5 is {}'.format(n))

            devaccs =[]
            for i in range(args.slice_n):
                gro = grouping(i, dev)
                if gro:
                    acc = compute_acc(gro)
                    devaccs.append((acc, len(gro), i))

            error_slices = [d[2] for d in devaccs if d[0]<0.5]
            predicted_wrong = [[test['sentence'][i], test['label'][i], test['probs'][i], max(test['log_domino_slices'][i]), max(test['resp'][i]), test['predicted_group'][i]] for i in range(len(test['sentence'])) if test['predicted_group'][i] in error_slices]
            acc = compute_acc(predicted_wrong)
            logger.log('number of predicted wrong is {}, accuracy of predicted wrong is {}'.format(len(predicted_wrong), acc))

        #################### run standard Gaussian Mixture ####################
        logger.log('train standard Gaussian')
        gm = GaussianMixture(n_components=args.n_components, random_state=0, covariance_type='diag').fit(np.array(dev_embeddings.cpu()))
        gold_labels = gm.predict(np.array(dev_embeddings.cpu())).tolist()
        gassuain_resp = gm.predict_proba(np.array(dev_embeddings.cpu()))

        y = torch.tensor(flatten_y(label_binarize(dev['label']))).cuda()
        y_hat = torch.tensor(dev['probs']).cuda()
        test_y = torch.tensor(flatten_y(label_binarize(test['label']))).cuda()
        test_y_hat = torch.tensor(test['probs']).cuda()
        if args.domino_type == 'MML':
            domino = DOMINO_MML(args, args.initialization, test=False)
            if args.use_em_init == 'none':
                domino.initialization(dev_embeddings, y, y_hat, outside_resp=None, outside_params=None)
            elif args.use_em_init == 'resp':
                domino.initialization(dev_embeddings, y, y_hat, outside_resp=em_resp, outside_params=None)
            elif args.use_em_init == 'param':
                domino.initialization(dev_embeddings, y, y_hat, outside_resp=em_resp, outside_params=domino_params)

            if args.load_trained_embedding:
                logger.log('load trained embedding model')
                domino.load_state_dict(torch.load('domino_MML_embedding.pt'))
        else:
            domino = DOMINO_DE(args, test=False, num_classes = 2)
        domino.cuda()


        '''
        ### extra from here
        domino.train()
        optimizer, scheduler = create_optimizer(args, domino)
        domino_loader = load_loaders(args, dev_embeddings, y, y_hat)
        ### test data
        test_y = torch.tensor(label_binarize(test['label'])).cuda()
        test_y_hat = torch.tensor(test['probs']).cuda()
        ### train model
        for e in range(args.epochs):
            for batch in domino_loader:
                domino.zero_grad()
                embeddings = batch[0]
                y = batch[1]
                y_hat = batch[2]
                if y.size(0) < args.domino_batch_size/2:
                    continue
                embedding_loss, mean_reg_loss, loss = domino.compute_loss(embeddings, y, y_hat)
                #print((embedding_loss,mean_reg_loss,loss))
                # print(domino.gaussian.mu)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(domino.parameters(), args.embedding_clip)
                optimizer.step()
                scheduler.step()
            if e%10 == 0:
                embedding_log_prob, y_log_prob, y_hat_log_prob, test_log = domino.predict(test_embeddings, test_y, test_y_hat)
                print(np.mean(torch.max(embedding_log_prob, dim=-1).values.tolist()))
        '''
        logger.log('train Domino MML')
        if args.domino_type == 'MML':
            domino = train_domino_MML(args, logger, domino, dev_embeddings, y, y_hat, test_embeddings, test_y, test_y_hat, em_dev_labels=em_dev_labels)
        else:
            domino = train_domino_DE(args, domino, dev_embeddings, y, y_hat)

        '''
        torch.save(domino.state_dict(), 'domino_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(
            args.y_log_likelihood_weight,
            args.y_hat_log_likelihood_weight,
            args.embedding_lr,
            args.non_embedding_lr,
            args.warmup_prop,
            args.noise_transition,
            args.mu_weight,
            args.var_weight,
            args.mu_kl_weight
        ))
        '''

    '''
    if args.domino_type == 'EM':    
        (
            weights_,
            means_,
            covariances_,
            y_probs,
            y_hat_probs,
            precisions_cholesky_,
        ) = domino.mm._get_parameters()
        with open('domino_result/original_domino_prior.json', 'w') as f:
            json.dump(weights_.tolist(), f)
        with open('domino_result/original_domino_mean.json', 'w') as f:
            json.dump(means_.tolist(), f)
        with open('domino_result/original_domino_covar.json', 'w') as f:
            json.dump(covariances_.tolist(), f)
        with open('domino_result/original_domino_y_probs.json', 'w') as f:
            json.dump(y_probs.tolist(), f)
        with open('domino_result/original_domino_y_hat_probs.json', 'w') as f:
            json.dump(y_hat_probs.tolist(), f)
    else:
        means = domino.gaussian.mu.tolist()[0]
        with open(
        "domino_result/domino_{}_mu_{}_{}_{}_{}_{}_{}_{}_{}.json".format(
            args.domino_type,
            args.y_log_likelihood_weight,
            args.y_hat_log_likelihood_weight,
            args.embedding_lr,
            args.non_embedding_lr,
            args.warmup_prop,
            args.noise_transition,
            args.mu_weight,
            args.var_weight
        ),
        "w",
        ) as f:
            json.dump(means, f)
        var = domino.gaussian.var.tolist()
        with open(
        "domino_result/domino_{}_var_{}_{}_{}_{}_{}_{}_{}_{}.json".format(
            args.domino_type,
            args.y_log_likelihood_weight,
            args.y_hat_log_likelihood_weight,
            args.embedding_lr,
            args.non_embedding_lr,
            args.warmup_prop,
            args.noise_transition,
            args.mu_weight,
            args.var_weight
        ),
        "w",
        ) as f:
            json.dump(var, f)
    '''
    #################### compute domino on validation ####################
    logger.log("compute domino membership for validation dataset")
    if args.domino_type =='EM':
        dev["log_domino_slices"], dev['resp'] = domino.predict_proba(
            data=dev, embeddings="embeddings", targets="label", pred_probs="probs", test=False
        )
        slice_prediction = dev["resp"][:]
        slice_group = np.argmax(slice_prediction, axis=1).tolist()
        dev["predicted_group"] = slice_group
        dev["resp"] = dev["resp"].tolist()
        dev["log_domino_slices"] = dev["log_domino_slices"].tolist()

    else:
        domino.eval()
        #embedding_log_prob, y_log_prob, y_hat_log_prob, dev_log = domino.predict(dev_embeddings, y, y_hat, test=False)
        dataset = domino_dataset(args, dev_embeddings, y, y_hat)
        domino_loader = DataLoader(dataset, batch_size=args.domino_batch_size, shuffle=False)
        all_dev_log = []
        all_resp = []
        for batch in domino_loader:
            embeddings = batch[0]
            batch_y = batch[1]
            batch_y_hat = batch[2]
            embedding_log_prob, y_log_prob, y_hat_log_prob, log_prob, resp = domino.predict(embeddings, batch_y, batch_y_hat, test=False)
            all_dev_log.append(log_prob)
            all_resp.append(resp)
        dev_log = torch.cat(all_dev_log, dim=0).squeeze()
        resp = torch.cat(all_resp, dim=0).squeeze()

        dev['log_domino_slices'] = dev_log.tolist()
        dev["resp"] = resp.tolist()
        dev["predicted_group"] = torch.max(resp, dim=-1).indices.tolist()

        #################### compare result with Gaussian Mixture ####################
        vscore = v_measure_score(np.array(gold_labels), np.array(dev["predicted_group"]))
        hscore = homogeneity_score(np.array(gold_labels), np.array(dev["predicted_group"]))
        cscore = completeness_score(np.array(gold_labels), np.array(dev["predicted_group"]))
        print('homogeneity score is {}, cscore is {}, vscore is {}'.format(hscore, cscore, vscore))

        vscore = v_measure_score(np.array(em_dev_labels), np.array(dev["predicted_group"]))
        hscore = homogeneity_score(np.array(em_dev_labels), np.array(dev["predicted_group"]))
        cscore = completeness_score(np.array(em_dev_labels), np.array(dev["predicted_group"]))
        print('EM homogeneity score is {}, cscore is {}, vscore is {}'.format(hscore, cscore, vscore))
        
    '''
    with open(
        "domino_result/domino_{}_dev_{}_{}_{}_{}_{}_{}_{}_{}.json".format(
            args.domino_type,
            args.y_log_likelihood_weight,
            args.y_hat_log_likelihood_weight,
            args.embedding_lr,
            args.non_embedding_lr,
            args.warmup_prop,
            args.noise_transition,
            args.mu_weight,
            args.var_weight,
        ),
        "w",
        ) as f:
            json.dump(dev, f)
    logger.log('finished computing dev slice')
    '''

    #################### compute domino on test ####################
    # eval on test
    if args.domino_type == "EM":
        test["log_domino_slices"], test['resp'] = domino.predict_proba(
            data=test, embeddings="embeddings", targets="label", pred_probs="probs", test=False
        )
        slice_prediction = test["resp"][:]
        slice_group = np.argmax(slice_prediction, axis=1).tolist()
        test["predicted_group"] = slice_group
        test["resp"] = test["resp"].tolist()
        test["log_domino_slices"] = test["log_domino_slices"].tolist()

    else:
        y = torch.tensor(flatten_y(label_binarize(test['label']))).cuda()
        y_hat = torch.tensor(test['probs']).cuda()

        embedding_log_prob, y_log_prob, y_hat_log_prob, log_prob, resp = domino.predict(test_embeddings, y, y_hat, test=True)
        
        #weighted_log_prob = test_log.clone()
        #log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        #test_log -= log_prob_norm.unsqueeze(-1)
        #test_log = torch.exp(test_log)
        test['log_domino_slices'] = log_prob.tolist()
        test['resp'] = resp.tolist()
        test["predicted_group"] = torch.max(resp, dim=-1).indices.tolist()   

        # print dev result first 
        n = 0
        for i in range(args.n_components):
            gro = grouping(i, dev)
            if gro:
                acc = compute_acc(gro)
                if acc < 0.5:
                    n += 1

        logger.log('number of dev slices with acc < 0.5 is {}'.format(n))

        n = 0
        for i in range(args.n_components):
            gro = grouping(i, dev)
            if gro:
                acc = compute_acc(gro)
                if acc < 0.5:
                    n += len(gro)
        logger.log('number of dev datapoints in slices with acc < 0.5 is {}'.format(n))

        '''
        with open(
        "domino_result/{}_{}_{}_{}_gamma{}_dev.json".format(
            args.task,
            args.noise_transition,
            args.domino_type,
            args.slice_n,
            args.y_hat_log_likelihood_weight
        ),
        "w",
        ) as f:
            json.dump(dev, f)
        '''

        # print test result 
        n = 0
        for i in range(args.n_components):
            gro = grouping(i, test)
            if gro:
                acc = compute_acc(gro)
                if acc < 0.5:
                    n += 1

        logger.log('number of test slices with acc < 0.5 is {}'.format(n))

        n = 0
        for i in range(args.n_components):
            gro = grouping(i, test)
            if gro:
                acc = compute_acc(gro)
                if acc < 0.5:
                    n += len(gro)
        logger.log('number of test datapoints in slices with acc < 0.5 is {}'.format(n))


    logger.log('finish computing slices on test')

    '''
    with open(
        "domino_result/{}_{}_{}_{}_gamma{}_test.json".format(
            args.task,
            args.noise_transition,
            args.domino_type,
            args.slice_n,
            args.y_hat_log_likelihood_weight
        ),
        "w",
        ) as f:
            json.dump(test, f)
    '''
