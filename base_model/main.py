import argparse
import sys
import os
import json

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    RobertaTokenizer,
    RobertaModel,
    ElectraTokenizer,
    ElectraModel,
)

from data_process import collect_tsv_data, load_dataloaders
from model import binary_classifier

import transformers

transformers.logging.set_verbosity_error()


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
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


def construct_optimizer(args, model, num_train_examples):
    no_weight_decay = ["LayerNorm.weight", "bias"]
    optimized_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(np in n for np in no_weight_decay)
            ],
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(np in n for np in no_weight_decay)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    # Implements linear decay of the learning rate.
    # default to be AdamW based on tensorflow, AdamWeightDecayOptimizer
    # parameters are using default
    optimizer = torch.optim.Adam(optimized_parameters, lr=args.learning_rate)

    num_training_steps = int(
        args.epochs * num_train_examples / (args.batch_size * args.accumulate_steps)
    )
    num_warmup_steps = int(args.warm_up_proportion * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    return optimizer, scheduler


def main(args, logger):
    # if args.model_type == "bert":
    #    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # elif args.model_type == "roberta":
    #    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    # elif args.model_type == "electra":
    #    tokenizer = ElectraTokenizer.from_pretrained(
    #        "google/electra-large-discriminator"
    #    )
    logger.log(str(args))

    # load data
    logger.log("import data")
    num_train_examples = len(collect_tsv_data(args)[0])
    train_loader, dev_loader, test_loader = load_dataloaders(args)
    if args.task == "MNLI":
        num_classes = 3
    elif args.task == "SST-5":
        num_classes = 5
    else:
        num_classes = 2

    # load model
    logger.log("initialize model")
    if args.model_type == "bert":
        encoder = BertModel.from_pretrained("bert-large-uncased").cuda()
    elif args.model_type == "roberta":
        encoder = RobertaModel.from_pretrained("roberta-large").cuda()
    elif args.model_type == "electra":
        encoder = ElectraModel.from_pretrained(
            "google/electra-large-discriminator"
        ).cuda()
    classifier = binary_classifier(args, num_classes, encoder).cuda()

    # loader optimizer
    optimizer, scheduler = construct_optimizer(args, classifier, num_train_examples)

    logger.log("start training")
    classifier.zero_grad()
    num_steps = 0
    best_accuracy = -float("inf")
    for epoch in range(args.epochs):
        # training part
        loss_value = 0
        classifier.train()
        for batch in train_loader:
            input_ids = batch[0].cuda()
            attn_mask = batch[1].cuda()
            token_type_ids = batch[2].cuda()
            labels = batch[3].cuda()

            logits = classifier(
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids,
            )

            loss = classifier.compute_loss(logits, labels)
            loss_value += loss.item()

            loss.backward()
            num_steps += 1

            if (num_steps + 1) % args.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                classifier.zero_grad()

            if (num_steps + 1) % args.logging_steps == 0:
                logger.log(
                    "loss value for steps {} to {} is {}".format(
                        num_steps - args.logging_steps, num_steps, loss_value
                    )
                )
                loss_value = 0

        # evaluation part
        classifier.eval()
        logger.log("evaluate validation dataset at epoch {}".format(epoch))
        accuracy, predictions, probs = evaluate(args, classifier, dev_loader, "dev")
        logger.log("validation accuracy at epoch {} is {}".format(epoch, accuracy))
        if accuracy > best_accuracy:
            logger.log("save trained model to {}".format(args.model_path))
            torch.save(classifier.state_dict(), args.model_path)
            with open(
                args.task + "_{}_{}_prediction.json".format("dev", args.model_type), "w"
            ) as f:
                json.dump(predictions, f)
            with open(
                args.task + "_{}_{}_probs.json".format("dev", args.model_type), "w"
            ) as f:
                json.dump(probs, f)
            logger.log("best accuracy {} -------> {}".format(best_accuracy, accuracy))
            best_accuracy = accuracy
            accuracy, _, _ = evaluate(args, classifier, test_loader, "test")
            logger.log("test accuracy is {}".format(accuracy))

    classifier.eval()
    classifier.load_state_dict(torch.load(args.model_path))
    accuracy, _, _ = evaluate(args, classifier, dev_loader, "dev")
    logger.log("dev accuracy is {}".format(accuracy))
    accuracy, _, _ = evaluate(args, classifier, test_loader, "test")
    logger.log("test accuracy is {}".format(accuracy))


def evaluate(args, model, data_loader, mode):
    model.eval()
    predictions = []
    probs = []
    num_correct = 0
    num_total = 0
    softmax_fc = nn.Softmax(dim=1)
    for batch in data_loader:
        input_ids = batch[0].cuda()
        attn_mask = batch[1].cuda()
        token_type_ids = batch[2].cuda()
        labels = batch[3].cuda()

        logits = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )

        # save results
        prediction = model.predict(logits).tolist()
        predictions += prediction
        prob = softmax_fc(logits).tolist()
        probs += prob

        for p, l in zip(prediction, labels):
            if p == l:
                num_correct += 1
        num_total += batch[0].size(0)

    with open(args.task + "_{}_{}_probs.json".format(mode, args.model_type), "w") as f:
        json.dump(probs, f)

    accuracy = num_correct / num_total

    return accuracy, predictions, probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="QQP")
    parser.add_argument("--data_dir", type=str, default="glue_data/")
    parser.add_argument("--toy", type=bool, default=False)
    parser.add_argument("--log_path", type=str, default="")
    parser.add_argument(
        "--model_type", type=str, default="bert", help="bert, roberta, electra"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warm_up_proportion", type=float, default=0.1)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.data_dir += args.task + "/"
    args.log_path = args.task + "_{}.log".format(args.model_type)
    args.model_path = (
        "trained_models/" + args.task + "_{}_meanpool.pt".format(args.model_type)
    )

    logger = Logger(args.log_path)

    main(args, logger)
