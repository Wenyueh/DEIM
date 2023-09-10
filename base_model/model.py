import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizer
import argparse
from data_process import load_dataloaders


class binary_classifier(nn.Module):
    def __init__(self, args, num_classes, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.num_classes = num_classes
        self.hidden_size = self.encoder.config.hidden_size
        self.linear_layer = nn.Linear(self.hidden_size, self.num_classes, bias=True)
        self.dropout = nn.Dropout(self.args.dropout_rate)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.args.model_type == "bert":
            embeddings = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )[0].mean(dim=1)
        else:
            embeddings = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )[0].mean(dim=1)
        embeddings = self.dropout(embeddings)
        logits = self.linear_layer(embeddings)
        return logits

    def compute_loss(self, logits, labels):
        loss_value = self.loss(logits, labels)
        return loss_value

    def predict(self, logits):
        assert logits.dim() == 2
        return torch.argmax(logits, dim=1)


class sentence_bert_classifier(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.hidden_size = self.args.hidden_size
        self.linear_layer = nn.Linear(self.hidden_size, 2, bias=True)
        self.dropout = nn.Dropout(self.args.dropout_rate)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        embeddings = self.dropout(embeddings)
        logits = self.linear_layer(embeddings)
        return logits

    def compute_loss(self, logits, labels):
        loss_value = self.loss(logits, labels)
        return loss_value

    def predict(self, logits):
        assert logits.dim() == 2
        return torch.argmax(logits, dim=1)


def evaluate(model, data_loader):
    model.eval()
    predictions = []
    num_correct = 0
    num_total = 0
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

        prediction = model.predict(logits).tolist()
        predictions += prediction
        for p, l in zip(prediction, labels):
            if p == l:
                num_correct += 1
        num_total += batch[0].size(0)
    accuracy = num_correct / num_total
    print(accuracy)

    return accuracy, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="glue_data/SST-2/")
    parser.add_argument("--toy", type=bool, default=True)
    parser.add_argument("--task", type=str, default="SST-2")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    encoder = AutoModel.from_pretrained("bert-base-uncased").cuda()
    classifier = binary_classifier(args, encoder).cuda()

    train_loader, dev_loader, test_loader = load_dataloaders(args)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    classifier.zero_grad()
    for e in range(100):
        loss_value = 0
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
            optimizer.step()
            classifier.zero_grad()
        # print(loss_value)
        loss_value = 0

        accuracy, _ = evaluate(classifier, train_loader)
        # print(accuracy)
