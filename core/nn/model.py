import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(
            -1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class NeuralNet(nn.Module):
    def __init__(self,
                 embedding_matrix,
                 n_classes=9,
                 hidden_size=64,
                 maxlen=150,
                 linear_size=100):
        super(NeuralNet, self).__init__()

        n_features, embed_size = embedding_matrix.shape
        self.embedding = nn.Embedding(n_features, embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.2)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(
            hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.attn = Attention(hidden_size * 2, maxlen)
        self.linear = nn.Linear(hidden_size * 2, linear_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(linear_size, n_classes)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        h_attn = self.attn(h_gru)
        linear = self.relu(self.linear(h_attn))
        h_drop = self.dropout(linear)
        out = self.linear2(h_drop)

        return out


def train_and_validate(X,
                       y,
                       X_test,
                       y_test,
                       embedding_matrix,
                       logger,
                       train_epochs,
                       device="cpu"):
    train_batch_size = 128
    val_batch_size = 512
    test_batch_size = 512

    x_test_cuda = torch.tensor(X_test, dtype=torch.long).to(device)
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=test_batch_size, shuffle=False)
    test_preds = []

    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (trn_index, val_index) in enumerate(fold.split(X, y)):
        X_train = torch.tensor(X[trn_index], dtype=torch.long).to(device)
        X_val = torch.tensor(X[val_index], dtype=torch.long).to(device)
        y_train = torch.tensor(
            y[trn_index, np.newaxis], dtype=torch.long).to(device)
        y_val = torch.tensor(
            y[val_index, np.newaxis], dtype=torch.long).to(device)

        model = NeuralNet(embedding_matrix)
        model.to(device)

        logger.info(f"Fold {i + 1}")
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_epochs)

        train = torch.utils.data.TensorDataset(X_train, y_train)
        valid = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=train_batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            valid, batch_size=val_batch_size, shuffle=False)
        best_score = -np.inf

        for epoch in range(train_epochs):
            model.train()
            avg_loss = 0.

            for (x_batch, y_batch) in train_loader:
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()
            valid_preds_epoch = np.zeros((X_val.size(0), len(y.unique())))
            avg_val_loss = 0.

            for j, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch).detach()
                avg_val_loss += loss_fn(
                    y_pred, y_batch.squeeze()).item() / len(valid_loader)
                valid_preds_epoch[j * val_batch_size:(
                    j + 1) * val_batch_size] = F.softmax(y_pred).cpu().numpy()

            f1 = f1_score(
                y_val.cpu().numpy(),
                np.argmax(valid_preds_epoch, axis=1),
                average="macro")
            logger.info(
                f"Epoch {epoch + 1} / {train_epochs} loss={avg_loss:.4f} val_loss={avg_val_loss:.4f} F1={f1:.4f}"
            )
            path = Path("bin/")
            path.mkdir(exist_ok=True)
            filename = path / f"best{i}.pt"
            if f1 > best_score:
                logger.info(f"Save model at epoch {epoch+1}")
                torch.save(model.state_dict(), filename)
                best_score = f1

            scheduler.step()

        avg_val_loss = 0.
        valid_preds_fold = np.zeros((X_val.size(0), len(y.unique())))
        test_preds_fold = np.zeros((X_test.shape[0], len(y.unique())))
        model.load_state_dict(torch.load(filename))
        model.eval()
        for j, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            avg_val_loss += loss_fn(
                y_pred, y_batch.squeeze()).item() / len(valid_loader)
            valid_preds_fold[j * val_batch_size:(j + 1) *
                             val_batch_size] = F.softmax(y_pred).cpu().numpy()
        f1 = f1_score(
            y_val.cpu().numpy(),
            np.argmax(valid_preds_fold, axis=1),
            average="macro")
        logger.info(f"Fold {i+1} Best Val F1={f1:.4f} Loss={avg_val_loss:.4f}")

        for k, (x_batch, ) in enumerate(test_loader):
            y_pred = model(x_batch).detach()
            test_preds_fold[k * test_batch_size:(k + 1) *
                            test_batch_size] = F.softmax(y_pred).cpu().numpy()
        test_preds.append(test_preds_fold)
    y_pred = np.zeros((y_test.shape[0], len(y_test.unique())))
    for t in test_preds:
        y_pred += t / 5
    f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average="macro")
    logger.info(f"Total F1={f1:.4f}")
