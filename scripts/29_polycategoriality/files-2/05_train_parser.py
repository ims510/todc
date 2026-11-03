"""Stage 5: train a small dependency parser.

============================================================================
Parser choice and motivation
============================================================================
We use a Dozat & Manning (2017) biaffine arc-factored parser, implemented
from scratch in PyTorch (~200 lines). Why this choice:

  1. Small footprint: ~2-3M parameters with the defaults below, so the
     model file is ~10MB and training fits on CPU. No BERT / no large
     pretrained encoders, so training is decoupled from massive pretrained
     features and the comparison between the original and collapsed label
     sets is clean.

  2. Standard baseline: the biaffine architecture is the canonical small
     UD parser and is what every recent paper compares against. Reasonable
     LAS on UD English-EWT (~80%+) without pretrained word vectors.

  3. Greedy decoding: for each dependent token we take argmax over heads,
     then argmax over labels conditional on the predicted head. UAS/LAS
     are per-token metrics, so we don't need the prediction to form a
     valid tree to score it. That keeps the decoder dead simple.

  4. Pure POS + word features: this is exactly the regime where the
     "deterministic given head_upos" claim from stage 3 is testable - the
     model has direct access to UPOS so it should easily learn the obl/nmod
     split when both labels are kept.

Alternatives considered:
  - SuPar (pip): same architecture, more polished, but a heavier dep tree.
  - Transition-based perceptron: smaller still, but noisier and decoding
    is non-trivial to implement cleanly.
  - Trankit / Stanza: BERT-based, ~100x more parameters. Overkill here.

Usage:
    # train on uncollapsed
    python 05_train_parser.py train <train.conllu> <dev.conllu> <model_dir> [--epochs N]

    # predict
    python 05_train_parser.py predict <model_dir> <test.conllu> <out.conllu>
============================================================================
"""
import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import parse_conllu


# ----------------------------- vocab ---------------------------------------

PAD, UNK, ROOT_TOK = "<pad>", "<unk>", "<root>"


class Vocab:
    def __init__(self, items, min_count=1, specials=(PAD, UNK)):
        c = Counter(items)
        self.itos = list(specials) + [w for w, n in c.most_common() if n >= min_count and w not in specials]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, w):
        return self.stoi.get(w, self.stoi.get(UNK, 0))

    def to_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        v = cls.__new__(cls)
        v.itos = d["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


def build_vocabs(sents, max_words=30000):
    words, pos, labels = [], [], []
    for s in sents:
        for tok in s:
            words.append(tok["form"].lower())
            pos.append(tok["upos"])
            labels.append(tok["deprel"])
    word_v = Vocab(words, min_count=2, specials=(PAD, UNK, ROOT_TOK))
    # cap word vocab
    if len(word_v) > max_words + 3:
        word_v.itos = word_v.itos[: max_words + 3]
        word_v.stoi = {w: i for i, w in enumerate(word_v.itos)}
    pos_v = Vocab(pos, specials=(PAD, UNK, ROOT_TOK))
    label_v = Vocab(labels, specials=(PAD,))
    return word_v, pos_v, label_v


# ----------------------------- model ---------------------------------------

class BiaffineParser(nn.Module):
    def __init__(self, n_words, n_pos, n_labels,
                 w_dim=64, p_dim=32, lstm_h=128, mlp_h=64, n_lstm=1, dropout=0.33):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, w_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(n_pos, p_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(w_dim + p_dim, lstm_h, num_layers=n_lstm,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_lstm > 1 else 0.0)
        H = 2 * lstm_h
        self.arc_h = nn.Linear(H, mlp_h)
        self.arc_d = nn.Linear(H, mlp_h)
        self.lab_h = nn.Linear(H, mlp_h)
        self.lab_d = nn.Linear(H, mlp_h)
        # biaffine arc weight: (mlp_h+1, mlp_h)  [+1 for bias term on head side]
        self.U_arc = nn.Parameter(torch.zeros(mlp_h + 1, mlp_h))
        nn.init.xavier_uniform_(self.U_arc)
        # biaffine label weight: per-label bilinear (mlp_h+1, n_labels, mlp_h+1)
        self.U_lab = nn.Parameter(torch.zeros(mlp_h + 1, n_labels, mlp_h + 1))
        nn.init.xavier_uniform_(self.U_lab.view(mlp_h + 1, -1))

    def encode(self, words, pos):
        x = torch.cat([self.word_emb(words), self.pos_emb(pos)], dim=-1)
        x = self.dropout(x)
        h, _ = self.lstm(x)
        return self.dropout(h)

    def arc_scores(self, h):
        # h: (B, T, H)
        h_a = F.relu(self.arc_h(h))   # (B, T, M)
        d_a = F.relu(self.arc_d(h))   # (B, T, M)
        ones = torch.ones(*h_a.shape[:-1], 1, device=h_a.device)
        h_a1 = torch.cat([h_a, ones], dim=-1)  # (B, T, M+1)
        # score[b, dep, head] = h_a1[b, head, :] @ U_arc @ d_a[b, dep, :]
        # einsum: 'bhi,ij,bdj->bdh'
        return torch.einsum("bhi,ij,bdj->bdh", h_a1, self.U_arc, d_a)

    def label_scores(self, h, heads):
        h_l = F.relu(self.lab_h(h))   # (B, T, M)
        d_l = F.relu(self.lab_d(h))   # (B, T, M)
        ones_h = torch.ones(*h_l.shape[:-1], 1, device=h_l.device)
        ones_d = torch.ones(*d_l.shape[:-1], 1, device=d_l.device)
        h_l1 = torch.cat([h_l, ones_h], dim=-1)  # (B, T, M+1)
        d_l1 = torch.cat([d_l, ones_d], dim=-1)  # (B, T, M+1)
        # gather head representation per dep:  h_sel[b, d, :] = h_l1[b, heads[b,d], :]
        idx = heads.unsqueeze(-1).expand(-1, -1, h_l1.size(-1))
        h_sel = torch.gather(h_l1, 1, idx)  # (B, T, M+1)
        # score[b, d, label] = h_sel[b,d,:] @ U_lab[:, label, :] @ d_l1[b,d,:]
        return torch.einsum("bdi,ilj,bdj->bdl", h_sel, self.U_lab, d_l1)


# ----------------------------- data plumbing -------------------------------

def encode_sentence(sent, word_v, pos_v, label_v, with_root=True):
    """Returns (words_ids, pos_ids, head_ids, label_ids, length).

    A virtual ROOT is prepended at index 0 so heads can point at 0 = ROOT.
    """
    words, pos, heads, labels = [], [], [], []
    if with_root:
        words.append(word_v.encode(ROOT_TOK))
        pos.append(pos_v.encode(ROOT_TOK))
        heads.append(0)
        labels.append(label_v.encode(PAD))  # pad label for ROOT
    for tok in sent:
        words.append(word_v.encode(tok["form"].lower()))
        pos.append(pos_v.encode(tok["upos"]))
        heads.append(max(0, tok["head"]))
        labels.append(label_v.encode(tok["deprel"]))
    return words, pos, heads, labels


def make_batch(batch, word_v, pos_v, label_v, device):
    encoded = [encode_sentence(s, word_v, pos_v, label_v) for s in batch]
    T = max(len(e[0]) for e in encoded)
    B = len(encoded)
    words = torch.zeros(B, T, dtype=torch.long)
    pos = torch.zeros(B, T, dtype=torch.long)
    heads = torch.zeros(B, T, dtype=torch.long)
    labels = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, (w, p, h, l) in enumerate(encoded):
        L = len(w)
        words[i, :L] = torch.tensor(w)
        pos[i, :L] = torch.tensor(p)
        heads[i, :L] = torch.tensor(h)
        labels[i, :L] = torch.tensor(l)
        mask[i, 1:L] = True  # ignore ROOT position when computing loss
    return (
        words.to(device),
        pos.to(device),
        heads.to(device),
        labels.to(device),
        mask.to(device),
    )


# ----------------------------- train / predict -----------------------------

def train_loop(train_sents, dev_sents, model, word_v, pos_v, label_v,
               epochs, batch_size, lr, device, log_path):
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.9))
    log = open(log_path, "w")
    best_las = 0.0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        # shuffle
        order = torch.randperm(len(train_sents)).tolist()
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            batch = [train_sents[i] for i in order[start : start + batch_size]]
            words, pos, gold_heads, gold_labels, mask = make_batch(
                batch, word_v, pos_v, label_v, device
            )
            h = model.encode(words, pos)
            arc_logits = model.arc_scores(h)  # (B, T, T)  [b, dep, head]
            lab_logits = model.label_scores(h, gold_heads)  # (B, T, n_labels)
            arc_loss = F.cross_entropy(
                arc_logits[mask], gold_heads[mask]
            )
            lab_loss = F.cross_entropy(
                lab_logits[mask], gold_labels[mask]
            )
            loss = arc_loss + lab_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        # eval on dev
        uas, las = evaluate(model, dev_sents, word_v, pos_v, label_v, device)
        msg = f"epoch {ep:3d}  loss {total_loss/max(1,n_batches):.4f}  dev UAS {uas:.4f}  LAS {las:.4f}"
        print(msg)
        log.write(msg + "\n")
        log.flush()
        if las > best_las:
            best_las = las
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    log.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_las


@torch.no_grad()
def predict_sentences(model, sents, word_v, pos_v, label_v, device, batch_size=32):
    model.eval()
    preds = []  # list of (heads_per_token, labels_per_token) per sentence
    for start in range(0, len(sents), batch_size):
        batch = sents[start : start + batch_size]
        words, pos, _, _, mask = make_batch(batch, word_v, pos_v, label_v, device)
        h = model.encode(words, pos)
        arc_logits = model.arc_scores(h)  # (B, T, T)
        # block self-attachment by setting diag to -inf (a token can't be its own head)
        T = arc_logits.size(1)
        diag = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0)
        arc_logits = arc_logits.masked_fill(diag, float("-inf"))
        pred_heads = arc_logits.argmax(dim=-1)  # (B, T)
        lab_logits = model.label_scores(h, pred_heads)  # (B, T, n_labels)
        pred_labels = lab_logits.argmax(dim=-1)  # (B, T)
        for i, sent in enumerate(batch):
            L = len(sent)
            ph = pred_heads[i, 1 : L + 1].tolist()
            pl = pred_labels[i, 1 : L + 1].tolist()
            preds.append((ph, pl))
    return preds


def evaluate(model, sents, word_v, pos_v, label_v, device):
    preds = predict_sentences(model, sents, word_v, pos_v, label_v, device)
    n = 0
    uas = 0
    las = 0
    for sent, (ph, pl) in zip(sents, preds):
        for tok, h, l in zip(sent, ph, pl):
            if tok["deprel"] == "punct":
                continue  # standard UD eval excludes punct
            n += 1
            if h == tok["head"]:
                uas += 1
                if label_v.itos[l] == tok["deprel"]:
                    las += 1
    return uas / max(1, n), las / max(1, n)


# ----------------------------- I/O helpers ---------------------------------

def write_predictions(input_path, output_path, sents, preds, label_v):
    """Re-emit input_path with HEAD and DEPREL columns replaced by preds."""
    si = -1
    pred_iter = iter(preds)
    cur_sent = None
    cur_pred = None
    cur_idx = 0
    with open(input_path, encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                if not line.strip() and cur_sent is not None:
                    cur_sent = None  # sentence ended
                fout.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 10 or "-" in fields[0] or "." in fields[0]:
                fout.write(line)
                continue
            if cur_sent is None:
                si += 1
                cur_sent = sents[si]
                cur_pred = preds[si]
                cur_idx = 0
            ph, pl = cur_pred
            fields[6] = str(ph[cur_idx])
            fields[7] = label_v.itos[pl[cur_idx]]
            cur_idx += 1
            fout.write("\t".join(fields) + "\n")


def save_model(model_dir, model, word_v, pos_v, label_v, config):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    with open(os.path.join(model_dir, "vocabs.json"), "w") as f:
        json.dump(
            {
                "word": word_v.to_dict(),
                "pos": pos_v.to_dict(),
                "label": label_v.to_dict(),
                "config": config,
            },
            f,
        )


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "vocabs.json")) as f:
        d = json.load(f)
    word_v = Vocab.from_dict(d["word"])
    pos_v = Vocab.from_dict(d["pos"])
    label_v = Vocab.from_dict(d["label"])
    cfg = d["config"]
    model = BiaffineParser(len(word_v), len(pos_v), len(label_v), **cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    return model, word_v, pos_v, label_v


# ----------------------------- main ----------------------------------------

def cmd_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    train_sents = list(parse_conllu(args.train))
    dev_sents = list(parse_conllu(args.dev))
    print(f"Train sents: {len(train_sents)}  Dev sents: {len(dev_sents)}")
    word_v, pos_v, label_v = build_vocabs(train_sents)
    print(f"|V_word|={len(word_v)}  |V_pos|={len(pos_v)}  |V_label|={len(label_v)}")
    config = dict(w_dim=64, p_dim=32, lstm_h=128, mlp_h=64, n_lstm=1, dropout=0.33)
    model = BiaffineParser(len(word_v), len(pos_v), len(label_v), **config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")
    log_path = os.path.join(args.model_dir, "training_log.txt")
    os.makedirs(args.model_dir, exist_ok=True)
    best = train_loop(train_sents, dev_sents, model,
                      word_v, pos_v, label_v,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, device=device, log_path=log_path)
    save_model(args.model_dir, model, word_v, pos_v, label_v, config)
    print(f"Best dev LAS: {best:.4f}")
    print(f"Saved model -> {args.model_dir}")


def cmd_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, word_v, pos_v, label_v = load_model(args.model_dir, device)
    sents = list(parse_conllu(args.input))
    preds = predict_sentences(model, sents, word_v, pos_v, label_v, device)
    write_predictions(args.input, args.output, sents, preds, label_v)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("train")
    pt.add_argument("dev")
    pt.add_argument("model_dir")
    pt.add_argument("--epochs", type=int, default=10)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=2e-3)
    pt.add_argument("--cpu", action="store_true")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict")
    pp.add_argument("model_dir")
    pp.add_argument("input")
    pp.add_argument("output")
    pp.add_argument("--cpu", action="store_true")
    pp.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)
