"""Single-split training for all models. Selects on balanced accuracy."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from features import TIME_FRAMES, N_MELS, mel_spectrogram, mfcc_mean  # noqa: E402
from models import get  # noqa: E402
from preprocess import load_wav  # noqa: E402

LABELS = {"normal": 0, "abnormal": 1}
SERVICE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SERVICE_DIR / "artifacts" / "cardio_model.pt"


def gather(data):
    recs = []
    for cls, y in LABELS.items():
        for p in (Path(data) / cls).glob("*.wav"):
            recs.append((str(p), y))
    assert recs, f"no wavs under {data}"
    return recs


class SpecDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        m = mel_spectrogram(load_wav(p))  # (64,128) @2000Hz/2s
        return torch.from_numpy(m[None]).float(), torch.tensor(y).long()


def evaluate(model, loader, dev):
    model.eval()
    P, L = [], []
    with torch.no_grad():
        for x, y in loader:
            P += model(x.to(dev)).argmax(1).cpu().tolist()
            L += y.tolist()
    sens = recall_score(L, P, pos_label=1, zero_division=0)
    spec = recall_score(L, P, pos_label=0, zero_division=0)
    return sens, spec, 0.5 * (sens + spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.getenv("DATA_DIR", "dataset/train"))
    ap.add_argument("--model", default="cnn_transformer", choices=["cnn", "cnn_transformer", "dt", "svm"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=os.getenv("MODEL_OUT", str(DEFAULT_OUT)))
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", ""))
    ap.add_argument("--weighted", action="store_true", help="class-weighted cross-entropy")
    ap.add_argument("--audit", action="store_true", help="run the v2 quality audit before training")
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.audit:
        from quality import audit
        audit(a.data)

    items = gather(a.data)
    paths = [p for p, _ in items]
    labels = [y for _, y in items]
    tr_p, te_p, tr_y, te_y = train_test_split(paths, labels, test_size=0.2,
                                              stratify=labels, random_state=42)
    # quick classical path on MFCC-mean (no torch loop)
    if a.model in ("dt", "svm"):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        Xtr = np.stack([mfcc_mean(load_wav(p)) for p in tr_p])
        Xte = np.stack([mfcc_mean(load_wav(p)) for p in te_p])
        clf = DecisionTreeClassifier(random_state=42) if a.model == "dt" else SVC(kernel="rbf")
        clf.fit(Xtr, tr_y)
        r = recall_score(te_y, clf.predict(Xte), pos_label=1, zero_division=0)
        print(f"{a.model} recall_abnormal={r:.4f} n_train={len(tr_p)}")
        import pickle
        Path(a.out).with_suffix(".pkl").write_bytes(pickle.dumps(clf))
        return

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = DataLoader(SpecDS(list(zip(tr_p, tr_y))), batch_size=a.batch, shuffle=True)
    te = DataLoader(SpecDS(list(zip(te_p, te_y))), batch_size=a.batch)
    model = get(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if a.weighted:
        cnt = np.bincount(np.asarray(tr_y), minlength=2).astype(float)
        w = torch.tensor(cnt.sum() / (2.0 * cnt), dtype=torch.float32, device=dev)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        print("class weights:", [round(v, 3) for v in w.tolist()])
    else:
        lossf = torch.nn.CrossEntropyLoss()
    try:
        import mlflow
        if a.tracking_uri:
            mlflow.set_tracking_uri(a.tracking_uri)
        mlflow.set_experiment("cardio-audio")
        mlflow.start_run(run_name=a.model)
        mlflow.log_params({"model": a.model, "sr": 2000, "window": "2s"})
        use_mlflow = True
    except Exception:
        use_mlflow = False
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        sens, spec, bal = evaluate(model, te, dev)
        print(f"epoch {ep + 1}/{a.epochs} loss={tot / max(len(tr), 1):.4f} "
              f"sens={sens:.4f} spec={spec:.4f} bal={bal:.4f}")
        if use_mlflow:
            mlflow.log_metrics({"loss": tot / max(len(tr), 1), "sensitivity": sens,
                                "specificity": spec, "balanced_acc": bal}, step=ep)
        if bal > best:
            best = bal
            torch.save(model.state_dict(), a.out)
    if use_mlflow:
        mlflow.log_metric("best_balanced_acc", best)
        mlflow.end_run()
    print(f"best balanced_acc={best:.4f} -> {a.out}")
    # ONNX export (best torch model, same Mel input -> audio-engine container)
    try:
        m = get(a.model)
        m.load_state_dict(torch.load(a.out, map_location="cpu"))
        m.eval()
        # Legacy exporter keeps the service image free of the onnxscript dep.
        torch.onnx.export(m, torch.randn(1, 1, N_MELS, TIME_FRAMES),
                          Path(a.out).with_suffix(".onnx"),
                          input_names=["mel"], output_names=["logits"], dynamo=False)
        print("onnx ->", Path(a.out).with_suffix(".onnx"))
    except Exception as e:
        print("onnx skipped:", e)


if __name__ == "__main__":
    main()
