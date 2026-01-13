import os, json, random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from callivision.config import load_cfg
from callivision.data.dataset import build_loaders
from callivision.data.transforms import build_transforms
from callivision.models.resnet import build_resnet18

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())
    return accuracy_score(ys, ps)

def main(cfg_path: str = "configs/baseline.yaml"):
    cfg = load_cfg(cfg_path)
    set_seed(cfg["seed"])

    device = torch.device(
        "cuda" if (cfg.get("device") == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    tfm = build_transforms(cfg["data"]["img_size"])
    train_loader, val_loader, class_to_idx = build_loaders(
        cfg["data"]["root"],
        tfm,
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
    )

    # num_classes 优先从数据集推断，避免配置写错
    num_classes = len(class_to_idx)

    model = build_resnet18(num_classes).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    best_acc = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg['train']['epochs']}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

        print(f"[epoch {epoch+1}] val_acc={val_acc:.4f} best={best_acc:.4f}")

if __name__ == "__main__":
    main()
