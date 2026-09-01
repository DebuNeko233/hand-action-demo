from __future__ import annotations
import sys, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from config import *
from src.dataset import HandSequenceDataset
from src.model import HandActionLSTM

def main() -> None:
    seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    base = HandSequenceDataset(DATASET_DIR, CLASS_NAMES, augment=False)
    if len(base) < 12: raise RuntimeError("Dataset is too small. Collect samples first.")
    idx=np.arange(len(base)); np.random.shuffle(idx); split=int(.8*len(idx)); train_idx,val_idx=idx[:split],idx[split:]
    train_ds=HandSequenceDataset(DATASET_DIR, CLASS_NAMES, augment=True)
    train_loader=DataLoader(Subset(train_ds, train_idx.tolist()), batch_size=32, shuffle=True)
    val_loader=DataLoader(Subset(base, val_idx.tolist()), batch_size=32)
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu")
    model=HandActionLSTM(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYERS,NUM_CLASSES).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=torch.nn.CrossEntropyLoss(); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=4,factor=.5)
    best=-1.0; hist={"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
    OUTPUT_DIR.mkdir(exist_ok=True); MODEL_PATH.parent.mkdir(exist_ok=True)
    for epoch in range(1,51):
        model.train(); tl=tc=tn=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step()
            tl += loss.item()*len(y); tc += (logits.argmax(1)==y).sum().item(); tn += len(y)
        model.eval(); vl=vc=vn=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); logits=model(x); loss=loss_fn(logits,y)
                vl+=loss.item()*len(y); vc+=(logits.argmax(1)==y).sum().item(); vn+=len(y)
        trl,tra,vloss,vacc=tl/tn,tc/tn,vl/vn,vc/vn; scheduler.step(vloss)
        for k,v in [("train_loss",trl),("train_acc",tra),("val_loss",vloss),("val_acc",vacc)]: hist[k].append(v)
        print(f"Epoch {epoch:02d} train_loss={trl:.4f} train_acc={tra:.3f} val_loss={vloss:.4f} val_acc={vacc:.3f}")
        if vacc>best:
            best=vacc; torch.save({"model_state_dict":model.state_dict(),"class_names":CLASS_NAMES,"input_size":INPUT_SIZE,"sequence_length":SEQUENCE_LENGTH,"hidden_size":HIDDEN_SIZE,"num_layers":NUM_LAYERS},MODEL_PATH)
    plt.figure(); plt.plot(hist["train_loss"],label="train loss"); plt.plot(hist["val_loss"],label="val loss"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"training_curve.png"); plt.close()
    ckpt=torch.load(MODEL_PATH,map_location=device); model.load_state_dict(ckpt["model_state_dict"]); model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in val_loader:
            p=model(x.to(device)).argmax(1).cpu().numpy(); ps.extend(p); ys.extend(y.numpy())
    print(classification_report(ys,ps,labels=list(range(NUM_CLASSES)),target_names=CLASS_NAMES,zero_division=0)); print(confusion_matrix(ys,ps,labels=list(range(NUM_CLASSES))))

if __name__ == "__main__": main()
