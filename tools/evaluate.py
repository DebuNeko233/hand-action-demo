from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from config import CLASS_NAMES, DATASET_DIR, MODEL_PATH
from src.dataset import HandSequenceDataset
from src.predictor import Predictor

def main() -> None:
    ds=HandSequenceDataset(DATASET_DIR,CLASS_NAMES,augment=False); predictor=Predictor(MODEL_PATH); ys=[]; ps=[]
    for x,y in ds:
        r=predictor.predict(x.numpy()); ys.append(int(y)); ps.append(r.class_id)
    print(classification_report(ys,ps,labels=list(range(len(CLASS_NAMES))),target_names=CLASS_NAMES,zero_division=0)); print(confusion_matrix(ys,ps))
if __name__=="__main__": main()
