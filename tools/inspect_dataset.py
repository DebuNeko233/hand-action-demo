from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from config import CLASS_NAMES,DATASET_DIR,SEQUENCE_LENGTH,INPUT_SIZE
for c in CLASS_NAMES:
    files=list((DATASET_DIR/c).glob("*.npy")); bad=[]
    for p in files:
        try:
            x=np.load(p)
            if x.shape!=(SEQUENCE_LENGTH,INPUT_SIZE) or not np.isfinite(x).all(): bad.append(p.name)
        except Exception: bad.append(p.name)
    print(f"{c:8s}: {len(files):4d} samples, bad={len(bad)}")
