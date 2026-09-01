import numpy as np
from src.dataset import HandSequenceDataset

def test_dataset(tmp_path):
    for c in ["idle","grab"]: (tmp_path/c).mkdir()
    np.save(tmp_path/"grab"/"000001.npy",np.zeros((30,66),dtype=np.float32))
    ds=HandSequenceDataset(tmp_path,["idle","grab"])
    x,y=ds[0]
    assert tuple(x.shape)==(30,66) and int(y)==1
