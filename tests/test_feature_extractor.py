import numpy as np
from src.feature_extractor import StatefulFeatureExtractor

def hand(offset=(0,0,0)):
    pts=np.zeros((21,3),dtype=np.float32)
    for i in range(21): pts[i]=np.array([i*.01,i*.005,i*.002],dtype=np.float32)+np.array(offset,dtype=np.float32)
    return pts

def test_shape_and_finite():
    f=StatefulFeatureExtractor().extract(hand())
    assert f.shape==(66,) and np.isfinite(f).all()

def test_translation_invariance_of_local_pose():
    a=StatefulFeatureExtractor().extract(hand((0,0,0)))[:63]
    b=StatefulFeatureExtractor().extract(hand((2,3,4)))[:63]
    assert np.allclose(a,b,atol=1e-5)
