import torch
from src.model import HandActionLSTM

def test_model_shape():
    m=HandActionLSTM()
    assert m(torch.randn(4,30,66)).shape==(4,6)
