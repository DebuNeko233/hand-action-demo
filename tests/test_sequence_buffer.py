import numpy as np
from src.sequence_buffer import SequenceBuffer

def test_buffer():
    b=SequenceBuffer(30,66)
    for _ in range(30): b.append(np.zeros(66,dtype=np.float32))
    assert b.is_ready() and b.get_sequence().shape==(30,66)
    b.clear(); assert len(b)==0 and not b.is_ready()
