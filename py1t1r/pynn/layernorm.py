import numpy as np

def LayerNorm(y, C, eps=1e-5, gamma=None, beta=None):
    """
    y: (B, N, C)；对最后一维 C 做 LayerNorm
    C: 通道数 (= y.shape[-1])
    gamma, beta: 可选，(C,)；不传则用 1/0
    """
    y = y.astype(np.float32, copy=False)
    mu  = y.mean(axis=-1, keepdims=True)          # (B,N,1)
    var = y.var(axis=-1, keepdims=True)           # (B,N,1)
    yhat = (y - mu) / np.sqrt(var + eps)
    if gamma is None:
        gamma = np.ones(C, dtype=np.float32)
    if beta is None:
        beta  = np.zeros(C, dtype=np.float32)
    return yhat * gamma.reshape(1,1,C) + beta.reshape(1,1,C)