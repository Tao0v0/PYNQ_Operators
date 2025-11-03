import numpy as np

# —— 向量化的高精度 erf 近似（A&S 7.1.26），仅用到 exp、加减乘除 —— #
def _erf_approx(z: np.ndarray) -> np.ndarray:
    # 处理 dtype，尽量沿用输入精度（float32/float64）
    dtype = z.dtype if np.issubdtype(z.dtype, np.floating) else np.float32
    z = z.astype(dtype, copy=False)

    # 常数（按 A&S 7.1.26）
    p  = dtype.type(0.3275911)
    a1 = dtype.type(0.254829592)
    a2 = dtype.type(-0.284496736)
    a3 = dtype.type(1.421413741)
    a4 = dtype.type(-1.453152027)
    a5 = dtype.type(1.061405429)

    # erf(-z) = -erf(z)
    x  = np.abs(z)
    t  = 1.0 / (1.0 + p * x)
    # Horner 形式
    poly = (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t)
    y = 1.0 - poly * np.exp(-x*x)
    return np.copysign(y, z)  # 恢复符号

# —— 最接近 PyTorch F.gelu（approximate='none'）的实现（支持 B,N,C 任意形状） —— #
def gelu(x: np.ndarray) -> np.ndarray:
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32
    inv_sqrt2 = dtype.type(1.0/np.sqrt(2.0))
    return 0.5 * x * (1.0 + _erf_approx(x * inv_sqrt2))