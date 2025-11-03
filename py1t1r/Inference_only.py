from pathlib import Path
import numpy as np
import scipy.io

from pynn.model1 import model
from pynn.backend import software
from pynn.layer import dense
from pynn.layer import conv2d
from pynn.layernorm import LayerNorm
# ===== 1) 数据（按原来的写法）=====
DATASET_DIR = Path(__file__).resolve().parent / "pynn" / "dataset"
TEST_X_MAT  = DATASET_DIR / "mnist_test_8x8_0d2.mat"
mat         = scipy.io.loadmat(str(TEST_X_MAT))
xs_test     = mat["data_0d2"][:, :]                # 形状：(60, N)，每列一个样本
xs_test     = np.delete(xs_test, [0, 7, 56, 63], 0).astype(np.float32)

test = np.random.rand(60, 1000).astype(np.float32)  # 随便造点数据试试


# ===== 2) 最小网络骨架 =====

# IN_DIM, HID_DIM, OUT_DIM = 60, 40, 10

# def build_model():
#     m = model(backend=software()) # 选择纯CPU执行
#     m.add(dense(HID_DIM, input_dim=IN_DIM, activation="relu",           bias_config=[0,0]))     # 加一个全连接隐藏层
#     m.add(dense(OUT_DIM,                 activation="stable_softmax",   bias_config=[0,0]))     # 加一个全连接输出层
#     return m

# # ===== 3) 随便初始化权重（固定seed，或你改成全0/全1都行）=====
# def init_dummy_weights(m):
#     rng = np.random.default_rng(0)  # 改成 None 就是完全随机
#     m.backend.W[0] = rng.standard_normal((HID_DIM, IN_DIM)).astype(np.float32)  # (40,60)
#     m.backend.W[1] = rng.standard_normal((OUT_DIM, HID_DIM)).astype(np.float32) # (10,40)

# # ===== 4) 只做前向 =====
# def run(x):
#     m = build_model()
#     init_dummy_weights(m)
#     return m.predict(x)

# if __name__ == "__main__":
#     y = run(test)                       
#     print("输出形状:", y.shape)       # 预期：(10, N)
#     print("前5个样本的前3类概率:\n", y[:3, :5])


# conv2d test
# 超参数

B, C, H, W = 8, 3, 80, 80

m = model(backend=software())
m1 = model(backend=software())
# Conv1: 3->64, 7x7, s=4, p=3   输出 H/4, W/4
m.add(conv2d(
    out_channels=64, in_channels=3,
    kernel_size=7, stride=4, padding=3,
    bias_config=[0,0],
    in_h=H, in_w=W,
    return_nchw=True
))

# 计算第一层输出尺寸
Ho1 = (H + 2*3 - 7)//4 + 1  # = 20
Wo1 = (W + 2*3 - 7)//4 + 1  # = 20

# Conv2: 64->64, 3x3, s=2, p=1  再下采样一半
m.add(conv2d(
    out_channels=64, in_channels=64,
    kernel_size=3, stride=2, padding=1,
    bias_config=[0,0],
    in_h=Ho1, in_w=Wo1,
    return_nchw=True
))

Ho2 = (Ho1 + 2*1 - 3)//2 + 1  # = 10
Wo2 = (Wo1 + 2*1 - 3)//2 + 1  # = 10

# Conv3: 64->64, 3x3, s=1, p=1  分辨率不变
m.add(conv2d(
    out_channels=64, in_channels=64,
    kernel_size=3, stride=1, padding=1,
    bias_config=[0,0],
    in_h=Ho2, in_w=Wo2,
    return_nchw=True
))

Ho3 = (Ho2 + 2*1 - 3)//1 + 1  # = 10
Wo3 = (Wo2 + 2*1 - 3)//1 + 1  # = 10

# ------- 随机权重（只为跑通）-------
# Conv1 权重: (64, 3*7*7)
m.backend.W[0] = np.random.randn(64, 3*7*7).astype(np.float32)
# Conv2 权重: (64, 64*3*3)
m.backend.W[1] = np.random.randn(64, 64*3*3).astype(np.float32)
# Conv3 权重: (64, 64*3*3)
m.backend.W[2] = np.random.randn(64, 64*3*3).astype(np.float32)

# ------- 推理 -------
x = np.random.randn(B, C, H, W).astype(np.float32)
y = m.predict(x)  # 由于最后一层 return_nchw=True，形状应为 (B, 64, Ho3, Wo3)

print("Conv1 -> (B,64,{},{}):".format(Ho1, Wo1))    # = (B,64,20,20)
print("Conv2 -> (B,64,{},{}):".format(Ho2, Wo2))    # = (B,64,10,10)
print("Conv3 -> (B,64,{},{}):".format(Ho3, Wo3))    # = (B,64,10,10)
print("y.shape =", y.shape)     # 预期(8,100,64)

y = LayerNorm(y,C=y.shape[-1])
print("y.shape =", y.shape)    

 

 

# 新建一个“纯全连接”的模型（和前面的 m 完全独立）
m_new = model(backend=software())

num_classes = 10

m_new.add(conv2d(
    out_channels=128, in_channels=64,
    kernel_size=3, stride=1, padding=1,
    bias_config=[0,0],
    in_h=10, in_w=10,
    return_nchw=False   # 输出(Cout*Ho*Wo, B)
))

# 一定要加到 m_new 上！并且输入维度要是 81*128
m_new.add(dense(
    output_dim=num_classes,
    input_dim=100*128,
    activation="linear",
    bias_config=[0,0]
))

# 初始化权重（现在 backend.W 有两层：W[0] conv，W[1] dense）
m_new.backend.W[0] = np.random.randn(128, 64*3*3).astype(np.float32)   # conv
m_new.backend.W[1] = np.random.randn(num_classes, 100*128).astype(np.float32)  # dense

# 推理
print("FEED y.shape =", y.shape)                 # 应该打印 (8, 64, 10, 10)
l0 = m_new.layer_list[0]
print("CONV expects:", l0.in_channels, l0.in_h, l0.in_w)  # 应该是 64 10 10
logits = m_new.predict(y)

print("logits:", logits.shape)   # 期望 (10, 8) 或 (8, 10) 取决于实现
