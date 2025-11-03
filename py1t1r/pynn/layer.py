#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Abstract class for different layers
from abc import ABC, abstractmethod # 抽象基类，ABC表示一个抽象类，abstractmethod表示一个装饰器，用于声明抽象方法
import numpy as np
from pynn.activations import activations


class layer(ABC):                   
    @property                       #   property表示把方法变成普通属性
    @abstractmethod                 #   被@abstractmethod装饰的方法在子类中必须实现
    # Layer ID in the backend
    def nlayer(self):
        raise NotImplementedError

    @property                       
    @abstractmethod
    # Backend obj
    def backend(self):
        raise NotImplementedError

    @property
    @abstractmethod
    # Input dimension
    def input_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    # Output dimension
    def output_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    # Weight dimension
    def weight_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    # Activation name
    def act_name(self):
        raise NotImplementedError

    @abstractmethod
    # Foward pass
    def call(self, x_in):
        pass

    @abstractmethod
    # Calculated error gradient
    def calc_gradients(self, dy):
        pass


class dense(layer):
    # Current layer ID in the network
    nlayer = []             # 表示当前层在网络中的编号

    # Backend obj
    backend = []            # 表示当前层使用的后端对象

    input_dim = []          # 表示输入维度
    output_dim = []         # 表示输出维度
    weight_dim = []         # 表示权重矩阵维度
    act_name = []           # 表示激活函数名称

    # Instantiation
    def __init__(           # 初始化类的属性
        self, output_dim, input_dim=np.nan, activation="linear", bias_config=[1, 1],
    ):
        self.output_dim = output_dim
        self.input_dim = input_dim
        # Activation func name
        self.act_name = activation
        # A 1x2 vector: [ratio(bias value) rep(use how many inputs)]
        self.bias_config = bias_config      # bias_config 是形如 [ratio, rep] 的 1×2  ration为1表示偏置值为1，rep为1表示偏置一行

        self.set_weight_dim()               # 设置权重矩阵维度
        self.x_in_history = []              # 初始化输入历史记录列表
        self.y_out_history = []             # 初始化输出历史记录列表

    # Set weights dimension
    def set_weight_dim(self):
        # Num of rows = output dim
        # Num of col = input dim + 1 (if physical bias)

        self.weight_dim = [                       
            self.output_dim, 
            self.input_dim + self.bias_config[1],
        ]

    # CALL The forward pass of the layer
    def call(self, x_in):
        """
        支持：
        - 3D: (B, N, C_in) -> (B, N, C_out)  （你要的）
        - 2D: (C_in, B)    -> (C_out, B)     （保留原有）
        只做推理：不存历史、不算梯度。
        """
        # ---------- 3D 分支： (B, N, C_in) ----------
        if x_in.ndim == 3:
            B, N, Cin = x_in.shape

            # 若 input_dim 尚未确定，则用真实 Cin
            if hasattr(self, "input_dim") and (self.input_dim is None or (isinstance(self.input_dim, float) and np.isnan(self.input_dim))):
                self.input_dim = Cin
                # 若你的 dense 有 set_weight_dim() 之类的初始化，请保持原始调用
                if hasattr(self, "set_weight_dim"):
                    self.set_weight_dim()

            # 形状检查
            if hasattr(self, "input_dim"):
                assert Cin == self.input_dim, f"[dense] Cin={Cin} 与层的 input_dim={self.input_dim} 不一致"

            # (B, N, C_in) -> (Cin, B*N)
            x2 = x_in.reshape(B * N, Cin).T  # (Cin, B*N)

            # 处理 bias 行（沿用原 bias_config 逻辑）
            if getattr(self, "bias_config", None) is not None and self.bias_config[1] != 0:
                bias_val, repeat_rows = self.bias_config  # e.g., [0, 0] 或 [1, 1]
                # bias_val 可能是标量或长度为1的数组；tile 成 (rep, B*N)
                bias_rows = np.tile(bias_val, (repeat_rows, x2.shape[1]))  # (rep, B*N)
                x_in_full = np.concatenate((x2, bias_rows), axis=0)        # (Cin+rep, B*N)
            else:
                x_in_full = x2

            # 乘法：得到 (C_out, B*N)
            y2 = self.backend.multiply(x_in_full, self.nlayer)

            # 激活
            act = activations.get(self.act_name, "act")
            y2 = act(y2)

            # 还原形状 (B, N, C_out)
            y_out = y2.T.reshape(B, N, self.output_dim)
            return y_out

        # ---------- 2D 分支： (C_in, B) ----------
        elif x_in.ndim == 2:
            n = np.size(x_in, 1)  # batch 列数

            if getattr(self, "bias_config", None) is not None and self.bias_config[1] != 0:
                x_in_full = np.concatenate(
                    (x_in, np.tile(self.bias_config[0], (self.bias_config[1], n))),
                    axis=0,
                )
            else:
                x_in_full = x_in

            y_out = self.backend.multiply(x_in_full, self.nlayer)

            act = activations.get(self.act_name, "act")
            y_out = act(y_out)
            return y_out

        else:
            raise ValueError(f"[dense] 不支持的输入形状 {x_in.shape}，仅支持 3D(B,N,C) 或 2D(C,B)")

    def calc_gradients(self, dy):
        raise NotImplementedError("conv2d: 推理-only")
        # y_out = self.y_out_history.pop(-1)      # 这一层的输出，激活函数后的结果
        # x_in = self.x_in_history.pop(-1)        # 这一层的输入，拼接偏置后的输入

        # # Activation
        # if "softmax" not in self.act_name:      # 如果当前层用了激活函数，除了softmax+交叉熵在最后一层不用激活函数

        #     act = activations.get(self.act_name, "deriv")
        #     dy = dy * act(y_out)                # 把梯度还原到“激活前

        # # Calculate the gradient
        # grads = np.matmul(dy, x_in.T)

        # # Back propogation (new deltas)
        # dx = self.backend.multiply_reverse(dy, self.nlayer)

        # return grads, dx        # 输出当前层的权重梯度和传给前一层的梯度


# ======= dense 风格卷积层（推理-only） ，支持返回 NCHW=======
def _pair(x):
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


class conv2d(layer):
    """
    im2col 风格的卷积；支持 groups；偏置采用“输入拼 1 行常数再乘”的风格（与 dense 一致）。
    backend 权重张量应为：
        W.shape == (out_channels, Cin_g*Kh*Kw + (1 if bias else 0))
    其中 Cin_g = in_channels // groups。
    """

    # ===== 必需：抽象属性 =====
    @property
    def act_name(self):
        # 卷积常用线性输出，后面谁要激活再单独加层
        return self._act_name

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def weight_dim(self):
        return self._weight_dim

    # ===== 其余公共属性（工程里别的代码会访问）=====
    @property
    def nlayer(self): return self._nlayer
    @nlayer.setter
    def nlayer(self, v): self._nlayer = v

    @property
    def backend(self): return self._backend
    @backend.setter
    def backend(self, v): self._backend = v

    # ===== 构造 =====
    def __init__(
        self,
        out_channels,
        in_channels=3,
        kernel_size=7,
        stride=4,
        padding=3,
        bias_config=[1, 1],     # ← 默认带偏置（必须）
        in_h=None,
        in_w=None,
        return_nchw=False,
        groups=1,
    ):
        self.name         = "conv2d"
        self._act_name    = "linear"   # 抽象属性需要一个值
        self.out_channels = int(out_channels)
        self.in_channels  = int(in_channels)
        self.kernel_size  = _pair(kernel_size)
        self.stride       = _pair(stride)
        self.padding      = _pair(padding)
        self.bias_config  = bias_config
        self.return_nchw  = bool(return_nchw)
        self.groups       = int(groups)

        if in_h is None or in_w is None:
            raise ValueError("conv2d: 需要在构造时提供 in_h 与 in_w")
        self.in_h, self.in_w = int(in_h), int(in_w)

        # 计算输出尺寸与登记 input_dim/output_dim
        Ho, Wo = self._infer_output_hw(self.in_h, self.in_w)
        self._input_dim  = self.in_channels * self.in_h * self.in_w
        self._output_dim = self.out_channels * Ho * Wo

        # 按当前配置登记权重尺寸（抽象属性需要）
        self.set_weight_dim()

    # ===== 内部工具 =====
    def _infer_output_hw(self, H, W):
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        Ph, Pw = self.padding
        Ho = (H + 2*Ph - Kh)//Sh + 1
        Wo = (W + 2*Pw - Kw)//Sw + 1
        if Ho <= 0 or Wo <= 0:
            raise ValueError("conv2d: 非法输出尺寸，请检查 kernel/stride/padding 与输入尺寸")
        return Ho, Wo

    def _im2col_nchw(self, x, Kh, Kw, Sh, Sw, Ph, Pw):
        """
        x: (B,C,H,W) -> (B, C*Kh*Kw, Ho*Wo)
        """
        B, C, H, W = x.shape
        Ho = (H + 2*Ph - Kh)//Sh + 1
        Wo = (W + 2*Pw - Kw)//Sw + 1

        if Ph or Pw:
            x_pad = np.pad(x, ((0,0),(0,0),(Ph,Ph),(Pw,Pw)), mode="constant")
        else:
            x_pad = x

        cols = np.empty((B, C*Kh*Kw, Ho*Wo), dtype=x.dtype)
        idx = 0
        for i in range(0, Ho*Sh, Sh):
            for j in range(0, Wo*Sw, Sw):
                patch = x_pad[:, :, i:i+Kh, j:j+Kw]     # (B,C,Kh,Kw)
                cols[:, :, idx] = patch.reshape(B, C*Kh*Kw)
                idx += 1
        return cols

    # ===== 权重维度登记（抽象属性需要）=====
    def set_weight_dim(self, input_dim=None):
        Kh, Kw = self.kernel_size
        use_bias = (self.bias_config is not None and self.bias_config[1] != 0)
        Cin_g = self.in_channels // self.groups
        self._w_cols_per_group = Cin_g * Kh * Kw + (1 if use_bias else 0)
        self._w_rows_per_group = self.out_channels // self.groups
        self._weight_dim = (self.out_channels, self._w_cols_per_group)
        return self._weight_dim

    # ===== 前向 =====
    def call(self, x_in):
        # 统一输入；框架里常见两种： (B,C,H,W) 或 (C*H*W, B)
        if x_in.ndim == 2:
            C, H, W = self.in_channels, self.in_h, self.in_w
            B = x_in.shape[1]
            x_bc_hw = x_in.reshape(C, H, W, B).transpose(3, 0, 1, 2)  # (B,C,H,W)
        elif x_in.ndim == 4:
            B, C, H, W = x_in.shape
            if not (C == self.in_channels and H == self.in_h and W == self.in_w):
                raise ValueError(f"conv2d: 输入 {x_in.shape} 与构造 (C={self.in_channels},H={self.in_h},W={self.in_w}) 不匹配")
            x_bc_hw = x_in
        else:
            raise ValueError(f"conv2d.call: 不支持的输入形状 {x_in.shape}")

        Ho, Wo = self._infer_output_hw(H, W)
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        Ph, Pw = self.padding
        HWo = Ho * Wo

        # im2col — NCHW -> (B, C*Kh*Kw, Ho*Wo)
        cols = self._im2col_nchw(x_bc_hw, Kh, Kw, Sh, Sw, Ph, Pw)  # (B, C*Kh*Kw, HWo)

        if self.groups == 1:
            K = self.in_channels * Kh * Kw
            x_in_mat = cols.transpose(1, 0, 2).reshape(K, B*HWo)     # (K, B*HWo)

            # 按“偏置=多一行常数”风格拼输入
            if self.bias_config is not None and self.bias_config[1] != 0:
                bias_val, repeat_rows = self.bias_config
                bias_rows = np.full((repeat_rows, x_in_mat.shape[1]), fill_value=bias_val, dtype=x_in_mat.dtype)
                x_full = np.concatenate((x_in_mat, bias_rows), axis=0)  # (K+rep, B*HWo)
            else:
                x_full = x_in_mat

            # 走后端乘法（后端权重已是 (Cout, K+rep)）
            y = self.backend.multiply(x_full, self.nlayer)            # (Cout, B*HWo)

        else:
            Cin_g  = self.in_channels // self.groups
            Cout_g = self.out_channels // self.groups
            Kg     = Cin_g * Kh * Kw
            use_bias = (self.bias_config is not None and self.bias_config[1] != 0)

            # 取该层整块权重（由 backend 初始化/加载）
            W_all = self.backend.W[self.nlayer]                       # (Cout, Kg + rep)
            expect_cols = Kg + (1 if use_bias else 0)
            if W_all.shape != (self.out_channels, expect_cols):
                raise ValueError(f"conv2d(groups): 期望权重 {(self.out_channels, expect_cols)}，实际 {W_all.shape}")

            ys = []
            for g in range(self.groups):
                # 本组输入列（B, Kg, HWo）-> (Kg, B*HWo)
                cols_g = cols[:, g*Kg:(g+1)*Kg, :]                     # (B, Kg, HWo)
                xg = cols_g.transpose(1, 0, 2).reshape(Kg, B*HWo)      # (Kg, B*HWo)

                # 拼偏置行
                if use_bias:
                    bias_val, repeat_rows = self.bias_config
                    ones = np.full((repeat_rows, xg.shape[1]), fill_value=bias_val, dtype=xg.dtype)
                    xg_full = np.concatenate((xg, ones), axis=0)       # (Kg+rep, B*HWo)
                else:
                    xg_full = xg

                # 本组权重（行切片）
                Wg = W_all[g*Cout_g:(g+1)*Cout_g, :]                   # (Cout_g, Kg+rep)

                # 直接 numpy 乘，避免维度错位
                yg = np.matmul(Wg, xg_full)                             # (Cout_g, B*HWo)
                ys.append(yg)

            y = np.concatenate(ys, axis=0)                             # (Cout, B*HWo)

        # 输出整理
        if self.return_nchw:
            y = y.reshape(self.out_channels, B, HWo).transpose(1, 0, 2).reshape(B, self.out_channels, Ho, Wo)
            return y
        else:
            y = y.transpose(1, 0).reshape(B, Ho*Wo, self.out_channels)  # (B, N, C)
            return y

    def calc_gradients(self, dy, x_in, y_out):
        raise NotImplementedError("conv2d: 推理-only")



# ---- LayerNorm over tokens: (B, N, C) ----
class layernorm_tokens(layer):
    """
    在 (B, N, C) 上对最后一维 C 归一化。
    参数存放在 backend.W[self.nlayer]，形状 (2, C)：
        第 0 行 = gamma（缩放），第 1 行 = beta（偏置）
    """
    def __init__(self, C, eps=1e-5, name="layernorm_tokens"):
        super().__init__()
        # —— 抽象属性占位（dense/conv2d 同风格）——
        self._nlayer     = []
        self._backend    = []
        self._input_dim  = []          # LN 不改变特征维度，add() 时会被上一层覆盖
        self._output_dim = []
        self._weight_dim = (2, int(C))
        self._act_name   = "linear"

        # 业务参数
        self.C   = int(C)
        self.eps = float(eps)
        self.name = name

    # ===== abstract properties 的 getter/setter（必须有）=====
    @property
    def nlayer(self): return self._nlayer
    @nlayer.setter
    def nlayer(self, v): self._nlayer = v

    @property
    def backend(self): return self._backend
    @backend.setter
    def backend(self, v): self._backend = v

    @property
    def input_dim(self): return self._input_dim
    @input_dim.setter
    def input_dim(self, v):
        self._input_dim = v
        # LN 不改变形状，维持 output_dim 与 input_dim 一致
        self._output_dim = v

    @property
    def output_dim(self): return self._output_dim
    @output_dim.setter
    def output_dim(self, v): self._output_dim = v

    @property
    def weight_dim(self): return self._weight_dim
    @weight_dim.setter
    def weight_dim(self, v): self._weight_dim = v

    @property
    def act_name(self): return self._act_name
    @act_name.setter
    def act_name(self, v): self._act_name = v

    # ===== 给 model.add() 用来决定权重形状（必须有）=====
    def set_weight_dim(self, input_dim=None):
        # γ/β 两行，每行 C 列
        self._weight_dim = (2, self.C)
        # LN 不改变特征维，保持 in/out 一致
        self._output_dim = self._input_dim
        return self._weight_dim

    # ===== 前向 =====
    def call(self, x_in):
        # 只接受 (B, N, C)
        if not (x_in.ndim == 3 and x_in.shape[-1] == self.C):
            raise ValueError(f"{self.name}: 期望输入 (B,N,{self.C})，但拿到 {x_in.shape}")

        # 取出 γ/β
        W = self.backend.W[self.nlayer]     # (2, C)
        gamma = W[0]                        # (C,)
        beta  = W[1]                        # (C,)

        y = x_in.astype(np.float32, copy=False)
        mu  = y.mean(axis=-1, keepdims=True)               # (B,N,1)
        var = y.var(axis=-1, keepdims=True)                # (B,N,1)
        yhat = (y - mu) / np.sqrt(var + self.eps)          # (B,N,C)
        return yhat * gamma.reshape(1,1,self.C) + beta.reshape(1,1,self.C)

    # ===== 反向（先占位，和 conv2d 一样推理-only）=====
    def calc_gradients(self, dy):
        raise NotImplementedError("layernorm_tokens: 推理-only")
    



