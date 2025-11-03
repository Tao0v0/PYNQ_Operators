import numpy as np
import time
import re
from pynn.model1 import model
from pynn.backend import software
from pynn.layer import conv2d,layernorm_tokens,dense
from pynn.layernorm import LayerNorm
from pynn.gelu import gelu
from pathlib import Path


class PatchEmbed:
    """
    Conv2d(c1->c2, k,stride,pad, [groups]) + LayerNorm(to tokens)
    内部自带一个 model: self.m
    前向输入:  (B, C, H, W)
    前向输出:  (B, N, C_out)  其中 N = H_out * W_out
    """

    def __init__(self, c1, c2, kernel_size=7, stride=4, padding=3, eps=1e-5, groups=1, in_h=None, in_w=None, use_bias=True):
        if in_h is None or in_w is None:
            raise ValueErro

        self.c1 = int(c1)
        self.c2 = int(c2)
        self.kh = self.kw = int(kernel_size) if isinstance(kernel_size, int) else int(kernel_size[0])
        self.stride = int(stride)
        self.padding = int(padding)
        self.eps = float(eps)
        self.groups = int(groups)
        self.in_h = int(in_h)
        self.in_w = int(in_w)
        self.use_bias = bool(use_bias)


        self.m = model(backend=software())

        self.m.add(conv2d(
            out_channels=self.c2, in_channels=self.c1,
            kernel_size=(self.kh, self.kw),
            stride=self.stride, padding=self.padding,
            in_h=self.in_h, in_w=self.in_w,
            groups=self.groups,
            return_nchw=False,                    # 直接输出 (B, N, C)
            bias_config=[1,1] if self.use_bias else [0,0]
        ))
        self.conv_idx = len(self.m.backend.W) - 1

        self.m.add(layernorm_tokens(C=self.c2, eps=self.eps))
        self.ln_idx = len(self.m.backend.W) - 1


        self.out_h = (self.in_h + 2*self.padding - self.kh) // self.stride + 1
        self.out_w = (self.in_w + 2*self.padding - self.kw) // self.stride + 1

    # ---------------- 权重加载（与其它模块统一：load_conv(W, b)） ----------------
    def load_conv(self, W, b=None):
        """
        统一入口：
          - 若 W 是 4D：形状 (C_out, C_in_g, Kh, Kw)
          - 若 W 是 2D：形状 (C_out, C_in_g*Kh*Kw)
          - b 为 (C_out,) 或 None；当 use_bias=False 时忽略 b
        会自动拼出 backend 需要的 full 矩阵：
          (C_out, C_in_g*Kh*Kw + (bias?1:0))
        """
        Cin_g = self.c1 // self.groups
        Kh, Kw = self.kh, self.kw

        W = np.asarray(W, dtype=np.float32)
        if W.ndim == 4:
            expect4 = (self.c2, Cin_g, Kh, Kw)
            if tuple(W.shape) != expect4:
                raise ValueError(f"conv 4D weight shape {W.shape} != {expect4}")
            W_core = W.reshape(self.c2, Cin_g * Kh * Kw)  # 展平
        elif W.ndim == 2:
            expect2 = (self.c2, Cin_g * Kh * Kw)
            if tuple(W.shape) != expect2:
                raise ValueError(f"conv 2D weight shape {W.shape} != {expect2}")
            W_core = W
        else:
            raise ValueError(f"W.ndim 必须为 2 或 4，当前 {W.ndim}")

        full_cols = W_core.shape[1] + (1 if self.use_bias else 0)
        W_full = np.zeros((self.c2, full_cols), dtype=np.float32)
        W_full[:, :W_core.shape[1]] = W_core

        if self.use_bias:
            if b is None:
                # 没给 b 就按 0 处理
                pass
            else:
                b = np.asarray(b, dtype=np.float32).reshape(-1)
                if b.shape[0] != self.c2:
                    raise ValueError(f"bias length {b.shape[0]} != C_out {self.c2}")
                W_full[:, -1] = b

        self.m.backend.W[self.conv_idx] = W_full

    def load_ln(self, gamma, beta):
        """
        layernorm_tokens 的 backend 权重约定:
        backend.W[self.ln_idx] 形状通常为 (2, C): 第0行=gamma, 第1行=beta
        """
        if gamma.shape != (self.c2,) or beta.shape != (self.c2,):
            raise ValueError(f"LN参数需为 ({self.c2},)")
        Wln = self.m.backend.W[self.ln_idx]
        Wln[0, :self.c2] = np.asarray(gamma, np.float32)
        Wln[1, :self.c2] = np.asarray(beta,  np.float32)


    def load_conv_full(self, W_full):
        Cin_g = self.c1 // self.groups
        expect_cols = Cin_g * self.kh * self.kw + (1 if self.use_bias else 0)
        if tuple(W_full.shape) != (self.c2, expect_cols):
            raise ValueError(f"conv full weight shape {W_full.shape} != {(self.c2, expect_cols)}")
        self.m.backend.W[self.conv_idx] = np.asarray(W_full, np.float32)

    def load_conv_core_bias(self, W_core, bias=None):
        self.load_conv(W_core, bias)

    # ---------------- 前向 ----------------
    def forward(self, x_nchw):
        """
        x_nchw: (B, C_in, H, W)
        返回:   (B, N, C_out)
        """
        if x_nchw.ndim != 4:
            raise ValueError(f"PatchEmbed.forward 期望 (B,C,H,W)，得到 {x_nchw.shape}")
        return self.m.predict(x_nchw)

    def out_shape_bnc(self, B):
        return (B, self.out_h * self.out_w, self.c2)

class MLP:
    def __init__(self, C, H, W, expansion=4, kernel_size=3, padding=None, use_bias=True):
        self.C = int(C)
        self.H = int(H)
        self.W = int(W)
        self.expansion = int(expansion)
        self.K = int(kernel_size)
        self.P = int(padding) if padding is not None else self.K // 2
        self.use_bias = bool(use_bias)

        # 三个子模型：升维全连接、深度可分卷积、降维全连接
        self.m_proj = model(backend=software())
        self.m_dw   = model(backend=software())
        self.m_back = model(backend=software())

        self._build_graph()

    def _bias_cfg(self):
        return [1, 1] if self.use_bias else [0, 0]

    def _build_graph(self):
        C  = self.C
        eC = self.expansion * C

        # 1) proj_up: (B,N,C) -> (B,N,eC)
        self.m_proj.add(
            dense(output_dim=eC, input_dim=C, activation="linear", bias_config=self._bias_cfg())
        )
        self.proj_idx = len(self.m_proj.backend.W) - 1

        # 2) dwconv: (B,eC,H,W) -> (B,N,eC)  (groups=eC, return_nchw=False)
        self.m_dw.add(
            conv2d(
                out_channels=eC, in_channels=eC,
                kernel_size=self.K, stride=1, padding=self.P,
                groups=eC, in_h=self.H, in_w=self.W,
                return_nchw=False, bias_config=self._bias_cfg()
            )
        )
        self.dw_idx = len(self.m_dw.backend.W) - 1

        # 3) proj_back: (B,N,eC) -> (B,N,C)
        self.m_back.add(
            dense(output_dim=C, input_dim=eC, activation="linear", bias_config=self._bias_cfg())
        )
        self.back_idx = len(self.m_back.backend.W) - 1

    # ---------------- 权重加载----------------
    def load_proj(self, weight, bias=None):
        """ weight:(eC, C)，bias:(eC,) 或 None """
        eC, C = self.expansion * self.C, self.C
        if weight.shape != (eC, C):
            raise ValueError(f"proj_up weight shape {weight.shape} != {(eC, C)}")
        W = np.zeros((eC, C + (1 if self.use_bias else 0)), dtype=np.float32)
        W[:, :C] = weight.astype(np.float32, copy=False)
        if self.use_bias:
            W[:, -1] = 0.0 if bias is None else np.asarray(bias, np.float32).reshape(-1)
        self.m_proj.backend.W[self.proj_idx] = W

    def load_dwconv(self, weight, bias=None):
        """depthwise conv:
        支持 weight 形状：
            - (eC, 1, K, K)  —— 来自 PyTorch
            - (eC, K*K)      —— 展平后
        bias: (eC,) 或 None
        """
        eC = self.expansion * self.C
        w = np.asarray(weight, np.float32)
        if w.ndim == 4:
            ec, cin_g, kh, kw = w.shape
            if not (ec == eC and cin_g == 1 and kh == self.K and kw == self.K):
                raise ValueError(f"dwconv 4D 权重维度不匹配: got {w.shape}, expect {(eC,1,self.K,self.K)}")
            core = w.reshape(eC, self.K * self.K)
        elif w.ndim == 2:
            if w.shape != (eC, self.K * self.K):
                raise ValueError(f"dwconv 2D 权重维度不匹配: got {w.shape}, expect {(eC, self.K*self.K)}")
            core = w
        else:
            raise ValueError("dwconv 权重必须是 2D 或 4D")

        cols = core.shape[1] + (1 if self.use_bias else 0)
        W = np.zeros((eC, cols), dtype=np.float32)
        W[:, :core.shape[1]] = core
        if self.use_bias:
            W[:, -1] = 0.0 if bias is None else np.asarray(bias, np.float32).reshape(-1)
        self.m_dw.backend.W[self.dw_idx] = W

    def load_proj_back(self, weight, bias=None):
        """ weight:(C, eC)，bias:(C,) 或 None """
        C, eC = self.C, self.expansion * self.C
        if weight.shape != (C, eC):
            raise ValueError(f"proj_back weight shape {weight.shape} != {(C, eC)}")
        W = np.zeros((C, eC + (1 if self.use_bias else 0)), dtype=np.float32)
        W[:, :eC] = weight.astype(np.float32, copy=False)
        if self.use_bias:
            W[:, -1] = 0.0 if bias is None else np.asarray(bias, np.float32).reshape(-1)
        self.m_back.backend.W[self.back_idx] = W

    # ---------------- 前向 ----------------
    def forward(self, x_bnc):
        """
        x_bnc: (B,N,C) 且 N == H*W
        返回   (B,N,C)
        """
        if x_bnc.ndim != 3:
            raise ValueError(f"MLP.forward 期望 (B,N,C)，得到 {x_bnc.shape}")
        B, N, C = x_bnc.shape
        if C != self.C or N != self.H * self.W:
            raise ValueError(f"shape 不匹配：输入 {x_bnc.shape}，但 C={self.C} 且 H*W={self.H*self.W}")

        # (B,N,C) --dense--> (B,N,eC)
        y = self.m_proj.predict(x_bnc)                               # 利用 dense 的 3D 分支，原生支持 (B,N,C) :contentReference[oaicite:0]{index=0}

        # (B,N,eC) -> (B,eC,H,W) 做 DWConv -> 回到 (B,N,eC)
        eC = self.expansion * self.C
        y_nchw = y.transpose(0, 2, 1).reshape(B, eC, self.H, self.W)
        y_dw   = self.m_dw.predict(y_nchw)                           # conv2d 支持 groups，权重列数 = Cin_g*Kh*Kw + bias列 :contentReference[oaicite:1]{index=1}

        # GELU（逐元素，形状保持 (B,N,eC)）
        y_act  = gelu(y_dw)

        # (B,N,eC) --dense--> (B,N,C)
        y_out  = self.m_back.predict(y_act)
        return y_out

class Attention:
    """
    无 metric 版本；两次 matmul 都走 pynn 的 model.predict：
      - S = Q @ K^T   -> _vmm_bnc(Q, W=K)
      - O = Attn @ V  -> _vmm_bnc(Attn, W=V^T)
    SR 分支：Conv2d(kernel=stride=sr) + LayerNorm(tokens)
    变更点：SR 分支延迟到 forward 按 H/W 动态构建；支持提前缓存 SR 权重与 LN 参数。
    """
    def __init__(self, dim, head=1, sr_ratio=1, H=None, W=None, use_bias=True, eps=1e-5):
        assert dim % head == 0, "dim 必须能被 head 整除"
        self.dim      = int(dim)
        self.head     = int(head)
        self.sr_ratio = int(sr_ratio)
        self.scale    = (self.dim // self.head) ** -0.5
        self.H        = int(H) if H is not None else None
        self.W        = int(W) if W is not None else None
        self.use_bias = bool(use_bias)
        self.eps      = float(eps)

        # 线性层：Q / KV / OUT（保持你原来的参数风格，不显式传 activation）
        self.m_q   = model(backend=software())
        self.m_kv  = model(backend=software())
        self.m_out = model(backend=software())

        self.m_q.add(  dense(output_dim=self.dim,   input_dim=self.dim,
                             bias_config=[1,1] if use_bias else [0,0]))
        self.m_kv.add( dense(output_dim=2*self.dim, input_dim=self.dim,
                             bias_config=[1,1] if use_bias else [0,0]))
        self.m_out.add(dense(output_dim=self.dim,   input_dim=self.dim,
                             bias_config=[1,1] if use_bias else [0,0]))

        # SR 分支改为懒构建：此处不再强制要求 H/W，不抛错
        self.m_sr = None
        # 提前加载 SR 权重/LN 时的缓存
        self._sr_cache = {"w": None, "b": None, "gamma": None, "beta": None}

    # -------- 懒构建 SR：在 forward 时根据 H/W 动态创建，并写入已缓存权重 --------
    def _ensure_sr(self, H, W):
        if self.sr_ratio <= 1:
            return
        if (self.m_sr is None) or (self.H != H) or (self.W != W):
            self.H, self.W = int(H), int(W)
            self.m_sr = model(backend=software())
            self.m_sr.add(conv2d(
                out_channels=self.dim, in_channels=self.dim,
                kernel_size=self.sr_ratio, stride=self.sr_ratio, padding=0,
                groups=1, in_h=self.H, in_w=self.W, return_nchw=False,
                bias_config=[1,1] if self.use_bias else [0,0]
            ))
            self.m_sr.add(layernorm_tokens(C=self.dim, eps=self.eps))
            # 写入缓存参数
            if self._sr_cache["w"] is not None:
                self._write_sr_conv(self._sr_cache["w"], self._sr_cache["b"])
            if (self._sr_cache["gamma"] is not None) and (self._sr_cache["beta"] is not None):
                self.m_sr.backend.W[1][0, :] = self._sr_cache["gamma"]
                self.m_sr.backend.W[1][1, :] = self._sr_cache["beta"]

    # 把 SR 卷积权重写入 backend；支持 (Cout,Cin,kh,kw) 或 (Cout, Cin*kh*kw)
    def _write_sr_conv(self, W_core, b=None):
        Kh = Kw = self.sr_ratio
        expect_cols = self.dim * Kh * Kw
        W_core = np.asarray(W_core, np.float32)
        if W_core.ndim == 4:
            Cout, Cin, kh, kw = W_core.shape
            if (Cout != self.dim) or (Cin != self.dim) or (kh != Kh) or (kw != Kw):
                raise ValueError(f"SR权重维度不匹配: got {W_core.shape}, expect {(self.dim,self.dim,Kh,Kw)}")
            W_core = W_core.reshape(self.dim, expect_cols)
        elif W_core.ndim == 2:
            if W_core.shape != (self.dim, expect_cols):
                raise ValueError(f"SR权重维度不匹配: got {W_core.shape}, expect {(self.dim, expect_cols)}")
        else:
            raise ValueError("不支持的 SR 权重维度")

        out = np.zeros((self.dim, expect_cols + (1 if self.use_bias else 0)), np.float32)
        out[:, :expect_cols] = W_core
        if self.use_bias and b is not None:
            out[:, -1] = np.asarray(b, np.float32).reshape(-1)
        self.m_sr.backend.W[0] = out

    # ========================= 权重加载（接口保持不变） =========================
    def load_q(self, W, b=None):
        W = np.asarray(W, np.float32)
        if W.shape != (self.dim, self.dim):
            raise ValueError(f"Q权重应为 {(self.dim, self.dim)}，得到 {W.shape}")
        out = np.zeros((self.dim, self.dim + (1 if self.use_bias else 0)), np.float32)
        out[:, :self.dim] = W
        if self.use_bias and b is not None:
            b = np.asarray(b, np.float32).reshape(-1)
            if b.shape[0] != self.dim: raise ValueError("Q 偏置长度不匹配")
            out[:, -1] = b
        self.m_q.backend.W[0] = out

    def load_kv(self, W, b=None):
        W = np.asarray(W, np.float32)
        if W.shape != (2*self.dim, self.dim):
            raise ValueError(f"KV权重应为 {(2*self.dim, self.dim)}，得到 {W.shape}")
        out = np.zeros((2*self.dim, self.dim + (1 if self.use_bias else 0)), np.float32)
        out[:, :self.dim] = W
        if self.use_bias and b is not None:
            b = np.asarray(b, np.float32).reshape(-1)
            if b.shape[0] != 2*self.dim: raise ValueError("KV 偏置长度不匹配")
            out[:, -1] = b
        self.m_kv.backend.W[0] = out

    def load_proj(self, W, b=None):
        W = np.asarray(W, np.float32)
        if W.shape != (self.dim, self.dim):
            raise ValueError(f"PROJ权重应为 {(self.dim, self.dim)}，得到 {W.shape}")
        out = np.zeros((self.dim, self.dim + (1 if self.use_bias else 0)), np.float32)
        out[:, :self.dim] = W
        if self.use_bias and b is not None:
            b = np.asarray(b, np.float32).reshape(-1)
            if b.shape[0] != self.dim: raise ValueError("PROJ 偏置长度不匹配")
            out[:, -1] = b
        self.m_out.backend.W[0] = out

    def load_sr_conv(self, W_core, b=None):
        if self.sr_ratio <= 1:
            raise ValueError("sr_ratio==1 时无 SR 分支")
        # 先缓存；若 m_sr 已建则直接写入
        self._sr_cache["w"] = np.asarray(W_core, np.float32)
        self._sr_cache["b"] = None if b is None else np.asarray(b, np.float32)
        if self.m_sr is not None:
            self._write_sr_conv(self._sr_cache["w"], self._sr_cache["b"])

    def load_sr_ln(self, gamma, beta):
        if self.sr_ratio <= 1:
            raise ValueError("sr_ratio==1 时无 SR 分支")
        self._sr_cache["gamma"] = np.asarray(gamma, np.float32).reshape(-1)
        self._sr_cache["beta"]  = np.asarray(beta,  np.float32).reshape(-1)
        if self.m_sr is not None:
            self.m_sr.backend.W[1][0, :] = self._sr_cache["gamma"]
            self.m_sr.backend.W[1][1, :] = self._sr_cache["beta"]

    # ------------- 用“一层 dense”做 (B,N,Cin)×(Cout,Cin) 的 VMM -------------
    def _vmm_bnc(self, x_bnc, W_no_bias):
        """
        x_bnc: (B,N,Cin) 或 (N,Cin)
        W_no_bias: (Cout, Cin)，直接写入 backend.W[0]
        返回: (B,N,Cout)
        """
        x = np.asarray(x_bnc, np.float32)
        if x.ndim == 2:
            x = x[None, ...]  # (N,C) -> (1,N,C)
        B, N, Cin = x.shape
        W = np.asarray(W_no_bias, np.float32)
        Cout, Cin_w = W.shape
        if Cin_w != Cin:
            raise ValueError(f"_vmm_bnc: Cin({Cin}) 与权重列({Cin_w})不一致")

        m = model(backend=software())
        m.add(dense(output_dim=Cout, input_dim=Cin, bias_config=[0,0]))
        m.backend.W[0] = W
        return m.predict(x)  # (B,N,Cout)

    # ========================= 前向 =========================
    def forward(self, x, H=None, W=None):
        """
        x: (B, N, C=dim)，N=H*W
        返回: (B, N, C)
        """
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x 形状应为 (B,N,{self.dim})，得到 {x.shape}")
        B, N, C = x.shape

        # 确定 H, W；sr_ratio>1 时在此构建 SR 分支
        H_eff = int(H if H is not None else self.H)
        W_eff = int(W if W is not None else self.W)
        if self.sr_ratio > 1:
            if H_eff is None or W_eff is None:
                raise ValueError("sr_ratio>1 需要在 forward 时提供 H/W")
            if H_eff * W_eff != N:
                raise ValueError(f"H*W 必须等于 N，收到 H={H_eff}, W={W_eff}, N={N}")
            self._ensure_sr(H_eff, W_eff)

        # 1) Q
        q = self.m_q.predict(x)  # (B,N,C)

        # 2) K,V（带 SR）
        if self.sr_ratio > 1:
            x_nchw = x.transpose(0, 2, 1).reshape(B, C, H_eff, W_eff)
            xs = self.m_sr.predict(x_nchw)       # (B,Ns,C)  因为 return_nchw=False
        else:
            xs = x                                # (B,N,C)
        kv = self.m_kv.predict(xs)                 # (B,Ns,2C)
        k, v = np.split(kv, 2, axis=-1)            # (B,Ns,C), (B,Ns,C)

        # 3) 多头拆分
        Dh = C // self.head
        q = q.reshape(B, N, self.head, Dh).transpose(0, 2, 1, 3)   # (B,head,N,Dh)
        k = k.reshape(B, -1, self.head, Dh).transpose(0, 2, 1, 3)  # (B,head,Ns,Dh)
        v = v.reshape(B, -1, self.head, Dh).transpose(0, 2, 1, 3)  # (B,head,Ns,Dh)

        # 4) 注意力（两次乘法都走 _vmm_bnc）
        outs = []
        for b in range(B):
            heads_out = []
            for h in range(self.head):
                q_bh = q[b, h]             # (N, Dh)
                k_bh = k[b, h]             # (Ns, Dh)
                v_bh = v[b, h]             # (Ns, Dh)

                # S = q @ k^T -> (N,Ns)；dense 需要权重 (Ns, Dh)
                scores = self._vmm_bnc(q_bh, W_no_bias=k_bh).squeeze(0)  # (N,Ns)
                scores = scores * self.scale
                scores = scores - scores.max(axis=-1, keepdims=True)     # 防溢出
                attn = np.exp(scores).astype(np.float32)
                denom = np.clip(attn.sum(axis=-1, keepdims=True), 1e-12, None)
                attn = attn / denom                                      # (N,Ns)

                # O = attn @ v -> (N,Dh)；dense 需要权重 (Dh, Ns)
                out_bh = self._vmm_bnc(attn, W_no_bias=v_bh.T).squeeze(0)  # (N,Dh)
                heads_out.append(out_bh)
            outs.append(np.concatenate(heads_out, axis=-1))               # (N,C)
        y = np.stack(outs, axis=0)                                        # (B,N,C)

        # 5) 输出投影
        y = self.m_out.predict(y)                                         # (B,N,C)
        return y

class Block:
    """
    结构：Pre-LN + Attention + 残差 → Pre-LN + MLP + 残差
    构造只需要 dim, head, sr_ratio；H/W 在 forward 时提供，以便：
      - Attention 的 SR 分支
      - MLP 的 DWConv (需要 H,W)
    """
    def __init__(self, dim, head=1, sr_ratio=1):
        self.dim = int(dim)
        self.head = int(head)
        self.sr_ratio = int(sr_ratio)

        self._eps = 1e-5
        self._use_bias = True
        self._mlp_expansion = 4
        self._mlp_kernel = 3
        self._mlp_padding = None  # 让 MLP 内部用 K//2

        # LN1 / LN2
        self.m_ln1 = model(backend=software())
        self.m_ln1.add(layernorm_tokens(C=self.dim, eps=self._eps))

        self.m_ln2 = model(backend=software())
        self.m_ln2.add(layernorm_tokens(C=self.dim, eps=self._eps))

        # 注意力：使用你当前文件里的 Attention（已改成懒构建 SR）
        self.attn = Attention(dim=self.dim, head=self.head, sr_ratio=self.sr_ratio,
                              H=None, W=None, use_bias=self._use_bias, eps=self._eps)

        # MLP 懒构建：第一次 forward(H,W) 才实例化
        self.mlp = None
        self._mlp_cache = {"proj": None, "proj_b": None,
                           "dw": None, "dw_b": None,
                           "back": None, "back_b": None}

    # -------- 懒构建 MLP --------
    def _ensure_mlp(self, H, W):
        if (self.mlp is None) or (self.mlp.H != H) or (self.mlp.W != W):
            self.mlp = MLP(
                C=self.dim, H=int(H), W=int(W),
                expansion=self._mlp_expansion,
                kernel_size=self._mlp_kernel,
                padding=self._mlp_padding,
                use_bias=self._use_bias
            )
            # 如之前已缓存权重，这里一次性灌入
            if self._mlp_cache["proj"] is not None:
                self.mlp.load_proj(self._mlp_cache["proj"], self._mlp_cache["proj_b"])
            if self._mlp_cache["dw"] is not None:
                self.mlp.load_dwconv(self._mlp_cache["dw"], self._mlp_cache["dw_b"])
            if self._mlp_cache["back"] is not None:
                self.mlp.load_proj_back(self._mlp_cache["back"], self._mlp_cache["back_b"])

    # ---------------- 前向 ----------------
    def forward(self, x, H=None, W=None):
        """
        x: (B, N, C) ；N=H*W
        """
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"Block.forward 期望 (B,N,{self.dim})，得到 {x.shape}")
        if self.sr_ratio > 1:
            if H is None or W is None:
                raise ValueError("sr_ratio>1 时需要 forward 提供 H/W")
            if int(H) * int(W) != x.shape[1]:
                raise ValueError(f"H*W 必须等于 N，收到 H={H}, W={W}, N={x.shape[1]}")
        else:
            # MLP 的 DWConv 也需要 H/W，即使 sr_ratio==1，这里同样要求传
            if H is None or W is None:
                raise ValueError("MLP 的 DWConv 需要 H/W，请在 forward 传入")

        # 1) Pre-LN + Attn + 残差
        y = self.m_ln1.predict(x)              # (B,N,C)
        y = self.attn.forward(y, H=H, W=W)     # (B,N,C)
        x = x + y

        # 2) Pre-LN + MLP + 残差（先确保 MLP 已按 H/W 构建）
        self._ensure_mlp(H, W)
        y2 = self.m_ln2.predict(x)             # (B,N,C)
        y2 = self.mlp.forward(y2)              # (B,N,C)
        out = x + y2
        return out

    # ---------------- 权重加载（便于外部从 npz/pth 灌入） ----------------
    # LN
    def load_ln1(self, gamma, beta):
        self.m_ln1.backend.W[0][0, :] = np.asarray(gamma, np.float32).reshape(-1)
        self.m_ln1.backend.W[0][1, :] = np.asarray(beta,  np.float32).reshape(-1)

    def load_ln2(self, gamma, beta):
        self.m_ln2.backend.W[0][0, :] = np.asarray(gamma, np.float32).reshape(-1)
        self.m_ln2.backend.W[0][1, :] = np.asarray(beta,  np.float32).reshape(-1)

    # Attention（直接转发给子模块 Attention 的加载接口）
    def load_attn_q(self, W, b=None):        self.attn.load_q(W, b)
    def load_attn_kv(self, W, b=None):       self.attn.load_kv(W, b)
    def load_attn_proj(self, W, b=None):     self.attn.load_proj(W, b)
    def load_attn_sr_conv(self, W, b=None):  self.attn.load_sr_conv(W, b)
    def load_attn_sr_ln(self, gamma, beta):  self.attn.load_sr_ln(gamma, beta)

    # MLP（支持先缓存、后 forward 再一次性写入）
    def load_mlp_proj(self, W, b=None):
        self._mlp_cache["proj"]   = np.asarray(W, np.float32)
        self._mlp_cache["proj_b"] = None if b is None else np.asarray(b, np.float32)
        if self.mlp is not None:
            self.mlp.load_proj(self._mlp_cache["proj"], self._mlp_cache["proj_b"])

    def load_mlp_dw(self, W, b=None):
        self._mlp_cache["dw"]   = np.asarray(W, np.float32)
        self._mlp_cache["dw_b"] = None if b is None else np.asarray(b, np.float32)
        if self.mlp is not None:
            self.mlp.load_dwconv(self._mlp_cache["dw"], self._mlp_cache["dw_b"])

    def load_mlp_back(self, W, b=None):
        self._mlp_cache["back"]   = np.asarray(W, np.float32)
        self._mlp_cache["back_b"] = None if b is None else np.asarray(b, np.float32)
        if self.mlp is not None:
            self.mlp.load_proj_back(self._mlp_cache["back"], self._mlp_cache["back_b"])




# ---------- 小工具 ----------
def _has(pack, key: str) -> bool:
    return key in pack.files

# ========= 扫描权重中的结构信息 =========
def _find_block_indices(pack, stage):
    pat = re.compile(rf"^block{stage}_(\d+)_norm1_weight$")
    idx = [int(m.group(1)) for k in pack.files if (m := pat.match(k))]
    return sorted(idx)

def _get_patch_specs(pack, stage):
    w = pack[f"patch_embed{stage}_proj_weight"]   # (Cout, Cin, Kh, Kw)
    Cout, Cin, Kh, Kw = w.shape
    k = int(Kh)
    return Cin, Cout, k

def _get_sr_ratio_from_weight(pack, stage, first_block_idx=0):
    # 权重可能没有 sr 分支的 key（当 sr_ratio==1 时），需稳健处理
    k_sr = f"block{stage}_{first_block_idx}_attn_sr_weight"
    if _has(pack, k_sr):
        return int(pack[k_sr].shape[-1])  # 通常 (C, C, sr, sr)
    return 1

# ========= 通用加载器 =========
def load_patch_from_pack(pe, pack, stage):
    pe.load_conv(pack[f"patch_embed{stage}_proj_weight"], pack[f"patch_embed{stage}_proj_bias"])
    pe.load_ln(  pack[f"patch_embed{stage}_norm_weight"], pack[f"patch_embed{stage}_norm_bias"])

def load_block_from_pack(blk, pack, stage, i, sr_ratio: int):
    pfx = f"block{stage}_{i}_"

    # LN (token LN)
    blk.load_ln1(pack[pfx + "norm1_weight"], pack[pfx + "norm1_bias"])
    blk.load_ln2(pack[pfx + "norm2_weight"], pack[pfx + "norm2_bias"])

    # Attention (Q/KV/Proj)
    blk.load_attn_q(   pack[pfx + "attn_q_weight"],    pack[pfx + "attn_q_bias"])
    blk.load_attn_kv(  pack[pfx + "attn_kv_weight"],   pack[pfx + "attn_kv_bias"])
    blk.load_attn_proj(pack[pfx + "attn_proj_weight"], pack[pfx + "attn_proj_bias"])

    # SR 分支：当 sr_ratio==1（或权重不存在）时跳过
    if sr_ratio > 1:
        k_sr_w = pfx + "attn_sr_weight"
        k_sr_b = pfx + "attn_sr_bias"
        k_ln_w = pfx + "attn_norm_weight"
        k_ln_b = pfx + "attn_norm_bias"

        if _has(pack, k_sr_w) and _has(pack, k_sr_b):
            blk.load_attn_sr_conv(pack[k_sr_w], pack[k_sr_b])
        # 有的导出会省略 SR-LN，同样做存在性检查
        if _has(pack, k_ln_w) and _has(pack, k_ln_b):
            blk.load_attn_sr_ln(pack[k_ln_w], pack[k_ln_b])

    # MLP（DWConv 4D→2D 兼容）
    blk.load_mlp_proj(pack[pfx + "mlp_fc1_weight"], pack[pfx + "mlp_fc1_bias"])
    dw = pack[pfx + "mlp_dwconv_dwconv_weight"]
    if dw.ndim == 4:
        eC, one, Kh, Kw = dw.shape
        dw = dw.reshape(eC, Kh * Kw)
    blk.load_mlp_dw(dw, pack[pfx + "mlp_dwconv_dwconv_bias"])
    blk.load_mlp_back(pack[pfx + "mlp_fc2_weight"], pack[pfx + "mlp_fc2_bias"])

def apply_last_ln(y_bnc, pack, stage, eps=1e-5):
    C = y_bnc.shape[-1]
    m = model(backend=software())
    m.add(layernorm_tokens(C=C, eps=eps))
    m.backend.W[0][0, :] = pack[f"norm{stage}_weight"]
    m.backend.W[0][1, :] = pack[f"norm{stage}_bias"]
    return m.predict(y_bnc)

# ========= 跑一个 stage（head / sr_ratio 可由 cfg 指定）=========
def run_stage(
    *, pack, stage, cfg, x_nchw=None, y_prev_bnc=None,
    B=None, H_in=None, W_in=None
):
    assert (x_nchw is None) ^ (y_prev_bnc is None), "x_nchw 与 y_prev_bnc 必须二选一"

    # 基本规格：通道&核由权重决定；stride 允许外部 cfg 指定（默认：stage1=4，其它=2）
    Cin_w, Cout_w, k = _get_patch_specs(pack, stage)
    stride = cfg.get("stride", 4 if stage == 1 else 2)
    pad = k // 2

    # block 列表（可从 cfg 指定，否则扫描权重）
    blocks = cfg.get("blocks", _find_block_indices(pack, stage))

    # sr_ratio：优先 cfg；否则从权重推断；如果权重没有 sr key，则为 1
    sr_from_w = _get_sr_ratio_from_weight(pack, stage, first_block_idx=blocks[0] if blocks else 0)
    sr_ratio = cfg.get("sr_ratio", sr_from_w)

    # 若权重提供了 sr 分支（sr_from_w>1），做一致性校验；否则权重无 sr，sr_ratio 自动降为 1
    if sr_from_w > 1:
        assert sr_ratio == sr_from_w, f"Stage{stage} sr_ratio={sr_ratio} 与权重推断 {sr_from_w} 不一致"
    else:
        sr_ratio = 1  # 权重无 sr 分支；强制关闭 SR

    head = cfg.get("head", 1)

    t0 = time.perf_counter()

    # 准备输入（若来自上游 tokens，需要还原为 NCHW）
    if x_nchw is not None:
        B, C_in, H_in, W_in = x_nchw.shape
        assert C_in == Cin_w, f"Stage{stage} 输入通道不符：x={C_in}, weight={Cin_w}"
    else:
        assert (B is not None) and (H_in is not None) and (W_in is not None), "还原 NCHW 需要 B/H/W"
        Bp, Np, Cp = y_prev_bnc.shape
        assert (Bp == B) and (Cp == Cin_w) and (Np == H_in * W_in), \
            f"Stage{stage} 还原失败: B {Bp}!={B} or C {Cp}!={Cin_w} or N {Np}!={H_in*W_in}"
        x_nchw = y_prev_bnc.transpose(0, 2, 1).reshape(B, Cin_w, H_in, W_in)

    # PatchEmbed
    pe = PatchEmbed(c1=Cin_w, c2=Cout_w, kernel_size=k, stride=stride, padding=pad,
                    eps=1e-5, groups=1, in_h=H_in, in_w=W_in, use_bias=True)
    load_patch_from_pack(pe, pack, stage)
    y = pe.forward(x_nchw)             # (B, N, Cout)
    H_out, W_out = pe.out_h, pe.out_w
    print(f"Stage{stage} PatchEmbed 输出:", y.shape, f"(H_out={H_out}, W_out={W_out})")

    # Blocks
    for i in blocks:
        blk = Block(dim=Cout_w, head=head, sr_ratio=sr_ratio)
        load_block_from_pack(blk, pack, stage, i, sr_ratio=sr_ratio)
        y = blk.forward(y, H=H_out, W=W_out)
        print(f"Stage{stage} Block{i+1} 输出:", y.shape)

    # Last LN
    y = apply_last_ln(y, pack, stage=stage, eps=1e-5)
    t1 = time.perf_counter()

    print(f"\nStage{stage} 最终输出:", y.shape)
    print(f"Stage{stage} 前30个数：", y.ravel()[:30])
    print(f"Stage{stage} 前向耗时: {(t1 - t0)*1000:.2f} ms\n")
    return y, (B, Cout_w, H_out, W_out)

# ========= 串起 Stage1 → Stage2 → Stage3 → Stage4（每个 stage 的 head/sr_ratio/stride 独立配置）=========
if __name__ == "__main__":
    PACK_PATH = "cmnext_stages1_4_weights.npz"
    pack = np.load(PACK_PATH)

    # 每个 stage 的可配参数（按你的模型改这里）
    STAGE_CFG = {
        1: dict(head=1, sr_ratio=8, stride=4),
        2: dict(head=2, sr_ratio=4, stride=2),
        3: dict(head=5, sr_ratio=2, stride=2),
        4: dict(head=8, sr_ratio=1, stride=2),  # sr=1 → 自动跳过 attn_sr_* / attn_norm_*
    }

    # Stage1：输入全 1
    B1, C1_in, H1_in, W1_in = 1, 3, 440, 640
    x1 = np.ones((B1, C1_in, H1_in, W1_in), np.float32)
    y1, (B1o, C1o, H1o, W1o) = run_stage(pack=pack, stage=1, cfg=STAGE_CFG[1], x_nchw=x1)

    # Stage2
    y2, (B2o, C2o, H2o, W2o) = run_stage(pack=pack, stage=2, cfg=STAGE_CFG[2],
                                         y_prev_bnc=y1, B=B1o, H_in=H1o, W_in=W1o)

    # Stage3
    y3, (B3o, C3o, H3o, W3o) = run_stage(pack=pack, stage=3, cfg=STAGE_CFG[3],
                                         y_prev_bnc=y2, B=B2o, H_in=H2o, W_in=W2o)

    # Stage4（sr=1 → 自动跳过 SR 分支的权重加载）
    y4, (B4o, C4o, H4o, W4o) = run_stage(pack=pack, stage=4, cfg=STAGE_CFG[4],
                                         y_prev_bnc=y3, B=B3o, H_in=H3o, W_in=W3o)

"""
stage4 输出前30个值: [0.161532, -0.039586, 0.544022, 0.03888, -0.055521, 0.102239, 0.029269, -0.111304, 0.048153, -3.301864,
                     0.272838, -0.053807, 0.314758, 0.061528, -0.00965, -0.095456, 0.052761, 0.213863, 0.302559, -0.068218, 
                    -0.125488, 0.568426, 0.217442, 0.168265, 0.397994, 0.262744, 0.139586, -0.008916, -0.033475, -0.037849]
"""







# stage1_pack_path = "cmnext_stages1_4_weights.npz"
# B_stage1, C_in_stage1, C_out_stage1 = 1, 3, 64
# H_stage1, W_stage1 = 440, 640
# stage1_num_blocks, stage1_head, stage1_sr_ratio = 3, 1, 8

# stage1_pack = np.load(stage1_pack_path)

# # PatchEmbed (Stage1)
# pe_stage1 = PatchEmbed(
#     c1=C_in_stage1, c2=C_out_stage1,
#     kernel_size=7, stride=4, padding=3,
#     eps=1e-5, groups=1, in_h=H_stage1, in_w=W_stage1,
#     use_bias=True
# )
# pe_stage1.load_conv(stage1_pack["patch_embed1_proj_weight"], stage1_pack["patch_embed1_proj_bias"])
# pe_stage1.load_ln(  stage1_pack["patch_embed1_norm_weight"], stage1_pack["patch_embed1_norm_bias"])

# x_stage1 = np.ones((B_stage1, C_in_stage1, H_stage1, W_stage1), np.float32)
# t0_stage1 = time.perf_counter()
# y_stage1 = pe_stage1.forward(x_stage1)  # (B, N, C_out)
# print("Stage1 PatchEmbed 输出:", y_stage1.shape)

# # 直接用 pe 的空间尺寸
# H1_stage1, W1_stage1 = pe_stage1.out_h, pe_stage1.out_w

# # Blocks（for 循环）
# for i in range(stage1_num_blocks):
#     blk_stage1 = Block(dim=C_out_stage1, head=stage1_head, sr_ratio=stage1_sr_ratio)
#     load_block_from_pack(blk_stage1, stage1_pack, i)
#     y_stage1 = blk_stage1.forward(y_stage1, H=H1_stage1, W=W1_stage1)
#     print(f"Stage1 Block{i+1} 输出:", y_stage1.shape)

# # stage1 末尾的 LN
# m_last_stage1 = model(backend=software())
# m_last_stage1.add(layernorm_tokens(C=C_out_stage1, eps=1e-5))
# m_last_stage1.backend.W[0][0, :] = stage1_pack["norm1_weight"]
# m_last_stage1.backend.W[0][1, :] = stage1_pack["norm1_bias"]

# y_final_stage1 = m_last_stage1.predict(y_stage1)
# t1_stage1 = time.perf_counter()

# print("\nStage1 最终输出:", y_final_stage1.shape)
# print("Stage1 前 30 个数：", y_final_stage1.ravel()[:30])
# print(f"Stage1 前向耗时: {(t1_stage1 - t0_stage1)*1000:.2f} ms")




# # 可选：如果你没在文件更上方导入过，再保留这两行
# # import numpy as np
# # import time


# stage2_pack_path = "cmnext_stages_4_weights.npz"
# B_stage2, C_in_stage2, C_out_stage2 = 1, 64, 128
# H_stage2, W_stage2 = 440/4, 640/4
# stage2_num_blocks, stage2_head, stage2_sr_ratio = 4, 2, 4

# stage2_pack = np.load(stage2_pack_path)

# # PatchEmbed (stage2)
# pe_stage2 = PatchEmbed(
#     c1=C_in_stage2, c2=C_out_stage2,
#     kernel_size=3, stride=2, padding=1,
#     eps=1e-5, groups=1, in_h=H_stage2, in_w=W_stage2,
#     use_bias=True
# )
# pe_stage2.load_conv(stage2_pack["patch_embed2_proj_weight"], stage2_pack["patch_embed2_proj_bias"])
# pe_stage2.load_ln(  stage2_pack["patch_embed2_norm_weight"], stage2_pack["patch_embed2_norm_bias"])



# x_stage2 = y_final_stage1.reshape(B_stage2, C_in_stage2, int(H_stage2), int(W_stage2))

# t0_stage2 = time.perf_counter()
# y_stage2 = pe_stage2.forward(x_stage2)  # (B, N, C_out)
# print("stage2 PatchEmbed 输出:", y_stage2.shape)

# # 直接用 pe 的空间尺寸
# H1_stage2, W1_stage2 = pe_stage2.out_h, pe_stage2.out_w

# # Blocks（for 循环）
# for i in range(stage2_num_blocks):
#     blk_stage2 = Block(dim=C_out_stage2, head=stage2_head, sr_ratio=stage2_sr_ratio)
#     load_block_from_pack(blk_stage2, stage2_pack, i)
#     y_stage2 = blk_stage2.forward(y_stage2, H=H1_stage2, W=W1_stage2)
#     print(f"stage2 Block{i+1} 输出:", y_stage2.shape)

# # stage2 末尾的 LN
# m_last_stage2 = model(backend=software())
# m_last_stage2.add(layernorm_tokens(C=C_out_stage2, eps=1e-5))
# m_last_stage2.backend.W[0][0, :] = stage2_pack["norm2_weight"]
# m_last_stage2.backend.W[0][1, :] = stage2_pack["norm2_bias"]

# y_final_stage2 = m_last_stage2.predict(y_stage2)
# t1_stage2 = time.perf_counter()

# print("\nstage2 最终输出:", y_final_stage2.shape)
# print("stage2 前 30 个数：", y_final_stage2.ravel()[:30])
# print(f"stage2 前向耗时: {(t1_stage2 - t0_stage2)*1000:.2f} ms")


