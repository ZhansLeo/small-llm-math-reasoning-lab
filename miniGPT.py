import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. RoPE
# ============================================================

def rotate_half(x):
    """
    LLaMA/Hugging Face 风格的 rotate_half。

    假设最后一个维度 D = 8：

    x = [x0, x1, x2, x3 | x4, x5, x6, x7]

    rotate_half(x)
    = [-x4, -x5, -x6, -x7 | x0, x1, x2, x3]

    它本身不是完整的旋转，
    而是配合：

        x * cos + rotate_half(x) * sin

    完成二维旋转。
    """

    half = x.shape[-1] // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat(
        [-x2, x1],
        dim=-1
    )


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, base=10000.0):
        super().__init__()

        assert dim % 2 == 0

        self.dim = dim
        self.base = base

        # ----------------------------------------------------
        # 不需要训练的频率
        #
        # shape:
        # [D/2]
        # ----------------------------------------------------

        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32
                ) / dim
            )
        )

        # buffer:
        # 随模型一起移动到 CPU/GPU
        # 但不是可训练参数
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

    def forward(self, position_ids):
        """
        position_ids:
        [B, T]

        返回：

        cos:
        [B, T, D]

        sin:
        [B, T, D]
        """

        # position_ids:
        # [B,T]

        # inv_freq:
        # [D/2]

        # 得到：
        # [B,T,D/2]
        freqs = torch.einsum(
            "bt,d->btd",
            position_ids.float(),
            self.inv_freq
        )

        # 复制到 D 维
        #
        # [B,T,D/2]
        #      ↓
        # [B,T,D]
        emb = torch.cat(
            [freqs, freqs],
            dim=-1
        )

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q:
        [B, Hq, T, D]

    k:
        [B, Hkv, T, D]

    cos/sin:
        [B, T, D]

    通过 unsqueeze(1)：

        [B,T,D]
            ↓
        [B,1,T,D]

    这样可以和不同数量的 attention heads 做 broadcasting。
    """

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (
        q * cos
        + rotate_half(q) * sin
    )

    k_embed = (
        k * cos
        + rotate_half(k) * sin
    )

    return q_embed, k_embed


# ============================================================
# 2. RMSNorm
# ============================================================

class RMSNorm(nn.Module):

    def __init__(self, C, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(C)
        )

        self.eps = eps

    def forward(self, x):

        # x:
        # [B,T,C]

        rms = torch.sqrt(
            x.pow(2).mean(
                dim=-1,
                keepdim=True
            )
            + self.eps
        )

        x = x / rms

        return self.weight * x


# ============================================================
# 3. GQA + RoPE + KV Cache
# ============================================================

class GQA(nn.Module):

    def __init__(
        self,
        C,
        num_q_heads,
        num_kv_heads,
        rope_base=10000.0
    ):
        super().__init__()

        # Q head 必须能平均分
        assert C % num_q_heads == 0

        # Q head 必须能平均分组给 KV head
        assert num_q_heads % num_kv_heads == 0

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        # 每个 head 的 hidden dimension
        self.head_dim = C // num_q_heads

        # RoPE 需要偶数维
        assert self.head_dim % 2 == 0

        # ----------------------------------------------------
        # Q / K / V projection
        #
        # Q:
        # C -> num_q_heads * head_dim
        #
        # K/V:
        # C -> num_kv_heads * head_dim
        # ----------------------------------------------------

        self.q_proj = nn.Linear(
            C,
            num_q_heads * self.head_dim
        )

        self.k_proj = nn.Linear(
            C,
            num_kv_heads * self.head_dim
        )

        self.v_proj = nn.Linear(
            C,
            num_kv_heads * self.head_dim
        )

        self.out_proj = nn.Linear(
            C,
            C
        )

        # RoPE
        #
        # 注意：
        # RoPE 的维度是 head_dim
        #
        # 因为 RoPE 是在每个 head 内部进行的。
        self.rope = RotaryEmbedding(
            self.head_dim,
            base=rope_base
        )

    def forward(
        self,
        x,
        position_ids,
        past_k=None,
        past_v=None
    ):
        """
        x:
            [B,T,C]

        position_ids:
            [B,T]

        past_k / past_v:

            None
            或：

            [B,Hkv,past_T,D]

        返回：

        out:
            [B,T,C]

        new_k:
            [B,Hkv,total_T,D]

        new_v:
            [B,Hkv,total_T,D]
        """

        B, T, C = x.shape

        # ====================================================
        # Step 1: Q / K / V projection
        # ====================================================

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # ====================================================
        # Step 2: 拆 head
        #
        # Q:
        # [B,T,C]
        # ↓
        # [B,T,Hq,D]
        #
        # K/V:
        # [B,T,Hkv,D]
        # ====================================================

        Q = Q.reshape(
            B,
            T,
            self.num_q_heads,
            self.head_dim
        )

        K = K.reshape(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        )

        V = V.reshape(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        )

        # ====================================================
        # Step 3:
        #
        # [B,T,H,D]
        #      ↓ transpose
        # [B,H,T,D]
        # ====================================================

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ====================================================
        # Step 4: RoPE
        #
        # 只有 Q/K 做旋转
        # V 不动
        # ====================================================

        cos, sin = self.rope(
            position_ids
        )

        Q, K = apply_rotary_pos_emb(
            Q,
            K,
            cos,
            sin
        )

        # ====================================================
        # Step 5: KV Cache
        #
        # 如果以前已经有 cache：
        #
        # past_k:
        # [B,Hkv,past_T,D]
        #
        # K:
        # [B,Hkv,T,D]
        #
        # 拼接：
        #
        # [B,Hkv,past_T + T,D]
        # ====================================================

        if past_k is not None:

            K = torch.cat(
                [past_k, K],
                dim=2
            )

            V = torch.cat(
                [past_v, V],
                dim=2
            )

        # ====================================================
        # Step 6: GQA
        #
        # 到这里为止：
        #
        # Q:
        # [B,Hq,T,D]
        #
        # K/V:
        # [B,Hkv,total_T,D]
        #
        # 现在把 K/V 复制给不同 Q groups。
        #
        # 注意：
        # 我们是在 cache 之后再复制。
        #
        # 这样 cache 里只保存真正的 Hkv 个 head，
        # 不会把重复的 K/V 也存进去。
        # ====================================================

        repeat_factor = (
            self.num_q_heads
            // self.num_kv_heads
        )

        K_for_attention = K.repeat_interleave(
            repeat_factor,
            dim=1
        )

        V_for_attention = V.repeat_interleave(
            repeat_factor,
            dim=1
        )

        # ====================================================
        # Step 7: QK^T
        #
        # Q:
        # [B,Hq,T,D]
        #
        # K:
        # [B,Hq,total_T,D]
        #
        # K^T:
        # [B,Hq,D,total_T]
        #
        # scores:
        # [B,Hq,T,total_T]
        # ====================================================

        scores = Q @ K_for_attention.transpose(
            -2,
            -1
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        # ====================================================
        # Step 8: causal mask
        #
        # 只有 Prefill / 普通训练需要。
        #
        # Decode 时：
        #
        # T = 1
        #
        # 只有一个新 token，
        # 它没有未来 token 可以偷看，
        # 因此不需要 tril mask。
        #
        # 我们这里假设：
        # past_k != None 时是单 token decode。
        # ====================================================

        if past_k is None and T > 1:

            mask = torch.tril(
                torch.ones(
                    T,
                    T,
                    device=x.device,
                    dtype=torch.bool
                )
            )

            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

        # ====================================================
        # Step 9: softmax
        # ====================================================

        weights = torch.softmax(
            scores,
            dim=-1
        )

        # ====================================================
        # Step 10:
        #
        # attention weights × V
        #
        # [B,H,T,total_T]
        # ×
        # [B,H,total_T,D]
        #
        # →
        # [B,H,T,D]
        # ====================================================

        out = weights @ V_for_attention

        # ====================================================
        # Step 11:
        #
        # [B,H,T,D]
        # ↓
        # [B,T,H,D]
        # ↓
        # [B,T,C]
        # ====================================================

        out = out.transpose(
            1,
            2
        )

        out = out.reshape(
            B,
            T,
            C
        )

        # 输出投影
        out = self.out_proj(out)

        # 返回：
        #
        # out:
        # 当前 hidden states
        #
        # K/V:
        # 更新后的 cache
        return out, K, V


# ============================================================
# 4. SwiGLU
# ============================================================

class SwiGLU(nn.Module):

    def __init__(self, C, hidden_dim):
        super().__init__()

        # gate 分支
        self.gate_proj = nn.Linear(
            C,
            hidden_dim
        )

        # information 分支
        self.up_proj = nn.Linear(
            C,
            hidden_dim
        )

        # hidden_dim -> C
        self.down_proj = nn.Linear(
            hidden_dim,
            C
        )

    def forward(self, x):

        # ----------------------------------------------------
        # gate 分支
        # ----------------------------------------------------

        gate = self.gate_proj(x)

        gate = F.silu(
            gate
        )

        # ----------------------------------------------------
        # up 分支
        # ----------------------------------------------------

        up = self.up_proj(x)

        # ----------------------------------------------------
        # 门控
        # ----------------------------------------------------

        x = gate * up

        # ----------------------------------------------------
        # 回到 C
        # ----------------------------------------------------

        x = self.down_proj(x)

        return x


# ============================================================
# 5. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        C,
        num_q_heads,
        num_kv_heads,
        hidden_dim
    ):
        super().__init__()

        # Pre-Norm
        self.norm1 = RMSNorm(C)

        self.attention = GQA(
            C,
            num_q_heads,
            num_kv_heads
        )

        self.norm2 = RMSNorm(C)

        self.mlp = SwiGLU(
            C,
            hidden_dim
        )

    def forward(
        self,
        x,
        position_ids,
        past_k=None,
        past_v=None
    ):

        # ====================================================
        # Attention sub-layer
        #
        # x -> RMSNorm -> Attention -> Residual
        # ====================================================

        attn_out, new_k, new_v = self.attention(
            self.norm1(x),
            position_ids=position_ids,
            past_k=past_k,
            past_v=past_v
        )

        x = x + attn_out

        # ====================================================
        # MLP sub-layer
        #
        # x -> RMSNorm -> SwiGLU -> Residual
        # ====================================================

        x = x + self.mlp(
            self.norm2(x)
        )

        return x, new_k, new_v


# ============================================================
# 6. Mini LLaMA / Qwen 风格模型
# ============================================================

class MiniLLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        C,
        num_q_heads,
        num_kv_heads,
        num_layers,
        hidden_dim
    ):
        super().__init__()

        # ----------------------------------------------------
        # Token Embedding
        #
        # [B,T]
        # ↓
        # [B,T,C]
        #
        # 注意：
        # 这里已经没有 position embedding。
        #
        # 因为位置由 RoPE 处理。
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            C
        )

        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([
            TransformerBlock(
                C,
                num_q_heads,
                num_kv_heads,
                hidden_dim
            )
            for _ in range(num_layers)
        ])

        # ----------------------------------------------------
        # 最后的 RMSNorm
        # ----------------------------------------------------

        self.final_norm = RMSNorm(C)

        # ----------------------------------------------------
        # LM Head
        #
        # [B,T,C]
        # ↓
        # [B,T,V]
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            C,
            vocab_size
        )

    def forward(
        self,
        input_ids,
        past_key_values=None
    ):
        """
        input_ids:
            [B,T]

        past_key_values:
            None
            或：

            [
                (K_layer0, V_layer0),
                (K_layer1, V_layer1),
                ...
            ]

        返回：

        logits:
            [B,T,V]

        new_key_values:
            每层更新后的 KV Cache
        """

        B, T = input_ids.shape

        # ====================================================
        # Step 1: Token Embedding
        # ====================================================

        x = self.token_embedding(
            input_ids
        )

        # x:
        # [B,T,C]

        # ====================================================
        # Step 2: 确定当前 token 的 position
        #
        # 第一次 Prefill：
        #
        # past_length = 0
        #
        # position:
        # [0,1,2,...,T-1]
        #
        # Decode：
        #
        # past_length = 已缓存 token 数
        #
        # 如果 cache 有 10 个 token，
        # 当前新 token 的 position 就是 10。
        # ====================================================

        if past_key_values is None:

            past_length = 0

        else:

            # 取第 0 层的 K
            #
            # [B,Hkv,past_T,D]

            past_length = (
                past_key_values[0][0].shape[2]
            )

        position_ids = torch.arange(
            past_length,
            past_length + T,
            device=input_ids.device
        ).unsqueeze(0)

        # position_ids:
        #
        # Prefill:
        # [0,1,2,...]
        #
        # Decode:
        # [past_length]
        #
        # shape:
        # [1,T]

        # ====================================================
        # Step 3: 通过所有 Transformer Blocks
        # ====================================================

        new_key_values = []

        for layer_idx, block in enumerate(
            self.blocks
        ):

            # 当前这一层过去的 cache

            if past_key_values is None:

                past_k = None
                past_v = None

            else:

                past_k, past_v = (
                    past_key_values[layer_idx]
                )

            # 当前 Block
            x, new_k, new_v = block(
                x,
                position_ids=position_ids,
                past_k=past_k,
                past_v=past_v
            )

            # 保存当前 layer 的新 cache

            new_key_values.append(
                (new_k, new_v)
            )

        # ====================================================
        # Step 4: Final Norm
        # ====================================================

        x = self.final_norm(x)

        # ====================================================
        # Step 5: LM Head
        #
        # [B,T,C]
        # ↓
        # [B,T,V]
        # ====================================================

        logits = self.lm_head(x)

        return logits, new_key_values


# ============================================================
# 7. Tokenizer
#
# 最简单的字符级 tokenizer
# 只是为了让我们专注于模型本身。
# ============================================================

text = "hello world! " * 100

chars = sorted(
    list(set(text))
)

stoi = {}

for i, ch in enumerate(chars):
    stoi[ch] = i

itos = {}

for i, ch in enumerate(chars):
    itos[i] = ch


def encode(text):
    ids = []

    for ch in text:
        ids.append(
            stoi[ch]
        )

    return ids


def decode(ids):
    chars = []

    for i in ids:
        chars.append(
            itos[i]
        )

    return "".join(chars)


# ============================================================
# 8. Dataset
# ============================================================

data = torch.tensor(
    encode(text),
    dtype=torch.long
)


def get_batch(
    data,
    batch_size,
    block_size
):

    # 随机选择 batch_size 个起点

    starts = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    # 输入
    #
    # data[i : i+block_size]

    x = torch.stack([
        data[
            i : i + block_size
        ]
        for i in starts
    ])

    # target 向右移动一位
    #
    # data[i+1 : i+block_size+1]

    y = torch.stack([
        data[
            i + 1 :
            i + block_size + 1
        ]
        for i in starts
    ])

    return x, y


# ============================================================
# 9. 创建模型
# ============================================================

model = MiniLLM(
    vocab_size=len(chars),

    # hidden size
    C=64,

    # Q heads
    num_q_heads=4,

    # KV heads
    #
    # 4 Q heads
    # 2 KV heads
    #
    # 每两个 Q head 共享一个 KV head
    num_kv_heads=2,

    # Transformer layers
    num_layers=2,

    # SwiGLU intermediate size
    hidden_dim=256
)


# ============================================================
# 10. Training
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

block_size = 32

for step in range(1000):

    x, y = get_batch(
        data,
        batch_size=16,
        block_size=block_size
    )

    # ----------------------------------------
    # 清除上一轮 gradient
    # ----------------------------------------

    optimizer.zero_grad()

    # ----------------------------------------
    # Training 时不使用 cache
    #
    # 每一轮都重新完整计算整个 sequence
    # ----------------------------------------

    logits, _ = model(
        x,
        past_key_values=None
    )

    B, T, V = logits.shape

    # ----------------------------------------
    # [B,T,V]
    # ↓
    # [B*T,V]
    #
    # 方便 CrossEntropy
    # ----------------------------------------

    logits = logits.reshape(
        B * T,
        V
    )

    y = y.reshape(
        B * T
    )

    # ----------------------------------------
    # Next-token prediction loss
    # ----------------------------------------

    loss = F.cross_entropy(
        logits,
        y
    )

    # ----------------------------------------
    # Backpropagation
    # ----------------------------------------

    loss.backward()

    # ----------------------------------------
    # 更新参数
    # ----------------------------------------

    optimizer.step()

    if step % 100 == 0:

        print(
            f"step={step}, "
            f"loss={loss.item():.4f}"
        )


# ============================================================
# 11. KV Cache Generation
# ============================================================

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens
):
    """
    idx:
        [B,T]

    第一次：
        Prefill

    后面：
        Decode
    """

    # ========================================================
    # ① Prefill
    #
    # 整个 prompt 一次性进入模型
    #
    # 同时建立每一层的 K/V Cache
    # ========================================================

    logits, past_key_values = model(
        idx,
        past_key_values=None
    )

    # 最后一个位置负责预测下一个 token

    logits = logits[:, -1, :]

    # Greedy decoding
    next_token = torch.argmax(
        logits,
        dim=-1,
        keepdim=True
    )

    # 把第一个新 token 接回去

    idx = torch.cat(
        [idx, next_token],
        dim=1
    )

    # ========================================================
    # ② Decode
    #
    # 每一次只输入刚刚生成的 token
    #
    # 输入：
    # [B,1]
    #
    # 而不是：
    # [B,整个序列]
    # ========================================================

    for _ in range(
        max_new_tokens - 1
    ):

        # 只拿最后生成的 token

        x = idx[:, -1:]

        # 进入模型
        #
        # past_key_values 不再是 None
        #
        # 所以：
        #
        # 只计算这个新 token 的 Q/K/V
        #
        # 旧 token 的 K/V 来自 cache

        logits, past_key_values = model(
            x,
            past_key_values=past_key_values
        )

        # 只取最后一个位置

        logits = logits[:, -1, :]

        # 选择概率最高的 token

        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

        # 拼回序列

        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

    return idx


# ============================================================
# 12. Inference
# ============================================================

prompt = "hello"

input_ids = torch.tensor([
    encode(prompt)
])

generated = generate(
    model,
    input_ids,
    max_new_tokens=30
)

generated_ids = generated[
    0
].tolist()

print(
    decode(generated_ids)
)