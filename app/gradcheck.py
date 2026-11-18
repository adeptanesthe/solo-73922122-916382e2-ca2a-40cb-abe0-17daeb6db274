"""梯度核对：float64 下比较自动求导与中心差分，验证自写因果注意力的反传可靠性。

流程：
  1. 构建目标模型的独立 float64 副本（与在线解码单例不共享任何张量）；
  2. 以「已知下一字符」在提示词末尾位置的负对数似然为损失，反向传播得到
     自动求导梯度；
  3. 对所选块的 Q/K/V 投影各取两个固定坐标、对提示词字符对应的绑定
     词嵌入/输出权重取两个坐标，分别做中心差分：
         g_fd ≈ (L(θ+h) − L(θ−h)) / (2h)
     扰动直接作用于共享参数本体，因此词嵌入坐标会同时经过输入（嵌入查表）
     与输出（LM head）两条路径——这正是权重绑定语义下真实梯度所覆盖的；
  4. 逐坐标比较，任一绝对误差 > 1e-5 即判 FAIL 并指出首个超差坐标。

全部计算都在独立副本上进行：在线模型的权重、KV 缓存与前向计数不受影响，
核对期间解码接口照常可用。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import tokenizer
from .model import CausalTransformer, build_target_copy

# 判定阈值：任一坐标 |自动求导 − 中心差分| 超过该值即判失败
TOLERANCE = 1e-5
# 中心差分扰动步长（float64 下截断误差 ~h²、舍入误差 ~1e-16/h，均远小于阈值）
FD_EPS = 1e-5

# Q/K/V 投影各自的两个固定坐标（在投影自身的 d_model×d_model 坐标系内，
# 取值任意但固定，保证每次核对同一批坐标、结果可复现）。
_QKV_COORDS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "q": ((0, 0), (53, 109)),
    "k": ((7, 31), (64, 64)),
    "v": ((12, 88), (127, 5)),
}


def _build_float64_model() -> CausalTransformer:
    """构建独立 float64 目标模型副本，并显式恢复嵌入/LM head 的权重绑定。"""
    model = build_target_copy()  # 固定种子，权重与在线单例逐位一致
    model = model.double()
    # 显式重新绑定：确保 tok_emb 与 lm_head 共享同一参数对象，
    # 差分扰动因而同时作用于输入与输出两条路径（与在线模型语义一致）。
    model.lm_head.weight = model.tok_emb.weight
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return model


def _nll_loss(model: CausalTransformer, prompt_ids: list[int],
              target_id: int) -> torch.Tensor:
    """提示词末尾位置上「已知下一字符」的负对数似然（float64 标量）。"""
    logits = model.forward(prompt_ids, use_cache=False)  # (T, V)，不触碰 KV 缓存
    logp = F.log_softmax(logits[-1], dim=-1)
    return -logp[target_id]


def _summarize(entries: list[dict]) -> tuple[str, float, dict | None]:
    """汇总判定：任一坐标误差超过阈值即 FAIL，并返回首个超差坐标。"""
    max_err = 0.0
    first_fail: dict | None = None
    for e in entries:
        err = e["abs_error"]
        max_err = max(max_err, err)
        if err > TOLERANCE and first_fail is None:
            first_fail = e
    status = "PASS" if first_fail is None else "FAIL"
    return status, max_err, first_fail


def run_gradcheck(prompt: str, next_char: str, block_idx: int) -> dict:
    """执行梯度核对，返回可 JSON 序列化的结果字典。

    参数：
      prompt:    1～24 个可打印 ASCII 字符（调用方已校验）；
      next_char: 单个可打印 ASCII 字符，作为已知下一字符计算 NLL 损失；
      block_idx: 目标模型块序号（0～3）。

    任何异常都会向上抛出，由 API 层转为「核对中断」错误响应，
    绝不会在中断时给出通过结论。
    """
    prompt_ids = tokenizer.encode(prompt)
    target_id = tokenizer.encode(next_char)[0]

    model = _build_float64_model()
    d = model.d_model
    layer = model.blocks[block_idx]
    in_proj = layer.self_attn.in_proj_weight  # (3d, d)：Q/K/V 按行块拼接
    tied = model.tok_emb.weight               # 与 lm_head.weight 为同一对象

    # 坐标清单（判定顺序即此顺序）：(显示名, 参数, 行, 列, 备注)
    specs: list[tuple[str, torch.Tensor, int, int, str]] = []
    for key, row_off in (("q", 0), ("k", d), ("v", 2 * d)):
        for r, c in _QKV_COORDS[key]:
            specs.append((f"block{block_idx + 1}.{key}_proj",
                          in_proj, row_off + r, c, f"投影内坐标 ({r},{c})"))
    for tid, c in ((prompt_ids[0], 0), (prompt_ids[-1], d - 1)):
        ch = tokenizer.render_token(tid)
        specs.append(("tied_emb/lm_head", tied, tid, c,
                      f"提示词字符 {ch!r}（id={tid}）对应行，绑定权重双路径"))

    # ---- 自动求导：一次前向 + 一次反向 ----
    model.zero_grad(set_to_none=True)
    loss = _nll_loss(model, prompt_ids, target_id)
    loss.backward()
    loss_value = loss.item()

    # ---- 中心差分：逐坐标扰动参数本体（绑定权重同时走输入/输出两条路径）----
    entries: list[dict] = []
    with torch.no_grad():
        for name, param, r, c, note in specs:
            auto = float(param.grad[r, c].item())
            orig = float(param[r, c].item())
            try:
                param[r, c] = orig + FD_EPS
                l_plus = float(_nll_loss(model, prompt_ids, target_id).item())
                param[r, c] = orig - FD_EPS
                l_minus = float(_nll_loss(model, prompt_ids, target_id).item())
            finally:
                param[r, c] = orig  # 无论是否中断都恢复副本参数
            fd = (l_plus - l_minus) / (2.0 * FD_EPS)
            entries.append({
                "name": name,
                "row": r,
                "col": c,
                "note": note,
                "autodiff": auto,
                "finite_diff": fd,
                "abs_error": abs(auto - fd),
            })

    status, max_err, first_fail = _summarize(entries)
    return {
        "status": status,
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "next_char": next_char,
        "next_token_id": target_id,
        "block": block_idx + 1,
        "dtype": "float64",
        "loss": loss_value,
        "fd_eps": FD_EPS,
        "tolerance": TOLERANCE,
        "max_abs_error": max_err,
        "first_failure": (
            {"name": first_fail["name"], "row": first_fail["row"],
             "col": first_fail["col"], "abs_error": first_fail["abs_error"]}
            if first_fail is not None else None
        ),
        "entries": entries,
    }
