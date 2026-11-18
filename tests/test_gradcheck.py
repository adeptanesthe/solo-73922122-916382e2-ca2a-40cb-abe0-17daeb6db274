"""梯度核对的关键功能测试：

  1. 固定用例下自动求导与中心差分一致（max |err| ≤ 1e-5），8 个坐标齐全；
  2. 绑定词嵌入/输出权重的梯度 = 输入路径贡献 + 输出路径贡献，
     且核对用模型确实保持权重绑定（差分扰动同时作用于两条路径）；
  3. 梯度核对不触碰在线单例：权重、KV 缓存、前向计数、全局 RNG 均不变，
     核对后原解码结果逐 token 不变；
  4. 判定逻辑：任一坐标误差 > 1e-5 即 FAIL 并指出首个超差坐标；
  5. API 校验与「中断绝不给出通过结论」。
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import tokenizer
from app.decoding import greedy_baseline
from app.gradcheck import TOLERANCE, _build_float64_model, _nll_loss, _summarize, run_gradcheck
from app.model import get_models

torch.set_num_threads(1)
DRAFT, TARGET = get_models()


def test_gradcheck_passes_fixed_cases() -> None:
    cases = [
        ("hello world", "!", 1),          # 块 2
        ("a", " ", 0),                    # 最短提示词（1 字符），块 1
        ("The quick brown fox!", "z", 2),   # 块 3
        ("0 1 1 2 3 5 8 13 21 34 5", "1", 3),   # 24 字符上限，块 4
    ]
    for prompt, ch, blk in cases:
        assert 1 <= len(prompt) <= 24
        r = run_gradcheck(prompt, ch, blk)
        assert r["status"] == "PASS", (prompt, ch, blk, r["max_abs_error"])
        assert len(r["entries"]) == 8, "Q/K/V 各 2 + 绑定嵌入 2，共 8 项"
        assert r["max_abs_error"] <= TOLERANCE
        assert r["first_failure"] is None
        assert r["block"] == blk + 1 and r["dtype"] == "float64"
        names = [e["name"] for e in r["entries"]]
        assert names.count(f"block{blk + 1}.q_proj") == 2
        assert names.count(f"block{blk + 1}.k_proj") == 2
        assert names.count(f"block{blk + 1}.v_proj") == 2
        assert names.count("tied_emb/lm_head") == 2
        for e in r["entries"]:
            assert e["abs_error"] == abs(e["autodiff"] - e["finite_diff"])
        # 绑定嵌入坐标的行必须对应提示词字符的 token id
        ids = tokenizer.encode(prompt)
        emb_rows = [e["row"] for e in r["entries"] if e["name"] == "tied_emb/lm_head"]
        assert emb_rows == [ids[0], ids[-1]]
        print(f"ok: 梯度核对 PASS prompt={prompt!r} next={ch!r} 块{blk + 1} "
              f"max_err={r['max_abs_error']:.3e}")


def test_tied_weight_gradient_is_two_path_sum() -> None:
    """绑定权重梯度必须同时包含输入（嵌入查表）与输出（LM head）两条路径。"""
    prompt_ids = tokenizer.encode("hello world")
    tid = tokenizer.encode("!")[0]
    row, col = prompt_ids[0], 0

    tied = _build_float64_model()
    # 核对用模型必须真的保持绑定（同一参数对象），否则差分扰动无法覆盖双路径
    assert tied.tok_emb.weight is tied.lm_head.weight
    tied.zero_grad()
    _nll_loss(tied, prompt_ids, tid).backward()
    g_tied = tied.tok_emb.weight.grad[row, col].item()

    untied = _build_float64_model()
    with torch.no_grad():
        untied.lm_head.weight = torch.nn.Parameter(
            untied.tok_emb.weight.detach().clone(), requires_grad=True)
    untied.zero_grad()
    _nll_loss(untied, prompt_ids, tid).backward()
    g_in = untied.tok_emb.weight.grad[row, col].item()   # 仅输入路径
    g_out = untied.lm_head.weight.grad[row, col].item()  # 仅输出路径

    assert abs(g_tied - (g_in + g_out)) < 1e-12, (
        f"绑定梯度 {g_tied} != 输入 {g_in} + 输出 {g_out}")
    print(f"ok: 绑定权重梯度 = 输入路径 + 输出路径（g={g_tied:.6e}）")


def test_singletons_untouched_by_gradcheck() -> None:
    """核对期间原解码可用，原权重、KV 缓存与前向计数不得改变。"""
    # 快照两个在线模型的全部权重、缓存状态、前向计数与全局 RNG
    w_target = [p.detach().clone() for p in TARGET.parameters()]
    w_draft = [p.detach().clone() for p in DRAFT.parameters()]
    cache_state = (TARGET.cache_len, DRAFT.cache_len,
                   TARGET._cache is None, DRAFT._cache is None)
    cache_tensors = None
    if TARGET._cache is not None:
        cache_tensors = [(k.clone(), v.clone()) for k, v in TARGET._cache]
    rng_state = torch.random.get_rng_state()

    # 核对前记录一次在线解码输出
    ids = tokenizer.encode("the quick brown ")
    before = greedy_baseline(TARGET, ids, 5).new_tokens
    counts_after_decode = (TARGET.forward_count, DRAFT.forward_count)

    run_gradcheck("hello world", "!", 1)
    run_gradcheck("The quick brown fox jumps!", "z", 3)

    # 权重逐位不变
    for p, w in zip(TARGET.parameters(), w_target):
        assert torch.equal(p, w), "目标模型权重被梯度核对修改"
    for p, w in zip(DRAFT.parameters(), w_draft):
        assert torch.equal(p, w), "草稿模型权重被梯度核对修改"
    # 前向计数不受核对影响（核对用的是独立副本）
    assert (TARGET.forward_count, DRAFT.forward_count) == counts_after_decode
    # KV 缓存状态不变
    assert (TARGET.cache_len, DRAFT.cache_len,
            TARGET._cache is None, DRAFT._cache is None) == cache_state
    if cache_tensors is not None:
        for (k, v), (k0, v0) in zip(TARGET._cache, cache_tensors):
            assert torch.equal(k, k0) and torch.equal(v, v0)
    # 全局 RNG 不被构建副本扰动
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    # 核对后原解码结果逐 token 不变
    after = greedy_baseline(TARGET, ids, 5).new_tokens
    assert before == after
    print("ok: 梯度核对后在线权重/KV/前向计数/RNG 均不变，解码输出逐 token 一致")


def test_fail_judgement_reports_first_coordinate() -> None:
    def entry(name, err):
        return {"name": name, "row": 0, "col": 0, "abs_error": err,
                "autodiff": 0.0, "finite_diff": err}

    status, max_err, first = _summarize([entry("a", 1e-9), entry("b", 1e-6)])
    assert status == "PASS" and first is None and max_err == 1e-6

    # 恰好等于阈值不算超差（判定规则为“超过”）
    status, _, first = _summarize([entry("a", TOLERANCE)])
    assert status == "PASS" and first is None

    entries = [entry("a", 1e-9), entry("b", 2e-5), entry("c", 9e-5)]
    status, max_err, first = _summarize(entries)
    assert status == "FAIL"
    assert first is entries[1], "必须指出首个超差坐标"
    assert max_err == 9e-5
    print("ok: 判定逻辑正确（超阈值即 FAIL 并指出首个坐标，等于阈值不判负）")


def test_api_gradcheck_ok_and_validation() -> None:
    from fastapi.testclient import TestClient

    from app.main import app
    client = TestClient(app)

    r = client.post("/api/gradcheck",
                    json={"prompt": "hello world", "next_char": "!", "block": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "PASS" and len(body["entries"]) == 8
    assert body["max_abs_error"] <= 1e-5

    # 参数校验：长度越界 → 422；非可打印 ASCII → 400
    assert client.post("/api/gradcheck", json={
        "prompt": "x" * 25, "next_char": "!", "block": 1}).status_code == 422
    assert client.post("/api/gradcheck", json={
        "prompt": "", "next_char": "!", "block": 1}).status_code == 422
    assert client.post("/api/gradcheck", json={
        "prompt": "a\nb", "next_char": "!", "block": 1}).status_code == 400
    assert client.post("/api/gradcheck", json={
        "prompt": "héllo", "next_char": "!", "block": 1}).status_code == 400
    assert client.post("/api/gradcheck", json={
        "prompt": "hi", "next_char": "!!", "block": 1}).status_code == 422
    assert client.post("/api/gradcheck", json={
        "prompt": "hi", "next_char": "é", "block": 1}).status_code == 400
    assert client.post("/api/gradcheck", json={
        "prompt": "hi", "next_char": "!", "block": 0}).status_code == 422
    assert client.post("/api/gradcheck", json={
        "prompt": "hi", "next_char": "!", "block": 5}).status_code == 422
    print("ok: API 正常返回 PASS，非法输入分别返回 400/422")


def test_api_interruption_never_passes() -> None:
    """计算中断时接口必须报错，绝不下发通过结论。"""
    from fastapi.testclient import TestClient

    import app.main as main_mod
    client = TestClient(main_mod.app)

    orig = main_mod.run_gradcheck

    def boom(*args, **kwargs):
        raise RuntimeError("模拟中断")

    main_mod.run_gradcheck = boom
    try:
        r = client.post("/api/gradcheck",
                        json={"prompt": "hello", "next_char": "!", "block": 1})
        assert r.status_code == 500
        body = r.json()
        assert body.get("status") == "ERROR"
        assert "PASS" not in str(body), "中断响应中不得出现通过结论"
    finally:
        main_mod.run_gradcheck = orig
    print("ok: 核对中断时返回 500 + ERROR，不含任何通过结论")


if __name__ == "__main__":
    test_gradcheck_passes_fixed_cases()
    test_tied_weight_gradient_is_two_path_sum()
    test_singletons_untouched_by_gradcheck()
    test_fail_judgement_reports_first_coordinate()
    test_api_gradcheck_ok_and_validation()
    test_api_interruption_never_passes()
    print("\n梯度核对全部测试通过 ✔")
