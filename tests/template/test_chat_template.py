"""对话模板: 渲染必须走官方模板, 不手搓 ChatML."""

import pytest

from minicpm_jev import BatchEngine
from minicpm_jev.template import render_chat

SYSTEM = "你是猫猫决策引擎, 只回答 yes 或 no。"


class _DummyModel:
    """只为验证参数校验路径; 显式给 template 时不会碰到底层模型."""

    model = None


def test_render_rejects_bad_messages() -> None:
    """字段缺失 / 消息为空 / 内容类型不对, 都要直接报错."""
    with pytest.raises(ValueError):
        render_chat(_DummyModel(), [{"content": "hi"}], template=b"x")
    with pytest.raises(ValueError):
        render_chat(_DummyModel(), [], template=b"x")
    with pytest.raises(TypeError):
        # 故意喂错类型的内容: 这里只放行静态检查, 运行时必须真的抛 TypeError
        render_chat(_DummyModel(), [{"role": "user", "content": 1}], template=b"x")  # type: ignore[dict-item]


def test_render_uses_official_template(engine: BatchEngine) -> None:
    """渲染结果要带官方模板标记, 并以 assistant 头加预填前缀收尾."""
    rendered = engine.render(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "问题: 1+1=2 吗?"},
        ],
        assistant_prefix="答案:",
    )
    assert rendered.startswith(b"<|im_start|>system")
    assert SYSTEM.encode("utf-8") in rendered
    assert rendered.endswith(b"<|im_start|>assistant\n" + "答案:".encode("utf-8"))


def test_render_without_assistant_header(engine: BatchEngine) -> None:
    """add_assistant=False 时不该补 assistant 头."""
    rendered = engine.render(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "问题: 1+1=2 吗?"},
        ],
        add_assistant=False,
    )
    assert not rendered.endswith(b"<|im_start|>assistant\n")
