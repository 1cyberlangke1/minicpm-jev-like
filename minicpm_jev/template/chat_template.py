"""官方对话模板渲染.

铁律: 一切 prompt 必须套 GGUF 官方对话模板. 这里只调用 llama.cpp 的库函数
llama_model_chat_template / llama_chat_apply_template, 不手搓 ChatML, 也不允许
裸文本 —— chat 模型在错模板下会降智, 测出来的分数全部无效.

关于 assistant 预填: 本模型模板会把末尾的 assistant 消息补上 <|im_end|> 再另起一个
assistant 头, 直接塞 assistant 消息拿不到「续写位」. 所以预填前缀走 assistant_prefix
参数: 模板渲染完 (结尾已是 "<|im_start|>assistant\n") 之后, 只往后接前缀内容, 不碰
任何模板标记.
"""

from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence

import llama_cpp

__all__ = ["ChatTemplateError", "get_chat_template", "render_chat"]


class ChatTemplateError(RuntimeError):
    """模板缺失或渲染失败. 直接抛错, 不静默回退到内置模板."""


def get_chat_template(model: object) -> bytes:
    """从 GGUF 元数据取 tokenizer.chat_template.

    输入: model -- llama_cpp._internals.LlamaModel
    输出: 模板字节串
    预期: 元数据里没有模板时抛 ChatTemplateError, 绝不裸文本回退
    """
    template = llama_cpp.llama_model_chat_template(model.model, None)  # type: ignore[attr-defined]
    if not template:
        raise ChatTemplateError("GGUF 元数据里没有 tokenizer.chat_template, 拒绝裸文本回退")
    return template


def _encode_messages(messages: Sequence[Mapping[str, str]]) -> tuple[list[bytes], list[bytes]]:
    """把消息列表编码成 UTF-8 的 role/content 字节串.

    输入: messages -- role/content 映射序列
    输出: (role 字节串列表, content 字节串列表)
    预期: 字段缺失或类型不对直接抛错, 不做字段补全
    """
    roles: list[bytes] = []
    contents: list[bytes] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"第 {index} 条消息不是 role/content 映射, 收到 {type(message).__name__}")
        try:
            role = message["role"]
            content = message["content"]
        except KeyError as err:
            raise ValueError(f"第 {index} 条消息缺少 {err.args[0]!r} 字段") from err
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError(f"第 {index} 条消息的 role/content 必须是 str")
        roles.append(role.encode("utf-8"))
        contents.append(content.encode("utf-8"))
    return roles, contents


def render_chat(
    model: object,
    messages: Sequence[Mapping[str, str]],
    *,
    add_assistant: bool = True,
    assistant_prefix: str = "",
    template: bytes | None = None,
) -> bytes:
    """按官方模板把消息列表渲染成 prompt 字节.

    输入: model -- LlamaModel; messages -- role/content 映射序列;
          add_assistant -- 是否以 assistant 头收尾 (决策位);
          assistant_prefix -- 接在 assistant 头之后的预填前缀 (只加内容);
          template -- 已取出的模板, 省一次元数据查询
    输出: 渲染后的 prompt 字节 (UTF-8)
    预期: 与 llama.cpp 内部渲染逐字节一致; 模板报错时抛 ChatTemplateError
    """
    resolved_template = template if template is not None else get_chat_template(model)
    if not messages:
        raise ValueError("messages 不能为空")
    roles, contents = _encode_messages(messages)

    chat = (llama_cpp.llama_chat_message * len(roles))()
    for index in range(len(roles)):
        chat[index].role = roles[index]
        chat[index].content = contents[index]

    # 第一遍只问长度 (buf=None), 第二遍才真写, 避免猜缓冲区大小
    needed = llama_cpp.llama_chat_apply_template(
        resolved_template, chat, len(roles), add_assistant, None, 0
    )
    if needed < 0:
        raise ChatTemplateError(f"llama_chat_apply_template 返回 {needed}")
    buffer = ctypes.create_string_buffer(needed + 1)
    written = llama_cpp.llama_chat_apply_template(
        resolved_template,
        chat,
        len(roles),
        add_assistant,
        ctypes.cast(buffer, ctypes.c_char_p),
        needed + 1,
    )
    if written < 0 or written > needed:
        raise ChatTemplateError(f"模板渲染长度异常: needed={needed} written={written}")
    return buffer.raw[:written] + assistant_prefix.encode("utf-8")
