"""Bearer 鉴权: 没配就不校验; 缺 key 403, 错 key 401 (照官方真机)."""

import pytest

from minicpm_jev.service import AuthenticationError, AuthGuard, MissingKeyError


def test_disabled_guard_passes_everything() -> None:
    """没配 key 时怎么调都放行 (本地默认不鉴权)."""
    guard = AuthGuard(None)
    assert guard.enabled is False
    guard.check(None)
    guard.check("Bearer whatever")
    guard.check("garbage")






def test_missing_header_is_403() -> None:
    """完全没带 key -> 403."""
    guard = AuthGuard("sekret")
    with pytest.raises(MissingKeyError) as error:
        guard.check(None)
    assert error.value.status_code == 403
    assert error.value.error_type == "authentication_error"
    with pytest.raises(MissingKeyError):
        guard.check("")


def test_wrong_key_is_401() -> None:
    """带了但不对 -> 401; 不做任何宽容."""
    guard = AuthGuard("sekret")
    with pytest.raises(AuthenticationError) as error:
        guard.check("Bearer nope")
    assert error.value.status_code == 401
    with pytest.raises(AuthenticationError):
        guard.check("sekret")
    with pytest.raises(AuthenticationError):
        guard.check("Bearer sekret ")
    with pytest.raises(AuthenticationError):
        guard.check("bearer sekret")


def test_error_body_shape_is_official() -> None:
    """错误体: {"detail": {"error_type", "message"}}, 消息是英文."""
    guard = AuthGuard("sekret")
    with pytest.raises(AuthenticationError) as error:
        guard.check("Bearer nope")
    detail = error.value.detail()
    assert set(detail) == {"detail"}
    # detail 字段在基类里是 dict、在 422 那类里是数组, 所以这里先按运行时形状收窄
    body = detail["detail"]
    assert isinstance(body, dict)
    assert body["error_type"] == "authentication_error"
    assert "API key" in body["message"]
