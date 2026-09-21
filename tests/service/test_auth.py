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


def test_empty_key_counts_as_disabled() -> None:
    """空串等于没配."""
    assert AuthGuard("").enabled is False


def test_enabled_guard_accepts_exact_header() -> None:
    """配了 key 就要 Bearer + key 完全一致."""
    guard = AuthGuard("sekret")
    assert guard.enabled is True
    guard.check("Bearer sekret")


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
    assert detail["detail"]["error_type"] == "authentication_error"
    assert "API key" in detail["detail"]["message"]
