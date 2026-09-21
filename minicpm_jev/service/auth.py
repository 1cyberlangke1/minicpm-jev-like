"""Bearer 鉴权: 没配 key 就不校验 (本地默认关闭).

真机实测官方是「缺 key -> 403, key 不对 -> 401」, 这里逐码照抄。
比较走 ``hmac.compare_digest``, 不按字符短路。
"""

from __future__ import annotations

import hmac

from .errors import AuthenticationError, MissingKeyError

__all__ = ["AuthGuard"]


class AuthGuard:
    """把请求头里的 Authorization 与本地 key 对齐."""

    def __init__(self, api_key: str | None) -> None:
        """输入: api_key -- 本地 key (None / 空串 = 不鉴权); 输出: 无."""
        self._api_key = api_key or None
        self._expected = f"Bearer {self._api_key}" if self._api_key else None

    @property
    def enabled(self) -> bool:
        """是否开启鉴权."""
        return self._expected is not None

    def check(self, authorization: str | None) -> None:
        """校验请求头.

        输入: authorization -- 请求头原文 (None 表示没带);
        输出: 无;
        预期: 未启用鉴权直接放行; 没带 key -> MissingKeyError(403);
              key 不对 -> AuthenticationError(401); 不做大小写 / 空白宽容。
        """
        expected = self._expected
        if expected is None:
            return
        if not authorization:
            raise MissingKeyError(
                "Must supply an API key. Check your request and try again."
            )
        if not hmac.compare_digest(authorization, expected):
            raise AuthenticationError(
                "Cannot authenticate with the server. "
                "Please check your API key and try again."
            )
