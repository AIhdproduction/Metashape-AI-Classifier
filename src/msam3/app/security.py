"""Only the window that the app opened itself may read or control a session."""

import secrets
from http.cookies import CookieError, SimpleCookie
from urllib.parse import parse_qs


def new_token() -> str:
    return secrets.token_urlsafe(32)


class SessionGuard:
    """ASGI middleware around every page, file and websocket of the app.

    The start URL carries the token once; it becomes an HttpOnly, SameSite=Strict cookie. Host and
    Origin are checked as well, because cookies do not distinguish the ports of 127.0.0.1.
    """

    def __init__(self, app, token: str, port: int):
        self.app = app
        self.token = token
        self.host = f"127.0.0.1:{port}"
        self.origin = f"http://{self.host}"
        self.cookie = f"msam3_{port}"
        self.policy = (
            "default-src 'self' data: blob:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            f"connect-src 'self' ws://{self.host}; "
            "frame-ancestors 'none'"
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {}
        for name, value in scope.get("headers", []):
            headers[name.decode("latin-1").lower()] = value.decode("latin-1")
        if headers.get("host") != self.host or headers.get("origin", self.origin) != self.origin:
            await self._refuse(scope, send)
            return
        if scope["type"] == "http" and scope["path"] == "/":
            query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
            if self._matches(query.get("token", [""])[0]):
                await self._enter(send)
                return
        if not self._matches(self._cookie_value(headers.get("cookie", ""))):
            await self._refuse(scope, send)
            return

        async def send_protected(message):
            if message["type"] == "http.response.start":
                message["headers"] = [
                    *message.get("headers", []),
                    (b"content-security-policy", self.policy.encode("latin-1")),
                    (b"referrer-policy", b"no-referrer"),
                ]
            await send(message)

        await self.app(scope, receive, send_protected)

    def _matches(self, value: str) -> bool:
        return secrets.compare_digest(value.encode(), self.token.encode())

    def _cookie_value(self, header: str) -> str:
        # SimpleCookie drops the whole header over one odd cookie of another local app.
        for part in header.split(";"):
            try:
                parsed = SimpleCookie(part.strip())
            except CookieError:
                continue
            if self.cookie in parsed:
                return parsed[self.cookie].value
        return ""

    async def _enter(self, send):
        cookie = f"{self.cookie}={self.token}; Path=/; HttpOnly; SameSite=Strict"
        headers = [
            (b"location", b"/"),
            (b"set-cookie", cookie.encode("latin-1")),
            (b"cache-control", b"no-store"),
            (b"referrer-policy", b"no-referrer"),
        ]
        await send({"type": "http.response.start", "status": 303, "headers": headers})
        await send({"type": "http.response.body", "body": b""})

    async def _refuse(self, scope, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = b"Metashape AI Classifier: no access without this session's link."
        headers = [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]
        await send({"type": "http.response.start", "status": 403, "headers": headers})
        await send({"type": "http.response.body", "body": body})
