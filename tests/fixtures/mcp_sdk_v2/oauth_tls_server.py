from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode

import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route


ACCESS_TOKEN = "agent-libos-oauth-tls-access-token"
AUTHORIZATION_CODE = "agent-libos-oauth-tls-code"
RESOURCE_URI = "fixture://oauth/status"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


class OAuthTlsFixture:
    def __init__(self, *, port: int, evidence_path: Path) -> None:
        self.origin = f"https://localhost:{port}"
        self.resource = f"{self.origin}/mcp"
        self.redirect_uri = "http://127.0.0.1:8765/oauth/callback"
        self.client_id = "agent-libos-oauth-tls-client"
        self.evidence_path = evidence_path
        self._evidence: dict[str, bool] = {}
        self._authorization: dict[str, str] | None = None

        server = MCPServer(
            "agent-libos-oauth-tls-fixture",
            version="2.0.0",
            log_level="ERROR",
        )

        @server.resource(
            RESOURCE_URI,
            name="oauth-status",
            description="A deterministic resource protected by the OAuth TLS fixture.",
            mime_type="text/plain",
        )
        def oauth_status() -> str:
            return "agent-libos OAuth TLS Runtime path authorized"

        app = server.streamable_http_app(
            streamable_http_path="/mcp",
            json_response=True,
            stateless_http=True,
            host="localhost",
        )
        app.router.routes[0:0] = [
            Route(
                "/.well-known/oauth-protected-resource/mcp",
                self.protected_resource_metadata,
                methods=["GET"],
            ),
            Route(
                "/.well-known/oauth-authorization-server",
                self.authorization_server_metadata,
                methods=["GET"],
            ),
            Route("/authorize", self.authorize, methods=["GET"]),
            Route("/token", self.token, methods=["POST"]),
        ]
        self.app = _BearerProtectedApp(app, fixture=self)

    def observe(self, **items: bool) -> None:
        self._evidence.update(items)
        temporary = self.evidence_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._evidence, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.evidence_path)

    async def protected_resource_metadata(self, _request: Request) -> Response:
        self.observe(protected_resource_metadata_served=True)
        return JSONResponse(
            {
                "resource": self.resource,
                "authorization_servers": [self.origin],
                "scopes_supported": ["mcp.read"],
            }
        )

    async def authorization_server_metadata(self, _request: Request) -> Response:
        self.observe(authorization_server_metadata_served=True)
        return JSONResponse(
            {
                "issuer": self.origin,
                "authorization_endpoint": f"{self.origin}/authorize",
                "token_endpoint": f"{self.origin}/token",
                "code_challenge_methods_supported": ["S256"],
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code"],
                "token_endpoint_auth_methods_supported": ["none"],
                "authorization_response_iss_parameter_supported": True,
                "scopes_supported": ["mcp.read"],
            }
        )

    async def authorize(self, request: Request) -> Response:
        query = request.query_params
        required = {
            "response_type",
            "client_id",
            "redirect_uri",
            "resource",
            "scope",
            "state",
            "code_challenge",
            "code_challenge_method",
        }
        if not required.issubset(query):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        valid = {
            "response_type": query["response_type"] == "code",
            "client_id": query["client_id"] == self.client_id,
            "redirect_uri": query["redirect_uri"] == self.redirect_uri,
            "resource": query["resource"] == self.resource,
            "scope": query["scope"] == "mcp.read",
            "state": bool(query["state"]),
            "challenge": len(query["code_challenge"]) == 43,
            "challenge_method": query["code_challenge_method"] == "S256",
        }
        if not all(valid.values()):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        self._authorization = {
            "state": query["state"],
            "code_challenge": query["code_challenge"],
        }
        self.observe(
            authorization_client_id_pinned=valid["client_id"],
            authorization_redirect_pinned=valid["redirect_uri"],
            authorization_resource_pinned=valid["resource"],
            authorization_scope_pinned=valid["scope"],
            authorization_state_present=valid["state"],
            authorization_pkce_s256=valid["challenge"] and valid["challenge_method"],
        )
        location = f"{self.redirect_uri}?{urlencode({'code': AUTHORIZATION_CODE, 'state': query['state'], 'iss': self.origin})}"
        return RedirectResponse(location, status_code=302)

    async def token(self, request: Request) -> Response:
        authorization = self._authorization
        if authorization is None:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        body = await request.body()
        try:
            form = parse_qs(body.decode("ascii"), strict_parsing=True)
            code_verifier = form["code_verifier"][0]
        except (KeyError, UnicodeDecodeError, ValueError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        derived_challenge = _b64url(hashlib.sha256(code_verifier.encode("ascii")).digest())
        valid = {
            "grant_type": form.get("grant_type") == ["authorization_code"],
            "code": form.get("code") == [AUTHORIZATION_CODE],
            "client_id": form.get("client_id") == [self.client_id],
            "redirect_uri": form.get("redirect_uri") == [self.redirect_uri],
            "resource": form.get("resource") == [self.resource],
            "pkce": derived_challenge == authorization["code_challenge"],
        }
        if not all(valid.values()):
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        self.observe(
            token_code_bound=valid["code"],
            token_client_id_pinned=valid["client_id"],
            token_redirect_pinned=valid["redirect_uri"],
            token_resource_pinned=valid["resource"],
            token_pkce_verified=valid["pkce"],
        )
        return JSONResponse(
            {
                "access_token": ACCESS_TOKEN,
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "mcp.read",
            }
        )


class _BearerProtectedApp:
    def __init__(self, app: Any, *, fixture: OAuthTlsFixture) -> None:
        self._app = app
        self._fixture = fixture

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/mcp":
            headers = {
                bytes(key).lower(): bytes(value)
                for key, value in scope.get("headers", ())
            }
            if headers.get(b"authorization") != f"Bearer {ACCESS_TOKEN}".encode("ascii"):
                response = JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": (
                            'Bearer resource_metadata="'
                            f"{self._fixture.origin}/.well-known/oauth-protected-resource/mcp"
                            '", scope="mcp.read"'
                        )
                    },
                )
                await response(scope, receive, send)
                return
            self._fixture.observe(mcp_bearer_verified=True)
        await self._app(scope, receive, send)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cert", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    fixture = OAuthTlsFixture(port=args.port, evidence_path=args.evidence)
    uvicorn.run(
        fixture.app,
        host="127.0.0.1",
        port=args.port,
        ssl_certfile=str(args.cert),
        ssl_keyfile=str(args.key),
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
