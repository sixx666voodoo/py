"""Minimal order confirmation site using the Python standard library."""

from urllib.parse import parse_qs
from wsgiref.simple_server import make_server
from wsgiref.util import setup_testing_defaults
import io
import os
import secrets


BASE_DIR = os.path.dirname(__file__)
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")


def render_template(name: str, **context: str) -> str:
    """Return the contents of an HTML template with simple replacements."""

    path = os.path.join(TEMPLATE_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    for key, value in context.items():
        content = content.replace(f"{{{{ {key} }}}}", value)
    return content


def application(environ, start_response):
    """WSGI application handling the order form workflow."""

    setup_testing_defaults(environ)
    method = environ["REQUEST_METHOD"]
    path = environ.get("PATH_INFO", "/")

    if path == "/" and method == "GET":
        body = render_template("index.html")
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [body.encode("utf-8")]

    if path == "/" and method == "POST":
        try:
            size = int(environ.get("CONTENT_LENGTH", 0))
        except ValueError:
            size = 0
        data = environ["wsgi.input"].read(size).decode("utf-8")
        params = parse_qs(data)
        email = params.get("email", [""])[0]
        code = secrets.token_hex(4).upper()
        body = render_template("confirm.html", email=email, code=code)
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [body.encode("utf-8")]

    start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
    return [b"Not Found"]


if __name__ == "__main__":  # pragma: no cover - manual run helper
    with make_server("", 8000, application) as server:
        print("Serving on http://localhost:8000")
        server.serve_forever()

