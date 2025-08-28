"""Tests for the minimal order confirmation site."""

import io

from wsgiref.util import setup_testing_defaults

from app import application


def call_app(method="GET", path="/", data=""):
    """Helper to call the WSGI app and capture the response."""

    environ = {}
    setup_testing_defaults(environ)
    environ["REQUEST_METHOD"] = method
    environ["PATH_INFO"] = path

    body = data.encode("utf-8")
    environ["wsgi.input"] = io.BytesIO(body)
    environ["CONTENT_LENGTH"] = str(len(body))

    result = {}

    def start_response(status, headers):
        result["status"] = status
        result["headers"] = headers

    response_body = b"".join(application(environ, start_response))
    result["body"] = response_body
    return result


def test_order_get():
    response = call_app()
    assert response["status"].startswith("200")
    assert b"<form" in response["body"]


def test_order_post_generates_code():
    response = call_app(method="POST", data="email=test@example.com")
    assert response["status"].startswith("200")
    assert b"Order Confirmation" in response["body"]

