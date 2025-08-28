from app import app


def test_order_get():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200


def test_order_post_generates_code():
    client = app.test_client()
    response = client.post('/', data={'email': 'test@example.com'})
    assert response.status_code == 200
    assert b'Order Confirmation' in response.data
