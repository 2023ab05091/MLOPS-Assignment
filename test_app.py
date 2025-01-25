import pytest
from flask.testing import FlaskClient
from app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_healthz(client: FlaskClient):
    response = client.get('/healthz')
    assert response.status_code == 200


def test_ready(client: FlaskClient):
    response = client.get('/ready')
    assert response.status_code == 200


def test_predict(client: FlaskClient, mocker):
    mocker.patch('app.model.predict_proba', return_value=[[0, 0.6]])
    response = client.post('/predict', data={'feature1': 1, 'feature2': 2})
    assert response.status_code == 200
    assert b'Your Forest is in Danger' in response.data

    mocker.patch('app.model.predict_proba', return_value=[[0, 0.4]])
    response = client.post('/predict', data={'feature1': 1, 'feature2': 2})
    assert response.status_code == 200
    assert b'Your Forest is safe' in response.data
