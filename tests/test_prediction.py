from src.models.predict import load_model


def test_model_loads():

    model = load_model()

    assert model is not None
