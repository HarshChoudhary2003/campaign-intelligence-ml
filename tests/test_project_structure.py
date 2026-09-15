from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_artifact_exists():

    model_path = (
        ROOT /
        "artifacts/models/"
        "final_xgboost_model.joblib"
    )

    assert model_path.exists()


def test_readme_exists():

    assert (
        ROOT / "README.md"
    ).exists()


def test_requirements_exists():

    assert (
        ROOT / "requirements.txt"
    ).exists()


def test_docker_files_exist():

    assert (
        ROOT / "Dockerfile"
    ).exists()

    assert (
        ROOT / "docker-compose.yml"
    ).exists()
