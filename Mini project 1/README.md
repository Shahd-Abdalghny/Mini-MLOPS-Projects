<!-- @format -->

# Laptop Price Prediction API

## Overview

Laptop Price Prediction API is an end-to-end MLOps project for predicting laptop prices from hardware and software specifications. It covers data preparation, feature engineering, model training and evaluation, model export, inference, automated testing, logging, FastAPI serving, and Docker containerization.

## Features

- Data handling and preprocessing
- Feature engineering
- Machine learning model training and inference
- Model export and parity testing
- FastAPI prediction service with input validation
- Configurable logging
- Automated tests with `pytest`
- Dependency management with `uv`
- Multi-stage Docker image running as a non-root user

## Project Architecture

The project separates model development from serving concerns. Training and preprocessing code lives in the `prodml` package, exported artifacts are stored in `models/`, and the API exposes prediction functionality through FastAPI. Tests verify application behavior and model consistency.

## Project Structure

```text
mini-project-1/
├── data/
├── models/
├── notebooks/
├── src/
│   └── prodml/
│       ├── api/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── parity_test.py
│       ├── predict.py
│       └── train.py
├── tests/
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

Important components:

- `data/`: Project datasets.
- `models/`: Trained or exported model artifacts.
- `notebooks/`: Exploratory analysis and experimentation notebooks.
- `src/prodml/`: Main Python package.
- `src/prodml/api/`: API-related components.
- `train.py`: Model training.
- `predict.py`: Model inference and prediction.
- `features.py`: Feature engineering and preprocessing.
- `data.py`: Data-related functionality.
- `export.py`: Model export.
- `config.py`: Project configuration.
- `logging_conf.py`: Logging configuration.
- `parity_test.py`: Model parity and consistency testing.
- `tests/`: Automated tests.
- `Dockerfile`: Docker image and container setup.
- `pyproject.toml`: Project metadata, dependencies, and configuration.
- `uv.lock`: Locked dependency versions.
- `.dockerignore`, `.gitignore`, `.python-version`: Build exclusions, Git exclusions, and Python-version configuration.

## Machine Learning Workflow

```text
Data
↓
Data preprocessing
↓
Feature engineering
↓
Model training
↓
Model evaluation
↓
Model export
↓
Prediction/inference
↓
FastAPI
↓
Docker
```

The data is prepared and transformed into model-ready features. A model is trained and evaluated, then exported for inference. The prediction logic is exposed through FastAPI and can be packaged in a Docker container.

## Model

The model predicts laptop prices from the following specifications:

- Company
- TypeName
- ScreenResolution
- Cpu
- Ram
- Memory
- Gpu
- OpSys
- Weight
- Inches

## Model Performance

- **R² score:** `0.8871`
- **MAE on log-transformed price:** `0.1596`

The R² score indicates how much variance in the target is explained by the model. The MAE is measured on the log-transformed price, so it represents the average absolute error in that transformed scale. These results describe the reported evaluation and do not imply production readiness.

## API

The FastAPI application accepts laptop specifications, validates the request, and returns a predicted laptop price. The exact endpoint path, port, and response schema depend on the application configuration and are represented here with placeholders.

Example request:

```json
{
  "Company": "HP",
  "TypeName": "Ultrabook",
  "ScreenResolution": "Full HD 1920x1080",
  "Cpu": "Intel Core i7 8550U 1.8GHz",
  "Ram": "16GB",
  "Memory": "512GB SSD",
  "Gpu": "Intel UHD Graphics 620",
  "OpSys": "Windows 10",
  "Weight": "1.3kg",
  "Inches": 14.0
}
```

Use the configured `/predict` to submit the request.

FastAPI may also provide interactive documentation, such as Swagger UI, at the configured documentation URL: `http://localhost:8000/docs`.

## Running Locally

From the project root, install `uv` if necessary, then create or synchronize the project environment:

```bash
uv sync
```

Activate the environment if desired:

```bash
 # macOS/Linux
source .venv/bin/activate

 # Windows PowerShell
.venv\Scripts\Activate.ps1
```

Train or export the model using the project’s configured module/script entry points as required. The exact invocation depends on the package configuration; consult `pyproject.toml` and the relevant source module.

Run the FastAPI application using the configured API module:

```bash
uv run uvicorn prodml.api.main:app --reload
```

## Testing

Run the automated test suite with:

```bash
uv run pytest
```

The tests are intended to verify data and prediction behavior, API-related functionality, and other project components. `src/prodml/parity_test.py` supports model parity and consistency checks so that exported/inference behavior remains consistent with the reference model behavior.

## Docker

The current image name is `laptop-price-api:0.1.0`.

Build the image:

```bash
docker build -f Dockerfile -t laptop-price-api:0.1.0 .
```

Run the container, replacing `<PORT>` like `8000` with the port configured by the application:

```bash
docker run --rm -p <PORT>:<PORT> laptop-price-api:0.1.0
```

Check running containers:

```bash
docker ps
```

Access the API from the host machine at:

```text
http://localhost:<PORT>/<API_ENDPOINT>
```

You can also pull and run the image directly from Docker Hub:

```bash
docker run --rm -p 8000:8000 shahdabdelghany/laptop-price-api:0.1.0
```

The Dockerfile uses a multi-stage build to keep build dependencies separate from the runtime image and help reduce unnecessary image contents. Running as a non-root user limits privileges inside the container and improves its security posture.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI and Pydantic/Pydantic Settings
- Uvicorn
- ONNX, ONNX Runtime, and ONNXMLTools
- pytest
- Docker
- uv

## MLOps Practices

This project demonstrates:

- A reproducible source and artifact structure
- Locked dependency management with `uv`
- Model training, export, and inference separation
- Automated testing with `pytest`
- Model parity and consistency checks
- Centralized logging configuration
- API-based model serving
- Docker containerization
- Separation between development/training and serving workflows

CI/CD, cloud deployment, experiment tracking, model monitoring, and a model registry are not included features of the current project.

## Future Improvements

Potential future work includes:

- Adding a CI/CD pipeline
- Introducing model versioning and a model registry
- Adding experiment tracking and stronger data validation
- Implementing model and API monitoring
- Automating model retraining
- Deploying to a cloud platform

## Author

Shahd Abdelghany
