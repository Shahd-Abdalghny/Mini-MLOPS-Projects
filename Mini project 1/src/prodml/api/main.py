import time
import uuid
from contextlib import asynccontextmanager
from prodml.config import settings
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.logging_conf import configure_logging, correlation_id_var
from prodml.predict import LaptopPricePredictor

import logging

configure_logging()
logger = logging.getLogger(__name__)

MODEL_VERSION = "0.1.0"

predictor = LaptopPricePredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """this function is called when the server starts and stops. It can be used to load the model into memory."""
    
    logger.info("Loading model...")
    predictor.load()
    logger.info("Model loaded successfully")
    yield
    


app = FastAPI(title="Laptop Price Prediction API", lifespan=lifespan)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
async def health():
    if predictor.session is None:
        logger.warning("Health check failed: model not loaded")
        return JSONResponse(status_code=503, content={"status": "model not loaded"})
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata():
    return {
        "model_version": MODEL_VERSION,
        "framework": "scikit-learn (VotingRegressor)",
        "feature_names": settings.FEATURE_COLUMNS,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start = time.perf_counter()

    features = request.model_dump()
    price = predictor.predict_one(features)

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(f"prediction served: price={price:.2f} latency_ms={latency_ms:.2f}")

    return PredictionResponse(
        prediction=price,
        model_version=MODEL_VERSION,
        correlation_id=correlation_id_var.get(),
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    start = time.perf_counter()

    rows = [item.model_dump() for item in request.items]
    prices = predictor.predict_batch(rows)

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(f"batch prediction served: count={len(prices)} latency_ms={latency_ms:.2f}")

    return BatchPredictionResponse(
        predictions=prices,
        model_version=MODEL_VERSION,
        correlation_id=correlation_id_var.get(),
        latency_ms=latency_ms,
    )