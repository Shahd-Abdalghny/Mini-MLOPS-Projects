from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    Company: str = Field(..., examples=["Dell"])
    TypeName: str = Field(..., examples=["Notebook"])
    ScreenResolution: str = Field(..., examples=["Full HD 1920x1080"])
    Cpu: str = Field(..., examples=["Intel Core i5 7200U 2.5GHz"])
    Ram: str = Field(..., examples=["8GB"])
    Memory: str = Field(..., examples=["256GB SSD"])
    Gpu: str = Field(..., examples=["Intel HD Graphics 620"])
    OpSys: str = Field(..., examples=["Windows 10"])
    Weight: str = Field(..., examples=["1.5kg"])
    Inches: float = Field(..., gt=0, lt=50, examples=[15.6])


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float