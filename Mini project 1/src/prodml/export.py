import pickle

from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert.xgboost.shape_calculators.Regressor import (
    calculate_linear_regressor_output_shapes,  
)
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from xgboost import XGBRegressor

from prodml.config import settings

update_registered_converter(
    XGBRegressor,
    "XGBoostXGBRegressor",
    calculate_linear_regressor_output_shapes,  
    convert_xgboost,
)

INITIAL_TYPES = [
    ("Company", StringTensorType([None, 1])),
    ("TypeName", StringTensorType([None, 1])),
    ("Ram", Int64TensorType([None, 1])),
    ("Weight", FloatTensorType([None, 1])),
    ("Touchscreen", Int64TensorType([None, 1])),
    ("Ips", Int64TensorType([None, 1])),
    ("Cpu brand", StringTensorType([None, 1])),
    ("Gpu brand", StringTensorType([None, 1])),
    ("os", StringTensorType([None, 1])),
    ("HDD", Int64TensorType([None, 1])),
    ("SSD", Int64TensorType([None, 1])),
    ("ppi", FloatTensorType([None, 1])),
]


def export_to_onnx(output_path: str = settings.onnx_model_path) -> None:
    with open(settings.model_path, "rb") as f:
        pipe = pickle.load(f)

    onnx_model = to_onnx(
    pipe,
    initial_types=INITIAL_TYPES,
    target_opset={"": 15, "ai.onnx.ml": 3},
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model saved to {output_path}")


if __name__ == "__main__":
    export_to_onnx()