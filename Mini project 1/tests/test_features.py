import pandas as pd

from prodml.features import (
    build_features,
    categorize_os,
    preprocess_cpu,
    preprocess_gpu,
    preprocess_screen_resolution,
)
import pytest

@pytest.mark.parametrize(
    "input_value, expected",
       [
        ("Windows 10", "Windows"),
        ("Mac OS X", "Mac"),
        ("Linux", "Others/No OS/Linux"),
        ("Other OS", "Others/No OS/Linux"),
    ],
)
def test_categorize_os(input_value, expected):
    assert categorize_os(input_value) == expected

@pytest.mark.parametrize(
    "input_value, expected",
    [
        ("IPS Panel Touchscreen 3840x2160", 1),
        ("Full HD 1920x1080", 0),
    ],
)
def test_preprocess_screen_resolution_touchscreen(input_value, expected):
    df = pd.DataFrame({"ScreenResolution": [input_value]})
    result = preprocess_screen_resolution(df)
    assert result["Touchscreen"].iloc[0] == expected
    

@pytest.mark.parametrize(
    "input_value, expected",
   [
        ("Intel Core i5 7200U 2.5GHz", "Intel Core i5"),
        ("AMD Ryzen 5 3500U", "AMD"),
        ("Intel Celeron N3050", "Other Intel Processor"),
    ],
)
def test_preprocess_cpu(input_value, expected):
    df = pd.DataFrame({"Cpu": [input_value]})
    result = preprocess_cpu(df)
    assert result["Cpu brand"].iloc[0] == expected


def test_preprocess_gpu_removes_arm():
    df = pd.DataFrame({"Gpu": ["ARM Mali T860 MP4", "Intel HD Graphics 620"]})
    result = preprocess_gpu(df)
    assert len(result) == 1
    assert result["Gpu brand"].iloc[0] == "Intel"


def test_preprocess_gpu_keeps_non_arm():
    df = pd.DataFrame({"Gpu": ["Intel HD Graphics 620", "Nvidia GeForce GTX 1050"]})
    result = preprocess_gpu(df)
    assert len(result) == 2
    assert list(result["Gpu brand"]) == ["Intel", "Nvidia"]