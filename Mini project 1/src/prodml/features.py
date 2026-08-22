import pandas as pd


def preprocess_screen_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the 'ScreenResolution' column in the DataFrame.
    This function extracts touch screen and IPS information from the 'ScreenResolution' column."""
    df = df.copy()
    df["Touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)
    df["Ips"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)

    new = df["ScreenResolution"].str.split("x", n=1, expand=True)
    df["X_res"] = new[0]
    df["Y_res"] = new[1]
    df["X_res"] = (
        df["X_res"].str.replace(",", "").str.findall(r"(\d+\.?\d+)").apply(lambda x: x[0])
    )
    df["X_res"] = df["X_res"].astype(int)
    df["Y_res"] = df["Y_res"].astype(int)

    df = df.drop(columns=["ScreenResolution"])
    return df

def _extract_processor(text: str) -> str:
    if text in ("Intel Core i7", "Intel Core i5", "Intel Core i3"):
        return text
    if text.split()[0] == "Intel":
        return "Other Intel Processor"
    return "AMD"


def preprocess_cpu(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the 'Cpu' column in the DataFrame.
    This function extracts the CPU brand from the 'Cpu' column."""
    df = df.copy()
    cpu_name = df["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))
    df["Cpu brand"] = cpu_name.apply(_extract_processor)
    df = df.drop(columns=["Cpu"])
    return df

def preprocess_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the 'Gpu' column in the DataFrame.
    This function extracts the GPU brand from the 'Gpu' column and filters out rows with 'ARM' GPUs."""
    df = df.copy()
    df["Gpu brand"] = df["Gpu"].apply(lambda x: x.split()[0])
    df = df[df["Gpu brand"] != "ARM"]
    df = df.drop(columns=["Gpu"])
    return df

def categorize_os(value: str) -> str:
    """Categorize the operating system based on its name."""
    if value in ("Windows 10", "Windows 7", "Windows 10 S"):
        return "Windows"
    if value in ("macOS", "Mac OS X"):
        return "Mac"
    return "Others/No OS/Linux"

def preprocess_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Memory"] = df["Memory"].astype(str).replace(r"\.0", "", regex=True)
    df["Memory"] = df["Memory"].str.replace("GB", "", regex=False)
    df["Memory"] = df["Memory"].str.replace("TB", "000", regex=False)

    new = df["Memory"].str.split("+", n=1, expand=True)
    first = new[0].str.strip()
    second = new[1]

    df["Layer1HDD"] = first.apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer1SSD"] = first.apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer1Hybrid"] = first.apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer1Flash_Storage"] = first.apply(lambda x: 1 if "Flash Storage" in x else 0)
    first = first.str.replace(r"\D", "", regex=True)

    second = second.fillna("0")
    df["Layer2HDD"] = second.apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer2SSD"] = second.apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer2Hybrid"] = second.apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer2Flash_Storage"] = second.apply(lambda x: 1 if "Flash Storage" in x else 0)
    second = second.str.replace(r"\D", "", regex=True)

    first = first.replace("", "0").astype(int)
    second = second.replace("", "0").astype(int)

    df["HDD"] = first * df["Layer1HDD"] + second * df["Layer2HDD"]
    df["SSD"] = first * df["Layer1SSD"] + second * df["Layer2SSD"]
    df["Hybrid"] = first * df["Layer1Hybrid"] + second * df["Layer2Hybrid"]
    df["Flash_Storage"] = first * df["Layer1Flash_Storage"] + second * df["Layer2Flash_Storage"]

    df = df.drop(columns=[
        "first", "second",
        "Layer1HDD", "Layer1SSD", "Layer1Hybrid", "Layer1Flash_Storage",
        "Layer2HDD", "Layer2SSD", "Layer2Hybrid", "Layer2Flash_Storage",
        "Memory",
    ], errors="ignore")
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features for the DataFrame by applying various preprocessing functions."""
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df = preprocess_screen_resolution(df)
    df = preprocess_cpu(df)
    df = preprocess_gpu(df)
    df["os"] = df["OpSys"].apply(categorize_os)
    df = df.drop(columns=["OpSys"])
    df = preprocess_memory(df)

    df["ppi"] = (((df["X_res"] ** 2) + (df["Y_res"] ** 2)) ** 0.5 / df["Inches"]).astype(float)
    df = df.drop(columns=["X_res", "Y_res", "Inches"])

    df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
    df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)
    df.drop(columns=["Hybrid","Flash_Storage"], inplace=True)

    return df