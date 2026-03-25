import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


@dataclass
class HorizonConfig:
    name: str
    horizon_days: int
    weight_return_col: str
    alpha: float
    rsi_window: int
    willr_window: int
    bb_window: int
    features: list[str]


CONFIG_1D = HorizonConfig(
    name="1d",
    horizon_days=1,
    weight_return_col="Return_1d",
    alpha=345,
    rsi_window=2,
    willr_window=7,
    bb_window=5,
    features=[
        "Return_1d",
        "Return_5d",
        "Return_10d",
        "RSI",
        "WILLR",
        "MA_5",
        "MA_10",
        "BB_percent_B",
    ],
)

CONFIG_7D = HorizonConfig(
    name="7d",
    horizon_days=7,
    weight_return_col="Return_7d",
    alpha=200,
    rsi_window=14,
    willr_window=14,
    bb_window=14,
    features=[
        "Return_1d",
        "MA_5",
        "MA_10",
        "STD_5",
        "STD_10",
        "RSI_14",
        "MACD",
        "WILLR_14",
        "Momentum_7",
    ],
)

CONFIG_30D = HorizonConfig(
    name="30d",
    horizon_days=30,
    weight_return_col="Return_30d",
    alpha=290,
    rsi_window=30,
    willr_window=43,
    bb_window=20,
    features=[
        "MA_5",
        "MA_10",
        "STD_5",
        "STD_10",
        "Momentum_35",
        "MACD",
        "RSI_30",
        "MA_30",
        "STD_30",
        "Return_35d",
        "Return_40d",
    ],
)


def download_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    if "Volume" in df.columns:
        df = df.drop(columns=["Volume"])
    return df


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Return_1d"] = out["Close"].pct_change(1)
    out["Return_5d"] = out["Close"].pct_change(5)
    out["Return_7d"] = out["Close"].pct_change(7)
    out["Return_10d"] = out["Close"].pct_change(10)
    out["Return_30d"] = out["Close"].pct_change(30)
    out["Return_35d"] = out["Close"].pct_change(35)
    out["Return_40d"] = out["Close"].pct_change(40)

    out["MA_5"] = out["Close"].rolling(5).mean()
    out["MA_10"] = out["Close"].rolling(10).mean()
    out["MA_30"] = out["Close"].rolling(30).mean()

    out["STD_5"] = out["Close"].rolling(5).std()
    out["STD_10"] = out["Close"].rolling(10).std()
    out["STD_30"] = out["Close"].rolling(30).std()

    out["Momentum_7"] = out["Close"] - out["Close"].shift(7)
    out["Momentum_35"] = out["Close"] - out["Close"].shift(35)

    macd = ta.trend.MACD(out["Close"])
    out["MACD"] = macd.macd_diff()

    out["RSI_14"] = ta.momentum.RSIIndicator(close=out["Close"], window=14).rsi()
    out["RSI_30"] = ta.momentum.RSIIndicator(close=out["Close"], window=30).rsi()

    out["WILLR_14"] = ta.momentum.WilliamsRIndicator(
        high=out["High"], low=out["Low"], close=out["Close"], lbp=14
    ).williams_r()

    out["WILLR_43"] = ta.momentum.WilliamsRIndicator(
        high=out["High"], low=out["Low"], close=out["Close"], lbp=43
    ).williams_r()

    return out


def add_dynamic_indicators(df: pd.DataFrame, cfg: HorizonConfig) -> pd.DataFrame:
    out = df.copy()

    out["RSI"] = ta.momentum.RSIIndicator(close=out["Close"], window=cfg.rsi_window).rsi()
    out["WILLR"] = ta.momentum.WilliamsRIndicator(
        high=out["High"],
        low=out["Low"],
        close=out["Close"],
        lbp=cfg.willr_window,
    ).williams_r()
    bb = ta.volatility.BollingerBands(close=out["Close"], window=cfg.bb_window, window_dev=2)
    out["BB_percent_B"] = bb.bollinger_pband()

    return out


def prepare_dataset(df: pd.DataFrame, cfg: HorizonConfig) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    out = add_base_features(df)
    out = add_dynamic_indicators(out, cfg)

    out["Target"] = (out["Close"].shift(-cfg.horizon_days) > out["Close"]).astype(int)

    change = out[cfg.weight_return_col].abs()
    out["Weight"] = 1 / (1 + cfg.alpha * change)

    required_cols = cfg.features + ["Target", "Weight"]
    out = out.dropna(subset=required_cols)

    X = out[cfg.features]
    y = out["Target"]
    w = out["Weight"]

    return X, y, w


def train_and_evaluate(
    symbol: str,
    cfg: HorizonConfig,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    random_state: int = 42,
):
    df = download_data(symbol, start=start, end=end)
    X, y, w = prepare_dataset(df, cfg)

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, w, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        eval_metric="logloss",
        random_state=random_state,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    )

    model.fit(X_train, y_train, sample_weight=w_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    p_value = binomtest(int(round(acc * len(y_test))), len(y_test), p=0.5, alternative="greater").pvalue

    sim_returns = np.where(y_pred == y_test.values, 1, -1)
    cumulative_score = int(sim_returns.sum())

    return {
        "symbol": symbol,
        "horizon": cfg.name,
        "rows": len(X),
        "test_rows": len(y_test),
        "accuracy": acc,
        "p_value_vs_coinflip": p_value,
        "cumulative_score": cumulative_score,
        "report": report,
        "model": model,
        "features": cfg.features,
    }


def transfer_evaluate(
    train_symbol: str,
    test_symbol: str,
    cfg: HorizonConfig,
    start_train: str = "2002-01-01",
    end_train: str = "2025-08-20",
    start_test: str = "2010-01-01",
    end_test: str = "2024-12-31",
):
    train_df = download_data(train_symbol, start=start_train, end=end_train)
    X_train_all, y_train_all, w_train_all = prepare_dataset(train_df, cfg)

    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    model.fit(X_train_all, y_train_all, sample_weight=w_train_all)

    test_df = download_data(test_symbol, start=start_test, end=end_test)
    X_test_all, y_test_all, _ = prepare_dataset(test_df, cfg)

    split_idx = int(len(X_test_all) * 0.8)
    X_test = X_test_all.iloc[split_idx:]
    y_test = y_test_all.iloc[split_idx:]

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    p_value = binomtest(int(round(acc * len(y_test))), len(y_test), p=0.5, alternative="greater").pvalue

    return {
        "train_symbol": train_symbol,
        "test_symbol": test_symbol,
        "horizon": cfg.name,
        "test_rows": len(y_test),
        "accuracy": acc,
        "p_value_vs_coinflip": p_value,
        "report": report,
    }


def print_result(result: dict) -> None:
    print("=" * 80)
    if "symbol" in result:
        print(f"Symbol: {result['symbol']} | Horizon: {result['horizon']}")
    else:
        print(
            f"Transfer {result['train_symbol']} -> {result['test_symbol']} | Horizon: {result['horizon']}"
        )

    print(f"Test rows: {result['test_rows']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"P-value vs coinflip: {result['p_value_vs_coinflip']:.3e}")

    if "cumulative_score" in result:
        print(f"Cumulative +/-1 score: {result['cumulative_score']}")

    print("Classification report:")
    print(result["report"])


def main() -> None:
    experiments = [
        ("EURUSD=X", CONFIG_1D),
        ("EURUSD=X", CONFIG_7D),
        ("EURUSD=X", CONFIG_30D),
        ("COPUSD=X", CONFIG_1D),
        ("^IXIC", CONFIG_1D),
    ]

    for symbol, cfg in experiments:
        result = train_and_evaluate(symbol=symbol, cfg=cfg)
        print_result(result)

    transfer_result = transfer_evaluate(
        train_symbol="EURUSD=X",
        test_symbol="COPUSD=X",
        cfg=CONFIG_1D,
    )
    print_result(transfer_result)


if __name__ == "__main__":
    main()
