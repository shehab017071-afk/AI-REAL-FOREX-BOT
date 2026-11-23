
import os
import sys
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

SYMBOL = "EURUSD=X"

FEATURES = [
    "rsi", "ema_gap", "trend", "return1",
    "return3", "momentum", "volatility",
    "candle_body", "range_pct"
]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Telegram secrets are missing")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        },
        timeout=20
    )

    response.raise_for_status()

    if not response.json().get("ok"):
        raise RuntimeError("Telegram delivery failed")

    print("Telegram message sent successfully")


def main():
    print("AI MACHINE LEARNING FOREX BOT")
    print("Pair:", SYMBOL)

    data = yf.download(
        SYMBOL,
        period="5d",
        interval="5m",
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if data.empty or len(data) < 200:
        raise ValueError("Market data unavailable or insufficient")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
        high = data["High"].iloc[:, 0]
        low = data["Low"].iloc[:, 0]
        opening = data["Open"].iloc[:, 0]
    else:
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        opening = data["Open"]

    # Ignore the latest unfinished candle
    close = close.iloc[:-1]
    high = high.iloc[:-1]
    low = low.iloc[:-1]
    opening = opening.iloc[:-1]

    df = pd.DataFrame({"close": close})

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))
    df["ema_gap"] = ema9 / ema21 - 1
    df["trend"] = ema21 / ema50 - 1
    df["return1"] = close.pct_change()
    df["return3"] = close.pct_change(3)
    df["momentum"] = close / close.shift(6) - 1
    df["volatility"] = df["return1"].rolling(12).std()
    df["candle_body"] = (close - opening) / close
    df["range_pct"] = (high - low) / close

    future_return = close.shift(-3) / close - 1

    target = pd.Series(np.nan, index=df.index)
    target.loc[future_return > 0.00015] = 1
    target.loc[future_return < -0.00015] = -1
    target.loc[
        future_return.between(-0.00015, 0.00015)
    ] = 0

    training = df[FEATURES].copy()
    training["target"] = target

    training = training.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    latest = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().tail(1)

    if len(training) < 150 or latest.empty:
        raise ValueError("Not enough clean market data")

    split = int(len(training) * 0.8)
    train = training.iloc[:split-3]
    test = training.iloc[split:]

    if len(train) < 80 or len(test) < 20:
        raise ValueError("Insufficient validation data")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[FEATURES],
        train["target"].astype(int)
    )

    predictions = model.predict(test[FEATURES])

    accuracy = accuracy_score(
        test["target"].astype(int),
        predictions
    )

    model.fit(
        training[FEATURES],
        training["target"].astype(int)
    )

    probabilities = model.predict_proba(latest)[0]
    probability_map = dict(zip(model.classes_, probabilities))

    up = float(probability_map.get(1, 0))
    down = float(probability_map.get(-1, 0))
    neutral = float(probability_map.get(0, 0))

    if up >= 0.60 and up > down:
        signal = "UP"
    elif down >= 0.60 and down > up:
        signal = "DOWN"
    else:
        signal = "WAIT"

    price = float(close.iloc[-1])
    candle_time = str(df.index[-1])

    print("Validation accuracy:", round(accuracy * 100, 2), "%")
    print("UP:", round(up * 100, 2), "%")
    print("DOWN:", round(down * 100, 2), "%")
    print("NEUTRAL:", round(neutral * 100, 2), "%")
    print("AI SIGNAL:", signal)

    message = (
        "🤖 AI REAL FOREX BOT\n\n"
        f"Pair: {SYMBOL}\n"
        "Timeframe: 5 Minutes\n"
        "Expiry: 1 Minute (not validated)\n\n"
        f"Signal: {signal}\n"
        f"Price: {price:.5f}\n"
        f"UP probability: {up * 100:.2f}%\n"
        f"DOWN probability: {down * 100:.2f}%\n"
        f"NEUTRAL probability: {neutral * 100:.2f}%\n"
        f"Validation accuracy: {accuracy * 100:.2f}%\n"
        f"Candle: {candle_time}\n\n"
        "Educational signal only. Not a guarantee of profit."
    )

    send_telegram(message)
    print("Bot finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("BOT ERROR:", error)
        sys.exit(1)
