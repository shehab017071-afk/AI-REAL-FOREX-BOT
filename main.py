
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
TIMEFRAME = "5m"
PERIOD = "5d"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

FEATURES = [
    "rsi",
    "ema_gap",
    "trend",
    "return1",
    "return3",
    "momentum",
    "volatility",
    "candle_body",
    "range_pct"
]


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Telegram secrets are missing")

    url = (
        "https://api.telegram.org/bot"
        + TELEGRAM_TOKEN
        + "/sendMessage"
    )

    try:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            },
            timeout=20
        )

        result = response.json()

        if response.status_code != 200 or not result.get("ok"):
            raise RuntimeError("Telegram delivery failed")

        print("Telegram message sent successfully")

    except Exception:
        raise RuntimeError("Telegram message could not be sent")


def make_features(data):
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    opening = data["Open"].astype(float)

    df = pd.DataFrame(index=data.index)

    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    df["ema_gap"] = (ema9 - ema21) / close
    df["trend"] = (ema9 > ema21).astype(int)

    df["return1"] = close.pct_change()
    df["return3"] = close.pct_change(3)

    df["momentum"] = close - close.shift(5)
    df["volatility"] = df["return1"].rolling(10).std()

    df["candle_body"] = (close - opening) / close
    df["range_pct"] = (high - low) / close

    return df.replace([np.inf, -np.inf], np.nan)


def main():
    print("AI MACHINE LEARNING FOREX BOT")
    print("Pair:", SYMBOL)

    data = yf.download(
        SYMBOL,
        period=PERIOD,
        interval=TIMEFRAME,
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if data.empty:
        raise RuntimeError("Market data unavailable")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    for column in required:
        if column not in data.columns:
            raise RuntimeError("Required market data missing")

    data = data[required].dropna()

    if len(data) < 200:
        raise RuntimeError("Not enough market data")

    features = make_features(data)

    future_return = data["Close"].shift(-3) / data["Close"] - 1

    target = pd.Series(
        np.select(
            [
                future_return > 0.00015,
                future_return < -0.00015
            ],
            [1, -1],
            default=0
        ),
        index=data.index
    )

    dataset = features.copy()
    dataset["target"] = target

    training = dataset.dropna().iloc[:-3].copy()

    latest = features.dropna().iloc[[-1]].copy()

    if len(training) < 150:
        raise RuntimeError("Not enough clean training data")

    split = int(len(training) * 0.8)

    train = training.iloc[:split]
    test = training.iloc[split:]

    if len(train) < 80 or len(test) < 20:
        raise RuntimeError("Not enough validation data")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1
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

    probabilities = model.predict_proba(latest[FEATURES])[0]

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

    price = float(data["Close"].iloc[-1])
    candle_time = str(data.index[-1])

    print("------------------------------")
    print("Candle:", candle_time)
    print("Price:", price)
    print("Validation accuracy:", round(accuracy * 100, 2), "%")
    print("UP probability:", round(up * 100, 2), "%")
    print("DOWN probability:", round(down * 100, 2), "%")
    print("NEUTRAL probability:", round(neutral * 100, 2), "%")
    print("AI SIGNAL:", signal)

    message = (
        "🤖 AI REAL FOREX SIGNAL BOT\n\n"
        f"Pair: {SYMBOL}\n"
        f"Timeframe: {TIMEFRAME}\n"
        "Expiry: 1 Minute (not validated)\n\n"
        f"Signal: {signal}\n"
        f"Price: {price:.5f}\n"
        f"UP probability: {up * 100:.2f}%\n"
        f"DOWN probability: {down * 100:.2f}%\n"
        f"NEUTRAL probability: {neutral * 100:.2f}%\n"
        f"Validation accuracy: {accuracy * 100:.2f}%\n"
        f"Candle: {candle_time}\n\n"
        "Educational signal only.\n"
        "Not a guarantee of profit."
    )

    send_telegram(message)
    print("Bot finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("BOT ERROR:", error)
        sys.exit(1)
