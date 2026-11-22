

import yfinance as yf
import pandas as pd
import time

PAIR = "EURUSD=X"

def analyze():
    data = yf.download(
        PAIR,
        period="1d",
        interval="1m",
        progress=False
    )

    if data.empty or len(data) < 60:
        print("Waiting for market data...")
        return

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data.copy().iloc[:-1]

    if len(df) < 55:
        return

    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    change = df["Close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))

    last = df.iloc[-1]
    previous = df.iloc[-2]

    up = (
        last["EMA9"] > last["EMA21"] > last["EMA50"]
        and last["Close"] > last["EMA9"]
        and last["Close"] > last["Open"]
        and 50 < last["RSI"] < 68
    )

    down = (
        last["EMA9"] < last["EMA21"] < last["EMA50"]
        and last["Close"] < last["EMA9"]
        and last["Close"] < last["Open"]
        and 32 < last["RSI"] < 50
    )

    print("\n===== REAL FOREX SIGNAL =====")
    print("Pair:", PAIR)
    print("Price:", round(float(last["Close"]), 5))
    print("RSI:", round(float(last["RSI"]), 2))

    if up:
        print("SIGNAL: UP 🟢")
        print("Expiry: Next 1-minute candle")
    elif down:
        print("SIGNAL: DOWN 🔴")
        print("Expiry: Next 1-minute candle")
    else:
        print("SIGNAL: NO TRADE ⚪")

    print("=============================")

while True:
    try:
        analyze()
    except Exception as e:
        print("Error:", e)

    time.sleep(60)
