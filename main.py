
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

SYMBOL = "EURUSD=X"

print("AI MACHINE LEARNING FOREX BOT")

try:
    data = yf.download(
        SYMBOL,
        period="5d",
        interval="5m",
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError("Market data unavailable")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]

    close = close.dropna()

    df = pd.DataFrame({"close": close})

    ema9 = df["close"].ewm(span=9).mean()
    ema21 = df["close"].ewm(span=21).mean()
    ema50 = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (100 / (1 + rs))
    df["ema_gap"] = ema9 / ema21 - 1
    df["trend"] = ema21 / ema50 - 1
    df["return1"] = df["close"].pct_change()
    df["return3"] = df["close"].pct_change(3)
    df["momentum"] = df["close"] / df["close"].shift(6) - 1
    df["volatility"] = df["return1"].rolling(12).std()

    future_return = df["close"].shift(-3) / df["close"] - 1

    df["target"] = np.select(
        [future_return > 0.00015,
         future_return < -0.00015],
        [1, -1],
        default=0
    )

    features = [
        "rsi", "ema_gap", "trend",
        "return1", "return3",
        "momentum", "volatility"
    ]

    clean = df.dropna().copy()

    if len(clean) < 150:
        raise ValueError("Not enough training data")

    X = clean[features]
    y = clean["target"].astype(int)

    split = int(len(clean) * 0.8)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X.iloc[:split], y.iloc[:split])

    test_prediction = model.predict(X.iloc[split:])
    accuracy = accuracy_score(y.iloc[split:], test_prediction)

    model.fit(X, y)

    latest = df[features].iloc[[-1]]

    if latest.isna().any().any():
        raise ValueError("Latest indicators unavailable")

    probabilities = model.predict_proba(latest)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))

    up = class_probabilities.get(1, 0)
    down = class_probabilities.get(-1, 0)

    print("Pair:", SYMBOL)
    print("Price:", float(close.iloc[-1]))
    print("Test accuracy:", round(accuracy * 100, 2), "%")
    print("UP probability:", round(up * 100, 2), "%")
    print("DOWN probability:", round(down * 100, 2), "%")

    if up >= 0.55 and up > down:
        print("AI SIGNAL: UP")
    elif down >= 0.55 and down > up:
        print("AI SIGNAL: DOWN")
    else:
        print("AI SIGNAL: WAIT")

except Exception as error:
    print("ERROR:", error)

print("Bot finished.")
