
import yfinance as yf
import pandas as pd

SYMBOL = "EURUSD=X"

print("AI Real Forex Bot Starting...")

try:
    data = yf.download(
        SYMBOL,
        period="5d",
        interval="5m",
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        print("ERROR: Forex data not available.")
    else:
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].iloc[:, 0]
        else:
            close = data["Close"]

        close = close.dropna()

        if len(close) < 30:
            print("Not enough market data. Try again later.")
        else:
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()

            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.ewm(
                alpha=1/14, adjust=False, min_periods=14
            ).mean()

            avg_loss = loss.ewm(
                alpha=1/14, adjust=False, min_periods=14
            ).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            price = float(close.iloc[-1])
            fast = float(ema9.iloc[-1])
            slow = float(ema21.iloc[-1])
            current_rsi = float(rsi.iloc[-1])

            print("Pair:", SYMBOL)
            print("Price:", price)
            print("RSI:", round(current_rsi, 2))

            if fast > slow and 50 < current_rsi < 70:
                print("SIGNAL: UP")
            elif fast < slow and 30 < current_rsi < 50:
                print("SIGNAL: DOWN")
            else:
                print("SIGNAL: WAIT")

except Exception as e:
    print("ERROR:", e)

print("Bot finished.")
