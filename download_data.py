import os
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AVGO", "AMD", "NFLX",
    "JPM", "BAC", "WMT", "COST", "KO",
    "PEP", "MCD", "DIS", "V", "MA",
    "XOM", "CVX", "CAT", "BA", "GE",
    "INTC", "CSCO", "ADBE", "CRM", "ORCL",
    "BTC-USD",
]

os.makedirs("data/historical", exist_ok=True)

for ticker in TICKERS:
    print(f"Downloading {ticker}...")

    try:
        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            print(f"  No data for {ticker}")
            continue

        # Remove yfinance MultiIndex
        if hasattr(df.columns, "levels"):
            df = df.xs(ticker, axis=1, level=1)

        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]

        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.rename(columns={"date": "timestamp"})

        filename = ticker.replace("-", "_") + ".csv"
        path = os.path.join("data", "historical", filename)

        df.to_csv(path, index=False)

        print(f"  Saved {len(df)} candles -> {path}")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDownload complete.")