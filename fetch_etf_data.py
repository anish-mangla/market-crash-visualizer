import yfinance as yf
import os

# Create a data folder if it doesn’t exist
os.makedirs("data", exist_ok=True)

# Define tickers (S&P 500 + sector ETFs + VIX)
tickers = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "^VIX"]

# Download daily data (Jan–Apr 2020)
data = yf.download(tickers, start="2020-01-01", end="2020-04-30")

# Save each ticker’s data to its own CSV
for ticker in tickers:
    df = data.xs(ticker, axis=1, level=1)
    df.to_csv(f"data/{ticker.replace('^', '')}.csv")
