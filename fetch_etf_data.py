import yfinance as yf
import os

# Create a data folder if it doesn’t exist
os.makedirs("data", exist_ok=True)

# Define tickers (S&P 500 + sector ETFs + VIX)
tickers = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "^VIX"]

# Download daily data (2020-2023)
# Extended date range to cover full COVID-19 period and recovery
data = yf.download(tickers, start="2020-01-01", end="2023-12-31", progress=True)

# Save each ticker's data to its own CSV
print("\nSaving data to CSV files...")
for ticker in tickers:
    df = data.xs(ticker, axis=1, level=1)
    csv_filename = f"data/{ticker.replace('^', '')}.csv"
    df.to_csv(csv_filename)
    print(f"  ✓ Saved {ticker}: {len(df)} rows ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

print("\n✅ Data download complete!")
