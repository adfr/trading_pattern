import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the ticker symbol
ticker = "QQQ"

# Define time period (7 days of minute data)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

print(f"Downloading {ticker} minute data from {start_date} to {end_date}...")

# Download minute-level data
data = yf.download(
    tickers=ticker,
    start=start_date,
    end=end_date,
    interval="1m",
    progress=False
)

# For a single ticker, yfinance doesn't use multi-index columns, but we'll handle it just in case
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[1] if col[1] else col[0] for col in data.columns]

# Reset index to make datetime a column
data = data.reset_index()

# Save to CSV with a clean format
output_file = f"{ticker}_minute_data.csv"
data.to_csv(output_file, index=False)

print(f"Downloaded {len(data)} rows of minute data")
print(f"Data saved to {output_file}")
print(f"Sample data:")
print(data.head()) 