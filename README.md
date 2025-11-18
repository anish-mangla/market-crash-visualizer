# market-crash-visualizer
In this project, we explore the following question:
"How did different sectors of the U.S. stock market respond to, crashed during, and recovered after the COVID-19 shock (Feb–Apr 2020), and what patterns in volatility and correlations emerged"

## 🗂️ Data Sources & Preparation

All datasets used in this project are publicly available and were collected from reliable financial and economic data providers.

### 1. Market & Sector Data
- **Source:** [Yahoo Finance](https://finance.yahoo.com/) using the Python `yfinance` library  
- **Tickers:**  
  `SPY` (S&P 500), `XLK` (Tech), `XLF` (Financials), `XLE` (Energy), `XLV` (Health Care),  
  `XLY` (Consumer Discretionary), `XLP` (Consumer Staples), `XLI` (Industrials), `^VIX` (Volatility Index)  
- **Process:** Daily data from **Jan 1 – Apr 30 2020** was downloaded via a short Python script and saved as CSVs in `./data/` (e.g., `data/SPY.csv`, `data/XLK.csv`, etc.).

### 2. Macroeconomic Data
- **Dataset:** `UNRATE` – Civilian Unemployment Rate  
- **Source:** [FRED – Federal Reserve Economic Data](https://fred.stlouisfed.org/series/UNRATE)  
- **Process:** Downloaded manually as `data/UNRATE.csv` (monthly U.S. unemployment rate).

### 3. COVID-19 Timeline
- **File:** `data/covid_timeline.csv`  
- **Source:** Manually compiled from public reports (CDC, WHO, news outlets)  
- **Purpose:** Used for annotation in visualizations to align market movements with major pandemic events.

### Visualization Designs and Layouts
1. **Sector Comparison Plot** - Normalized price comparison across all sectors
2. **Volatility Overlay Plot** - Sector prices with VIX volatility index overlay
3. **Individual Sector Comparison** - Grid layout showing each sector separately vs SPY
4. **Correlation Heatmap** - Correlation matrix of daily returns between sectors
5. **Returns Visualization** - Cumulative percentage returns for each sector
6. **Unemployment Overlay** - Sector prices with unemployment rate overlay
7. **Raw Values Plot** - Actual price values (USD) for all sectors
8. **Summary Statistics** - Bar charts showing max drawdown, total return, and volatility
9. **View All Visualizations** - Display all visualizations sequentially

This uses the data in /data and (OPTIONAL) user input from the terminal to create specified visualization. 

Run the script using `python visualize_sectors.py` and then follow the instructions on the terminal to see the visualizations. 
