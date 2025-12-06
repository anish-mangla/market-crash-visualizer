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

The project includes two versions of the visualization script:

#### Static Version (`visualize_sectors.py`)
- Basic static visualizations without interactive tooltips
- Run with: `python visualize_sectors.py`

#### Interactive Version (`visualize_sectors_interactive.py`) - **Recommended**
- All static visualizations plus interactive tooltips on hover
- Run with: `python visualize_sectors_interactive.py`

**Available Visualizations:**

1. **Sector Comparison Plot** - Normalized price comparison across all sectors, showing how each sector performed relative to a base date (Jan 2, 2020 = 100)
2. **Volatility Overlay Plot** - Sector prices with VIX volatility index overlay in a two-panel layout, showing the relationship between sector performance and market volatility
3. **Individual Sector Comparison** - Grid layout (2x4) showing each sector separately compared to SPY, making it easy to see individual sector patterns
4. **Correlation Heatmap** - Correlation matrix of daily returns between sectors, displayed as a color-coded heatmap showing which sectors moved together during the crash
5. **Returns Visualization** - Cumulative percentage returns for each sector from the base date, showing total gains/losses over the period
6. **Unemployment Overlay** - Sector prices with unemployment rate overlay in a two-panel layout, showing the relationship between market performance and unemployment data
7. **Raw Values Plot** - Actual price values (USD) for all sectors without normalization, showing the absolute price levels
8. **Summary Statistics** - Three bar charts showing max drawdown, total return, and volatility metrics for all sectors side-by-side
9. **Volume Analysis** - Trading volume analysis showing volume trends for each sector as filled area charts with lines. Displays volume in millions of shares, making it easy to see volume spikes during the market crash period
10. **Drawdown Timeline** - Small multiples visualization (3x3 grid) showing drawdown percentage for each sector over time. Each subplot displays daily drawdown percentages with bars extending downward from the 0% baseline (positive values extend above). All subplots include date axes and event markers
11. **View All Visualizations** - Display all visualizations sequentially

This uses the data in `/data` and user input from the terminal to create specified visualizations. 

**Usage:**
```bash
# Interactive version (recommended)
python visualize_sectors_interactive.py

# Static version
python visualize_sectors.py
```

Follow the on-screen menu to select which visualization to display. The interactive version includes hover tooltips for detailed data exploration. 
