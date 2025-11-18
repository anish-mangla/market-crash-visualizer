import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Configuration
SECTOR_TICKERS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials'
}

COLORS = {
    'XLK': '#1f77b4',  # Blue
    'XLF': '#ff7f0e',  # Orange
    'XLE': '#2ca02c',  # Green
    'XLV': '#d62728',  # Red
    'XLY': '#9467bd',  # Purple
    'XLP': '#8c564b',  # Brown
    'XLI': '#e377c2',  # Pink
    'SPY': '#7f7f7f',  # Gray
    'VIX': '#bcbd22'   # Yellow-green
}

def load_data():
    """Load all sector ETF data, SPY, VIX, COVID timeline, and unemployment data."""
    data = {}
    
    for ticker in SECTOR_TICKERS.keys():
        df = pd.read_csv(f'data/{ticker}.csv', parse_dates=['Date'], index_col='Date')
        data[ticker] = df['Close']
    
    spy = pd.read_csv('data/SPY.csv', parse_dates=['Date'], index_col='Date')
    data['SPY'] = spy['Close']
    
    vix = pd.read_csv('data/VIX.csv', parse_dates=['Date'], index_col='Date')
    data['VIX'] = vix['Close']
    
    covid_timeline = pd.read_csv('data/covid_timeline.csv', parse_dates=['Date'])
    
    try:
        unrate = pd.read_csv('data/UNRATE.csv', parse_dates=['observation_date'], index_col='observation_date')
        unrate.index.name = 'Date'
    except:
        unrate = None
    
    df = pd.DataFrame(data)
    df = df.sort_index()
    
    return df, covid_timeline, unrate

def normalize_prices(df, base_date=None):
    """Normalize all prices to 100 at the base date for comparison.
    
    Args:
        df: DataFrame with price data
        base_date: Date string or None. If None, uses first date in dataframe.
    """
    normalized = df.copy()
    
    if base_date is None:
        base_date = df.index[0]
    elif isinstance(base_date, str):
        if base_date not in df.index:
            try:
                base_date = pd.to_datetime(base_date)
                available_dates = df.index[df.index >= base_date]
                if len(available_dates) > 0:
                    base_date = available_dates[0]
                else:
                    base_date = df.index[0]
            except:
                base_date = df.index[0]
    
    base_values = df.loc[base_date]
    
    for col in df.columns:
        if col != 'VIX':
            normalized[col] = (df[col] / base_values[col]) * 100
    
    return normalized, base_date

def format_date_axis(ax, date_range):
    """Format x-axis dates based on the date range length."""
    date_span = (date_range[-1] - date_range[0]).days
    
    if date_span <= 90:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    elif date_span <= 365:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    elif date_span <= 730:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

def create_sector_comparison_plot(df_normalized, covid_timeline, base_date, figsize=(14, 8)):
    """Create main sector comparison plot with normalized prices."""
    fig, ax = plt.subplots(figsize=figsize)
    
    for ticker, sector_name in SECTOR_TICKERS.items():
        ax.plot(df_normalized.index, df_normalized[ticker], 
                label=sector_name, color=COLORS[ticker], linewidth=2, alpha=0.8)
    
    ax.plot(df_normalized.index, df_normalized['SPY'], 
            label='S&P 500 (SPY)', color=COLORS['SPY'], 
            linewidth=2.5, linestyle='--', alpha=0.9)
    
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_normalized.index:
            ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(event_date, ax.get_ylim()[1] * 0.98, event['Event'], 
                   rotation=90, verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    base_date_str = pd.to_datetime(base_date).strftime('%b %d, %Y')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Normalized Price ({base_date_str} = 100)', fontsize=12, fontweight='bold')
    
    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')
    ax.set_title(f'Sector Performance Comparison ({start_date} - {end_date})', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    format_date_axis(ax, df_normalized.index)
    
    plt.tight_layout()
    return fig, ax

def create_volatility_overlay_plot(df_normalized, covid_timeline, base_date, figsize=(14, 10)):
    """Create plot with sector prices and VIX volatility overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
    
    for ticker, sector_name in SECTOR_TICKERS.items():
        ax1.plot(df_normalized.index, df_normalized[ticker], 
                label=sector_name, color=COLORS[ticker], linewidth=2, alpha=0.8)
    
    ax1.plot(df_normalized.index, df_normalized['SPY'], 
            label='S&P 500 (SPY)', color=COLORS['SPY'], 
            linewidth=2.5, linestyle='--', alpha=0.9)
    
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_normalized.index:
            ax1.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax1.text(event_date, ax1.get_ylim()[1] * 0.98, event['Event'], 
                    rotation=90, verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    base_date_str = pd.to_datetime(base_date).strftime('%b %d, %Y')
    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')
    
    ax1.set_ylabel(f'Normalized Price ({base_date_str} = 100)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Sector Performance with Volatility Overlay ({start_date} - {end_date})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', framealpha=0.9, fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2.plot(df_normalized.index, df_normalized['VIX'], 
            color=COLORS['VIX'], linewidth=2.5, label='VIX Volatility Index', alpha=0.9)
    ax2.fill_between(df_normalized.index, df_normalized['VIX'], 
                     alpha=0.3, color=COLORS['VIX'])
    
    ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Moderate Volatility (20)')
    ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, linewidth=1, label='High Volatility (30)')
    
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_normalized.index:
            ax2.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('VIX Level', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    format_date_axis(ax2, df_normalized.index)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def create_individual_sector_comparison(df_normalized, covid_timeline, base_date, figsize=(16, 10)):
    """Create detailed comparison with individual sector subplots."""
    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    
    sectors = list(SECTOR_TICKERS.items())
    
    for idx, (ticker, sector_name) in enumerate(sectors):
        ax = axes[idx]
        
        ax.plot(df_normalized.index, df_normalized[ticker], 
               color=COLORS[ticker], linewidth=2.5, label=sector_name)
        
        ax.plot(df_normalized.index, df_normalized['SPY'], 
               color=COLORS['SPY'], linewidth=2, linestyle='--', 
               alpha=0.7, label='S&P 500')
        
        for _, event in covid_timeline.iterrows():
            event_date = event['Date']
            if event_date in df_normalized.index:
                ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.4, linewidth=0.8)
        
        ax.set_title(sector_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8)
        
        if idx >= 4:
            format_date_axis(ax, df_normalized.index)
    
    fig.delaxes(axes[7])
    
    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')
    fig.suptitle(f'Individual Sector Performance ({start_date} - {end_date})', 
                fontsize=14, fontweight='bold', y=0.995)
    
    base_date_str = pd.to_datetime(base_date).strftime('%b %d, %Y')
    fig.text(0.04, 0.5, f'Normalized Price ({base_date_str} = 100)', 
            va='center', rotation='vertical', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0.05, 0, 1, 0.99])
    return fig, axes

def create_correlation_heatmap(df_normalized, figsize=(10, 8)):
    """Create correlation matrix heatmap for sectors."""
    returns = df_normalized.pct_change().dropna()
    
    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    returns_sectors = returns[sector_cols]

    corr_matrix = returns_sectors.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = [SECTOR_TICKERS.get(col, col) if col != 'SPY' else 'S&P 500' for col in corr_matrix.columns]
    
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')
    
    ax.set_title('Sector Correlation Matrix (Daily Returns)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, ax

def create_returns_visualization(df_normalized, covid_timeline, base_date, figsize=(14, 8)):
    """Create percentage returns visualization."""
    returns = ((df_normalized / df_normalized.loc[base_date]) - 1) * 100
    
    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    returns_sectors = returns[sector_cols]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for ticker in SECTOR_TICKERS.keys():
        sector_name = SECTOR_TICKERS[ticker]
        ax.plot(returns_sectors.index, returns_sectors[ticker], 
                label=sector_name, color=COLORS[ticker], linewidth=2, alpha=0.8)
    
    ax.plot(returns_sectors.index, returns_sectors['SPY'], 
            label='S&P 500 (SPY)', color=COLORS['SPY'], 
            linewidth=2.5, linestyle='--', alpha=0.9)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in returns_sectors.index:
            ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    
    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')
    ax.set_title(f'Sector Cumulative Returns ({start_date} - {end_date})', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    format_date_axis(ax, df_normalized.index)
    
    plt.tight_layout()
    return fig, ax

def create_unemployment_overlay(df_normalized, unrate, covid_timeline, base_date, figsize=(14, 10)):
    """Create plot with sector prices and unemployment rate overlay."""
    if unrate is None:
        return None, None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
    
    for ticker, sector_name in SECTOR_TICKERS.items():
        ax1.plot(df_normalized.index, df_normalized[ticker], 
                label=sector_name, color=COLORS[ticker], linewidth=2, alpha=0.8)
    
    ax1.plot(df_normalized.index, df_normalized['SPY'], 
            label='S&P 500 (SPY)', color=COLORS['SPY'], 
            linewidth=2.5, linestyle='--', alpha=0.9)
    
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_normalized.index:
            ax1.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    base_date_str = pd.to_datetime(base_date).strftime('%b %d, %Y')
    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')
    
    ax1.set_ylabel(f'Normalized Price ({base_date_str} = 100)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Sector Performance with Unemployment Rate Overlay ({start_date} - {end_date})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', framealpha=0.9, fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2.plot(unrate.index, unrate['UNRATE'], 
            color='#d62728', linewidth=2.5, marker='o', markersize=6, 
            label='Unemployment Rate (%)', alpha=0.9)
    ax2.fill_between(unrate.index, unrate['UNRATE'], 
                     alpha=0.3, color='#d62728')
    
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_normalized.index:
            ax2.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    format_date_axis(ax2, df_normalized.index)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def create_raw_values_plot(df, covid_timeline, figsize=(14, 8)):
    """Create raw price values visualization (not normalized)."""
    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    df_sectors = df[sector_cols]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for ticker in SECTOR_TICKERS.keys():
        sector_name = SECTOR_TICKERS[ticker]
        ax.plot(df_sectors.index, df_sectors[ticker], 
                label=sector_name, color=COLORS[ticker], linewidth=2, alpha=0.8)
    
    ax.plot(df_sectors.index, df_sectors['SPY'], 
            label='S&P 500 (SPY)', color=COLORS['SPY'], 
            linewidth=2.5, linestyle='--', alpha=0.9)
    
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_sectors.index:
            ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    start_date = df_sectors.index[0].strftime('%b %Y')
    end_date = df_sectors.index[-1].strftime('%b %Y')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax.set_title(f'Sector Raw Price Values ({start_date} - {end_date})', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    format_date_axis(ax, df_sectors.index)
    
    plt.tight_layout()
    return fig, ax

def create_summary_statistics(df_normalized, base_date, figsize=(12, 8)):
    """Create bar chart with summary statistics (max drawdown, total return, etc.)."""
    end_date = df_normalized.index[-1]
    
    stats = {}
    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    
    for ticker in sector_cols:
        prices = df_normalized[ticker]
        peak = prices.expanding(min_periods=1).max()
        drawdown = ((prices - peak) / peak).min() * 100
        total_return = ((prices.loc[end_date] / prices.loc[base_date]) - 1) * 100

        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        stats[ticker] = {
            'Max Drawdown': drawdown,
            'Total Return': total_return,
            'Volatility': volatility
        }
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    labels = [SECTOR_TICKERS.get(t, 'S&P 500') if t != 'SPY' else 'S&P 500' for t in sector_cols]
    colors_list = [COLORS[t] for t in sector_cols]
    
    drawdowns = [stats[t]['Max Drawdown'] for t in sector_cols]
    axes[0].barh(labels, drawdowns, color=colors_list, alpha=0.8)
    axes[0].set_xlabel('Maximum Drawdown (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    returns = [stats[t]['Total Return'] for t in sector_cols]
    axes[1].barh(labels, returns, color=colors_list, alpha=0.8)
    axes[1].set_xlabel('Total Return (%)', fontsize=11, fontweight='bold')
    
    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')
    axes[1].set_title(f'Total Return ({start_date} - {end_date})', fontsize=12, fontweight='bold')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    volatilities = [stats[t]['Volatility'] for t in sector_cols]
    axes[2].barh(labels, volatilities, color=colors_list, alpha=0.8)
    axes[2].set_xlabel('Annualized Volatility (%)', fontsize=11, fontweight='bold')
    axes[2].set_title('Volatility', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Sector Performance Summary Statistics', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig, axes

def display_menu():
    """Display the visualization menu."""
    print("\n" + "="*60)
    print("SECTOR VISUALIZATION MENU")
    print("="*60)
    print("1. Sector Comparison Plot")
    print("2. Volatility Overlay Plot")
    print("3. Individual Sector Comparison")
    print("4. Correlation Heatmap")
    print("5. Returns Visualization")
    print("6. Unemployment Overlay")
    print("7. Raw Values Plot")
    print("8. Summary Statistics")
    print("9. View All Visualizations")
    print("0. Exit")
    print("="*60)

def main():
    """Main function to generate visualizations interactively."""
    print("Loading data...")
    df, covid_timeline, unrate = load_data()
    
    print("Normalizing prices...")
    df_normalized, base_date = normalize_prices(df)
    print(f"Using base date: {pd.to_datetime(base_date).strftime('%Y-%m-%d')}")
    
    while True:
        display_menu()
        choice = input("\nSelect a visualization (0-9): ").strip()
        
        if choice == '0':
            print("Exiting...")
            break
        elif choice == '1':
            print("\nCreating sector comparison plot...")
            fig, ax = create_sector_comparison_plot(df_normalized, covid_timeline, base_date)
            plt.show()
        elif choice == '2':
            print("\nCreating volatility overlay plot...")
            fig, axes = create_volatility_overlay_plot(df_normalized, covid_timeline, base_date)
            plt.show()
        elif choice == '3':
            print("\nCreating individual sector comparison...")
            fig, axes = create_individual_sector_comparison(df_normalized, covid_timeline, base_date)
            plt.show()
        elif choice == '4':
            print("\nCreating correlation heatmap...")
            fig, ax = create_correlation_heatmap(df_normalized)
            plt.show()
        elif choice == '5':
            print("\nCreating returns visualization...")
            fig, ax = create_returns_visualization(df_normalized, covid_timeline, base_date)
            plt.show()
        elif choice == '6':
            print("\nCreating unemployment overlay...")
            fig, axes = create_unemployment_overlay(df_normalized, unrate, covid_timeline, base_date)
            if fig is not None:
                plt.show()
            else:
                print("Unemployment data not available.")
        elif choice == '7':
            print("\nCreating raw values plot...")
            fig, ax = create_raw_values_plot(df, covid_timeline)
            plt.show()
        elif choice == '8':
            print("\nCreating summary statistics...")
            fig, axes = create_summary_statistics(df_normalized, base_date)
            plt.show()
        elif choice == '9':
            print("\nDisplaying all visualizations...")
            print("(Close each window to view the next)")
            
            print("\n1/8: Sector Comparison Plot...")
            fig1, ax1 = create_sector_comparison_plot(df_normalized, covid_timeline, base_date)
            plt.show()
            
            print("2/8: Volatility Overlay Plot...")
            fig2, axes2 = create_volatility_overlay_plot(df_normalized, covid_timeline, base_date)
            plt.show()
            
            print("3/8: Individual Sector Comparison...")
            fig3, axes3 = create_individual_sector_comparison(df_normalized, covid_timeline, base_date)
            plt.show()
            
            print("4/8: Correlation Heatmap...")
            fig4, ax4 = create_correlation_heatmap(df_normalized)
            plt.show()
            
            print("5/8: Returns Visualization...")
            fig5, ax5 = create_returns_visualization(df_normalized, covid_timeline, base_date)
            plt.show()
            
            if unrate is not None:
                print("6/8: Unemployment Overlay...")
                fig6, axes6 = create_unemployment_overlay(df_normalized, unrate, covid_timeline, base_date)
                if fig6 is not None:
                    plt.show()
            
            print("7/8: Raw Values Plot...")
            fig7, ax7 = create_raw_values_plot(df, covid_timeline)
            plt.show()
            
            print("8/8: Summary Statistics...")
            fig8, axes8 = create_summary_statistics(df_normalized, base_date)
            plt.show()
            
            print("\nAll visualizations displayed!")
        else:
            print("Invalid choice. Please select a number between 0-9.")

if __name__ == '__main__':
    main()

