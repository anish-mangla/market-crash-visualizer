import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from matplotlib.lines import Line2D

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
    'XLK': '#1f77b4',  # blue
    'XLF': '#ff7f0e',  # orange
    'XLE': '#2ca02c',  # green
    'XLV': '#d62728',  # red
    'XLY': '#9467bd',  # purple
    'XLP': '#8c564b',  # brown
    'XLI': '#e377c2',  # pink
    'SPY': '#7f7f7f',  # gray
    'VIX': '#17becf'   # teal
}

COVID_EVENT_LABELS = {
    '2020-01-21': 'First confirmed COVID-19 case in the US',
    '2020-02-19': 'US markets begin steep decline',
    '2020-03-11': 'WHO declares COVID-19 a pandemic',
    '2020-03-12': 'Stock market experiences largest one-day drop since 1987',
    '2020-03-27': 'US CARES Act stimulus signed',
    '2020-04-16': 'Peak unemployment filings (22 million claims)',
}


def format_date_axis(ax, dates):
    """Format x-axis with appropriate date labeling."""
    years = pd.DatetimeIndex(dates).year
    if len(set(years)) > 3:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def highlight_crash_period(ax, base_date):
    """Highlight key COVID crash period."""
    crash_start = pd.to_datetime('2020-02-19')
    crash_end = pd.to_datetime('2020-03-23')

    ax.axvspan(crash_start, crash_end, color='red', alpha=0.08, label='Crash Period')

    annotations = {
        '2020-02-19': 'Market Peak',
        '2020-03-23': 'Market Bottom'
    }

    for date_str, label in annotations.items():
        date = pd.to_datetime(date_str)
        if date >= base_date:
            ax.axvline(x=date, color='red', linestyle=':', alpha=0.7)

    if ax.get_lines():
        format_date_axis(ax, ax.get_lines()[0].get_xdata())


def load_data():
    """Load all sector ETF data, SPY, VIX, COVID timeline, and unemployment data."""
    data = {}
    volume_data = {}

    for ticker in SECTOR_TICKERS.keys():
        df = pd.read_csv(f'data/{ticker}.csv', parse_dates=['Date'], index_col='Date')
        data[ticker] = df['Close']
        volume_data[ticker] = df['Volume']

    spy = pd.read_csv('data/SPY.csv', parse_dates=['Date'], index_col='Date')
    data['SPY'] = spy['Close']
    volume_data['SPY'] = spy['Volume']

    vix = pd.read_csv('data/VIX.csv', parse_dates=['Date'], index_col='Date')
    data['VIX'] = vix['Close']
    volume_data['VIX'] = vix['Volume']

    covid_timeline = pd.read_csv('data/covid_timeline.csv', parse_dates=['Date'])

    try:
        unrate = pd.read_csv(
            'data/UNRATE.csv',
            parse_dates=['observation_date'],
            index_col='observation_date'
        )
        unrate.index.name = 'Date'
    except Exception:
        unrate = None

    df = pd.DataFrame(data).sort_index()
    df_volume = pd.DataFrame(volume_data).sort_index()
    return df, df_volume, covid_timeline, unrate


def normalize_prices(df, base_date=None):
    """Normalize all prices to 100 at the base date for comparison."""
    normalized = df.copy()

    if base_date is None:
        base_date = df.index[0]
    elif isinstance(base_date, str):
        if base_date not in df.index:
            try:
                tmp = pd.to_datetime(base_date)
                available_dates = df.index[df.index >= tmp]
                if len(available_dates) > 0:
                    base_date = available_dates[0]
                else:
                    base_date = df.index[0]
            except Exception:
                base_date = df.index[0]

    base_date = pd.to_datetime(base_date)

    base_values = normalized.loc[base_date]
    normalized = (normalized / base_values) * 100.0
    normalized = normalized[normalized.index >= base_date]

    return normalized, base_date


# ==========================
# Static plotting functions
# ==========================

def create_sector_comparison_plot_static(df_normalized, covid_timeline, base_date, figsize=(14, 8)):
    """Create main sector comparison plot with normalized prices."""
    fig, ax = plt.subplots(figsize=figsize)

    for ticker, sector_name in SECTOR_TICKERS.items():
        ax.plot(
            df_normalized.index,
            df_normalized[ticker],
            label=sector_name,
            color=COLORS[ticker],
            linewidth=2,
            alpha=0.8,
        )

    ax.plot(
        df_normalized.index,
        df_normalized['SPY'],
        label='S&P 500 (SPY)',
        color=COLORS['SPY'],
        linewidth=2.5,
        linestyle='--',
        alpha=0.9,
    )

    # Yellow event callouts
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_normalized.index:
            label = COVID_EVENT_LABELS.get(
                event_date.strftime('%Y-%m-%d'),
                event['Event']
            )
            ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(
                event_date,
                ax.get_ylim()[1] * 0.98,
                label,
                rotation=90,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            )

    base_date_str = pd.to_datetime(base_date).strftime('%b %d, %Y')
    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Normalized Price ({base_date_str} = 100)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Sector Performance Comparison ({start_date} - {end_date})',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.legend(loc='lower left', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    highlight_crash_period(ax, base_date)
    format_date_axis(ax, df_normalized.index)

    fig.tight_layout()
    return fig, ax


def create_volatility_overlay_plot_static(df_normalized, covid_timeline, base_date, figsize=(14, 10)):
    """Create plot with sectors and VIX overlay."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=figsize,
        height_ratios=[3, 1],
        sharex=True
    )

    # ----- Top panel: sectors -----
    for ticker, sector_name in SECTOR_TICKERS.items():
        ax1.plot(
            df_normalized.index,
            df_normalized[ticker],
            label=sector_name,
            color=COLORS[ticker],
            linewidth=2,
            alpha=0.8,
        )

    ax1.plot(
        df_normalized.index,
        df_normalized['SPY'],
        label='S&P 500 (SPY)',
        color=COLORS['SPY'],
        linewidth=2.5,
        linestyle='--',
        alpha=0.9,
    )

    # Yellow event callouts on top panel
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_normalized.index:
            label = COVID_EVENT_LABELS.get(
                event_date.strftime('%Y-%m-%d'),
                event['Event']
            )
            ax1.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax1.text(
                event_date,
                ax1.get_ylim()[1] * 0.98,
                label,
                rotation=90,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            )

    base_date_str = pd.to_datetime(base_date).strftime('%b %d, %Y')
    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')

    ax1.set_ylabel(f'Normalized Price ({base_date_str} = 100)', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'Sector Performance with Volatility Overlay ({start_date} - {end_date})',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')

    highlight_crash_period(ax1, base_date)

    # ----- Bottom panel: VIX (original style) -----
    vix = df_normalized['VIX']

    ax2.plot(
        df_normalized.index,
        vix,
        label='VIX Volatility Index',
        color=COLORS['VIX'],
        linewidth=2.5,
        alpha=0.9,
    )
    ax2.fill_between(
        df_normalized.index,
        vix,
        alpha=0.3,
        color=COLORS['VIX'],
    )

    ax2.axhline(
        20,
        color='orange',
        linestyle='--',
        linewidth=1,
        alpha=0.7,
        label='Moderate Volatility (20)',
    )
    ax2.axhline(
        30,
        color='red',
        linestyle='--',
        linewidth=1,
        alpha=0.7,
        label='High Volatility (30)',
    )

    # Only red lines (no text) in bottom panel
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_normalized.index:
            ax2.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)

    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('VIX Level', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    format_date_axis(ax2, df_normalized.index)
    plt.tight_layout()
    return fig, (ax1, ax2)


def create_individual_sector_comparison_static(df_normalized, covid_timeline, base_date, figsize=(16, 10)):
    """Create detailed comparison with individual sector subplots."""
    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    for i, ticker in enumerate(sector_cols):
        ax = axes[i]
        if ticker == 'SPY':
            ax.plot(
                df_normalized.index,
                df_normalized[ticker],
                label='S&P 500',
                color=COLORS['SPY'],
                linewidth=2.5,
                linestyle='--',
                alpha=0.9,
            )
        else:
            sector_name = SECTOR_TICKERS[ticker]
            ax.plot(
                df_normalized.index,
                df_normalized[ticker],
                label=sector_name,
                color=COLORS[ticker],
                linewidth=2,
                alpha=0.8,
            )
            ax.plot(
                df_normalized.index,
                df_normalized['SPY'],
                label='S&P 500',
                color=COLORS['SPY'],
                linewidth=2,
                linestyle='--',
                alpha=0.7,
            )

        # Vertical lines only (no boxes) to keep subplots clean
        for _, event in covid_timeline.iterrows():
            event_date = event['Date']
            if event_date in df_normalized.index:
                ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.3, linewidth=1)

        ax.set_title(SECTOR_TICKERS.get(ticker, 'S&P 500'), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)

        if i >= 4:
            ax.set_xlabel('Date', fontsize=9)
        if i % 4 == 0:
            ax.set_ylabel('Normalized Price', fontsize=9)

        highlight_crash_period(ax, base_date)

    for ax in axes:
        format_date_axis(ax, df_normalized.index)

    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')
    base_date_str = pd.to_datetime(base_date).strftime('%b %d, %Y')

    fig.suptitle(
        f'Individual Sector Performance ({start_date} - {end_date})',
        fontsize=14,
        fontweight='bold',
        y=0.995,
    )
    fig.text(
        0.04,
        0.5,
        f'Normalized Price ({base_date_str} = 100)',
        va='center',
        rotation='vertical',
        fontsize=12,
        fontweight='bold',
    )

    plt.tight_layout(rect=[0.05, 0, 1, 0.99])
    return fig, axes


def create_correlation_heatmap_static(df_normalized, figsize=(10, 8)):
    """Create correlation heatmap of daily returns."""
    returns = df_normalized.pct_change().dropna()
    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    returns_sectors = returns[sector_cols]
    corr_matrix = returns_sectors.corr()

    fig, ax = plt.subplots(figsize=figsize)
    labels = [
        SECTOR_TICKERS.get(col, col) if col != 'SPY' else 'S&P 500'
        for col in corr_matrix.columns
    ]

    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f'{corr_value:.2f}',
                ha='center',
                va='center',
                color='black',
                fontsize=9,
                fontweight='bold',
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')

    ax.set_title(
        'Sector Correlation Matrix (Daily Returns)',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    plt.tight_layout()
    return fig, ax


def create_returns_visualization_static(df_normalized, covid_timeline, base_date, figsize=(14, 8)):
    """Create cumulative returns visualization."""
    returns = df_normalized.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=figsize)

    for ticker, sector_name in SECTOR_TICKERS.items():
        ax.plot(
            cumulative_returns.index,
            cumulative_returns[ticker] * 100,
            label=sector_name,
            color=COLORS[ticker],
            linewidth=2,
            alpha=0.8,
        )

    ax.plot(
        cumulative_returns.index,
        cumulative_returns['SPY'] * 100,
        label='S&P 500 (SPY)',
        color=COLORS['SPY'],
        linewidth=2.5,
        linestyle='--',
        alpha=0.9,
    )

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # Yellow event callouts
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in cumulative_returns.index:
            label = COVID_EVENT_LABELS.get(
                event_date.strftime('%Y-%m-%d'),
                event['Event']
            )
            ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(
                event_date,
                ax.get_ylim()[1] * 0.98,
                label,
                rotation=90,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            )

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')

    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')

    ax.set_title(
        f'Sector Cumulative Returns ({start_date} - {end_date})',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=4,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.1),
    )

    ax.grid(True, alpha=0.3, linestyle='--')

    highlight_crash_period(ax, base_date)
    format_date_axis(ax, cumulative_returns.index)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig, ax


def create_unemployment_overlay_static(df_normalized, unrate, covid_timeline, base_date, figsize=(14, 10)):
    """Create plot with sector prices and unemployment rate overlay."""
    if unrate is None:
        print("Unemployment data not found. Skipping unemployment overlay plot.")
        return None, None

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=figsize,
        height_ratios=[3, 1],
        sharex=True
    )

    # ----- Top panel: sectors -----
    for ticker, sector_name in SECTOR_TICKERS.items():
        ax1.plot(
            df_normalized.index,
            df_normalized[ticker],
            label=sector_name,
            color=COLORS[ticker],
            linewidth=2,
            alpha=0.8,
        )

    ax1.plot(
        df_normalized.index,
        df_normalized['SPY'],
        label='S&P 500 (SPY)',
        color=COLORS['SPY'],
        linewidth=2.5,
        linestyle='--',
        alpha=0.9,
    )

    # Yellow event callouts on top panel
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_normalized.index:
            label = COVID_EVENT_LABELS.get(
                event_date.strftime('%Y-%m-%d'),
                event['Event']
            )
            ax1.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax1.text(
                event_date,
                ax1.get_ylim()[1] * 0.98,
                label,
                rotation=90,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            )

    base_date_str = pd.to_datetime(base_date).strftime('%b %d, %Y')
    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')

    ax1.set_title(
        f'Sector Performance with Unemployment Rate Overlay ({start_date} - {end_date})',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax1.set_ylabel(f'Normalized Price ({base_date_str} = 100)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9, fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')

    highlight_crash_period(ax1, base_date)

    # ----- Bottom panel: unemployment (original style) -----
    ax2.plot(
        unrate.index,
        unrate['UNRATE'],
        color='#d62728',
        linewidth=2.5,
        marker='o',
        markersize=6,
        label='Unemployment Rate (%)',
        alpha=0.9,
    )
    ax2.fill_between(
        unrate.index,
        unrate['UNRATE'],
        alpha=0.3,
        color='#d62728',
    )

    # Only red lines (no yellow boxes) in bottom panel
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


def create_raw_values_plot_static(df, covid_timeline, figsize=(14, 8)):
    """Create raw price values visualization (not normalized)."""
    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    df_sectors = df[sector_cols]

    fig, ax = plt.subplots(figsize=figsize)

    for ticker in SECTOR_TICKERS.keys():
        sector_name = SECTOR_TICKERS[ticker]
        ax.plot(
            df_sectors.index,
            df_sectors[ticker],
            label=sector_name,
            color=COLORS[ticker],
            linewidth=2,
            alpha=0.8,
        )

    ax.plot(
        df_sectors.index,
        df_sectors['SPY'],
        label='S&P 500 (SPY)',
        color=COLORS['SPY'],
        linewidth=2.5,
        linestyle='--',
        alpha=0.9,
    )

    # Yellow event callouts
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_sectors.index:
            label = COVID_EVENT_LABELS.get(
                event_date.strftime('%Y-%m-%d'),
                event['Event']
            )
            ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(
                event_date,
                ax.get_ylim()[1] * 0.98,
                label,
                rotation=90,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            )

    start_date = df_sectors.index[0].strftime('%b %Y')
    end_date = df_sectors.index[-1].strftime('%b %Y')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Sector Raw Price Values ({start_date} - {end_date})',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        framealpha=0.9,
        fontsize=9,
    )
    ax.grid(True, alpha=0.3, linestyle='--')

    highlight_crash_period(ax, df_sectors.index[0])
    format_date_axis(ax, df_sectors.index)

    plt.tight_layout()
    return fig, ax


def create_summary_statistics_static(df_normalized, base_date, figsize=(12, 8)):
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
            'Volatility': volatility,
        }

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    labels = [
        SECTOR_TICKERS.get(t, t) if t != 'SPY' else 'S&P 500'
        for t in sector_cols
    ]
    colors_list = [COLORS.get(t, '#7f7f7f') for t in sector_cols]

    drawdowns = [stats[t]['Max Drawdown'] for t in sector_cols]
    axes[0].barh(labels, drawdowns, color=colors_list, alpha=0.8)
    axes[0].set_xlabel('Maximum Drawdown (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    total_returns = [stats[t]['Total Return'] for t in sector_cols]
    axes[1].barh(labels, total_returns, color=colors_list, alpha=0.8)
    axes[1].set_xlabel('Total Return (%)', fontsize=11, fontweight='bold')

    start_date = df_normalized.index[0].strftime('%b %Y')
    end_date = df_normalized.index[-1].strftime('%b %Y')
    axes[1].set_title(
        f'Total Return ({start_date} - {end_date})',
        fontsize=12,
        fontweight='bold',
    )
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    axes[1].grid(True, alpha=0.3, axis='x')

    volatilities = [stats[t]['Volatility'] for t in sector_cols]
    axes[2].barh(labels, volatilities, color=colors_list, alpha=0.8)
    axes[2].set_xlabel('Annualized Volatility (%)', fontsize=11, fontweight='bold')
    axes[2].set_title('Volatility', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')

    fig.suptitle(
        'Sector Performance Summary Statistics',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()
    return fig, axes


def create_volume_analysis_static(df_volume, covid_timeline, figsize=(14, 8)):
    """Create volume analysis visualization showing trading volume as area chart for each sector."""
    # Include all sectors + SPY
    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    df_volume_sectors = df_volume[sector_cols]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot area chart for each sector (filled area under line)
    for ticker in SECTOR_TICKERS.keys():
        sector_name = SECTOR_TICKERS[ticker]
        ax.fill_between(
            df_volume_sectors.index,
            df_volume_sectors[ticker],
            alpha=0.6,
            color=COLORS[ticker],
            label='',  # Don't label fill_between (creates internal children)
        )
        # Add line on top for better visibility - this will have the proper label
        line = ax.plot(
            df_volume_sectors.index,
            df_volume_sectors[ticker],
            color=COLORS[ticker],
            linewidth=1.5,
            alpha=0.9,
            label=sector_name,  # Set label on the actual line
        )

    # Add SPY with different style (dashed line, no fill or lighter fill)
    ax.fill_between(
        df_volume_sectors.index,
        df_volume_sectors['SPY'],
        alpha=0.4,
        color=COLORS['SPY'],
        label='',  # Don't label fill_between
    )
    ax.plot(
        df_volume_sectors.index,
        df_volume_sectors['SPY'],
        color=COLORS['SPY'],
        linewidth=2,
        linestyle='--',
        alpha=0.9,
        label='S&P 500 (SPY)',  # Set label on the actual line
    )

    # Yellow event callouts
    for _, event in covid_timeline.iterrows():
        event_date = event['Date']
        if event_date in df_volume_sectors.index:
            label = COVID_EVENT_LABELS.get(
                event_date.strftime('%Y-%m-%d'),
                event['Event']
            )
            ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(
                event_date,
                ax.get_ylim()[1] * 0.98,
                label,
                rotation=90,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            )

    start_date = df_volume_sectors.index[0].strftime('%b %Y')
    end_date = df_volume_sectors.index[-1].strftime('%b %Y')

    # Format y-axis to show values in millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trading Volume (millions of shares)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Sector Trading Volume Analysis ({start_date} - {end_date})',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        framealpha=0.9,
        fontsize=9,
    )
    ax.grid(True, alpha=0.3, linestyle='--')

    highlight_crash_period(ax, df_volume_sectors.index[0])
    format_date_axis(ax, df_volume_sectors.index)

    plt.tight_layout()
    return fig, ax


def create_drawdown_timeline_static(df_normalized, covid_timeline, base_date, figsize=(18, 14)):
    """Create drawdown timeline visualization with separate subplots for each sector (small multiples)."""
    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    df_sectors = df_normalized[sector_cols]
    
    # Calculate drawdown for each sector
    drawdown_data = {}
    for ticker in sector_cols:
        prices = df_sectors[ticker]
        # Calculate running peak (highest price so far)
        peak = prices.expanding(min_periods=1).max()
        # Calculate drawdown: (current - peak) / peak * 100
        drawdown = ((prices - peak) / peak) * 100
        drawdown_data[ticker] = drawdown
    
    df_drawdown = pd.DataFrame(drawdown_data)
    
    # Create subplots: 3 rows x 3 columns (8 sectors + SPY = 9 total)
    # Use sharex=False to ensure all subplots show their own date labels
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=False, sharey=True, dpi=100)
    axes = axes.flatten()
    
    # Remove the 9th subplot (empty one) - hide it instead of deleting to avoid indexing issues
    axes[8].set_visible(False)
    
    # Convert dates to numeric for bar positioning
    dates = df_drawdown.index
    date_nums = mdates.date2num(dates)
    n_days = len(dates)
    
    # Calculate bar width (days between dates)
    if n_days > 1:
        day_spacing = (date_nums[-1] - date_nums[0]) / (n_days - 1) if n_days > 1 else 1
    else:
        day_spacing = 1
    bar_width = day_spacing * 0.8  # Each bar takes 80% of available space
    
    # Plot each sector in its own subplot
    for idx, ticker in enumerate(sector_cols):
        ax = axes[idx]
        sector_name = SECTOR_TICKERS.get(ticker, 'S&P 500')
        drawdown_values = df_drawdown[ticker].values
        
        # Plot bars extending downward from 0 (drawdowns are negative)
        if ticker == 'SPY':
            # SPY with solid, clear style (more visible)
            ax.bar(
                date_nums,
                drawdown_values,  # Negative values, bars go down
                width=bar_width,
                label=sector_name,
                color=COLORS[ticker],
                alpha=0.6,  # More opaque for better visibility
                edgecolor=COLORS[ticker],
                linewidth=4.0,  # Thicker outline
                linestyle='-',  # Solid line instead of dashed
            )
        else:
            # Regular sectors with outlined bars
            ax.bar(
                date_nums,
                drawdown_values,  # Negative values, bars go down
                width=bar_width,
                label=sector_name,
                color='none',  # Transparent fill (outlined style)
                edgecolor=COLORS[ticker],
                linewidth=3.0,
                alpha=1.0,
            )
        
        # Reference line at 0% (no drawdown) - bold line in middle
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2.5, alpha=0.9, zorder=10)
        
        # Event markers - clean vertical lines on all subplots (no labels)
        for _, event in covid_timeline.iterrows():
            event_date = event['Date']
            if event_date in df_drawdown.index:
                ax.axvline(x=event_date, color='red', linestyle=':', alpha=0.4, linewidth=1.2, zorder=5)
        
        # Set subplot title
        ax.set_title(sector_name, fontsize=12, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        
        # Format y-axis to show negative values clearly
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        # Set y-axis limits - allow both positive and negative values
        # Positive values extend above 0%, negative values extend below 0%
        y_min = df_drawdown.min().min() * 1.1  # Most negative value (e.g., -66)
        y_max = max(df_drawdown.max().max() * 1.1, 5) if df_drawdown.max().max() > 0 else 5  # Allow space for positive values
        ax.set_ylim(y_min, y_max)  # (-66, 5) -> 0% in middle, negatives below, positives above
        
        # Format x-axis on ALL subplots (show dates on all)
        # Make sure x-axis is visible and formatted
        ax.tick_params(axis='x', labelsize=9, rotation=45, bottom=True, labelbottom=True)
        format_date_axis(ax, df_drawdown.index)
        
        # Highlight crash period
        highlight_crash_period(ax, base_date)
        
        # Improve tick label clarity
        ax.tick_params(axis='y', labelsize=9)
    
    start_date = df_drawdown.index[0].strftime('%b %Y')
    end_date = df_drawdown.index[-1].strftime('%b %Y')
    
    # Set overall title
    fig.suptitle(
        f'Sector Drawdown Timeline ({start_date} - {end_date})',
        fontsize=18,
        fontweight='bold',
        y=0.995,
    )
    
    # Set common labels
    fig.text(0.5, 0.015, 'Date', ha='center', fontsize=13, fontweight='bold')
    fig.text(0.015, 0.5, 'Drawdown (%)', va='center', rotation='vertical', fontsize=13, fontweight='bold')
    
    plt.tight_layout(rect=[0.025, 0.025, 0.995, 0.99])
    return fig, axes


# ==========================
# Interactive helpers
# ==========================

def _attach_line_tooltips(ax, lines, formatter=None, pixel_radius=8.0):
    """Attach shared tooltip to multiple Line2D objects (datetime-safe)."""
    fig = ax.figure
    valid_lines = []
    for ln in lines:
        if not isinstance(ln, Line2D):
            continue
        xdata = ln.get_xdata()
        if xdata is None or len(xdata) == 0:
            continue
        valid_lines.append(ln)

    if not valid_lines:
        return

    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="w", ec="#333"),
        arrowprops=dict(arrowstyle="->", color="#333"),
    )
    ann.set_visible(False)

    cache = []
    for ln in valid_lines:
        x_raw = np.asarray(ln.get_xdata())
        y = np.asarray(ln.get_ydata(), dtype=float)

        if x_raw.size == 0 or y.size == 0:
            continue

        try:
            if np.issubdtype(x_raw.dtype, np.datetime64):
                x = mdates.date2num(x_raw)
            else:
                x = x_raw.astype(float)
        except (TypeError, ValueError):
            x = mdates.date2num(x_raw)

        pts = np.column_stack((x, y))
        cache.append((ln, pts))

    if not cache:
        return

    def default_fmt(ln, x_val, y_val):
        label = ln.get_label() or "series"
        try:
            x_disp = mdates.num2date(x_val).strftime("%Y-%m-%d")
        except Exception:
            x_disp = str(x_val)
        return f"{label}\nDate: {x_disp}\nValue: {y_val:,.2f}"

    fmt = formatter or default_fmt
    last_line = None
    last_idx = None

    def on_move(event):
        nonlocal last_line, last_idx

        if event.inaxes is not ax or event.x is None or event.y is None:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            last_line, last_idx = None, None
            return

        ex, ey = event.x, event.y
        best = None

        for ln, pts in cache:
            if not ln.get_visible():
                continue

            px = ax.transData.transform(pts)
            dx = px[:, 0] - ex
            dy = px[:, 1] - ey
            d2 = dx * dx + dy * dy
            i = int(np.argmin(d2))
            dist = float(np.sqrt(d2[i]))
            if dist <= pixel_radius and (best is None or dist < best[0]):
                x_val, y_val = pts[i]
                best = (dist, ln, i, x_val, y_val)

        if best is None:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            last_line, last_idx = None, None
            return

        _, ln, idx, xv, yv = best
        if ln is last_line and idx == last_idx and ann.get_visible():
            return

        ann.xy = (xv, yv)
        ann.set_text(fmt(ln, xv, yv))
        ann.set_visible(True)
        fig.canvas.draw_idle()
        last_line, last_idx = ln, idx

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def _attach_bar_tooltips(ax, bars, labels):
    """
    Attach hover tooltips to bar charts.
    Works for both vertical (bar) and horizontal (barh) bars.
    """
    fig = ax.figure
    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="w", ec="#333"),
        arrowprops=dict(arrowstyle="->", color="#333"),
    )
    ann.set_visible(False)

    def on_move(event):
        if event.inaxes is not ax:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            return

        hit_any = False
        for bar, label in zip(bars, labels):
            contains, _ = bar.contains(event)
            if not contains:
                continue

            w = bar.get_width()
            h = bar.get_height()
            horizontal = abs(w) >= abs(h)

            if horizontal:
                value = w
                x = bar.get_x() + w
                y = bar.get_y() + h / 2.0
            else:
                value = h
                x = bar.get_x() + w / 2.0
                y = bar.get_y() + h

            ann.xy = (x, y)
            ann.set_text(f"{label}\n{value:.2f}")
            ann.set_visible(True)
            fig.canvas.draw_idle()
            hit_any = True
            break

        if not hit_any and ann.get_visible():
            ann.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


# ==========================
# Interactive wrappers
# ==========================

def create_sector_comparison_plot(df_normalized, covid_timeline, base_date, figsize=(14, 8)):
    fig, ax = create_sector_comparison_plot_static(df_normalized, covid_timeline, base_date, figsize)
    lines = ax.get_lines()
    _attach_line_tooltips(ax, lines)
    return fig, ax


def create_volatility_overlay_plot(df_normalized, covid_timeline, base_date, figsize=(14, 10)):
    fig, (ax1, ax2) = create_volatility_overlay_plot_static(df_normalized, covid_timeline, base_date, figsize)
    _attach_line_tooltips(ax1, ax1.get_lines())
    _attach_line_tooltips(ax2, ax2.get_lines())
    return fig, (ax1, ax2)


def create_individual_sector_comparison(df_normalized, covid_timeline, base_date, figsize=(16, 10)):
    fig, axes = create_individual_sector_comparison_static(df_normalized, covid_timeline, base_date, figsize)
    for ax in axes.flatten():
        _attach_line_tooltips(ax, ax.get_lines())
    return fig, axes


def create_correlation_heatmap(df_normalized, figsize=(10, 8)):
    """Return a static heatmap (no interactivity)."""
    fig, ax = create_correlation_heatmap_static(df_normalized, figsize)
    return fig, ax


def create_returns_visualization(df_normalized, covid_timeline, base_date, figsize=(14, 8)):
    fig, ax = create_returns_visualization_static(df_normalized, covid_timeline, base_date, figsize)
    _attach_line_tooltips(ax, ax.get_lines())
    return fig, ax


def create_unemployment_overlay(df_normalized, unrate, covid_timeline, base_date, figsize=(14, 10)):
    fig, axes = create_unemployment_overlay_static(df_normalized, unrate, covid_timeline, base_date, figsize)
    if fig is None:
        return None, None
    ax1, ax2 = axes
    _attach_line_tooltips(ax1, ax1.get_lines())
    _attach_line_tooltips(ax2, ax2.get_lines())
    return fig, axes


def create_raw_values_plot(df, covid_timeline, figsize=(14, 8)):
    fig, ax = create_raw_values_plot_static(df, covid_timeline, figsize)
    _attach_line_tooltips(ax, ax.get_lines())
    return fig, ax


def create_summary_statistics(df_normalized, base_date, figsize=(12, 8)):
    fig, axes = create_summary_statistics_static(df_normalized, base_date, figsize)
    for ax in axes:
        bars = ax.patches
        tick_labels = [t.get_text() for t in ax.get_yticklabels()] or [t.get_text() for t in ax.get_xticklabels()]
        if len(tick_labels) == len(bars):
            labels = tick_labels
        else:
            labels = [f"Bar {i+1}" for i in range(len(bars))]
        _attach_bar_tooltips(ax, bars, labels)
    return fig, axes


def create_volume_analysis(df_volume, covid_timeline, figsize=(14, 8)):
    fig, ax = create_volume_analysis_static(df_volume, covid_timeline, figsize)
    
    # Attach tooltips to lines (area charts have lines on top)
    # Filter out internal matplotlib child objects (from fill_between)
    all_lines = ax.get_lines()
    valid_lines = [ln for ln in all_lines if not ln.get_label().startswith('_child')]
    
    # Custom formatter for volume tooltips (show in millions)
    def volume_formatter(ln, x_val, y_val):
        label = ln.get_label() or "series"
        try:
            x_disp = mdates.num2date(x_val).strftime("%Y-%m-%d")
        except Exception:
            x_disp = str(x_val)
        volume_millions = y_val / 1e6
        return f"{label}\nDate: {x_disp}\nVolume: {volume_millions:.2f}M shares"
    
    _attach_line_tooltips(ax, valid_lines, formatter=volume_formatter)
    return fig, ax


def create_drawdown_timeline(df_normalized, covid_timeline, base_date, figsize=(18, 14)):
    fig, axes = create_drawdown_timeline_static(df_normalized, covid_timeline, base_date, figsize)
    
    # Attach tooltips to bars in each subplot
    sector_cols = list(SECTOR_TICKERS.keys()) + ['SPY']
    n_days = len(df_normalized.index)
    
    # Only process the 8 subplots (9th was removed)
    for idx, ax in enumerate(axes[:8]):
        bars = ax.patches
        if len(bars) > 0:
            # Each subplot has bars for one sector, so label them with the sector name
            sector_name = SECTOR_TICKERS.get(sector_cols[idx], 'S&P 500')
            bar_labels = [sector_name] * len(bars)
            _attach_bar_tooltips(ax, bars, bar_labels)
    
    return fig, axes


# ==========================
# Menu + main
# ==========================

def display_menu():
    print("\n" + "="*60)
    print("SECTOR VISUALIZATION MENU")
    print("="*60)
    print("\n1. Sector Comparison Plot")
    print("2. Volatility Overlay Plot")
    print("3. Individual Sector Comparison")
    print("4. Correlation Heatmap")
    print("5. Returns Visualization")
    print("6. Unemployment Overlay")
    print("7. Raw Values Plot")
    print("8. Summary Statistics")
    print("9. Volume Analysis")
    print("10. Drawdown Timeline")
    print("11. View All Visualizations")
    print("0. Exit")
    print("\n" + "="*60)


def main():
    print("Loading data...")
    df, df_volume, covid_timeline, unrate = load_data()

    print("Normalizing prices...")
    df_normalized, base_date = normalize_prices(df, base_date='2020-01-02')

    print(f"Using base date: {pd.to_datetime(base_date).strftime('%Y-%m-%d')}")

    while True:
        display_menu()
        choice = input("\nSelect a visualization (0-11): ")

        if choice == '0':
            print("\nExiting visualization tool. Goodbye!")
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
            print("\nCreating volume analysis...")
            fig, ax = create_volume_analysis(df_volume, covid_timeline)
            plt.show()
        elif choice == '10':
            print("\nCreating drawdown timeline...")
            fig, axes = create_drawdown_timeline(df_normalized, covid_timeline, base_date)
            plt.show()
        elif choice == '11':
            print("\nDisplaying all visualizations...")
            print("(Close each window to view the next)")

            print("\n1/10: Sector Comparison Plot...")
            fig1, ax1 = create_sector_comparison_plot(df_normalized, covid_timeline, base_date)
            plt.show()

            print("2/10: Volatility Overlay Plot...")
            fig2, axes2 = create_volatility_overlay_plot(df_normalized, covid_timeline, base_date)
            plt.show()

            print("3/10: Individual Sector Comparison...")
            fig3, axes3 = create_individual_sector_comparison(df_normalized, covid_timeline, base_date)
            plt.show()

            print("4/10: Correlation Heatmap...")
            fig4, ax4 = create_correlation_heatmap(df_normalized)
            plt.show()

            print("5/10: Returns Visualization...")
            fig5, ax5 = create_returns_visualization(df_normalized, covid_timeline, base_date)
            plt.show()

            print("6/10: Unemployment Overlay...")
            fig6, axes6 = create_unemployment_overlay(df_normalized, unrate, covid_timeline, base_date)
            if fig6 is not None:
                plt.show()
            else:
                print("Unemployment data not available.")

            print("7/10: Raw Values Plot...")
            fig7, ax7 = create_raw_values_plot(df, covid_timeline)
            plt.show()

            print("8/10: Summary Statistics...")
            fig8, axes8 = create_summary_statistics(df_normalized, base_date)
            plt.show()

            print("9/10: Volume Analysis...")
            fig9, ax9 = create_volume_analysis(df_volume, covid_timeline)
            plt.show()

            print("10/10: Drawdown Timeline...")
            fig10, axes10 = create_drawdown_timeline(df_normalized, covid_timeline, base_date)
            plt.show()

            print("\nAll visualizations displayed!")
        else:
            print("Invalid choice. Please select a number between 0-11.")


if __name__ == '__main__':
    main()
