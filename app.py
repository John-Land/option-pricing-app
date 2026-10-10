import datetime
import time
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import t, genpareto
import streamlit as st
import yfinance as yf

# ==============================================================================
# STREAMLIT PAGE CONFIG & CSS STYLING
# ==============================================================================
st.set_page_config(page_title="Historic Asset Return Fat Tail Analysis", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .chart-desc {
        font-size: 0.875rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .section-header {
        margin-top: 1.5rem;
        padding-top: 0.5rem;
        border-top: 2px solid #e2e8f0;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    div.stButton > button[kind="primary"] {
        background-color: #265d74 !important;
        border-color: #265d74 !important;
        color: white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1e4b5d !important;
        border-color: #1e4b5d !important;
    }
    /* Uniformly adjust metric values */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 1.4rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# DATA FETCHING & SPLICING LOGIC (From option_pricing_app.py)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data_v2(ticker_symbol):
    for attempt in range(3):
        try:
            data = yf.Ticker(ticker_symbol).history(period="max", interval="1d")
            if data is not None and not data.empty and len(data) > 252:
                data = data.dropna(subset=["Close"])
                if len(data) > 252:
                    return data
        except Exception:
            pass
        time.sleep(2)
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_recent_data_v2(ticker_symbol):
    for attempt in range(3):
        try:
            data = yf.Ticker(ticker_symbol).history(period="5d", interval="1d")
            if data is not None and not data.empty:
                return data.dropna(subset=["Close"])
        except Exception:
            pass
        time.sleep(2)
    return pd.DataFrame()

def load_historical_csv(ticker_symbol):
    file_map = {
        "GC=F": "data/gold.csv",
        "SI=F": "data/silver.csv",
        "JPY=X": "data/usd_jpy.csv"
    }
    if ticker_symbol not in file_map:
        return pd.DataFrame()
        
    filepath = file_map[ticker_symbol]
    if not os.path.exists(filepath):
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        cols = ["Open", "High", "Low", "Close"]
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        df = df.dropna(subset=["Close"])
        return df
    except Exception as e:
        print(f"CSV Load Error for {ticker_symbol}: {e}")
        return pd.DataFrame()


# ==============================================================================
# SIDEBAR / LEFT PANEL
# ==============================================================================
st.sidebar.title("Asset Selection")

ticker = st.sidebar.text_input("Underlying Ticker (e.g., SPY)", value="SPY").upper()

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = True  # Changed from False to True to auto-run on load

if st.sidebar.button("Fetch Data and Analyse Returns", type="primary"):
    st.session_state.run_analysis = True


if st.session_state.run_analysis:
    etf_to_index_map = {
        "SPY": "^GSPC",  
        "QQQ": "^NDX",   
        "IWM": "^RUT",   
        "DIA": "^DJI",   
        "GLD": "GC=F",   
        "SLV": "SI=F",
        "USD/JPY": "JPY=X"
    }
    fetch_ticker = etf_to_index_map.get(ticker, ticker)

    with st.spinner(f"Fetching max historical data for {ticker} via underlying proxy {fetch_ticker}..."):
        # 1. Fetch from Yahoo Finance
        data = fetch_historical_data_v2(fetch_ticker)
        if data.empty:
            st.error(f"No historical data found for {fetch_ticker}.")
            st.stop()
            
        data.index = data.index.tz_localize(None)

        # 2. Historical CSV Splicing Logic
        csv_data = load_historical_csv(fetch_ticker)
        if not csv_data.empty:
            min_yf_date = data.index.min()
            csv_data_filtered = csv_data[csv_data.index < min_yf_date].copy()
            if not csv_data_filtered.empty:
                target_cols = ["Open", "High", "Low", "Close", "Volume"]
                for col in target_cols:
                    if col not in csv_data_filtered.columns:
                        csv_data_filtered[col] = 0.0
                csv_data_filtered = csv_data_filtered[target_cols]
                
                available_yf_cols = [col for col in target_cols if col in data.columns]
                data = data[available_yf_cols]
                for col in target_cols:
                    if col not in data.columns:
                        data[col] = 0.0
                data = data[target_cols]
                data = pd.concat([csv_data_filtered, data]).sort_index()

        # 3. Fetch recent ETF price for scaling
        etf_data = fetch_recent_data_v2(ticker)
        if etf_data.empty:
            st.error(f"Could not fetch current live price for ETF {ticker}.")
            st.stop()
            
        current_etf_price = float(etf_data["Close"].iloc[-1])
        valid_index_prices = data["Close"].dropna()
        if isinstance(valid_index_prices, pd.DataFrame):
            valid_index_prices = valid_index_prices.iloc[:, 0]
            
        current_index_price = float(valid_index_prices.iloc[-1])
        price_ratio = current_etf_price / current_index_price

        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns:
                data[col] = data[col] * price_ratio

        # Metrics Extraction
        start_date = data.index.min().strftime("%Y-%m-%d")
        end_date = data.index.max().strftime("%Y-%m-%d")
        total_days = len(data)

        close_prices = data["Close"]
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]
        close_array = close_prices.to_numpy(dtype=float).flatten()
        S_0 = float(close_array[-1])

        returns = np.log(close_array[1:] / close_array[:-1])
        returns = returns[~np.isnan(returns)]
        
        daily_avg = np.mean(returns)
        daily_vol = np.std(returns)
        annualized_vol = daily_vol * np.sqrt(252)
        years_of_data = len(returns) / 252.0

        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Historic Data Description")
        st.sidebar.metric("Spot Price", f"${S_0:.2f}")
        st.sidebar.metric("Historical Daily Avg", f"{daily_avg*100:.4f}%")
        st.sidebar.metric("Historical Daily Vol", f"{daily_vol*100:.2f}%")
        st.sidebar.metric("Historical Annualized Vol", f"{annualized_vol*100:.2f}%")
        st.sidebar.metric("Underlying Source", fetch_ticker)
        st.sidebar.metric("Start Date", start_date)
        st.sidebar.metric("End Date", end_date)
        st.sidebar.metric("Trading Years", f"{years_of_data:.1f}")
        st.sidebar.metric("Trading Days", f"{total_days:,}")

    # ==============================================================================
    # MAIN SECTION
    # ==============================================================================
    st.markdown("<h1 class='main-header'>Asset Return Tail Diagnostics</h1>", unsafe_allow_html=True)
    
    # ------------------------------------------------------------------------------
    # SECTION 1: Historic Daily Returns
    # ------------------------------------------------------------------------------
    st.markdown("<h2 class='section-header'>Historic Daily Returns</h2>", unsafe_allow_html=True)
    
    r_mean = np.mean(returns)
    r_std = np.std(returns)
    r_min = np.min(returns)
    r_max = np.max(returns)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical Daily Avg", f"{r_mean*100:.4f}%")
    c2.metric("Daily Std of Return", f"{r_std*100:.2f}%")
    c3.metric("Min Daily Return", f"{r_min*100:.2f}%")
    c4.metric("Max Daily Return", f"{r_max*100:.2f}%")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns * 100, 
        marker_color='#265d74', 
        nbinsx=150,
        name="Daily % Returns"
    ))
    fig_hist.update_layout(
        xaxis_title="Daily Return (%) \u2192",
        yaxis_title="Frequency \u2191",
        margin=dict(l=40, r=20, t=40, b=40),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_hist.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    fig_hist.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    st.plotly_chart(fig_hist, use_container_width=True)

    # ------------------------------------------------------------------------------
    # SECTION 2: Model Fitting and Tail Diagnostics on Historic Daily Returns
    # ------------------------------------------------------------------------------
    st.markdown("<h2 class='section-header'>Model Fitting and Tail Diagnostics on Historic Daily Returns</h2>", unsafe_allow_html=True)

    # 1. Student-t Fitting
    df_mle, loc_mle, scale_mle = t.fit(returns)
    
    # 2. GPD Fitting (POT Method)
    pct_5 = np.percentile(returns, 5)
    pct_95 = np.percentile(returns, 95)
    
    exceedances_l = pct_5 - returns[returns < pct_5]
    exceedances_r = returns[returns > pct_95] - pct_95
    
    shape_l, _, scale_gpd_l = genpareto.fit(exceedances_l, floc=0) if len(exceedances_l) > 0 else (0.001, 0, 0.001)
    shape_r, _, scale_gpd_r = genpareto.fit(exceedances_r, floc=0) if len(exceedances_r) > 0 else (0.001, 0, 0.001)

    # Metrics Layout
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Student t Mean MLE", f"{loc_mle*100:.4f}%")
    m2.metric("Student t Scale MLE", f"{scale_mle*100:.2f}%")
    m3.metric("Student t df MLE", f"{df_mle:.3f}")
    m4.metric("Left Tail GPD \u03BE", f"{shape_l:.4f}")
    m5.metric("Right Tail GPD \u03BE", f"{shape_r:.4f}")

    # Analytic variables for empirical diagnostics
    zipf_alpha_l, zipf_alpha_r = None, None
    ms_finals = {1: 0, 2: 0, 3: 0, 4: 0}
    mep_slope_l, mep_slope_r = None, None
    min_exceedances = 10

    # Isolate empirical tails
    abs_returns = np.abs(returns)
    left_tail_empirical = np.abs(returns[returns < 0])
    right_tail_empirical = returns[returns > 0]

    # --- PARAMETRIC MODEL CHARTS ---
    col_t, col_gpd = st.columns(2)
    
    with col_t:
        st.markdown("### Student-t Distribution")
        st.caption(r"""Profiles the likelihood of the data being generated by a Student-t distribution across various Degrees of Freedom (df).
* **df $\ge$ 30:** Thin tails (Normal distribution).
* **df $\le$ 4:** Fat tails (Infinite kurtosis).
* **df $\le$ 2:** Extreme fat tails (Infinite variance).""")
        
        dfs = np.arange(0.5, 30.25, 0.25)
        ll_vals = []
        for df_val in dfs:
            ll = np.sum(t.logpdf(returns, df_val, loc=loc_mle, scale=scale_mle))
            ll_vals.append(ll)
            
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=dfs, y=ll_vals, 
            mode='lines', 
            name='Log-Likelihood',
            line=dict(color='#265d74', width=2.5)
        ))
        fig_t.add_vline(x=df_mle, line_dash="dash", line_color="#c81d25", annotation_text=f"MLE df={df_mle:.2f}")
        
        fig_t.update_layout(
            xaxis_title="Degrees of Freedom (df) \u2192",
            yaxis_title="Log-Likelihood \u2191",
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig_t.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False)
        fig_t.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False)
        st.plotly_chart(fig_t, use_container_width=True)

    with col_gpd:
        st.markdown("### Fitted GPD (POT Exceedances)")
        st.caption(r"""Models the probability density of extreme exceedances beyond a threshold using the Tail Index ($\xi$).
* **$\xi \le 0$:** Thin tails (Normal/Exponential bounds).
* **$0 < \xi < 0.5$:** Heavy tails (Power-law decay).
* **$\xi \ge 0.5$:** Infinite variance (Historical maximums do not limit future extremes).""")

        max_exc = max(np.max(exceedances_l), np.max(exceedances_r)) if len(exceedances_l)>0 and len(exceedances_r)>0 else 0.1
        x_gpd = np.linspace(0, max_exc, 200)
        
        pdf_l = genpareto.pdf(x_gpd, c=shape_l, loc=0, scale=scale_gpd_l)
        pdf_r = genpareto.pdf(x_gpd, c=shape_r, loc=0, scale=scale_gpd_r)
        
        fig_gpd = go.Figure()
        fig_gpd.add_trace(go.Scatter(
            x=x_gpd * 100, y=pdf_l, 
            mode='lines', 
            name=f'Left Tail (\u03BE={shape_l:.2f})',
            line=dict(color='#c81d25', width=2.5)
        ))
        fig_gpd.add_trace(go.Scatter(
            x=x_gpd * 100, y=pdf_r, 
            mode='lines', 
            name=f'Right Tail (\u03BE={shape_r:.2f})',
            line=dict(color='#265d74', width=2.5)
        ))
        
        fig_gpd.update_layout(
            xaxis_title="Absolute Exceedance (%) \u2192",
            yaxis_title="Likelihood (Density) \u2191",
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)")
        )
        fig_gpd.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False)
        fig_gpd.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False)
        st.plotly_chart(fig_gpd, use_container_width=True)

    # --- EMPIRICAL DIAGNOSTIC CHARTS ---
    col_left, col_mid, col_right = st.columns(3)
    
    with col_left:
        st.markdown("### Zipf Plot")
        st.caption(r"""Diagnoses the empirical rate of tail decay by plotting the Log of an absolute threshold $K$ (X-axis) against the Log of the probability of exceeding that threshold, $P(|X| > K)$ (Y-axis).
*   **Thin Tail:** Plunges almost vertically downward.
*   **Fat Tail:** Shallow, negatively sloping diagonal.""")
        
        fig_zipf = go.Figure()
        
        if len(left_tail_empirical) >= min_exceedances:
            l_sorted = np.sort(left_tail_empirical)
            l_ranks = np.arange(len(l_sorted), 0, -1)
            l_surv = l_ranks / len(l_sorted)
            l_mask = (l_sorted >= np.percentile(left_tail_empirical, 90)) & (l_ranks >= min_exceedances)
            
            if len(l_sorted[l_mask]) > 1:
                try:
                    slope_l, _ = np.polyfit(np.log(l_sorted[l_mask]), np.log(l_surv[l_mask]), 1)
                    zipf_alpha_l = -slope_l
                except:
                    pass
            
            fig_zipf.add_trace(go.Scatter(
                x=l_sorted[l_mask], y=l_surv[l_mask],
                mode='lines+markers', name='Left Tail',
                line=dict(color='#c81d25', width=2),
                marker=dict(size=4, opacity=0.6)
            ))
            
        if len(right_tail_empirical) >= min_exceedances:
            r_sorted = np.sort(right_tail_empirical)
            r_ranks = np.arange(len(r_sorted), 0, -1)
            r_surv = r_ranks / len(r_sorted)
            r_mask = (r_sorted >= np.percentile(right_tail_empirical, 90)) & (r_ranks >= min_exceedances)
            
            if len(r_sorted[r_mask]) > 1:
                try:
                    slope_r, _ = np.polyfit(np.log(r_sorted[r_mask]), np.log(r_surv[r_mask]), 1)
                    zipf_alpha_r = -slope_r
                except:
                    pass
            
            fig_zipf.add_trace(go.Scatter(
                x=r_sorted[r_mask], y=r_surv[r_mask],
                mode='lines+markers', name='Right Tail',
                line=dict(color='#265d74', width=2),
                marker=dict(size=4, opacity=0.6)
            ))
            
        fig_zipf.update_layout(
            xaxis_type="log",
            yaxis_type="log",
            xaxis_title="Absolute Return Threshold (K) [Log] \u2192",
            yaxis_title="P(|X| > K) [Log] \u2191",
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
        )
        fig_zipf.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
        fig_zipf.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
        st.plotly_chart(fig_zipf, use_container_width=True)

    with col_mid:
        st.markdown("### Maximum-to-Sum Plot")
        st.caption(r"""Visually tests if a moment is finite by plotting historical days $t$ (X-axis) against the ratio of the single maximum absolute return to the sum of all absolute returns (Y-axis), for moments $p=1, 2, 3, 4$.
*   **Thin Tail:** Lines drop toward 0 (finite moments).
*   **Fat Tail:** Higher moments jump erratically or hover above 0.""")
        
        ms_n = np.arange(1, len(abs_returns) + 1)
        fig_ms = go.Figure()
        colors_ms = {1: '#265d74', 2: '#c81d25', 3: '#7dcea0', 4: 'black'}
        
        step = max(1, len(abs_returns) // 1000)
        plot_idx = np.arange(0, len(abs_returns), step)
        
        for p in [1, 2, 3, 4]:
            pow_samples = np.power(abs_returns, p).astype(np.float64) 
            running_max = np.maximum.accumulate(pow_samples)
            running_sum = np.cumsum(pow_samples)
            out_array = np.zeros_like(running_max, dtype=np.float64)
            ratio = np.divide(running_max, running_sum, out=out_array, where=running_sum!=0)
            
            ms_finals[p] = float(ratio[-1]) if len(ratio) > 0 else 0.0
            
            fig_ms.add_trace(go.Scatter(
                x=ms_n[plot_idx], 
                y=ratio[plot_idx], 
                mode='lines', 
                name=f'p={p} (Moment {p})',
                line=dict(color=colors_ms[p], width=2)
            ))
            
        fig_ms.update_layout(
            xaxis_title="Historical Days (t) \u2192",
            yaxis_title="Max / Sum Ratio \u2191",
            yaxis=dict(range=[-0.05, 1.05]),
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)")
        )
        fig_ms.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
        fig_ms.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
        st.plotly_chart(fig_ms, use_container_width=True)

    with col_right:
        st.markdown("### Mean Excess Plot")
        st.caption(r"""Plots how severe deviations become *after* an initial absolute threshold $K$ (X-axis) is breached, measured against the expected excess beyond $K$, $e(K)$ (Y-axis).
*   **Thin Tail:** Downward slope (aggressive mean reversion).
*   **Fat Tail:** Upward slope (expected severity accelerates).""")
        
        k_start_mep = np.percentile(abs_returns, 90)
        
        valid_k_l, mep_l_vals = [], []
        if len(left_tail_empirical) >= min_exceedances:
            max_l = np.max(left_tail_empirical)
            k_vals_l = np.linspace(k_start_mep, max_l, 150)
            
            for k in k_vals_l:
                l_exc = -k - returns[returns < -k]
                if len(l_exc) >= min_exceedances:
                    mep_l_vals.append(np.mean(l_exc))
                    valid_k_l.append(k)
                    
            if len(valid_k_l) > 1:
                try:
                    mep_slope_l, _ = np.polyfit(valid_k_l, mep_l_vals, 1)
                except:
                    pass

        valid_k_r, mep_r_vals = [], []
        if len(right_tail_empirical) >= min_exceedances:
            max_r = np.max(right_tail_empirical)
            k_vals_r = np.linspace(k_start_mep, max_r, 150)
            
            for k in k_vals_r:
                r_exc = returns[returns > k] - k
                if len(r_exc) >= min_exceedances:
                    mep_r_vals.append(np.mean(r_exc))
                    valid_k_r.append(k)
                    
            if len(valid_k_r) > 1:
                try:
                    mep_slope_r, _ = np.polyfit(valid_k_r, mep_r_vals, 1)
                except:
                    pass
            
        fig_mep = go.Figure()
        if len(valid_k_l) > 0:
            fig_mep.add_trace(go.Scatter(
                x=valid_k_l, y=mep_l_vals,
                mode='lines+markers', name='Left Tail',
                line=dict(color='#c81d25', width=2),
                marker=dict(size=4)
            ))
        if len(valid_k_r) > 0:
            fig_mep.add_trace(go.Scatter(
                x=valid_k_r, y=mep_r_vals,
                mode='lines+markers', name='Right Tail',
                line=dict(color='#265d74', width=2),
                marker=dict(size=4)
            ))
        
        fig_mep.update_layout(
            xaxis_title="Absolute Threshold (K) \u2192",
            yaxis_title="Expected Excess e(K) \u2191",
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
        )
        fig_mep.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
        fig_mep.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
        st.plotly_chart(fig_mep, use_container_width=True)

    # ------------------------------------------------------------------------------
    # UNIFIED DATA-DRIVEN INTERPRETATION BLOCK (Using standard Markdown)
    # ------------------------------------------------------------------------------
    
    # Helper to safely format floats
    def fmt(val, decimals=2):
        return f"{val:.{decimals}f}" if val is not None and not np.isnan(val) else "N/A"

    # Fat tail global flag based on multiple signals
    is_fat = (
        (df_mle < 4.0) or 
        (shape_l > 0) or (shape_r > 0) or 
        (zipf_alpha_l is not None and zipf_alpha_l < 4.0) or 
        (zipf_alpha_r is not None and zipf_alpha_r < 4.0) or 
        (ms_finals.get(4, 0) > 0.05) or 
        (mep_slope_l is not None and mep_slope_l > 0.05) or 
        (mep_slope_r is not None and mep_slope_r > 0.05)
    )
    tail_type = "fat" if is_fat else "thin"

    # Text formatters for parametric models
    def get_t_text_styled(df):
        if df <= 2: return "**Extreme Fat Tails (Infinite Variance):** Severe structural tail risk. Historical standard deviation is mathematically meaningless."
        elif df <= 4: return "**Fat Tailed (Infinite Kurtosis):** Variance exists, but outlier events completely dominate long-term tail risk."
        else: return "**Thin Tailed:** More closely resembles normal distribution assumptions."

    def get_gpd_text_styled(xi):
        if xi >= 0.5: return "**Infinite Variance:** The tail is so fat that historical worst-case scenarios do not limit future worst-case scenarios."
        elif xi > 0: return "**Heavy Tails:** Extreme structural jumps are far more likely than standard distributions assume."
        else: return "**Thin Tails:** Normal/Exponential risk bounds."

    # Text formatters for empirical plots
    def get_zipf_text_styled(alpha):
        if alpha is None or np.isnan(alpha): return "**N/A:** Insufficient tail data to calculate exponent."
        if alpha <= 2.0: return "**Infinite Variance:** Extreme events decay so slowly that standard deviation is mathematically meaningless and highly unstable."
        elif alpha <= 4.0: return "**Fat Tailed (Infinite Kurtosis):** Variance exists, but outlier events completely dominate long-term tail risk."
        else: return "**Thin Tailed:** Outliers decay rapidly, closely resembling standard exponential or normal distributions."

    def get_ms_text_styled(finals):
        failed_ps = [p for p in [4, 3, 2, 1] if finals.get(p, 0) > 0.05]
        
        if not failed_ps:
            return f"**Thin Tailed (All $p \le 4$ ratios $\le 0.05$):** All evaluated moments smoothly converge near zero, indicating finite and stable statistics."
            
        severity_map = {1: "Undefined Mean", 2: "Infinite Variance", 3: "Infinite Skewness", 4: "Infinite Kurtosis"}
        severity = severity_map[min(failed_ps)]
        
        failed_details = ", ".join([f"$p={p}$ $\\approx {finals.get(p, 0):.3f}$" for p in failed_ps])
        
        return f"**{severity} ({failed_details}):** Historical outliers dominate the sum, mathematically preventing convergence of these moments."

    def get_mep_text_styled(slope):
        if slope is None or np.isnan(slope): return "**N/A:** Insufficient tail data to extract excess slope."
        if slope > 0.05: return "**Fat Tail (Accelerating):** The expected severity of a shock strictly increases the deeper into the tail you go."
        elif slope < -0.05: return "**Thin Tail (Dampening):** Extreme shocks encounter resistance and dampen out over time."
        else: return "**Exponential / Normal:** The expected excess severity remains relatively constant across thresholds."

    # Render purely using Streamlit's native st.info to guarantee markdown math and bullets render properly
    st.info(f"""
#### Data-Driven Interpretation

The historic daily return data for ticker **{ticker}** indicate a **{tail_type}** tailed distribution.

**Student-t Distribution (Degrees of Freedom):**
* **MLE ($df \\approx {fmt(df_mle)}$):** {get_t_text_styled(df_mle)}

**Fitted GPD (POT Exceedances):**
* **Left Tail ($\\xi \\approx {fmt(shape_l)}$):** {get_gpd_text_styled(shape_l)}
* **Right Tail ($\\xi \\approx {fmt(shape_r)}$):** {get_gpd_text_styled(shape_r)}

**Zipf Plot (Tail Exponent $\\alpha$):**
* **Left Tail ($\\alpha \\approx {fmt(zipf_alpha_l)}$):** {get_zipf_text_styled(zipf_alpha_l)}
* **Right Tail ($\\alpha \\approx {fmt(zipf_alpha_r)}$):** {get_zipf_text_styled(zipf_alpha_r)}

**Maximum-to-Sum Plot (Moment Convergence):**
* {get_ms_text_styled(ms_finals)}

**Mean Excess Plot (Expected Severity):**
* **Left Tail (Slope $\\approx {fmt(mep_slope_l, 3)}$):** {get_mep_text_styled(mep_slope_l)}
* **Right Tail (Slope $\\approx {fmt(mep_slope_r, 3)}$):** {get_mep_text_styled(mep_slope_r)}
    """)