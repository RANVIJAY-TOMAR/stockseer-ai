import streamlit as st # <--- ADDED THIS LINE
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objs as go 
import feedparser
import requests
from bs4 import BeautifulSoup
import numpy as np

# --- Data Fetching and Processing ---
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_symbol, period='3mo', interval='1d'):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period=period, interval=interval)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=3600)
def add_technical_indicators(df):
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    df_ta = df.copy()
    df_ta['SMA_20'] = ta.trend.sma_indicator(df_ta['Close'], window=20)
    df_ta['RSI'] = ta.momentum.rsi(df_ta['Close'], window=14)
    df_ta['MACD'] = ta.trend.macd_diff(df_ta['Close'])
    return df_ta

@st.cache_data(ttl=3600)
def generate_signal_basic(df):
    if df.empty or not all(k in df.columns for k in ['RSI', 'MACD']):
        return "N/A"
    latest = df.iloc[-1]
    if pd.isna(latest['RSI']) or pd.isna(latest['MACD']):
        return "N/A"
    if latest['RSI'] < 30 and latest['MACD'] > 0:
        return "BUY"
    elif latest['RSI'] > 70 and latest['MACD'] < 0:
        return "SELL"
    else:
        return "HOLD"

@st.cache_data(ttl=3600)
def generate_signal_detailed(df):
    if df.empty or not all(k in df.columns for k in ['RSI', 'MACD', 'SMA_20', 'Close']) or len(df) < 20:
        return "N/A", "Insufficient data for detailed signal generation (need ~20 days)."
    latest = df.iloc[-1]
    if len(df) < 2: return "N/A", "Not enough data for crossover signal logic."
    previous = df.iloc[-2] 
    rsi_val = latest['RSI']; macd_hist_val = latest['MACD']
    try:
        macd_line_series = ta.trend.macd(df['Close'])
        macd_signal_series = ta.trend.macd_signal(df['Close'])
        if len(macd_line_series) < 2 or len(macd_signal_series) < 2: return "N/A", "Not enough data for MACD line calculation."
        macd_line_val = macd_line_series.iloc[-1]; macd_signal_line_val = macd_signal_series.iloc[-1]
        prev_macd_line = macd_line_series.iloc[-2]; prev_macd_signal_line = macd_signal_series.iloc[-2]
    except Exception as e: return "N/A", f"Error calculating MACD lines: {e}"
    close_price = latest['Close']; sma20 = latest['SMA_20']
    reasons = []; buy_score = 0; sell_score = 0
    if pd.isna(rsi_val) or pd.isna(macd_hist_val) or pd.isna(macd_line_val) or pd.isna(macd_signal_line_val) or pd.isna(sma20):
        return "N/A", "Indicator data has NaNs."
    if rsi_val < 30: reasons.append(f"RSI ({rsi_val:.2f}) is in oversold territory (<30)."); buy_score += 2
    elif rsi_val < 40: reasons.append(f"RSI ({rsi_val:.2f}) is approaching oversold (<40)."); buy_score += 1
    elif rsi_val > 70: reasons.append(f"RSI ({rsi_val:.2f}) is in overbought territory (>70)."); sell_score += 2
    elif rsi_val > 60: reasons.append(f"RSI ({rsi_val:.2f}) is approaching overbought (>60)."); sell_score += 1
    else: reasons.append(f"RSI ({rsi_val:.2f}) is neutral (30-70).")
    if macd_line_val > macd_signal_line_val and prev_macd_line <= prev_macd_signal_line: reasons.append(f"MACD Line ({macd_line_val:.2f}) crossed above Signal Line ({macd_signal_line_val:.2f}) (Bullish Crossover)."); buy_score += 2
    elif macd_line_val < macd_signal_line_val and prev_macd_line >= prev_macd_signal_line: reasons.append(f"MACD Line ({macd_line_val:.2f}) crossed below Signal Line ({macd_signal_line_val:.2f}) (Bearish Crossover)."); sell_score += 2
    elif macd_line_val > macd_signal_line_val: reasons.append(f"MACD Line ({macd_line_val:.2f}) is above Signal Line ({macd_signal_line_val:.2f}) (Bullish)."); buy_score +=1
    elif macd_line_val < macd_signal_line_val: reasons.append(f"MACD Line ({macd_line_val:.2f}) is below Signal Line ({macd_signal_line_val:.2f}) (Bearish)."); sell_score +=1
    if macd_hist_val > 0: reasons.append(f"MACD Histogram ({macd_hist_val:.2f}) is positive (Bullish momentum)."); buy_score += 0.5
    elif macd_hist_val < 0: reasons.append(f"MACD Histogram ({macd_hist_val:.2f}) is negative (Bearish momentum)."); sell_score += 0.5
    if close_price > sma20: reasons.append(f"Price (${close_price:.2f}) is above 20-Day SMA (${sma20:.2f}) (Short-term uptrend)."); buy_score += 1
    elif close_price < sma20: reasons.append(f"Price (${close_price:.2f}) is below 20-Day SMA (${sma20:.2f}) (Short-term downtrend)."); sell_score += 1
    final_signal = "HOLD"
    if buy_score > sell_score + 1.5 : final_signal = "STRONG BUY"
    elif sell_score > buy_score + 1.5: final_signal = "STRONG SELL"
    elif buy_score > sell_score : final_signal = "BUY"
    elif sell_score > buy_score: final_signal = "SELL"
    return final_signal, " ".join(reasons) if reasons else "Neutral signals or insufficient conviction."

@st.cache_data(ttl=86400)
def get_company_info_yfinance(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol); info = stock.info
        if not info: return "Info not available.", "N/A", "N/A", None, "N/A", {}, pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame(), pd.DataFrame()
        analyst_recs, analyst_price_target, insider_tx_summary_df, inst_holders_df = None, None, pd.DataFrame(), pd.DataFrame()
        try: analyst_recs = stock.recommendations
        except: pass
        try: analyst_price_target = stock.analyst_price_target
        except: pass
        try:
            holders = stock.institutional_holders
            if holders is not None and not holders.empty: inst_holders_df = holders
        except: pass
        try:
            summary = stock.insider_transactions_summary
            if summary is not None and not summary.empty: insider_tx_summary_df = summary
        except: pass
        return (info.get('longBusinessSummary', "No summary available via yfinance API."), info.get('sector', 'N/A'), info.get('industry', 'N/A'),
                info.get('marketCap'), info.get('exchange', 'N/A'), info,
                stock.quarterly_financials if not stock.quarterly_financials.empty else stock.financials,
                stock.quarterly_earnings if not stock.quarterly_earnings.empty else stock.earnings,
                analyst_recs, analyst_price_target, insider_tx_summary_df, inst_holders_df)
    except Exception as e:
        return (f"yfinance info retrieval failed: {e}", "N/A", "N/A", None, "N/A", {}, 
                pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame(), pd.DataFrame())

@st.cache_data(ttl=86400)
def get_company_profile_scraping(ticker_symbol):
    try:
        search_query = f"{ticker_symbol} stock company profile site:finance.yahoo.com"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}&hl=en"
        search_response = requests.get(search_url, headers=headers, timeout=10)
        search_response.raise_for_status()
        soup = BeautifulSoup(search_response.text, 'html.parser')
        profile_url = None
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            if href and "finance.yahoo.com/quote" in href and "/profile" in href:
                if href.startswith("/url?q="):
                    profile_url = href.split("/url?q=")[1].split("&sa=")[0]
                    from urllib.parse import unquote
                    profile_url = unquote(profile_url)
                    break
        if not profile_url: return "Couldn't find Yahoo Finance profile link on Google."
        profile_response = requests.get(profile_url, headers=headers, timeout=10)
        profile_response.raise_for_status()
        profile_soup = BeautifulSoup(profile_response.text, 'html.parser')
        summary_section = profile_soup.find('section', attrs={'data-testid': 'qsp-profile'}) # Updated based on Yahoo's current structure (might change)
        if summary_section:
            # Try to find the paragraph(s) within the specific profile description section
            profile_description_div = summary_section.find('div', class_=lambda x: x and 'description' in x.lower()) # Heuristic
            if profile_description_div:
                summary_p = profile_description_div.find('p')
                if summary_p: return summary_p.get_text(separator="\n", strip=True)
            # Fallback within the section if specific div not found
            summary_p_fallback = summary_section.find('p')
            if summary_p_fallback: return summary_p_fallback.get_text(separator="\n", strip=True)

        all_paragraphs = profile_soup.find_all('p') # General fallback
        for p_tag in all_paragraphs:
            if len(p_tag.get_text(strip=True)) > 200: return p_tag.get_text(separator="\n", strip=True)
        return "Company profile section/paragraph not found on Yahoo page."
    except requests.exceptions.RequestException as e: return f"Network error retrieving company profile: {e}"
    except Exception as e: return f"Error scraping company profile: {e}"

@st.cache_data(ttl=1800)
def get_stock_news_feedparser(ticker_symbol):
    feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}®ion=US&lang=en-US"
    try:
        news_feed = feedparser.parse(feed_url)
        if news_feed.bozo: return [], f"News feed for {ticker_symbol} is not well-formed."
        return news_feed.entries[:5], None
    except Exception as e: return [], f"Could not fetch news using feedparser: {e}"

def plot_stock_chart_simple(df, symbol): # Kept if needed, but main app uses complex chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='white', width=2)))
    if 'SMA_20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange', dash='dash')))
    fig.update_layout(template='plotly_dark', title={'text': f"📈 {symbol} Stock Price & SMA 20", 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor':'top'},
                      xaxis_title='Date', yaxis_title='Price (USD)', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(l=40, r=40, t=80, b=40), height=500, plot_bgcolor='black', paper_bgcolor='black')
    fig.update_xaxes(showgrid=False, color='white'); fig.update_yaxes(showgrid=False, color='white')
    return fig