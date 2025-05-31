import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
import ta
import numpy as np
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="StockSeer.AI", layout="wide", page_icon="📈")

# --- SENTIMENT MODEL ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
sentiment_analyzer = load_sentiment_model()

# --- STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        color: #e0e0e0;
        background-color: #0a0a0a;
        background-image:
            url("data:image/svg+xml,%3Csvg viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.6' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.03'/%3E%3C/svg%3E"),
            linear-gradient(150deg, #0a0a0a, #1a1a1a);
        min-height: 100vh;
    }
    .stApp {
        background-color: #0a0a0a;
        background-image:
            url("data:image/svg+xml,%3Csvg viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.6' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.03'/%3E%3C/svg%3E"),
            linear-gradient(150deg, #0a0a0a, #1a1a1a);
        min-height: 100vh;
    }
    .metric-box {
        background: linear-gradient(145deg, #222222, #111111);
        border: 1px solid #39ff14;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.8);
        transition: all 0.3s ease-in-out;
    }
    .metric-box:hover {
        transform: translateY(-7px);
        box-shadow: 0 10px 30px rgba(57, 255, 20, 0.4);
        border-color: #5eff40;
    }
    .tag {
        display: inline-block; background-color: #39ff14; color: #0e0e0e;
        padding: 6px 12px; margin-right: 10px; margin-bottom: 10px;
        font-weight: 600; font-size: 14px; border-radius: 8px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.5); letter-spacing: 0.5px;
    }
    a { color: #39ff14; text-decoration: none; transition: color 0.2s ease; }
    a:hover { color: #5eff40; text-decoration: underline; }
    .stMetricValue { font-size: 2.2rem !important; font-weight: 700; color: #39ff14; text-shadow: 0 0 10px rgba(57, 255, 20, 0.7); }
    .stMetricLabel { font-size: 1.1rem !important; color: #b0b0b0; font-weight: 400; letter-spacing: 0.5px; }
    .stMetricDelta { font-size: 1.1rem !important; font-weight: 600; letter-spacing: 0.5px; }
    h1, h2, h3, h4, h5, h6 {
        color: #39ff14; text-shadow: 0 0 12px rgba(57, 255, 20, 0.5);
        font-weight: 700; margin-top: 2rem; margin-bottom: 1.2rem; letter-spacing: 1px;
    }
    h1 { font-size: 2.8rem; } h2 { font-size: 2.2rem; } h3 { font-size: 1.8rem; } h4 { font-size: 1.5rem; }
    p { color: #e0e0e0; line-height: 1.7; font-weight: 300; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem; color: #e0e0e0; font-weight: 400; transition: color 0.3s ease;
    }
    .stTabs [data-baseweb="tab-list"] button:hover [data-testid="stMarkdownContainer"] p { color: #5eff40; }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #1a1a1a; border-radius: 10px 10px 0 0; margin-right: 8px;
        padding: 10px 20px; border: 1px solid #2b2b2b; border-bottom: none;
        transition: background-color 0.3s ease, border 0.3s ease;
    }
    .stTabs [data-baseweb="tab-list"] button:hover { background-color: #2b2b2b; border-color: #39ff14; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #0e0e0e; border-bottom: 3px solid #39ff14; color: #39ff14;
        font-weight: 600; box-shadow: 0 -2px 10px rgba(57, 255, 20, 0.2);
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p { color: #39ff14; font-weight: 600; }
    .css-1d391kg { /* Sidebar */
        background: linear-gradient(180deg, #050505, #101010);
        border-right: 1px solid #39ff14; box-shadow: 5px 0 15px rgba(0,0,0,0.7);
    }
    .css-1d391kg .stTextInput > div > div > input {
        background-color: #1a1a1a; color: #39ff14; border: 1px solid #39ff14;
        border-radius: 8px; padding: 10px;
    }
    .css-1d391kg .stSelectbox > label, .css-1d391kg .stCheckbox > label { color: #39ff14; font-weight: 600; }
    .sentiment-positive { background-color: #39ff14; color: #0e0e0e; }
    .sentiment-negative { background-color: #c0392b; color: #fff; }
    .sentiment-neutral { background-color: #888; color: #fff; }
    [data-testid="stAppViewContainer"] > .main > div { background-color: transparent !important; }
    [data-testid="stAppViewBlockContainer"] { background-color: transparent !important; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def _render_metric_box(content):
    st.markdown(f"<div class='metric-box'>{content}</div>", unsafe_allow_html=True)

def _render_tag(tag_text, sentiment_class=""):
    return f"<div class='tag {sentiment_class}'>{tag_text}</div>"

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/44/Stock_chart_icon.png", width=100)
    st.title("StockSeer.AI")
    st.markdown("### 📊 Analyze | 💹 Predict | 📈 Grow")
    st.markdown("---")
    ticker = st.text_input("🔍 Enter Stock Ticker", "AAPL").upper()
    st.markdown("---")
    compare_ticker = st.text_input("🤝 Compare with Ticker (Optional)", "").upper()
    st.markdown("---")
    period_options = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","3Y": "3y", "5Y":"5y","Max":"max"}
    selected_label = st.sidebar.selectbox("Select Chart Period", list(period_options.keys()), index=3)
    selected_period = period_options[selected_label]

# --- UTILITY FUNCTIONS ---
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_symbol, period='1y', interval='1d'):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period=period, interval=interval)
    df.dropna(inplace=True)
    df.attrs['ticker_symbol'] = ticker_symbol
    return df

@st.cache_data(ttl=3600)
def add_technical_indicators(df):
    if df.empty or 'Close' not in df.columns: return pd.DataFrame()
    df_ta = df.copy()
    df_ta['SMA_20'] = ta.trend.sma_indicator(df_ta['Close'], window=20)
    df_ta['SMA_50'] = ta.trend.sma_indicator(df_ta['Close'], window=50)
    df_ta['RSI'] = ta.momentum.rsi(df_ta['Close'], window=14)
    df_ta['MACD_line'] = ta.trend.macd(df_ta['Close'])
    df_ta['MACD_signal'] = ta.trend.macd_signal(df_ta['Close'])
    df_ta['MACD_hist'] = ta.trend.macd_diff(df_ta['Close'])
    bb_indicator = ta.volatility.BollingerBands(close=df_ta['Close'], window=20, window_dev=2)
    df_ta['BB_High'] = bb_indicator.bollinger_hband()
    df_ta['BB_Mid'] = bb_indicator.bollinger_mavg()
    df_ta['BB_Low'] = bb_indicator.bollinger_lband()
    return df_ta

@st.cache_data(ttl=3600)
def generate_signal(df):
    if df.empty or not all(k in df.columns for k in ['RSI', 'MACD_hist', 'SMA_20', 'Close', 'MACD_line', 'MACD_signal']) or len(df) < 20:
        return "N/A", "Insufficient data for detailed signal generation (need ~20 days)."
    latest = df.iloc[-1]
    if len(df) < 2: return "N/A", "Not enough data for crossover signal logic."
    previous = df.iloc[-2] 
    rsi_val = latest['RSI']; macd_hist_val = latest['MACD_hist']
    try:
        macd_line_val = latest['MACD_line']; macd_signal_line_val = latest['MACD_signal']
        prev_macd_line = previous['MACD_line']; prev_macd_signal_line = previous['MACD_signal']
    except KeyError: return "N/A", "MACD components missing in DataFrame."
    except Exception as e: return "N/A", f"Error accessing MACD lines: {e}"
    close_price = latest['Close']; sma20 = latest['SMA_20']
    reasons = []; buy_score = 0; sell_score = 0
    if pd.isna(rsi_val) or pd.isna(macd_hist_val) or pd.isna(macd_line_val) or pd.isna(macd_signal_line_val) or pd.isna(sma20):
        return "N/A", "Indicator data has NaNs."
    if rsi_val < 30: reasons.append(f"RSI ({rsi_val:.2f}) is in oversold territory (<30)."); buy_score += 2
    elif rsi_val < 40: reasons.append(f"RSI ({rsi_val:.2f}) nears oversold."); buy_score += 1
    elif rsi_val > 70: reasons.append(f"RSI ({rsi_val:.2f}) is overbought (>70)."); sell_score += 2
    elif rsi_val > 60: reasons.append(f"RSI ({rsi_val:.2f}) nears overbought."); sell_score += 1
    else: reasons.append(f"RSI ({rsi_val:.2f}) is neutral.")
    if macd_line_val > macd_signal_line_val and prev_macd_line <= prev_macd_signal_line: reasons.append("MACD Bullish Crossover."); buy_score += 2
    elif macd_line_val < macd_signal_line_val and prev_macd_line >= prev_macd_signal_line: reasons.append("MACD Bearish Crossover."); sell_score += 2
    elif macd_line_val > macd_signal_line_val: reasons.append("MACD Line > Signal Line (Bullish)."); buy_score +=1
    elif macd_line_val < macd_signal_line_val: reasons.append("MACD Line < Signal Line (Bearish)."); sell_score +=1
    if macd_hist_val > 0: reasons.append(f"MACD Hist ({macd_hist_val:.2f}) positive (Bullish momentum)."); buy_score += 0.5
    elif macd_hist_val < 0: reasons.append(f"MACD Hist ({macd_hist_val:.2f}) negative (Bearish momentum)."); sell_score += 0.5
    if close_price > sma20: reasons.append(f"Price > SMA20 (Short-term uptrend)."); buy_score += 1
    elif close_price < sma20: reasons.append(f"Price < SMA20 (Short-term downtrend)."); sell_score += 1
    final_signal = "HOLD"
    if buy_score > sell_score + 1: final_signal = "BUY"
    elif sell_score > buy_score + 1: final_signal = "SELL"
    elif buy_score > sell_score + 2.5: final_signal = "STRONG BUY"
    elif sell_score > buy_score + 2.5: final_signal = "STRONG SELL"
    return final_signal, " ".join(reasons) if reasons else "Neutral signals."

@st.cache_data(ttl=3600)
def get_about_stock_info(ticker_symbol): # Removed Ownership related DFs from return
    description = "Info not available."
    sector, industry = "N/A", "N/A"
    market_cap, exchange = None, "N/A"
    info_dict = {}
    financials_df, earnings_df = pd.DataFrame(), pd.DataFrame()
    analyst_recs_df, analyst_price_target_dict = None, None 
    company_officers_list = []
    try:
        stock = yf.Ticker(ticker_symbol); info = stock.info
        if not info: 
            pass # Defaults are already set
        else:
            description = info.get('longBusinessSummary', "No summary available.")
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            market_cap = info.get('marketCap')
            exchange = info.get('exchange', 'N/A')
            info_dict = info 
            company_officers_list = info.get('companyOfficers', [])
        try:
            financials_df = stock.quarterly_financials if not stock.quarterly_financials.empty else stock.financials
        except Exception: pass 
        try:
            earnings_df = stock.quarterly_earnings if not stock.quarterly_earnings.empty else stock.earnings
        except Exception: pass
        try: analyst_recs_df = stock.recommendations
        except: pass
        try: analyst_price_target_dict = stock.analyst_price_target
        except: pass
                 
        return (description, sector, industry, market_cap, exchange, info_dict,
                financials_df, earnings_df, analyst_recs_df, analyst_price_target_dict,
                company_officers_list) # Return 11 items
    except Exception as e:
        return (f"Info retrieval failed: {e}", "N/A", "N/A", None, "N/A", {}, 
                pd.DataFrame(), pd.DataFrame(), None, None, [])

@st.cache_data(ttl=1800)
def get_stock_news_yfinance(ticker_symbol):
    news_items = []
    error_message = None
    try:
        stock_ticker_obj = yf.Ticker(ticker_symbol)
        fetched_news = stock_ticker_obj.news
        if not fetched_news:
            error_message = f"No news found for {ticker_symbol} via yfinance."
        else:
            for item in fetched_news:
                publish_time = item.get('providerPublishTime')
                publish_time_readable = "N/A"
                if publish_time:
                    try: publish_time_readable = pd.to_datetime(publish_time, unit='s').strftime('%Y-%m-%d %H:%M')
                    except ValueError: 
                        try: publish_time_readable = pd.to_datetime(publish_time).strftime('%Y-%m-%d %H:%M')
                        except: publish_time_readable = str(publish_time) if publish_time else "N/A"
                news_items.append({
                    'title': item.get('title', 'No Title Available'),
                    'link': item.get('link', '#'),
                    'published': publish_time_readable,
                    'publisher': item.get('publisher', 'N/A') })
            news_items = news_items[:10]
    except Exception as e: error_message = f"Failed to fetch news for {ticker_symbol} via yfinance. Error: {str(e)}"
    return news_items, error_message

@st.cache_data(ttl=3600)
def assess_volatility_and_risk(df, window=60):
    if df.empty or 'Close' not in df.columns or len(df) < window + 1: return None, "N/A", "Not enough data for volatility."
    daily_returns = df['Close'].pct_change().dropna()
    if len(daily_returns) < window: return None, "N/A", f"Not enough returns (need {window}, have {len(daily_returns)})."
    actual_window = min(window, len(daily_returns))
    if actual_window < 2: return None, "N/A", "Too few points for std dev."
    rolling_std_dev = daily_returns.rolling(window=actual_window).std().iloc[-1]
    annualized_volatility = rolling_std_dev * np.sqrt(252)
    if pd.isna(annualized_volatility): return None, "N/A", "Volatility NaN."
    vol_percent = annualized_volatility * 100
    risk_level, risk_explanation = "N/A", "Volatility helps understand price swings."
    if vol_percent < 15: risk_level, risk_explanation = "Low", "Relatively low price swings. Lower risk."
    elif vol_percent < 30: risk_level, risk_explanation = "Moderate", "Moderate price swings. Balanced risk/return."
    elif vol_percent < 50: risk_level, risk_explanation = "High", "Significant price swings. Higher risk."
    else: risk_level, risk_explanation = "Very High", "Extreme price swings. Very high risk."
    return vol_percent, risk_level, risk_explanation

@st.cache_data(ttl=3600)
def get_historical_volatility_data(df, window=30, trading_days=252):
    if df.empty or 'Close' not in df.columns or len(df) < window + 1: return None
    daily_returns = df['Close'].pct_change()
    rolling_std = daily_returns.rolling(window=window).std()
    historical_volatility = rolling_std * np.sqrt(trading_days) * 100
    return historical_volatility.dropna()

@st.cache_data(ttl=3600)
def analyze_sentiment_text(text):
    if not text or not isinstance(text, str): return {"label": "NEUTRAL", "score": 0.0}
    try:
        max_len = sentiment_analyzer.tokenizer.model_max_length
        truncated_text = text[:max_len] if len(text) > max_len else text
        if not truncated_text.strip(): return {"label": "NEUTRAL", "score": 0.0}
        result = sentiment_analyzer(truncated_text)[0]
        return result
    except Exception: return {"label": "NEUTRAL", "score": 0.0}

@st.cache_data(ttl=3600)
def get_correlation_data(ticker1_df, ticker2_symbol, main_ticker_symbol, period='1y', interval='1d'):
    try:
        ticker2_df = fetch_stock_data(ticker2_symbol, period=period, interval=interval)
        if ticker1_df.empty or ticker2_df.empty:
            return None, None, "Not enough data for one or both tickers for correlation."
        returns1 = ticker1_df['Close'].pct_change().rename(main_ticker_symbol)
        returns2 = ticker2_df['Close'].pct_change().rename(ticker2_symbol)
        combined_returns = pd.concat([returns1, returns2], axis=1).dropna()
        if len(combined_returns) < 20:
            return None, None, "Not enough overlapping data points for meaningful correlation."
        rolling_corr = combined_returns[main_ticker_symbol].rolling(window=30).corr(combined_returns[returns2.name])
        overall_corr = combined_returns[main_ticker_symbol].corr(combined_returns[returns2.name])
        return rolling_corr.dropna(), overall_corr, None
    except Exception as e:
        return None, None, f"Error calculating correlation with {ticker2_symbol}: {e}"

@st.cache_data(ttl=3600)
def calculate_historical_performance_and_cagr(df_hist, initial_investment=1000):
    if df_hist.empty or len(df_hist) < 2 or 'Close' not in df_hist.columns:
        return None, None, None, "Not enough historical data (need at least 2 data points)."
    start_price = df_hist['Close'].iloc[0]
    end_price = df_hist['Close'].iloc[-1]
    num_days = (df_hist.index[-1] - df_hist.index[0]).days
    num_years = num_days / 365.25
    if num_years < 0.1 : 
        current_value = initial_investment * (end_price/start_price) if start_price != 0 else initial_investment
        return initial_investment, current_value, None, "Data period too short for meaningful CAGR."
    total_return_multiple = end_price / start_price if start_price != 0 else 1
    final_value = initial_investment * total_return_multiple
    cagr = None
    if total_return_multiple > 0 and num_years > 0:
        cagr = ((total_return_multiple) ** (1/num_years)) - 1
    return initial_investment, final_value, cagr * 100 if cagr is not None else None, None

def project_future_value_cagr(initial_investment, cagr_percent, years_to_project):
    if cagr_percent is None or initial_investment is None or years_to_project is None:
        return None
    cagr_decimal = cagr_percent / 100
    future_value = initial_investment * ((1 + cagr_decimal) ** years_to_project)
    return future_value
    
# --- Chatbot Session State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_ticker_for_chat' not in st.session_state:
    st.session_state.current_ticker_for_chat = ""

# --- Chatbot Processing Function ---
def get_chatbot_response(user_query, stock_data_bundle_local, current_ticker_symbol):
    query = user_query.lower().strip()
    response = "I'm not sure how to answer that. You can ask about: current price, P/E ratio, RSI, MACD, latest news headlines (type 'news'), 52-week high/low, market cap, sector, or industry."
    
    s_info_chat = stock_data_bundle_local.get('s_info_full', {})
    df_ta_chat = stock_data_bundle_local.get('df_ta')
    current_price_chat = stock_data_bundle_local.get('current_price')
    news_items_chat = stock_data_bundle_local.get('news_items', []) 
    ticker_name_chat = s_info_chat.get('shortName', current_ticker_symbol)

    if not s_info_chat and (df_ta_chat is None or df_ta_chat.empty): 
        return f"Please ensure a stock ticker is loaded in the sidebar to ask questions about it."

    if "price" in query or "what is the current price" in query: 
        response = f"The current price of {ticker_name_chat} is ${current_price_chat:.2f}." if current_price_chat else "Current price data not available."
    elif "p/e" in query or "pe ratio" in query: 
        pe = s_info_chat.get('trailingPE')
        response = f"The P/E ratio (trailing) for {ticker_name_chat} is {pe:.2f}." if pe else "P/E ratio not available."
    elif "rsi" in query: 
        rsi_val = df_ta_chat['RSI'].iloc[-1] if df_ta_chat is not None and not df_ta_chat.empty and 'RSI' in df_ta_chat.columns else None
        response = f"The latest RSI for {ticker_name_chat} is {rsi_val:.2f}." if pd.notna(rsi_val) else "RSI data not available or not calculated yet."
    elif "macd" in query: 
        macd_val = df_ta_chat['MACD_hist'].iloc[-1] if df_ta_chat is not None and not df_ta_chat.empty and 'MACD_hist' in df_ta_chat.columns else None
        response = f"The latest MACD Histogram value for {ticker_name_chat} is {macd_val:.2f}." if pd.notna(macd_val) else "MACD data not available or not calculated yet."
    elif "52 week high" in query: 
        high = s_info_chat.get('fiftyTwoWeekHigh')
        response = f"The 52-week high for {ticker_name_chat} is ${high:.2f}." if high else "52-week high not available."
    elif "52 week low" in query: 
        low = s_info_chat.get('fiftyTwoWeekLow')
        response = f"The 52-week low for {ticker_name_chat} is ${low:.2f}." if low else "52-week low not available."
    elif "news" in query or "latest news" in query:
        if news_items_chat: 
            response = f"Here are up to 3 recent news headlines for {ticker_name_chat}:\n"
            for i, item in enumerate(news_items_chat[:3]): 
                response += f"\n{i+1}. {item.get('title','N/A')} (Source: {item.get('publisher','N/A')})"
        else:
            response = f"No recent news available for {ticker_name_chat} at the moment."
    elif "market cap" in query: 
        mcap = s_info_chat.get('marketCap')
        response = f"The market capitalization for {ticker_name_chat} is ${mcap:,.0f}." if mcap else "Market Cap N/A."
    elif "sector" in query: 
        sec_chat = s_info_chat.get('sector')
        response = f"{ticker_name_chat} is in the {sec_chat} sector." if sec_chat and sec_chat != 'N/A' else "Sector data not available."
    elif "industry" in query: 
        ind_chat = s_info_chat.get('industry')
        response = f"{ticker_name_chat} is in the {ind_chat} industry." if ind_chat and ind_chat != 'N/A' else "Industry data not available."
    elif "hello" in query or "hi" in query or "hey" in query:
        response = f"Hello! I am StockSeer, your AI assistant. How can I help you with {ticker_name_chat} today?"
    elif "thank you" in query or "thanks" in query:
        response = "You're welcome! Is there anything else I can help you with?"
    elif "bye" in query or "goodbye" in query:
        response = "Goodbye! Feel free to ask if you need more insights."
    
    return response

# --- MAIN APP LOGIC ---
if ticker:
    try:
        stock_info_main = yf.Ticker(ticker).info
        if not stock_info_main or stock_info_main.get('regularMarketPrice') is None:
            st.error(f"Essential data for **{ticker}** unavailable. Check ticker."); st.stop()
    except Exception as e: st.error(f"Error fetching initial data for {ticker}: {e}"); st.stop()

    current_price = stock_info_main.get('regularMarketPrice', stock_info_main.get('currentPrice'))
    previous_close = stock_info_main.get('regularMarketPreviousClose', stock_info_main.get('previousClose'))
    fifty_two_week_high = stock_info_main.get('fiftyTwoWeekHigh')
    fifty_two_week_low = stock_info_main.get('fiftyTwoWeekLow')
    volume_today = stock_info_main.get('regularMarketVolume', stock_info_main.get('volume'))
    today_change, today_change_percent = (None, None)
    if current_price and previous_close and previous_close != 0:
        today_change = current_price - previous_close
        today_change_percent = (today_change / previous_close) * 100

    with st.spinner(f"Fetching all data for {ticker}..."):
        df = fetch_stock_data(ticker, selected_period)
        if df.empty: st.error(f"No historical data for {ticker} ({selected_label})."); st.stop()
        
        df_ta = add_technical_indicators(df.copy())
        signal, signal_reason = generate_signal(df_ta)
        
        # Corrected unpacking to match the simplified get_about_stock_info (11 items)
        (about_info, sector, industry, mcap_val, exch_val,
         s_info_full, fin_df, earn_df, 
         analyst_recs, analyst_price_target_data, 
         company_officers) = get_about_stock_info(ticker)
        
        news_items, news_error_message = get_stock_news_yfinance(ticker)
        
        volatility_percent, risk_level, risk_explanation_text = assess_volatility_and_risk(df.copy(), window=60)
        hist_vol_series = get_historical_volatility_data(df.copy(), window=30)

        df_5y_for_calc = df 
        if selected_period not in ["5y", "max"]:
            if not df.empty and (df.index[-1] - df.index[0]).days / 365.25 < 4.9:
                try:
                    df_5y_for_calc_temp = fetch_stock_data(ticker, period="5y")
                    if not df_5y_for_calc_temp.empty: df_5y_for_calc = df_5y_for_calc_temp
                except: pass 
        hist_initial_investment, hist_final_value, hist_cagr, hist_perf_error = calculate_historical_performance_and_cagr(df_5y_for_calc)

        if not s_info_full and stock_info_main: # Fallback logic
            s_info_full = stock_info_main
            if "failed" in str(about_info).lower() or "not available" in str(about_info).lower() or "no summary" in str(about_info).lower():
                about_info = stock_info_main.get('longBusinessSummary', "No summary available.")
            if sector=='N/A': sector = stock_info_main.get('sector','N/A')
            if industry=='N/A': industry = stock_info_main.get('industry','N/A')
            if mcap_val is None : mcap_val = stock_info_main.get('marketCap')
            if exch_val=='N/A' : exch_val = stock_info_main.get('exchange','N/A')
            company_officers = stock_info_main.get('companyOfficers', [])


    tab_titles = ["📈 Chart", "📊 Fundamentals", "💡 Insights", "📚 About", "🔮 Performance", "💬 Chat", "🧠 AI, Risk & News"] # Ownership tab removed, News added to AI tab
    if compare_ticker and ticker != compare_ticker: tab_titles.insert(1, "🆚 Comparison")
    tabs = st.tabs(tab_titles)

    # --- TAB 0: STOCK CHART (Reverted to simpler overlay style) ---
    with tabs[0]:
        st.markdown(f"### {s_info_full.get('shortName', ticker)} - Live Dashboard")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Price",f"${current_price:.2f}" if current_price else "N/A", f"{today_change:.2f} ({today_change_percent:.2f}%)" if today_change_percent is not None else None)
        c2.metric("52W High",f"${fifty_two_week_high:.2f}" if fifty_two_week_high else "N/A")
        c3.metric("52W Low",f"${fifty_two_week_low:.2f}" if fifty_two_week_low else "N/A")
        c4.metric("Volume",f"{volume_today:,.0f}" if volume_today else "N/A")
        st.markdown("---"); st.markdown("### 📊 Price Chart & Technicals")
        
        co_s,co_r,co_m, co_bb_col = st.columns(4) 
        show_sma = co_s.checkbox("SMA 20",True,key='sma_chart_cb_reverted')
        show_rsi = co_r.checkbox("RSI",False,key='rsi_chart_cb_reverted')
        show_macd = co_m.checkbox("MACD",False,key='macd_chart_cb_reverted')
        show_bbands = co_bb_col.checkbox("Bollinger Bands", False, key='bb_chart_cb_reverted')
        show_earnings_dates_cb = st.checkbox("Show Earnings Dates", True, key='earnings_cb_reverted_chart')

        fig=go.Figure() 

        fig.add_trace(go.Candlestick(x=df_ta.index,open=df_ta['Open'],high=df_ta['High'],low=df_ta['Low'],close=df_ta['Close'],name='Price',increasing_line_color='#39ff14',decreasing_line_color='#c0392b'))
        
        if show_sma and 'SMA_20' in df_ta.columns and not df_ta['SMA_20'].isnull().all():
            fig.add_trace(go.Scatter(x=df_ta.index,y=df_ta['SMA_20'],name='SMA 20',line=dict(color='#87CEEB',dash='dash')))
        
        if show_bbands and all(col in df_ta.columns for col in ['BB_High', 'BB_Low', 'BB_Mid']) and \
           not df_ta['BB_High'].isnull().all() and not df_ta['BB_Low'].isnull().all():
            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_High'], line=dict(color='rgba(152,251,152,0.3)', width=1), name='BB High', legendgroup='bollinger', showlegend=False))
            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Low'], line=dict(color='rgba(152,251,152,0.3)', width=1), name='BB Low', fill='tonexty', fillcolor='rgba(152,251,152,0.1)', legendgroup='bollinger', showlegend=False))
            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Mid'], line=dict(color='rgba(240,230,140,0.7)', width=1.5, dash='dashdot'), name='BB Mid (20)', legendgroup='bollinger'))
        
        if show_rsi and 'RSI' in df_ta.columns and not df_ta['RSI'].isnull().all():
            fig.add_trace(go.Scatter(x=df_ta.index,y=df_ta['RSI'],name='RSI',line=dict(color='#FFA500'), yaxis="y2"))
        
        if show_macd and all(k in df_ta for k in ['MACD_line','MACD_signal','MACD_hist']):
            fig.add_trace(go.Scatter(x=df_ta.index,y=df_ta['MACD_line'],name='MACD Line',line=dict(color='#DA70D6'), yaxis="y3"))
            fig.add_trace(go.Scatter(x=df_ta.index,y=df_ta['MACD_signal'],name='Signal Line',line=dict(color='#FFD700',dash='dot'), yaxis="y3"))
            fig.add_trace(go.Bar(x=df_ta.index,y=df_ta['MACD_hist'],name='MACD Hist.',marker_color=np.where(df_ta['MACD_hist']>0,'#39ff14','#c0392b'), yaxis="y3", opacity=0.6))
            
        if show_earnings_dates_cb:
            try:
                earnings_data = yf.Ticker(ticker).earnings_dates 
                if earnings_data is not None and not earnings_data.empty:
                    min_c_date, max_c_date = df_ta.index.min(), df_ta.index.max()
                    if not isinstance(earnings_data.index, pd.DatetimeIndex): earnings_data.index = pd.to_datetime(earnings_data.index, errors='coerce').dropna()
                    relevant_e_dates = earnings_data[(earnings_data.index >= min_c_date) & (earnings_data.index <= max_c_date)]
                    for date_val in relevant_e_dates.index:
                        fig.add_vline(x=date_val, line_width=1, line_dash="longdash", line_color="rgba(200,200,200,0.6)", annotation_text="E", annotation_position="bottom right", annotation_font_size=10, annotation_font_color="rgba(200,200,200,0.9)")
            except Exception as e: st.sidebar.caption(f"Earnings dates error: {e}")

        if current_price and not df_ta.empty: fig.add_annotation(x=df_ta.index[-1],y=current_price,text=f"Current: ${current_price:.2f}",showarrow=True,arrowhead=2,ax=0,ay=-40,font=dict(color="#FFF",size=12,family="Poppins"),bgcolor="rgba(57,255,20,0.7)",bordercolor="#0a0a0a",borderwidth=1,borderpad=4,opacity=0.9)
        
        fig.update_layout(
            height=650, template='plotly_dark', plot_bgcolor='#0e0e0e', paper_bgcolor='#0e0e0e',
            hovermode='x unified',
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
            xaxis_title='Date', yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            xaxis=dict(rangeselector=dict(buttons=[dict(count=1,label="1m",step="month",stepmode="backward"),dict(count=3,label="3m",step="month",stepmode="backward"),dict(count=6,label="6m",step="month",stepmode="backward"),dict(count=1,label="YTD",step="year",stepmode="todate"),dict(count=1,label="1y",step="year",stepmode="backward"),dict(step="all")],font=dict(color="#39ff14",size=10),bgcolor="#1a1a1a",bordercolor="#39ff14",activecolor="#2b2b2b")),
            yaxis2=dict(title=dict(text='RSI', font=dict(size=10)), overlaying='y', side='right', showgrid=False, range=[0,100], visible=show_rsi, position=0.97, tickfont=dict(size=8)),
            yaxis3=dict(title=dict(text='MACD', font=dict(size=10)), overlaying='y', side='right', showgrid=False, visible=show_macd, position=0.90 if show_rsi else 0.97, tickfont=dict(size=8)),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        if show_rsi: fig.add_hline(y=30,line_dash="dash",line_color="green",opacity=0.5,yref="y2",layer="below"); fig.add_hline(y=70,line_dash="dash",line_color="red",opacity=0.5,yref="y2",layer="below")
        if show_macd: fig.add_hline(y=0,line_dash="dash",line_color="#888",opacity=0.5,yref="y3",layer="below")

        st.plotly_chart(fig,use_container_width=True)
        st.markdown("#### Last 10 Days Data"); 
        csv_export = df.to_csv(index=True).encode('utf-8')
        st.download_button(label="📥 Download Full Historical Data (CSV)", data=csv_export, file_name=f'{ticker}_historical_data_{selected_label.replace(" ","_")}.csv', mime='text/csv', key='download_hist_csv_reverted')
        cols_to_show_in_table = ['Open','High','Low','Close','Volume','SMA_20','RSI','MACD_hist']
        if 'BB_High' in df_ta: cols_to_show_in_table.extend(['BB_High', 'BB_Mid', 'BB_Low'])
        st.dataframe(df_ta[cols_to_show_in_table].tail(10).style.format(precision=2),use_container_width=True)


    # --- DYNAMIC TAB INDEXING & CONTENT (Rest of the tabs) ---
    cti = 1 
    if compare_ticker and ticker != compare_ticker:
        with tabs[cti]: # Comparison Tab
            st.markdown(f"### 🆚 Comparing {ticker} with {compare_ticker}")
            with st.spinner(f"Fetching data for {compare_ticker}..."):
                df_c = fetch_stock_data(compare_ticker,selected_period)
                # Adjusted unpacking
                c_about_info, s_c, i_c, _, _, si_c, _, _, _, _, _ = get_about_stock_info(compare_ticker) 
            if df_c.empty: st.error(f"No data for **{compare_ticker}**.")
            else:
                st.markdown("#### 📈 Price Comparison (Normalized)")
                if df.empty or df_c.empty or df['Close'].iloc[0]==0 or df_c['Close'].iloc[0]==0: st.warning("Cannot normalize prices.")
                else:
                    df_n=(df['Close']/df['Close'].iloc[0]*100);df_cn=(df_c['Close']/df_c['Close'].iloc[0]*100)
                    comp_chart_df=pd.concat([df_n.rename(ticker),df_cn.rename(compare_ticker)],axis=1).dropna()
                    if comp_chart_df.empty: st.warning("No overlapping data for comparison.")
                    else:
                        fig_c=go.Figure();fig_c.add_trace(go.Scatter(x=comp_chart_df.index,y=comp_chart_df[ticker],name=ticker,line=dict(color='#39ff14')));fig_c.add_trace(go.Scatter(x=comp_chart_df.index,y=comp_chart_df[compare_ticker],name=compare_ticker,line=dict(color='#00BFFF')))
                        fig_c.update_layout(title=f'{ticker} vs {compare_ticker} Price ({selected_label})',template='plotly_dark',plot_bgcolor='#0e0e0e',paper_bgcolor='#0e0e0e',height=500,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1));st.plotly_chart(fig_c,use_container_width=True)
                st.markdown("---");st.markdown("#### 📊 Key Metrics Side-by-Side")
                col_mc,col_cc=st.columns(2)
                def fmt_s_comparison(v,t):
                    if v is None or pd.isna(v) or not isinstance(v,(int,float,np.number)):return "N/A"
                    if t=="b": return f"${v/1e9:.2f}B"
                    elif t=="%": return f"{v*100:.2f}%"
                    elif t=="r": return f"{v:.2f}"
                    elif t=="$": return f"${v:.2f}"
                    return str(v)
                with col_mc:st.markdown(f"##### {s_info_full.get('shortName',ticker)}");_render_metric_box(f"""<p><b>MCap:</b>{fmt_s_comparison(s_info_full.get("marketCap"),"b")}</p><p><b>P/E:</b>{fmt_s_comparison(s_info_full.get("trailingPE"),"r")}</p><p><b>EPS:</b>{fmt_s_comparison(s_info_full.get("trailingEps"),"$")}</p><p><b>ROE:</b>{fmt_s_comparison(s_info_full.get("returnOnEquity"),"%")}</p><p><b>Sec:</b>{sector or 'N/A'}</p>""")
                with col_cc:st.markdown(f"##### {(si_c.get('shortName',compare_ticker) if si_c else compare_ticker)}");
                if si_c:_render_metric_box(f"""<p><b>MCap:</b>{fmt_s_comparison(si_c.get("marketCap"),"b")}</p><p><b>P/E:</b>{fmt_s_comparison(si_c.get("trailingPE"),"r")}</p><p><b>EPS:</b>{fmt_s_comparison(si_c.get("trailingEps"),"$")}</p><p><b>ROE:</b>{fmt_s_comparison(si_c.get("returnOnEquity"),"%")}</p><p><b>Sec:</b>{s_c or 'N/A'}</p>""")
                else:st.warning(f"Fundamentals for {compare_ticker} unavailable.")
        cti+=1

    with tabs[cti]: # Fundamentals Tab
        st.markdown(f"### 📊 Key Fundamentals for {s_info_full.get('shortName', ticker)}")
        def fmt_f_fundamentals(v,t):
            if v is None or pd.isna(v) or not isinstance(v,(int,float,np.number)): return "N/A"
            if t=="b": return f"${v/1e9:.2f}B"
            elif t=="m": return f"${v/1e6:.2f}M"
            elif t=="%": return f"{v*100:.2f}%"
            elif t=="r": return f"{v:.2f}"
            elif t=="$": return f"${v:.2f}"
            elif t=="i": return f"{v:,.0f}"
            return str(v)
        info_fund = s_info_full if s_info_full else stock_info_main
        if not info_fund or not isinstance(info_fund, dict): 
            st.error(f"Fundamental data for **{ticker}** is currently unavailable or in an unexpected format.")
        else:
            try:
                st.markdown("#### 💰 Valuation & Earnings")
                cf1,cf2,cf3=st.columns(3);cf1.metric("Market Cap",fmt_f_fundamentals(info_fund.get("marketCap"),"b"));cf2.metric("Ent. Value",fmt_f_fundamentals(info_fund.get("enterpriseValue"),"b"));cf3.metric("P/E (Trail)",fmt_f_fundamentals(info_fund.get("trailingPE"),"r"))
                cf4,cf5,cf6=st.columns(3);cf4.metric("P/E (Fwd)",fmt_f_fundamentals(info_fund.get("forwardPE"),"r"));cf5.metric("EPS (Trail)",fmt_f_fundamentals(info_fund.get("trailingEps"),"$"));cf6.metric("EPS (Fwd)",fmt_f_fundamentals(info_fund.get("forwardEps"),"$"))
                with st.expander("Learn about Valuation & Earnings Metrics"): st.markdown("- **Market Cap:** Total market value...\n- **Enterprise Value (EV):** Company's total value...\n- **P/E Ratio:** Share price relative to earnings...\n- **EPS:** Company's profit per share...")

                st.markdown("---");st.markdown("#### 📈 Profitability & Margins")
                cf7,cf8,cf9=st.columns(3);cf7.metric("ROE",fmt_f_fundamentals(info_fund.get("returnOnEquity"),"%"));cf8.metric("ROA",fmt_f_fundamentals(info_fund.get("returnOnAssets"),"%"));cf9.metric("Profit Margin",fmt_f_fundamentals(info_fund.get("profitMargins"),"%"))
                cf10,cf11,cf12=st.columns(3);cf10.metric("Gross Margin",fmt_f_fundamentals(info_fund.get("grossMargins"),"%"));cf11.metric("Oper. Margin",fmt_f_fundamentals(info_fund.get("operatingMargins"),"%"));cf12.metric("Beta",fmt_f_fundamentals(info_fund.get("beta"),"r"))
                with st.expander("Learn about Profitability & Margins"): st.markdown("- **ROE:** Profitability vs. equity...\n- **ROA:** Profitability vs. assets...\n- **Profit/Gross/Oper. Margin:** Efficiency levels...\n- **Beta:** Volatility vs. market...")
                
                st.markdown("---");st.markdown("#### 💧 Liquidity & Financial Health")
                cf13,cf14,cf15=st.columns(3);cf13.metric("Debt/Equity",fmt_f_fundamentals(info_fund.get("debtToEquity"),"r"));cf14.metric("Current Ratio",fmt_f_fundamentals(info_fund.get("currentRatio"),"r"));cf15.metric("Quick Ratio",fmt_f_fundamentals(info_fund.get("quickRatio"),"r"))
                with st.expander("Learn about Liquidity & Financial Health"): st.markdown("- **Debt/Equity:** Financial leverage...\n- **Current Ratio:** Short-term obligations...\n- **Quick Ratio:** Stricter short-term liquidity...")

                st.markdown("---");st.markdown("#### 💵 Dividends & Performance Averages")
                cf16,cf17,cf18=st.columns(3);cf16.metric("Div. Yield",fmt_f_fundamentals(info_fund.get("dividendYield"),"%"));cf17.metric("Payout Ratio",fmt_f_fundamentals(info_fund.get("payoutRatio"),"%"));cf18.metric("50-Day Avg",fmt_f_fundamentals(info_fund.get("fiftyDayAverage"),"$"))
                st.metric("200-Day Avg Price",fmt_f_fundamentals(info_fund.get("twoHundredDayAverage"),"$"))
                with st.expander("Learn about Dividends & Averages"): st.markdown("- **Dividend Yield:** Dividend relative to price...\n- **Payout Ratio:** Earnings paid as dividends...\n- **50/200-Day Avg:** Trend indicators...")

                st.markdown("---");st.markdown("#### 📉 Financial Statements Visualizations")
                if fin_df is not None and not fin_df.empty:
                    st.markdown("##### Quarterly Financials Overview")
                    fin_p=fin_df.T.sort_index(ascending=True)
                    try:fin_p.index=pd.to_datetime(fin_p.index).strftime('%Y-%m-%d')
                    except:fin_p.index=fin_p.index.astype(str)
                    if 'Total Revenue' in fin_p.columns and not fin_p['Total Revenue'].isnull().all():fig_r=go.Figure(go.Bar(x=fin_p.index,y=fin_p['Total Revenue']/1e6,marker_color='#39ff14',name='Revenue'));fig_r.update_layout(title='Quarterly Revenue (M)',template='plotly_dark',plot_bgcolor='#0e0e0e',paper_bgcolor='#0e0e0e',yaxis_title='Amount(M)');st.plotly_chart(fig_r,use_container_width=True)
                    else: st.info("Total Revenue data not available for plotting or contains all NaNs.")
                    if 'Net Income' in fin_p.columns and not fin_p['Net Income'].isnull().all():fig_n=go.Figure(go.Bar(x=fin_p.index,y=fin_p['Net Income']/1e6,marker_color='#87CEEB',name='Net Income'));fig_n.update_layout(title='Quarterly Net Income (M)',template='plotly_dark',plot_bgcolor='#0e0e0e',paper_bgcolor='#0e0e0e',yaxis_title='Amount(M)');st.plotly_chart(fig_n,use_container_width=True)
                    else: st.info("Net Income data not available for plotting or contains all NaNs.")
                else:st.info("Quarterly financials unavailable.")
                if earn_df is not None and not earn_df.empty:
                    st.markdown("##### Quarterly EPS")
                    earn_p=earn_df.T.sort_index(ascending=True);earn_p.index=earn_p.index.astype(str)
                    eps_c='EPS' if 'EPS' in earn_p.columns else 'Diluted EPS' if 'Diluted EPS' in earn_p.columns else 'Earnings' if 'Earnings' in earn_p.columns else None
                    if eps_c and not earn_p[eps_c].isnull().all():fig_e=go.Figure(go.Bar(x=earn_p.index,y=earn_p[eps_c],marker_color='#FFA500',name=eps_c));fig_e.update_layout(title=f'Quarterly {eps_c.replace("Earnings","Earnings")}',template='plotly_dark',plot_bgcolor='#0e0e0e',paper_bgcolor='#0e0e0e',yaxis_title=f'{eps_c}($)' );st.plotly_chart(fig_e,use_container_width=True)
                    else:st.info("EPS data column not found or all NaN.")
                else:st.info("Quarterly earnings unavailable.")
                st.markdown("---");_render_metric_box("<p><b>Fundamental Analysis:</b> ... <i>Always verify with official filings.</i></p>")
            except Exception as e:st.error(f"Error processing fundamentals for {ticker}: {e}")
        cti+=1

    with tabs[cti]: # Insights Tab
        st.markdown("### 💡 Technical Insights & Signals")
        sig_col = '#39ff14'; sh_col = 'rgba(57,255,20,0.7)' 
        if "SELL" in signal: sig_col = '#c0392b'; sh_col = 'rgba(192,57,43,0.7)'
        elif "HOLD" in signal: sig_col = '#e0e0e0'; sh_col = 'rgba(224,224,224,0.5)'
        if "STRONG" in signal: sig_col = '#FFD700' 
        st.markdown(f"<p style='font-size:1.3rem;font-weight:bold;color:#e0e0e0;'>Signal: <span style='font-size:1.9rem;font-weight:bold;color:{sig_col};text-shadow:0 0 10px {sh_col};margin-left:10px;'>{signal}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:1rem;color:#b0b0b0;'><i>Rationale:</i></p>", unsafe_allow_html=True)
        reasons_list = signal_reason.split(". ")
        for r_item in reasons_list:
            if r_item.strip(): st.markdown(f"<p style='font-size:1rem;color:#b0b0b0; margin-left:15px;'>- {r_item.strip()}.</p>", unsafe_allow_html=True)
        st.markdown("---");st.markdown("#### 📊 Key Indicator Values")
        if not df_ta.empty and len(df_ta)>0:
            latest_ind=df_ta.iloc[-1];cols_i=st.columns(3)
            cols_i[0].metric("RSI (14)",f"{latest_ind.get('RSI',float('nan')):.2f}")
            cols_i[1].metric("MACD (Hist)",f"{latest_ind.get('MACD_hist',float('nan')):.2f}")
            cols_i[2].metric("SMA (20)",f"${latest_ind.get('SMA_20',float('nan')):.2f}")
        else:st.info("Indicator data unavailable.")
        
        st.markdown("---")
        st.markdown("#### 🔗 Correlation Analysis")
        corr_ticker_options = ["SPY", "QQQ", "GLD", "BTC-USD", "VIX"]
        corr_ticker_selected = st.selectbox("Correlate with:", corr_ticker_options, index=0, key="corr_select")
        if corr_ticker_selected:
            with st.spinner(f"Calculating correlation with {corr_ticker_selected}..."):
                rolling_corr_data, overall_corr_val, corr_err = get_correlation_data(df.copy(), corr_ticker_selected, ticker, period=selected_period)
            if corr_err:
                st.warning(corr_err)
            elif rolling_corr_data is not None and overall_corr_val is not None:
                st.metric(f"Overall Correlation with {corr_ticker_selected} ({selected_label})", f"{overall_corr_val:.2f}")
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(x=rolling_corr_data.index, y=rolling_corr_data, mode='lines', name='30-Day Rolling Correlation', line=dict(color='cyan')))
                fig_corr.update_layout(title=f'{ticker} vs. {corr_ticker_selected} - 30D Rolling Correlation',
                                       template='plotly_dark', plot_bgcolor='#0e0e0e', paper_bgcolor='#0e0e0e',
                                       height=300, yaxis_title="Correlation Coefficient", margin=dict(t=40, b=40))
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info(f"Could not calculate correlation with {corr_ticker_selected}.")

        st.markdown("---");_render_metric_box("<p><b>Strategic Highlights:</b></p><ul><li>Momentum & Trend...</li><li>Context is Crucial...</li></ul><p><i><b>Disclaimer:</b> Not financial advice.</i></p>")
        cti+=1

    with tabs[cti]: # About Tab
        st.markdown(f"### 📚 About {s_info_full.get('shortName', ticker)}")
        tags_list=[]
        if mcap_val:
            if mcap_val > 2e11:tags_list.append(_render_tag("Mega Cap"))
            elif mcap_val > 1e10:tags_list.append(_render_tag("Large Cap"))
            else:tags_list.append(_render_tag("Small/Mid Cap"))
        info_s=s_info_full if s_info_full else stock_info_main
        if info_s:
            if exch_val and exch_val!='N/A':tags_list.append(_render_tag(f"Exch: {exch_val}"))
            if sector and sector!='N/A':tags_list.append(_render_tag(f"Sector: {sector}"))
            if industry and industry!='N/A':tags_list.append(_render_tag(f"Ind: {industry}"))
            emp=info_s.get('fullTimeEmployees')
            if emp:tags_list.append(_render_tag(f"{emp//1000}k+ Empl" if emp>=1000 else f"{emp} Empl"))
            q_type=info_s.get('quoteType');
            if q_type and q_type!="EQUITY":tags_list.append(_render_tag(q_type))
        st.markdown(" ".join(tags_list) if tags_list else _render_tag("General Info"),unsafe_allow_html=True)
        if info_s and info_s.get('website'):st.markdown(f"#### 🌐 Website: [{info_s.get('website')}]({info_s.get('website')})")
        if info_s and info_s.get('city'):st.markdown(f"#### 📍 HQ: {info_s.get('city','N/A')}, {info_s.get('state','')} {info_s.get('country','')}")
        
        if company_officers: # company_officers is now correctly unpacked
            st.markdown("---"); st.markdown("#### 🧑‍💼 Key Executives")
            exec_data = [{"Name": officer.get('name'), "Title": officer.get('title')} 
                         for officer in company_officers 
                         if isinstance(officer, dict) and officer.get('name') and officer.get('title')]
            if exec_data: st.dataframe(pd.DataFrame(exec_data).head(5), use_container_width=True, hide_index=True)
            else: st.info("Key executive info not in expected format.")
        # else: st.info("No key executive information found.") # This message can be noisy

        st.markdown("#### 🧾 Business Summary")
        desc_txt="No summary available."
        if isinstance(about_info,str) and "failed" not in about_info.lower() and "not available" not in about_info.lower():desc_txt=about_info.replace('\n','<br>')
        _render_metric_box(desc_txt)
        cti+=1
    
    # Ownership Tab was removed. cti will naturally flow to the next available tab.

    with tabs[cti]: # Performance & Projection Tab
        st.markdown(f"### 🔮 Historical Performance & Future Projection for {s_info_full.get('shortName', ticker)}")
        st.markdown("#### ⏳ 5-Year Historical Performance Review")
        if hist_perf_error: st.warning(hist_perf_error)
        elif hist_final_value is not None and hist_cagr is not None:
            col_hist1, col_hist2 = st.columns(2)
            with col_hist1: st.metric(label=f"${hist_initial_investment:,.0f} Invested ~5 Years Ago is Now", value=f"${hist_final_value:,.2f}")
            with col_hist2: st.metric(label="Approx. Compound Annual Growth (CAGR)", value=f"{hist_cagr:.2f}%")
            _render_metric_box("<p style='font-size:0.9em; color:#a0a0a0;'><i>Based on price appreciation over the last 5 years of available data (or maximum available if less than 5 years). Dividends are not included in this simple calculation. Past performance is not indicative of future results.</i></p>")
        else: st.info("Could not calculate 5-year historical performance (e.g., insufficient data or stock IPO'd recently).")

        st.markdown("---"); st.markdown("#### 🚀 Future Value Projection (Based on Historical CAGR)")
        proj_investment = st.number_input("If you invest (USD):", min_value=100, value=1000, step=100, key="proj_invest_input")
        proj_years = st.slider("Project for how many years?", min_value=1, max_value=10, value=5, key="proj_years_slider")
        if hist_cagr is not None:
            projected_value = project_future_value_cagr(proj_investment, hist_cagr, proj_years)
            if projected_value is not None:
                st.markdown(f"Based on the historical CAGR of **{hist_cagr:.2f}%**:")
                profit_or_loss = projected_value - proj_investment
                profit_percentage = (profit_or_loss / proj_investment) * 100 if proj_investment else 0
                delta_text = f"${profit_or_loss:,.2f} ({profit_percentage:.2f}%)"
                st.metric(label=f"Projected value of ${proj_investment:,.0f} in {proj_years} years", value=f"${projected_value:,.2f}", delta=delta_text if profit_or_loss != 0 else None)
            else: st.warning("Could not project future value based on CAGR.")
            _render_metric_box("<p style='font-size:0.9em; color:#a0a0a0;'><b>🚨 IMPORTANT DISCLAIMER:</b> This projection is purely illustrative, based on past CAGR and assumes this rate will continue. Stock markets are volatile, and past performance is NOT a guarantee of future results. This is NOT financial advice. Consult a qualified financial advisor.</p>")
        else: st.warning("Cannot provide future projection as historical CAGR could not be calculated.")
        cti+=1
        
    with tabs[cti]: # Chatbot Tab
        st.markdown(f"### 💬 Chat with StockSeer about {s_info_full.get('shortName', ticker)}")
        if st.session_state.current_ticker_for_chat != ticker:
            st.session_state.chat_history = []
            st.session_state.current_ticker_for_chat = ticker
            st.session_state.chat_history.append({"role": "assistant", "content": f"Hello! I'm StockSeer. How can I help you with {s_info_full.get('shortName', ticker)} today?"})
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input(f"Ask about {ticker}... (e.g., 'current price?', 'P/E ratio?', 'news')"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            stock_data_bundle = { "s_info_full": s_info_full, "df_ta": df_ta, "current_price": current_price, "news_items": news_items } 
            with st.spinner("StockSeer is thinking..."):
                bot_response = get_chatbot_response(prompt, stock_data_bundle, ticker) 
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"): st.markdown(bot_response)
        cti+=1

    with tabs[cti]: # AI, Risk & News Tab (Final Tab)
        st.markdown("### 🧠 AI Suggestion, Risk, Analyst View & News")
        sig_col_final = '#39ff14'; sh_col_final = 'rgba(57,255,20,0.7)' 
        if "SELL" in signal: sig_col_final = '#c0392b'; sh_col_final = 'rgba(192,57,43,0.7)'
        elif "HOLD" in signal: sig_col_final = '#e0e0e0'; sh_col_final = 'rgba(224,224,224,0.5)'
        if "STRONG" in signal: sig_col_final = '#FFD700'
        
        st.markdown(f"<p style='font-size:1.3rem;font-weight:bold;color:#e0e0e0;'>AI Technical Suggestion: <span style='font-size:1.9rem;font-weight:bold;color:{sig_col_final};text-shadow:0 0 10px {sh_col_final};margin-left:10px;'>{signal}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:1rem;color:#b0b0b0;'><i>Basis:</i></p>", unsafe_allow_html=True)
        detailed_reasons = signal_reason.split(". ")
        for dr_item in detailed_reasons:
            if dr_item.strip(): st.markdown(f"<p style='font-size:1rem;color:#b0b0b0; margin-left:15px;'>- {dr_item.strip()}.</p>", unsafe_allow_html=True)
        st.caption("🔔 AI suggestions are based on technical indicators. For informational purposes only.")
        
        st.markdown("---"); st.markdown("### 📈 Volatility & Risk Profile")
        if volatility_percent is not None:
            risk_color_map = {"Low":"green","Moderate":"orange","High":"red","Very High":"#8B0000"}
            risk_html_color = risk_color_map.get(risk_level,"grey")
            col_v, col_r_disp = st.columns(2)
            with col_v: st.metric(label="Annualized Volatility (~60D)", value=f"{volatility_percent:.2f}%")
            with col_r_disp: st.markdown(f"<p style='font-size:1.1rem;color:#b0b0b0;margin-bottom:0px;'>Qualitative Risk:</p><div class='tag' style='background-color:{risk_html_color}; color:white; font-size:1.5rem; padding: 8px 15px; margin-top:5px;'>{risk_level}</div>", unsafe_allow_html=True)
            _render_metric_box(f"<p style='font-weight:400;'>{risk_explanation_text}</p><p style='font-size:0.9em;color:#a0a0a0;'><i>Note: Volatility is based on daily returns over ~60 trading days. Higher volatility implies larger potential price swings.</i></p>")
            if hist_vol_series is not None and not hist_vol_series.empty:
                fig_hist_vol = go.Figure()
                fig_hist_vol.add_trace(go.Scatter(x=hist_vol_series.index, y=hist_vol_series, mode='lines', name='30-D Ann. Volatility', line=dict(color='#FFA500')))
                fig_hist_vol.update_layout(title_text='Historical Annualized Volatility Trend (%)', template='plotly_dark', plot_bgcolor='#0e0e0e', paper_bgcolor='#0e0e0e', height=300, yaxis_title="Volatility (%)", margin=dict(t=40, b=40))
                st.plotly_chart(fig_hist_vol, use_container_width=True)
            else: st.info("Not enough data to plot historical volatility trend.")
        else: st.info(f"Could not assess volatility: {risk_explanation_text}")

        if analyst_recs is not None and not analyst_recs.empty:
            st.markdown("---"); st.markdown("### 🎯 Analyst Recommendations & Price Targets")
            latest_recs_df = analyst_recs.tail(10).sort_index(ascending=False)
            recs_display_list = []
            for idx, row_data in latest_recs_df.iterrows():
                recs_display_list.append({
                    "Date": idx.strftime('%Y-%m-%d') if isinstance(idx, pd.Timestamp) else str(idx),
                    "Firm": row_data.get('Firm', 'N/A'), "Action": row_data.get('Action', 'N/A'),
                    "From": row_data.get('From Grade', ''), "To": row_data.get('To Grade', 'N/A')})
            if recs_display_list:
                st.markdown("##### Recent Analyst Actions:")
                st.dataframe(pd.DataFrame(recs_display_list), use_container_width=True, hide_index=True)
            pt_current_price_for_calc = current_price
            pt_target_mean_val, pt_target_high_val, pt_target_low_val, pt_num_analysts_val = None, None, None, None
            if analyst_price_target_data and isinstance(analyst_price_target_data, dict):
                pt_target_mean_val = analyst_price_target_data.get('targetMeanPrice')
                pt_target_high_val = analyst_price_target_data.get('targetHighPrice')
                pt_target_low_val = analyst_price_target_data.get('targetLowPrice')
                pt_num_analysts_val = analyst_price_target_data.get('numberOfAnalystOpinions')
            elif s_info_full:
                pt_target_mean_val = s_info_full.get('targetMeanPrice'); pt_target_high_val = s_info_full.get('targetHighPrice'); pt_target_low_val = s_info_full.get('targetLowPrice'); pt_num_analysts_val = s_info_full.get('numberOfAnalystOpinions')
            if pt_target_mean_val is not None:
                st.markdown("##### Price Target Summary:")
                pt_cols_disp = st.columns(3); upside = None
                if pt_current_price_for_calc and pt_target_mean_val > 0 and pt_current_price_for_calc > 0 : upside = f"{((pt_target_mean_val - pt_current_price_for_calc) / pt_current_price_for_calc * 100):.2f}%"
                pt_cols_disp[0].metric("Mean Target", f"${pt_target_mean_val:.2f}" if pt_target_mean_val else "N/A", upside if upside else None)
                pt_cols_disp[1].metric("High Target", f"${pt_target_high_val:.2f}" if pt_target_high_val else "N/A")
                pt_cols_disp[2].metric("Low Target", f"${pt_target_low_val:.2f}" if pt_target_low_val else "N/A")
                if pt_num_analysts_val: st.caption(f"Based on {pt_num_analysts_val} analyst opinion(s).")
            else: st.info("Price target data not available.")
        else: st.info(f"No analyst recommendation data found for {ticker}.")

        st.markdown("---")
        st.markdown(f"### 📰 Latest News & Sentiment for {s_info_full.get('shortName', ticker)}")
        if news_error_message:
            st.warning(news_error_message)
        if news_items:
            sentiment_summary = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
            news_display_cols = st.columns(2) 
            col_idx = 0
            for item in news_items:
                title = item.get('title', 'No Title Available')
                published_dt = item.get('published', 'N/A')
                publisher = item.get('publisher', 'N/A')
                link = item.get('link', '#')
                sentiment_result = analyze_sentiment_text(title) 
                lbl = sentiment_result.get('label','NEUTRAL').upper(); sc = sentiment_result.get('score',0.0)
                if lbl not in sentiment_summary: lbl = "NEUTRAL"
                sentiment_summary[lbl]+=1
                s_cls = "sentiment-neutral"
                if lbl == "POSITIVE": s_cls = "sentiment-positive"
                elif lbl == "NEGATIVE": s_cls = "sentiment-negative"
                with news_display_cols[col_idx % 2]:
                    _render_metric_box(f"""<p style='margin-bottom:5px;font-weight:600;font-size:1.05rem;color:#e0e0e0;'>{title}</p><p style='margin-bottom:8px;'><small style='color:#a0a0a0;'>{published_dt} - {publisher}</small> {_render_tag(f"{lbl} ({sc:.2f})",s_cls)}</p><a href="{link}" target="_blank" style='font-size:0.9rem;display:inline-block;'>Read More »</a>""")
                col_idx += 1
            st.markdown("---"); st.markdown("#### Overall News Sentiment Distribution:")
            sent_labels_chart=list(sentiment_summary.keys()); sent_values_chart=[sentiment_summary[k] for k in sent_labels_chart]
            sent_colors_chart=['#39ff14' if k=='POSITIVE' else '#c0392b' if k=='NEGATIVE' else '#888' for k in sent_labels_chart]
            if sum(sent_values_chart)>0:
                fig_sent_dist=go.Figure(data=[go.Bar(x=sent_labels_chart,y=sent_values_chart,marker_color=sent_colors_chart)])
                fig_sent_dist.update_layout(title_text='News Sentiment Counts',xaxis_title="Sentiment",yaxis_title="Articles",template='plotly_dark',plot_bgcolor='#0e0e0e',paper_bgcolor='#0e0e0e',showlegend=False,height=300, margin=dict(t=30,b=30)); st.plotly_chart(fig_sent_dist,use_container_width=True)
            else: st.info("No sentiment data for chart.")
            st.caption("Sentiment analysis classifies headlines. Scores indicate confidence.")
        elif not news_error_message: st.info(f"No recent news found for {ticker} via yfinance.")


else: # Welcome screen
    st.markdown("""<div style='text-align:center;padding:40px 20px;background:linear-gradient(145deg,#1a1a1a,#0f0f0f);border-radius:15px;margin:20px auto;max-width:800px;box-shadow:0 8px 25px rgba(0,0,0,0.7);border:1px solid #39ff14;'><img src="https://www.gstatic.com/images/branding/product/1x/finance_2020q4_48dp.png" alt="Logo" style="width:70px;margin-bottom:15px;"><h1 style='font-size:2.5rem;color:#39ff14;text-shadow:0 0 15px rgba(57,255,20,0.7);margin-bottom:15px;'>Welcome to StockSeer.AI!</h1><p style='font-size:1.2rem;color:#e0e0e0;margin-bottom:20px;'>Intelligent portal for stock insights.</p><p style='font-size:1rem;color:#b0b0b0;'>Enter a <strong>stock ticker</strong> in the sidebar to begin.</p><p style='font-size:1rem;color:#b0b0b0;'>Compare stocks with a second ticker!</p></div>""",unsafe_allow_html=True)