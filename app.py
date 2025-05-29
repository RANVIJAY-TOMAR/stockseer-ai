import streamlit as st
from streamlit_lottie import st_lottie
import requests
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import feedparser

# --- Load Lottie animation ---
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_intro = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Stockseer AI - Premium Stock Analysis", layout="wide")

# --- Session State Initialization ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'show_intro' not in st.session_state:
    st.session_state.show_intro = True
if 'news_symbol' not in st.session_state:
    st.session_state.news_symbol = "AAPL"
if 'about_symbol' not in st.session_state:
    st.session_state.about_symbol = "AAPL"

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigate", ["Stock Analysis", "News Feed", "About the Stock", "Settings"])

# --- Dark Mode Toggle ---
if page == "Settings":
    st.title("Settings")
    st.session_state.dark_mode = st.checkbox("Enable Dark Mode", value=st.session_state.dark_mode)

# --- Theming ---
if st.session_state.dark_mode:
    bg_color = "#121212"
    text_color = "#FFFFFF"
    secondary_text_color = "#CCCCCC"
    plotly_template = 'plotly_dark'
    table_header_bg = "#1f2937"
    table_cell_bg = "#273343"
else:
    bg_color = "#FFFFFF"
    text_color = "#111111"
    secondary_text_color = "#444444"
    plotly_template = 'plotly_white'
    table_header_bg = "#e2e8f0"
    table_cell_bg = "#f8fafc"

# --- Inject CSS ---
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {bg_color};
        color: {text_color};
    }}
    th, td {{
        background-color: {table_cell_bg} !important;
        color: {text_color} !important;
    }}
    .st-bf, .css-1d391kg {{
        color: {text_color} !important;
    }}
    </style>
    """, unsafe_allow_html=True
)

# --- Show Intro Animation Once ---
if st.session_state.show_intro:
    st_lottie(lottie_intro, speed=1, height=280)
    if st.button("Start Stockseer AI"):
        st.session_state.show_intro = False
    st.stop()

# --- Stock Analysis Page ---
if page == "Stock Analysis":
    st.title("📈 Stock Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA):", value="AAPL").upper()

    if symbol:
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="3mo", interval="1d")

            if df.empty:
                st.warning(f"No data found for symbol '{symbol}'. Try another.")
            else:
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    increasing_line_color='limegreen',
                    decreasing_line_color='red',
                    showlegend=False
                )])
                fig.update_layout(
                    template=plotly_template,
                    title=f"{symbol} - Last 3 Months Candlestick Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=600,
                    margin=dict(l=40, r=40, t=80, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Styled Data Table
                st.markdown(f"### 📋 Latest Raw Data for {symbol}")
                styled_df = df.style.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', table_header_bg), ('color', text_color), ('font-weight', 'bold')]},
                    {'selector': 'td', 'props': [('background-color', table_cell_bg), ('color', text_color)]},
                ]).format("{:.2f}")
                st.dataframe(styled_df, height=300)

        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")

# --- News Feed Page ---
if page == "News Feed":
    st.title("📰 News Feed")
    news_symbol = st.text_input("Enter Stock Symbol for News (e.g., AAPL):", st.session_state.news_symbol).upper()
    st.session_state.news_symbol = news_symbol

    try:
        feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={news_symbol}&region=US&lang=en-US"
        news_feed = feedparser.parse(feed_url)
        news_items = news_feed.entries[:5]

        if not news_items:
            st.warning("No news found for this stock.")
        else:
            for item in news_items:
                st.subheader(item.title)
                st.markdown(f"[Read More]({item.link})")
                st.caption(item.published)

    except Exception as e:
        st.error(f"Could not fetch news: {e}")

# --- About the Stock Page ---
if page == "About the Stock":
    st.title("🏢 About the Stock")
    about_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", st.session_state.about_symbol).upper()
    st.session_state.about_symbol = about_symbol

    try:
        query = f"{about_symbol} company overview site:finance.yahoo.com"
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        yahoo_link = None
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if "https://finance.yahoo.com/quote" in href and "/profile" in href:
                yahoo_link = href.split("&")[0].replace("/url?q=", "")
                break

        if yahoo_link:
            profile_page = requests.get(yahoo_link, headers=headers)
            profile_soup = BeautifulSoup(profile_page.text, "html.parser")
            summary = profile_soup.find("section")

            if summary:
                st.markdown("### 🧠 Quick Company Overview")
                st.write(summary.get_text(strip=True))
            else:
                st.info("Couldn't find a detailed overview. Try checking [Yahoo Finance](https://finance.yahoo.com).")
        else:
            st.info("Overview not found. Please ensure the stock symbol is correct.")

    except Exception as e:
        st.error(f"Error fetching about section: {e}")
