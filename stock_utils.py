# stock_utils.py
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go

import feedparser

import requests
from bs4 import BeautifulSoup

def get_about_stock_info(ticker):
    try:
        search_query = f"{ticker} stock company profile site:finance.yahoo.com"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        search_response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(search_response.text, 'html.parser')
        links = soup.find_all('a')
        profile_url = None
        for link in links:
            href = link.get('href')
            if href and "finance.yahoo.com/quote" in href and "profile" in href:
                href = href.replace("/url?q=", "").split("&")[0]
                profile_url = href
                break

        if not profile_url:
            return "Couldn't find detailed company profile for this stock."

        profile_response = requests.get(profile_url, headers=headers)
        profile_soup = BeautifulSoup(profile_response.text, 'html.parser')
        summary = profile_soup.find('section', attrs={'data-test': 'qsp-profile'})
        if not summary:
            summary = profile_soup.find('p')
        return summary.get_text(separator="\n").strip() if summary else "Company profile not available."
    except Exception as e:
        return f"Error retrieving company profile: {e}"


def get_stock_news(ticker):
    feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    news_feed = feedparser.parse(feed_url)
    news_items = news_feed.entries[:5]  # limit to 5 articles
    return news_items


def fetch_stock_data(ticker, period='3mo', interval='1d'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df.dropna(inplace=True)
    return df


def add_technical_indicators(df):
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.macd_diff(df['Close'])
    df['MACD'] = macd
    return df


def generate_signal(df):
    latest = df.iloc[-1]
    if latest['RSI'] < 30 and latest['MACD'] > 0:
        return "BUY"
    elif latest['RSI'] > 70 and latest['MACD'] < 0:
        return "SELL"
    else:
        return "HOLD"


def plot_stock_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], mode='lines', name='Close Price',
        line=dict(color='white', width=2)))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20',
        line=dict(color='orange', dash='dash')))

    fig.update_layout(
        template='plotly_dark',
        title={
            'text': f"📈 {symbol} Stock Price & SMA 20",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        height=500,
        plot_bgcolor='black',
        paper_bgcolor='black'
    )

    fig.update_xaxes(showgrid=False, color='white')
    fig.update_yaxes(showgrid=False, color='white')
    return fig


# News Section
import feedparser

def get_stock_news(ticker):
    feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    news_feed = feedparser.parse(feed_url)
    news_items = news_feed.entries[:5]  # limit to 5 articles
    return news_items
