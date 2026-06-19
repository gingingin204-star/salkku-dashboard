"""
Salkku-dashboardin koko backend yhdessä tiedostossa.

Sisältää: API-avainten luvun (env > config.json), neljä agenttia
(TA, uutiset, tulosrapsa, osinko), orchestratorin joka ajaa ne per
tikkeri, ja FastAPI-palvelimen joka tarjoaa dashboardin + JSON-API:n.

Käynnistys paikallisesti:   uvicorn app:app --reload --port 8000
Käynnistys pilvessä:        uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import base64
import csv
import json
import os
import secrets
from datetime import datetime, timedelta, timezone

import requests
import yfinance as yf
import pandas as pd
from anthropic import Anthropic
from fastapi import FastAPI, Body, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
PORTFOLIO_PATH = os.path.join(BASE_DIR, "portfolio_example.csv")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

DEFAULT_CONFIG = {"anthropic_api_key": "", "finnhub_api_key": ""}
ENV_VAR_NAMES = {"anthropic_api_key": "ANTHROPIC_API_KEY", "finnhub_api_key": "FINNHUB_API_KEY"}


# ---------------------------------------------------------------------------
# Asetukset / API-avaimet
# ---------------------------------------------------------------------------
def _file_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = DEFAULT_CONFIG.copy()
    merged.update(data)
    return merged


def load_config() -> dict:
    cfg = _file_config()
    for key, env_name in ENV_VAR_NAMES.items():
        env_val = os.environ.get(env_name)
        if env_val:
            cfg[key] = env_val
    return cfg


def is_from_env(key: str) -> bool:
    env_name = ENV_VAR_NAMES.get(key)
    return bool(env_name and os.environ.get(env_name))


def save_config(new_values: dict) -> dict:
    current = _file_config()
    for key in DEFAULT_CONFIG.keys():
        if is_from_env(key):
            continue
        if key in new_values and new_values[key] is not None:
            current[key] = new_values[key].strip()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    return current


def config_status() -> dict:
    cfg = load_config()
    return {
        key: {"set": bool(value), "source": "env" if is_from_env(key) else "file"}
        for key, value in cfg.items()
    }


# ---------------------------------------------------------------------------
# TA-agentti
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    def last(series):
        valid = series.dropna()
        return round(float(valid.iloc[-1]), 3) if not valid.empty else None

    return {
        "last_close": last(close), "rsi_14": last(rsi),
        "ma50": last(ma50), "ma200": last(ma200),
        "macd": last(macd), "macd_signal": last(macd_signal),
    }


def ta_agent_run(ticker: str, cfg: dict) -> dict:
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return {"error": f"Ei kurssidataa tikkerille '{ticker}'."}
        indicators = compute_indicators(df)
    except Exception as e:
        return {"error": f"yfinance-virhe: {e}"}

    api_key = cfg.get("anthropic_api_key")
    if not api_key:
        return {"indicators": indicators, "summary": "Aseta Anthropic API -avain saadaksesi tulkinnan."}
    try:
        client = Anthropic(api_key=api_key)
        prompt = (
            f"Olet tekninen analyytikko. Tikkeri: {ticker}.\nIndikaattorit: {indicators}\n\n"
            "Anna lyhyt suomenkielinen tulkinta (max 4 lausetta) trendistä ja siitä, "
            "viittaavatko luvut osto-, myynti- vai neutraaliin signaaliin."
        )
        msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=300,
                                      messages=[{"role": "user", "content": prompt}])
        summary = msg.content[0].text
    except Exception as e:
        summary = f"Claude API -virhe: {e}"
    return {"indicators": indicators, "summary": summary}


# ---------------------------------------------------------------------------
# Uutisagentti
# ---------------------------------------------------------------------------
def news_agent_run(ticker: str, cfg: dict) -> dict:
    finnhub_key = cfg.get("finnhub_api_key")
    if not finnhub_key:
        return {"error": "Finnhub API -avain puuttuu."}
    try:
        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=7)
        r = requests.get("https://finnhub.io/api/v1/company-news", params={
            "symbol": ticker, "from": from_date.isoformat(), "to": to_date.isoformat(), "token": finnhub_key
        }, timeout=10)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        return {"error": f"Finnhub-virhe: {e}"}

    if not raw:
        return {"articles": [], "summary": "Ei uutisia viimeisen 7 päivän ajalta."}
    articles = [{"title": a.get("headline"), "url": a.get("url"), "source": a.get("source")} for a in raw[:8]]

    api_key = cfg.get("anthropic_api_key")
    if not api_key:
        return {"articles": articles, "summary": "Aseta Anthropic API -avain saadaksesi yhteenvedon."}
    try:
        client = Anthropic(api_key=api_key)
        titles_text = "\n".join(f"- {a['title']}" for a in articles)
        prompt = (
            f"Olet talousuutisanalyytikko. Tikkeri: {ticker}.\nUutisotsikot viim. 7 pv:\n{titles_text}\n\n"
            "Tee lyhyt suomenkielinen yhteenveto (max 4 lausetta): mitä tapahtuu ja onko sävy "
            "positiivinen, negatiivinen vai neutraali."
        )
        msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=300,
                                      messages=[{"role": "user", "content": prompt}])
        summary = msg.content[0].text
    except Exception as e:
        summary = f"Claude API -virhe: {e}"
    return {"articles": articles, "summary": summary}


# ---------------------------------------------------------------------------
# Tulosrapsa-agentti
# ---------------------------------------------------------------------------
def earnings_agent_run(ticker: str, cfg: dict) -> dict:
    finnhub_key = cfg.get("finnhub_api_key")
    if not finnhub_key:
        return {"error": "Finnhub API -avain puuttuu."}
    try:
        r = requests.get("https://finnhub.io/api/v1/stock/earnings",
                          params={"symbol": ticker, "token": finnhub_key}, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"Finnhub-virhe: {e}"}

    if not data:
        return {"reports": [], "note": "Ei tulosdataa - tarkista yhtiön IR-sivu manuaalisesti."}
    recent = data[:4]

    api_key = cfg.get("anthropic_api_key")
    if not api_key:
        return {"reports": recent}
    try:
        client = Anthropic(api_key=api_key)
        prompt = (
            f"Tikkeri: {ticker}. Viimeisimmät tulosjulkistukset (actual vs. estimate EPS): {recent}\n\n"
            "Tee lyhyt suomenkielinen analyysi (max 4 lausetta): ylittikö/alittiko yhtiö odotukset ja "
            "näkyykö selkeä trendi."
        )
        msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=300,
                                      messages=[{"role": "user", "content": prompt}])
        summary = msg.content[0].text
    except Exception as e:
        summary = f"Claude API -virhe: {e}"
    return {"reports": recent, "summary": summary}


# ---------------------------------------------------------------------------
# Osinkoagentti
# ---------------------------------------------------------------------------
def dividend_agent_run(ticker: str, cfg: dict) -> dict:
    try:
        t = yf.Ticker(ticker)
        div_history = t.dividends
        try:
            info = t.info
        except Exception:
            info = {}
    except Exception as e:
        return {"error": f"yfinance-virhe: {e}"}

    if div_history is None or div_history.empty:
        return {"history": [], "note": "Ei osinkohistoriaa tälle tikkerille."}
    recent = div_history.tail(8)
    history = [{"date": str(idx.date()), "amount": round(float(val), 4)} for idx, val in recent.items()]
    dividend_yield = info.get("dividendYield")

    api_key = cfg.get("anthropic_api_key")
    if not api_key:
        return {"history": history, "dividend_yield": dividend_yield}
    try:
        client = Anthropic(api_key=api_key)
        prompt = (
            f"Tikkeri: {ticker}. Osinkohistoria: {history}. Osinkotuotto: {dividend_yield}\n\n"
            "Tee lyhyt suomenkielinen analyysi (max 3 lausetta) osingon vakaudesta ja trendistä."
        )
        msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=250,
                                      messages=[{"role": "user", "content": prompt}])
        summary = msg.content[0].text
    except Exception as e:
        summary = f"Claude API -virhe: {e}"
    return {"history": history, "dividend_yield": dividend_yield, "summary": summary}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _safe_filename(ticker: str) -> str:
    return ticker.replace(".", "_").replace("/", "_")


def run_for_ticker(ticker: str) -> dict:
    cfg = load_config()
    result = {
        "ticker": ticker,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ta": ta_agent_run(ticker, cfg),
        "news": news_agent_run(ticker, cfg),
        "earnings": earnings_agent_run(ticker, cfg),
        "dividend": dividend_agent_run(ticker, cfg),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f"{_safe_filename(ticker)}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def read_portfolio(path: str = PORTFOLIO_PATH) -> list:
    tickers = []
    if not os.path.exists(path):
        return tickers
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = (row.get("ticker") or "").strip()
            if ticker:
                tickers.append(ticker)
    return tickers


# ---------------------------------------------------------------------------
# FastAPI-palvelin
# ---------------------------------------------------------------------------
app = FastAPI(title="Osakesalkku-dashboard")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    password = os.environ.get("DASHBOARD_PASSWORD")
    if not password:
        return await call_next(request)
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            _, _, supplied = decoded.partition(":")
        except Exception:
            supplied = ""
        if secrets.compare_digest(supplied, password):
            return await call_next(request)
    return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Salkku"'},
                     content="Kirjautuminen vaaditaan.")


@app.get("/api/portfolio")
def get_portfolio():
    rows = []
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return {"holdings": rows}


@app.get("/api/data/{ticker}")
def get_data(ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.json")
    if not os.path.exists(path):
        return {"error": "Ei dataa vielä tälle tikkerille. Paina 'Päivitä'."}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/refresh/{ticker}")
def refresh_ticker(ticker: str):
    return run_for_ticker(ticker)


@app.post("/api/refresh-all")
def refresh_all():
    return {t: run_for_ticker(t) for t in read_portfolio(PORTFOLIO_PATH)}


@app.get("/api/config/status")
def get_config_status():
    return config_status()


@app.post("/api/config")
def update_config(payload: dict = Body(...)):
    save_config(payload)
    return config_status()


app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
