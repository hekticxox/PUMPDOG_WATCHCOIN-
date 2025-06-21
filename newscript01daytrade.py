import os
import sys
import time
import requests
import numpy as np
import logging
import json
import hmac
import base64
import hashlib
from dotenv import load_dotenv
from kucoin_universal_sdk.api import DefaultClient # type: ignore
from kucoin_universal_sdk.model import ClientOptionBuilder, TransportOptionBuilder, GLOBAL_API_ENDPOINT, GLOBAL_FUTURES_API_ENDPOINT, GLOBAL_BROKER_API_ENDPOINT # type: ignore

# === CONFIGURATION ===
with open("config.json") as f:
    config = json.load(f)

MIN_CANDLES = config["MIN_CANDLES"]
MIN_AVG_VOLUME = config["MIN_AVG_VOLUME"]
MIN_PRICE = config["MIN_PRICE"]
MAX_SPREAD_RATIO = config["MAX_SPREAD_RATIO"]
RSI_RANGE = tuple(config["RSI_RANGE"])
SCORE_WEIGHTS = config["SCORE_WEIGHTS"]
DEFAULT_LEVERAGE = config["DEFAULT_LEVERAGE"]
DEFAULT_SIZE = config["DEFAULT_SIZE"]

# === INIT ===
load_dotenv()
logging.basicConfig(filename='swingbot.log', level=logging.INFO, format='%(asctime)s %(message)s')

API_KEY = os.getenv("KUCOIN_API_KEY")
API_SECRET = os.getenv("KUCOIN_API_SECRET")
API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")
BASE_URL = os.getenv("KUCOIN_API_BASE_URL", "https://api-futures.kucoin.com")

http_transport_option = (
    TransportOptionBuilder()
    .set_keep_alive(True)
    .set_max_pool_size(10)
    .set_max_connection_per_pool(10)
    .build()
)

client_option = (
    ClientOptionBuilder()
    .set_key(API_KEY)
    .set_secret(API_SECRET)
    .set_passphrase(API_PASSPHRASE)
    .set_spot_endpoint(GLOBAL_API_ENDPOINT)
    .set_futures_endpoint(GLOBAL_FUTURES_API_ENDPOINT)
    .set_broker_endpoint(GLOBAL_BROKER_API_ENDPOINT)
    .set_transport_option(http_transport_option)
    .build()
)

client = DefaultClient(client_option)
rest = client.rest_service()

account = rest.account_service
futures = rest.futures_service

open_positions = {}  # symbol -> {'side': 'buy'/'sell', 'size': float, 'entry_price': float}

# === TECHNICAL INDICATORS ===
def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes)
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = 100 - 100 / (1 + rs)
    for delta in deltas[period:]:
        upval = max(delta, 0)
        downval = -min(delta, 0)
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi = 100 - 100 / (1 + rs)
    return rsi

def fetch_recent_candles(symbol, verbose=False):
    now = int(time.time())
    start = now - 15 * 60 * 30  # 30 x 15min candles
    granularities = [900, 300, 60]
    for granularity in granularities:
        endpoint = f"/api/v1/kline/query?symbol={symbol}&granularity={granularity}&startAt={start}&endAt={now}"
        url = BASE_URL + endpoint
        if verbose:
            print(f"Fetching candles for {symbol} with granularity {granularity}...")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "200000" and "data" in data:
                return data["data"][-30:]  # Use last 30 candles for all metrics
        time.sleep(0.1)  # Rate limit protection
    return []

def calculate_swing_levels(candles, verbose=False):
    if not candles:
        if verbose:
            print("No candle data to calculate swing levels.")
        return None, None
    closes = [float(c[2]) for c in candles]
    high = max([float(c[3]) for c in candles])
    low = min([float(c[4]) for c in candles])
    pivot = (high + low + closes[-1]) / 3
    support1 = (2 * pivot) - high
    resistance1 = (2 * pivot) - low
    if verbose:
        print(f"Calculated S1: {support1}, R1: {resistance1}")
    return support1, resistance1

# === CANDIDATE SELECTION ===
def get_improved_swing_candidates(verbose=False):
    contracts_url = BASE_URL + "/api/v1/contracts/active"
    contracts_resp = requests.get(contracts_url)
    contracts = contracts_resp.json().get("data", [])
    symbols = [c["symbol"] for c in contracts]

    candidates = []
    for i, symbol in enumerate(symbols):
        if verbose:
            print(f"Fetching candles for {symbol} ({i+1}/{len(symbols)})")
        candles = fetch_recent_candles(symbol, verbose=False)
        if len(candles) < MIN_CANDLES:
            continue
        closes = np.array([float(c[2]) for c in candles])
        highs = np.array([float(c[3]) for c in candles])
        lows = np.array([float(c[4]) for c in candles])
        volumes = np.array([float(c[5]) for c in candles])
        last_close = closes[-1]
        last_high = highs[-1]
        last_low = lows[-1]
        avg_volume = np.mean(volumes)
        if len(closes) < 21 or avg_volume < MIN_AVG_VOLUME or last_close < MIN_PRICE:
            continue

        # Spread filter (relative to last close)
        spread = (last_high - last_low) / last_close if last_close else 0
        if spread > MAX_SPREAD_RATIO:
            continue

        # Volatility (stddev of returns)
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns)

        # ATR (Average True Range)
        tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
        atr = np.mean(tr)

        # Trend: EMA 9 vs EMA 21
        ema9 = np.mean(closes[-9:])
        ema21 = np.mean(closes[-21:])
        trend_strength = ema9 - ema21

        # RSI
        rsi = calculate_rsi(closes)

        # Score: combine normalized metrics (weights can be tuned)
        score = (volatility * SCORE_WEIGHTS['volatility']) + \
                (atr * SCORE_WEIGHTS['atr']) + \
                (trend_strength * SCORE_WEIGHTS['trend']) + \
                (avg_volume * SCORE_WEIGHTS['volume'])

        # Filter for reasonable RSI and trend
        if not (RSI_RANGE[0] < rsi < RSI_RANGE[1]):
            print(f"{symbol} skipped: RSI {rsi:.2f} out of range {RSI_RANGE}")
            continue
        if abs(trend_strength) < 0.001:
            print(f"{symbol} skipped: trend_strength {trend_strength:.6f} too weak")
            continue

        candidates.append({
            'symbol': symbol,
            'score': score,
            'volatility': volatility,
            'atr': atr,
            'trend_strength': trend_strength,
            'avg_volume': avg_volume,
            'rsi': rsi,
            'spread': spread,
            'last_close': last_close
        })
        if verbose:
            print(f"{symbol}: score={score:.6f}, vol={volatility:.6f}, atr={atr:.6f}, trend={trend_strength:.6f}, rsi={rsi:.2f}, avg_vol={avg_volume:.2f}, spread={spread:.4f}, price={last_close:.4f}")

    candidates.sort(key=lambda x: x['score'], reverse=True)
    if verbose:
        print("Top 10 swing candidates by score:")
        for c in candidates[:10]:
            print(f"{c['symbol']}: score={c['score']:.6f}, vol={c['volatility']:.6f}, atr={c['atr']:.6f}, trend={c['trend_strength']:.6f}, rsi={c['rsi']:.2f}, avg_vol={c['avg_volume']:.2f}, spread={c['spread']:.4f}, price={c['last_close']:.4f}")
    return candidates[:10]

def get_fast_candidates(verbose=False):
    # Step 1: Get all tickers in one call
    tickers_url = BASE_URL + "/api/v1/ticker"
    tickers_resp = requests.get(tickers_url)
    tickers = tickers_resp.json().get("data", [])
    # Step 2: Filter by volume, price, etc.
    filtered = []
    for t in tickers:
        symbol = t["symbol"]
        last = float(t["last"])
        vol = float(t["vol"])
        if last >= MIN_PRICE and vol >= MIN_AVG_VOLUME:
            filtered.append(symbol)
    # Step 3: Only fetch candles for filtered symbols
    # (You can use ThreadPoolExecutor here, but keep max_workers low)
    return filtered

# === OPPORTUNITY DETECTION ===
def find_trade_opportunities(candidates, verbose=False):
    opportunities = []
    for c in candidates:
        symbol = c['symbol']
        candles = fetch_recent_candles(symbol, verbose=verbose)
        closes = np.array([float(c[2]) for c in candles])
        rsi = calculate_rsi(closes)
        s1, r1 = calculate_swing_levels(candles)
        last_price = closes[-1]
        ema9 = np.mean(closes[-9:])
        ema21 = np.mean(closes[-21:])
        trend_up = ema9 > ema21
        trend_down = ema9 < ema21

        # Debug: print all relevant values
        if verbose:
            print(f"{symbol}: price={last_price:.4f}, s1={s1}, r1={r1}, rsi={rsi:.2f}, ema9={ema9:.4f}, ema21={ema21:.4f}, trend_up={trend_up}, trend_down={trend_down}")

        # Buy signal: price near support, RSI low
        if s1 and last_price <= s1 * 1.03 and rsi < 50:
            opportunities.append({
                'symbol': symbol,
                'action': 'buy',
                'price': last_price,
                'support': s1,
                'resistance': r1,
                'rsi': rsi,
                'trend': 'up' if trend_up else 'down'
            })
        # Sell signal: price near resistance, RSI high
        elif r1 and last_price >= r1 * 0.99 and rsi > 55:
            opportunities.append({
                'symbol': symbol,
                'action': 'sell',
                'price': last_price,
                'support': s1,
                'resistance': r1,
                'rsi': rsi,
                'trend': 'down' if trend_down else 'up'
            })
        else:
            if verbose:
                reasons = []
                if not (s1 and last_price <= s1 * 1.03): reasons.append("not near support")
                if not (rsi < 50): reasons.append("RSI not low")
                if not trend_up: reasons.append("trend not up")
                if not (r1 and last_price >= r1 * 0.99): reasons.append("not near resistance")
                if not (rsi > 55): reasons.append("RSI not high")
                if not trend_down: reasons.append("trend not down")
                print(f"{symbol} skipped: {', '.join(reasons)}")
    return opportunities

# === TRADE EXECUTION & TRACKING ===
# Remove the build_order_request and use the REST API directly for order placement
def place_order(symbol, side, size, price, leverage=DEFAULT_LEVERAGE, verbose=False):
    try:
        order_type = "limit"
        endpoint = "/api/v1/orders"
        url = BASE_URL + endpoint
        body = {
            "symbol": symbol,
            "side": side,
            "leverage": str(leverage),
            "size": str(size),
            "price": str(price),
            "type": order_type
        }
        body_json = json.dumps(body)
        headers = get_kucoin_headers(endpoint, method="POST", body=body_json)
        if verbose:
            print(f"Placing {order_type} order: {side} {size} {symbol} at {price} with {leverage}x leverage")
        resp = requests.post(url, headers=headers, data=body_json)
        result = resp.json()
        if verbose:
            print(f"Order result: {result}")
        logging.info(f"Placed order: {result}")
        # KuCoin returns orderId under result['data']['orderId']
        if result.get('code') == '200000' and 'data' in result and 'orderId' in result['data']:
            open_positions[symbol] = {'side': side, 'size': size, 'entry_price': price}
        return result
    except Exception as e:
        if verbose:
            print(f"Error placing order: {e}")
        logging.error(f"Order error: {e}")
        return None

def close_position(symbol, verbose=False):
    pos = open_positions.get(symbol)
    if not pos:
        print(f"No open position for {symbol}")
        return
    side = 'sell' if pos['side'] == 'buy' else 'buy'
    print(f"Closing {pos['side']} position for {symbol} at market.")
    place_order(symbol, side, pos['size'], pos['entry_price'], verbose=verbose)
    log_pnl(symbol, pos['entry_price'])
    del open_positions[symbol]
    check_reversal_opportunity(symbol, pos['side'], verbose=verbose)

def log_pnl(symbol, exit_price):
    pos = open_positions.get(symbol)
    if not pos:
        return
    pnl = (exit_price - pos['entry_price']) * pos['size'] if pos['side'] == 'buy' else (pos['entry_price'] - exit_price) * pos['size']
    logging.info(f"Closed {pos['side']} {symbol} at {exit_price}. P&L: {pnl}")

def get_kucoin_headers(endpoint, method='GET', body=''):
    now = str(int(time.time() * 1000))
    str_to_sign = now + method + endpoint + body
    signature = base64.b64encode(hmac.new(API_SECRET.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()).decode()
    passphrase = base64.b64encode(hmac.new(API_SECRET.encode('utf-8'), API_PASSPHRASE.encode('utf-8'), hashlib.sha256).digest()).decode()
    return {
        "KC-API-KEY": API_KEY,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": now,
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json"
    }

def get_futures_balance(verbose=False):
    try:
        endpoint = '/api/v1/account-overview?currency=USDT'
        url = BASE_URL + endpoint
        headers = get_kucoin_headers(endpoint)
        resp = requests.get(url, headers=headers)
        data = resp.json()
        if verbose:
            print(f"Account overview response: {data}")
        if data.get('code') == '200000' and 'data' in data and 'availableBalance' in data['data']:
            return float(data['data']['availableBalance'])
        else:
            if verbose:
                print("Could not find availableBalance in response.")
            return 0.0
    except Exception as e:
        if verbose:
            print(f"Failed to fetch futures balance: {e}")
        return 0.0

def verify_and_execute_opportunities(opportunities, verbose=False):
    for opp in opportunities:
        print(f"\nOpportunity detected for {opp['symbol']}:")
        print(f"  Action: {opp['action'].upper()}")
        print(f"  Price: {opp['price']}")
        print(f"  Support: {opp['support']}")
        print(f"  Resistance: {opp['resistance']}")
        print(f"  RSI: {opp['rsi']:.2f}")
        print(f"  Trend: {opp['trend']}")

        # Show available balance
        available_balance = get_futures_balance(verbose=verbose)
        print(f"Available USDT balance: {available_balance:.2f}")

        # Ask user for trade amount in $ or %
        while True:
            amt_input = input("Enter trade amount (e.g. 50 for $50 or 10% for 10 percent of balance, blank for default): ").strip()
            if not amt_input:
                trade_amount = DEFAULT_SIZE
                break
            elif amt_input.endswith('%'):
                try:
                    pct = float(amt_input.rstrip('%'))
                    trade_amount = available_balance * pct / 100
                    break
                except Exception:
                    print("Invalid percent format. Try again.")
            else:
                try:
                    trade_amount = float(amt_input)
                    break
                except Exception:
                    print("Invalid dollar amount. Try again.")

        # Calculate position size based on trade_amount and price
        price = opp['price']
        side = opp['action']
        leverage = DEFAULT_LEVERAGE
        # Futures position size = (trade_amount * leverage) / price
        size = (trade_amount * leverage) / price

        print(f"Calculated position size: {size:.6f} contracts (amount: {trade_amount:.2f} USDT at {leverage}x)")

        confirm = input("Do you want to execute this trade? (y/n): ")
        if confirm.lower() == 'y':
            place_order(opp['symbol'], side, size, price, leverage, verbose=verbose)
        else:
            print("Trade not executed.")

# === REVERSAL OPPORTUNITY DETECTION ===
def check_reversal_opportunity(symbol, last_side, verbose=False):
    candles = fetch_recent_candles(symbol, verbose=verbose)
    closes = np.array([float(c[2]) for c in candles])
    rsi = calculate_rsi(closes)
    s1, r1 = calculate_swing_levels(candles)
    last_price = closes[-1]
    ema9 = np.mean(closes[-9:])
    ema21 = np.mean(closes[-21:])
    trend_up = ema9 > ema21
    trend_down = ema9 < ema21

    if last_side == 'buy' and r1 and last_price >= r1 * 0.99 and rsi > 55:
        # Close buy position and open sell position
        print(f"Reversal opportunity detected: close buy, open sell for {symbol}")
        close_position(symbol, verbose=verbose)
        place_order(symbol, 'sell', DEFAULT_SIZE, last_price, DEFAULT_LEVERAGE, verbose=verbose)
    elif last_side == 'sell' and s1 and last_price <= s1 * 1.03 and rsi < 50:
        # Close sell position and open buy position
        print(f"Reversal opportunity detected: close sell, open buy for {symbol}")
        close_position(symbol, verbose=verbose)
        place_order(symbol, 'buy', DEFAULT_SIZE, last_price, DEFAULT_LEVERAGE, verbose=verbose)
    else:
        if verbose:
            print(f"No reversal opportunity for {symbol}: last_side={last_side}, last_price={last_price}, s1={s1}, r1={r1}, rsi={rsi:.2f}, trend_up={trend_up}, trend_down={trend_down}")

# === MAIN STRATEGY LOOP ===
def run_strategy(verbose=False):
    print("\n=== SWING TRADING BOT ===")
    if verbose:
        print("Verbose mode enabled.")
    candidates = get_improved_swing_candidates(verbose=verbose)
    if not candidates:
        print("No candidates found. Exiting.")
        return
    opportunities = find_trade_opportunities(candidates, verbose=verbose)
    if not opportunities:
        print("No trade opportunities found. Exiting.")
        return
    verify_and_execute_opportunities(opportunities, verbose=verbose)
    if verbose:
        print("Checking for reversal opportunities...")
    for symbol in open_positions.keys():
        pos = open_positions[symbol]
        check_reversal_opportunity(symbol, pos['side'], verbose=verbose)

# For testing: run strategy once with verbose output
run_strategy(verbose=True)

# To run continuously, uncomment the following lines:
# while True:
#     run_strategy()
#     time.sleep(60)  # Wait 60 seconds between each full cycle
