# kucoin_scalper_bot.py
# A scalper bot for KuCoin Futures using EMA & RSI signals with live trading and trailing stop-loss

import time
import hmac
import base64
import hashlib
import requests
import json
import pandas as pd
import csv
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

# === LOAD ENVIRONMENT ===
load_dotenv("/home/hekticxox/Desktop/.env")

API_KEY = os.getenv("KUCOIN_API_KEY")
API_SECRET = os.getenv("KUCOIN_API_SECRET")
API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")
API_BASE_URL = 'https://api-futures.kucoin.com'

LEVERAGE = 50
RISK_PERCENT = 10
TAKE_PROFIT_PERCENT = 40
STOP_LOSS_PERCENT = 10
TRAILING_STOP_PERCENT = 0.3
MAX_TRADES = 7
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
TRADE_LOG_FILE = 'scalper_trade_log.csv'

# === UTILITIES ===
def get_headers(endpoint, method='GET', body=''):
    now = int(time.time() * 1000)
    str_to_sign = str(now) + method + endpoint + body
    signature = base64.b64encode(hmac.new(API_SECRET.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()).decode()
    passphrase = base64.b64encode(hmac.new(API_SECRET.encode('utf-8'), API_PASSPHRASE.encode('utf-8'), hashlib.sha256).digest()).decode()
    return {
        "KC-API-KEY": API_KEY,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": str(now),
        "KC-API-PASSPHRASE": passphrase,
        "KC-API-KEY-VERSION": "2",
        "Content-Type": "application/json"
    }

# === KUCOIN API HELPERS ===
def get_account_balance():
    endpoint = '/api/v1/account-overview?currency=USDT'
    url = API_BASE_URL + endpoint
    response = requests.get(url, headers=get_headers(endpoint))
    return float(response.json()['data']['availableBalance'])

def get_futures_symbols():
    endpoint = '/api/v1/contracts/active'
    url = API_BASE_URL + endpoint
    response = requests.get(url, headers=get_headers(endpoint))
    symbols = [item['symbol'] for item in response.json()['data'] if item['symbol'].endswith('USDTM')]
    return symbols

def get_candles(symbol, interval='1min', limit=100):
    endpoint = f"/api/v1/kline/query?symbol={symbol}&granularity=60&limit={limit}"
    url = API_BASE_URL + endpoint
    response = requests.get(url, headers=get_headers(endpoint))
    raw_data = response.json()['data']
    if not raw_data or not isinstance(raw_data[0], list):
        raise ValueError("Unexpected candle data format from API.")
    df = pd.DataFrame(raw_data, columns=['timestamp','open','high','low','close','volume'])
    df['close'] = pd.to_numeric(df['close'])
    return df

def calculate_indicators(df):
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def signal_from_indicators(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    print(f"Last close: {latest['close']:.6f}, EMA9: {latest['ema_fast']:.6f}, EMA21: {latest['ema_slow']:.6f}, RSI: {latest['rsi']:.2f}")
    if prev['ema_fast'] < prev['ema_slow'] and latest['ema_fast'] > latest['ema_slow'] and latest['rsi'] < RSI_OVERBOUGHT:
        return 'buy'
    elif prev['ema_fast'] > prev['ema_slow'] and latest['ema_fast'] < latest['ema_slow'] and latest['rsi'] > RSI_OVERSOLD:
        return 'sell'
    return None

def place_market_order(symbol, side, size):
    endpoint = '/api/v1/orders'
    url = API_BASE_URL + endpoint
    client_oid = str(uuid.uuid4())
    data = {
        "clientOid": client_oid,
        "symbol": symbol,
        "side": side,
        "type": "market",
        "size": size,
        "leverage": str(LEVERAGE)
    }
    response = requests.post(
        url,
        headers=get_headers(endpoint, 'POST', json.dumps(data)),
        data=json.dumps(data)
    )
    result = response.json()
    print(f"Order response for {symbol}: {json.dumps(result, indent=2)}")
    if not result.get('success', False):
        print(f"❌ Failed to place order on {symbol}: {result.get('code')} - {result.get('msg')}")
        return None
    return result.get('data', {}).get('orderId', client_oid)

def get_open_positions():
    endpoint = '/api/v1/positions'
    url = API_BASE_URL + endpoint
    response = requests.get(url, headers=get_headers(endpoint))
    return response.json().get('data', [])

def close_position(symbol, side):
    closing_side = 'sell' if side == 'buy' else 'buy'
    endpoint = '/api/v1/orders'
    url = API_BASE_URL + endpoint
    data = {
        "symbol": symbol,
        "side": closing_side,
        "type": "market",
        "closeOrder": True
    }
    response = requests.post(url, headers=get_headers(endpoint, 'POST', json.dumps(data)), data=json.dumps(data))
    return response.json()

def monitor_and_manage_positions():
    positions = get_open_positions()
    for pos in positions:
        symbol = pos['symbol']
        entry_price = float(pos.get('avgEntryPrice', 0))
        size = float(pos.get('currentQty', 0))
        side = 'buy' if size > 0 else 'sell'
        unrealized_pnl = float(pos.get('unrealisedPnl', 0))
        print(f"[MONITOR] {symbol} | Side: {side} | Entry: {entry_price} | PnL: {unrealized_pnl:.4f}")
        if unrealized_pnl >= TAKE_PROFIT_PERCENT / 100 * abs(entry_price * size):
            print(f"✅ TP Hit: Closing {symbol}")
            close_position(symbol, side)
        elif unrealized_pnl <= -STOP_LOSS_PERCENT / 100 * abs(entry_price * size):
            print(f"❌ SL Hit: Closing {symbol}")
            close_position(symbol, side)

def log_trade(data):
    file_exists = os.path.isfile(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# === MAIN LOGIC ===
def run_scalper():
    print("Starting KuCoin EMA+RSI Signal-Based Scalper Bot")
    balance = get_account_balance()
    print(f"Initial balance: {balance:.6f} USDT")

    if balance < 1:
        print("❗ Insufficient balance in Futures account. Please fund it before trading.")
        return

    trade_num = 1
    symbols = get_futures_symbols()
    print(f"Fetched {len(symbols)} tradable USDTM futures symbols.")

    while trade_num <= MAX_TRADES:
        trade_made = False
        monitor_and_manage_positions()
        for symbol in symbols:
            print(f"\nChecking {symbol} for signals...")
            try:
                df = get_candles(symbol)
                df = calculate_indicators(df)
                signal = signal_from_indicators(df)
            except Exception as e:
                print(f"⚠️ Skipping {symbol} due to error: {e}")
                continue

            if not signal:
                print(f"No signal on {symbol}.")
                continue

            print(f"=== TRADE #{trade_num} on {symbol} SIGNAL: {signal.upper()} ===")
            risk_capital = balance * (RISK_PERCENT / 100)
            tp_target = risk_capital * (TAKE_PROFIT_PERCENT / 100)
            sl_target = risk_capital * (STOP_LOSS_PERCENT / 100)
            trailing_stop = risk_capital * (TRAILING_STOP_PERCENT / 100)
            print(f"Risk Capital: {risk_capital:.4f}, TP: {tp_target:.4f}, SL: {sl_target:.4f}, Trailing SL: {trailing_stop:.4f}")

            order_size = round((risk_capital * LEVERAGE) / df.iloc[-1]['close'], 2)
            order_id = place_market_order(symbol, signal, order_size)
            if order_id:
                print(f"📤 Placed {signal.upper()} order on {symbol} | Size: {order_size} | Order ID: {order_id}")
            else:
                print(f"⚠️ Order on {symbol} was not successful. Skipping balance refresh.")

            log_trade({
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'signal': signal,
                'order_id': order_id or 'N/A',
                'risk_capital': round(risk_capital, 4),
                'take_profit': round(tp_target, 4),
                'stop_loss': round(sl_target, 4),
                'trailing_stop': round(trailing_stop, 4),
                'balance': round(balance, 4)
            })

            trade_num += 1
            trade_made = True

            if balance <= 0 or trade_num > MAX_TRADES:
                print("🚫 Ending session. Balance depleted or max trades hit.")
                return

        if not trade_made:
            print("No trades executed this round. Waiting 60 seconds...")
            time.sleep(60)

if __name__ == '__main__':
    run_scalper()
