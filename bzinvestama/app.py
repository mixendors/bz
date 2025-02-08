from flask import Flask, render_template, request, flash
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

def get_data(symbol, period="1mo", interval="1h"):
    """
    Ambil data historis menggunakan yfinance.
    Gunakan periode 1 bulan dengan interval 1 jam agar didapatkan data yang cukup untuk indikator.
    """
    try:
        data = yf.download(symbol, period=period, interval=interval)
        return data if not data.empty else None
    except Exception as e:
        print(f"[!] Error fetching data for {symbol}: {e}")
        return None

def calculate_trade_levels(data):
    """
    Hitung indikator teknikal dan level-level trade:
      - MA50 sebagai filter tren
      - RSI (14) untuk mengukur momentum
      - MACD (EMA12, EMA26, dan sinyal EMA9)
      - ATR (14) untuk mengukur volatilitas
      - Swing High/Low (dari 20 data terakhir) untuk menentukan level Fibonacci
      - Fibonacci retracement (38.2% dan 61.8%) serta extension sebagai level TP
    
    Syarat sinyal bullish (Buy) adalah:
      - Harga saat ini di atas MA50 (tren naik)
      - RSI di bawah 40 (oversold dalam konteks uptrend)
      - MACD berada di atas garis sinyal
    """
    # Pastikan data mencukupi (minimal 50 data agar MA50 valid)
    if len(data) < 50:
        return None

    close = data['Close']
    high = data['High']
    low = data['Low']

    # 1. MA50 (konversi ke float)
    ma50 = float(close.rolling(window=50).mean().iloc[-1])

    # 2. RSI periode 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = float(gain.rolling(window=14).mean().iloc[-1])
    avg_loss = float(loss.rolling(window=14).mean().iloc[-1])
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # 3. MACD: EMA12, EMA26, dan signal line (EMA9 dari MACD)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    # Pastikan nilai terakhir dikonversi ke float
    current_close = float(close.iloc[-1])
    macd_last = float(macd.iloc[-1])
    macd_signal_last = float(macd_signal.iloc[-1])

    # 4. ATR periode 14
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(window=14).mean().iloc[-1])

    # 5. Tentukan swing high dan swing low dari 20 data terakhir (konversi ke float)
    recent_data = data[-20:]
    swing_high = float(recent_data['High'].max())
    swing_low = float(recent_data['Low'].min())

    # 6. Hitung level Fibonacci retracement dan extension
    fib_382 = swing_low + 0.382 * (swing_high - swing_low)
    fib_618 = swing_low + 0.618 * (swing_high - swing_low)
    fib_ext = swing_high + 0.618 * (swing_high - swing_low)

    # 7. Syarat sinyal trade (bullish) dengan nilai yang sudah berupa float:
    #    - Harga saat ini > MA50
    #    - RSI < 40 (oversold dalam konteks tren naik)
    #    - MACD > MACD Signal
    trade_signal = "Buy" if (current_close > ma50 and rsi < 50 and macd_last > macd_signal_last) else "No Trade"

    if trade_signal == "Buy":
        # Entry disarankan di level Fibonacci 38.2 retracement
        entry = fib_382
        # Stop loss diletakkan di bawah swing low dengan margin 1.5 x ATR
        stop_loss = swing_low - 1.5 * atr
        # TP1 menggunakan risk:reward minimal 1:2
        tp1 = entry + 2 * (entry - stop_loss)
        # TP2 menggunakan level Fibonacci extension
        tp2 = fib_ext
        # TP3 diambil mendekati swing high
        tp3 = swing_high
    else:
        # Jika tidak memenuhi syarat, semua level disamakan dengan harga saat ini (tidak ada sinyal trade)
        entry = current_close
        stop_loss = current_close
        tp1 = current_close
        tp2 = current_close
        tp3 = current_close

    levels = {
        'Entry': float(entry),
        'Stop Loss': float(stop_loss),
        'TP1': float(tp1),
        'TP2': float(tp2),
        'TP3': float(tp3),
        'MA50': ma50,
        'RSI': float(rsi),
        'MACD': macd_last,
        'MACD_Signal': macd_signal_last,
        'ATR': atr,
        'Swing High': swing_high,
        'Swing Low': swing_low,
        'Fib_382': float(fib_382),
        'Fib_618': float(fib_618),
        'Fib_Ext': float(fib_ext),
        'Current Close': current_close,
        'Trade Signal': trade_signal
    }
    return levels

def analyze_stock(symbol):
    """
    Analisis saham dengan mengambil data historis, menghitung indikator dan level trade,
    lalu menyusun rekomendasi.
    """
    data = get_data(symbol)
    if data is None or len(data) < 50:
        return None

    levels = calculate_trade_levels(data)
    if levels is None:
        return None

    recommendation = {
        'Symbol': symbol,
        'Close': float(levels['Current Close']),
        'Entry': float(levels['Entry']),
        'Stop Loss': float(levels['Stop Loss']),
        'TP1': float(levels['TP1']),
        'TP2': float(levels['TP2']),
        'TP3': float(levels['TP3']),
        'Trade Signal': levels['Trade Signal'],
        'Indicators': {
            'MA50': float(levels['MA50']),
            'RSI': float(levels['RSI']),
            'MACD': float(levels['MACD']),
            'MACD_Signal': float(levels['MACD_Signal']),
            'ATR': float(levels['ATR']),
            'Swing High': float(levels['Swing High']),
            'Swing Low': float(levels['Swing Low']),
            'Fib_382': float(levels['Fib_382']),
            'Fib_618': float(levels['Fib_618']),
            'Fib_Ext': float(levels['Fib_Ext'])
        }
    }
    return recommendation

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper().strip()
        if not symbol:
            flash('Masukkan kode saham terlebih dahulu')
            return render_template('base.html')
        
        if not symbol.endswith('.JK'):
            symbol += '.JK'
            
        rec = analyze_stock(symbol)
        if rec:
            return render_template('base.html', recommendation=rec)
        else:
            flash(f'Data {symbol} tidak valid atau tidak cukup untuk analisis')
            return render_template('base.html')
    
    return render_template('base.html')

import csv

def get_candidates_from_csv(filepath):
    """
    Membaca file CSV dan mengembalikan daftar ticker.
    Pastikan file CSV memiliki header "Ticker".
    Jika ticker belum mengandung ".JK", maka akan ditambahkan.
    """
    candidates = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Mencari key 'Ticker' atau 'ticker'
            ticker = row.get('Ticker') or row.get('ticker')
            if ticker is None:
                continue
            ticker = ticker.strip()
            if not ticker.endswith('.JK'):
                ticker = ticker + '.JK'
            candidates.append(ticker)
    return candidates

def scan_stocks():
    """
    Memindai daftar saham kandidat dari file CSV dan mengembalikan 5 saham teratas
    yang memiliki sinyal "Buy" (menggunakan rumusan perhitungan buy yang sama).
    
    Hanya menampilkan:
      - Kode Saham
      - Harga Penutupan (Close)
      - Harga Entry
      - Swing High
      - Swing Low
    """
    csv_path = "daftar_saham.csv"  # Pastikan file CSV ada di direktori yang sama
    candidates = get_candidates_from_csv(csv_path)
    results = []
    for symbol in candidates:
        rec = analyze_stock(symbol)
        if rec and rec['Trade Signal'] == "Buy":
            filtered = {
                'Symbol': rec['Symbol'],
                'Close': rec['Close'],
                'Entry': rec['Entry'],
                'Swing High': rec['Indicators']['Swing High'],
                'Swing Low': rec['Indicators']['Swing Low']
            }
            results.append(filtered)
    return results[:5]

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        scan_results = scan_stocks()
        return render_template('scan.html', results=scan_results)
    else:
        return render_template('scan.html')




if __name__ == '__main__':
    app.run(debug=True)
