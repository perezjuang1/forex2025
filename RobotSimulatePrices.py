import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta

CSV_FILE = 'EUR_USD_m5_simulacion.csv'  # Ahora el archivo tiene 'simulacion' en el nombre
INSTRUMENT = 'EUR/USD'
INITIAL_PRICE = 1.1000
VOLATILITY = 0.0005  # Volatilidad estándar para el random walk
VOLUME_MEAN = 100
VOLUME_STD = 10
COLUMNS = ['Date', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose', 'Volume']


def generate_next_ohlcv(last_close):
    """
    Genera un nuevo registro OHLCV basado en el último precio de cierre.
    """
    open_price = last_close
    # Simula el movimiento de precios
    change = np.random.normal(0, VOLATILITY)
    close_price = open_price + change
    high_price = max(open_price, close_price) + abs(np.random.normal(0, VOLATILITY/2))
    low_price = min(open_price, close_price) - abs(np.random.normal(0, VOLATILITY/2))
    volume = int(np.random.normal(VOLUME_MEAN, VOLUME_STD))
    return open_price, high_price, low_price, close_price, volume


def get_last_valid_row(csv_file):
    if not os.path.exists(csv_file):
        return None
    try:
        df = pd.read_csv(csv_file, usecols=COLUMNS, dtype={
            'Date': str,
            'BidOpen': float,
            'BidHigh': float,
            'BidLow': float,
            'BidClose': float,
            'Volume': int
        }, on_bad_lines='skip')
        # Buscar la última fila válida (Date es string y BidClose es float)
        for i in range(len(df)-1, -1, -1):
            row = df.iloc[i]
            try:
                datetime.strptime(str(row['Date']), '%Y-%m-%d %H:%M:%S')
                float(row['BidClose'])
                return row
            except Exception:
                continue
        return None
    except Exception:
        return None


def main():
    print(f"Simulador de precios iniciado. Guardando en {CSV_FILE}")
    # Fecha de inicio: una semana atrás desde ahora
    start_time = datetime.now() - timedelta(days=7)
    # Si el archivo no existe, crea el encabezado y la primera fila
    if not os.path.exists(CSV_FILE):
        open_p, high_p, low_p, close_p, vol = generate_next_ohlcv(INITIAL_PRICE)
        data = {
            'Date': [start_time.strftime('%Y-%m-%d %H:%M:%S')],
            'BidOpen': [open_p],
            'BidHigh': [high_p],
            'BidLow': [low_p],
            'BidClose': [close_p],
            'Volume': [vol]
        }
        df = pd.DataFrame(data, columns=COLUMNS)
        df.to_csv(CSV_FILE, index=False)
        last_close = close_p
        last_time = start_time
    else:
        last_row = get_last_valid_row(CSV_FILE)
        if last_row is not None:
            last_close = float(last_row['BidClose'])
            last_time = datetime.strptime(str(last_row['Date']), '%Y-%m-%d %H:%M:%S')
        else:
            last_close = INITIAL_PRICE
            last_time = start_time

    while True:
        # Avanza un minuto desde la última fecha
        last_time = last_time + timedelta(minutes=1)
        open_p, high_p, low_p, close_p, vol = generate_next_ohlcv(last_close)
        new_row = {
            'Date': last_time.strftime('%Y-%m-%d %H:%M:%S'),
            'BidOpen': open_p,
            'BidHigh': high_p,
            'BidLow': low_p,
            'BidClose': close_p,
            'Volume': vol
        }
        # Agrega al CSV
        df_new = pd.DataFrame([new_row], columns=COLUMNS)
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
        print(f"Nuevo precio generado: {new_row}")
        last_close = close_p
        # Espera hasta el siguiente minuto real
        time.sleep(60)

if __name__ == '__main__':
    main() 