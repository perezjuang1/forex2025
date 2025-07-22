import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from datetime import datetime
from backports.zoneinfo import ZoneInfo
from ConfigurationOperation import ConfigurationOperation
from Price import Price

class BacktestRunner:
    """
    Clase para ejecutar backtesting sobre datos históricos usando la estrategia actual.
    """
    def __init__(self, instrument, timeframe, days, price_file=None):
        self.config = ConfigurationOperation()
        self.instrument = instrument
        self.timeframe = timeframe
        self.days = days
        self.price_file = price_file or f"{instrument.replace('/', '_')}_{timeframe}.csv"
        self.robot = Price(days, instrument, timeframe)

    def load_data(self):
        """
        Carga los datos históricos desde un archivo CSV.
        """
        if not os.path.exists(self.price_file):
            raise FileNotFoundError(f"Archivo de precios no encontrado: {self.price_file}")
        df = pd.read_csv(self.price_file)
        return df

    def run_backtest(self):
        """
        Ejecuta el backtesting sobre los datos cargados y simula las señales y operaciones.
        """
        df = self.load_data()
        df = self.robot.set_indicators(df)
        # Simulación de operaciones
        trades = []
        position = None
        for i in range(len(df)):
            signal = df[self.config.signal_col].iloc[i]
            price = df['bidclose'].iloc[i]
            date = df['date'].iloc[i]
            if signal == self.robot.SIGNAL_BUY:
                if position != 'BUY':
                    trades.append({'type': 'BUY', 'date': date, 'price': price})
                    position = 'BUY'
            elif signal == self.robot.SIGNAL_SELL:
                if position != 'SELL':
                    trades.append({'type': 'SELL', 'date': date, 'price': price})
                    position = 'SELL'
        return trades

    def summary(self, trades):
        """
        Genera un resumen de las operaciones simuladas.
        """
        print(f"Total operaciones: {len(trades)}")
        for t in trades:
            print(f"{t['type']} | Fecha: {t['date']} | Precio: {t['price']}")

if __name__ == "__main__":
    # Ejemplo de uso
    instrument = "EUR/USD"
    timeframe = "m1"
    days = 5
    runner = BacktestRunner(instrument, timeframe, days)
    trades = runner.run_backtest()
    runner.summary(trades)
