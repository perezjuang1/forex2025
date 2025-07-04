import pandas as pd


from datetime import datetime
from backports.zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import signal
import datetime as dt
from ConnectionFxcm import RobotConnection
import time
import logging
import os

class RobotPrice:
    """
    Clase principal para manejo de precios, indicadores y operaciones de trading.
    """

    def __init__(self, days: int, instrument: str, timeframe: str):
        """
        Inicializa el objeto RobotPrice.
        """
        self.instrument = instrument
        self.timeframe = timeframe
        self.pricedata = None
        self.days = days
        self.robotconnection = RobotConnection()
        self.connection = self.robotconnection.getConnection()
        self._setup_logging()

    def _setup_logging(self):
        """
        Configura el sistema de logging.
        """
        if not os.path.exists('logs'):
            os.makedirs('logs')
        self.logger = logging.getLogger('RobotPrice')
        self.logger.setLevel(logging.INFO)
        log_file = f'logs/robot_price_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s\n%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _log_message(self, message: str, level: str = 'info'):
        """
        Helper para logging con formato especial.
        """
        if isinstance(message, str):
            message = message.encode('utf-8').decode('utf-8')
        replacements = {
            '•': '*', 'ó': 'o', 'ñ': 'n', 'á': 'a', 'é': 'e', 'í': 'i', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ñ': 'N'
        }
        for old, new in replacements.items():
            message = message.replace(old, new)
        if level == 'info':
            self.logger.info(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'warning':
            self.logger.warning(message)

    def get_price_data(self, instrument: str, timeframe: str, days: int, connection) -> pd.DataFrame:
        """
        Obtiene y guarda los datos de precio, calcula indicadores y señales.
        """
        europe_London_datetime = datetime.now(ZoneInfo('Europe/London'))
        date_from = europe_London_datetime - dt.timedelta(days=days)
        date_to = europe_London_datetime
        history = connection.get_history(instrument, timeframe, date_from, date_to)
        current_unit, _ = connection.parse_timeframe(timeframe)
        self._log_message(
            f"DATOS PRECIO: Unidad={current_unit}, Timeframe={timeframe}, Instrumento={instrument}, Fecha={europe_London_datetime}"
        )
        pricedata = pd.DataFrame(history, columns=["Date", "BidOpen", "BidHigh", "BidLow", "BidClose", "Volume"])
        d = {
            'date': pricedata['Date'],
            'bidhigh': pricedata['BidHigh'],
            'bidlow': pricedata['BidLow'],
            'bidclose': pricedata['BidClose'],
            'bidopen': pricedata['BidOpen'],
            'tickqty': pricedata['Volume']
        }
        df = pd.DataFrame(data=d)
        df['timeframe'] = timeframe
        df['date'] = df['date'].astype(str).str.replace('-', '').str.replace(':', '').str.replace(' ', '').str[:-2]
        df['date'] = df['date'].apply(lambda x: int(x))
        self.pricedata = self.set_indicators(df)
        self.save_price_data_file(self.pricedata)
        return self.pricedata

    def save_price_data_file(self, pricedata: pd.DataFrame):
        """
        Guarda el DataFrame de precios en un archivo CSV y lo registra.
        """
        fileName = self.instrument.replace("/", "_") + "_" + self.timeframe + ".csv"
        pricedata.to_csv(fileName)
        self._log_message(f"Archivo guardado: {fileName}")

    def set_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula y agrega indicadores técnicos al DataFrame.
        """
        df = self.calculate_peaks(df)
        df = self.calculate_trend(df)
        df['ema'] = df['bidclose'].ewm(span=50).mean()
        df['ema_slow'] = df['bidclose'].ewm(span=100).mean()
        df['MediaPositionSell'] = np.where((df['ema'] < df['ema_slow']), 1, 0)
        df['MediaPositionBuy'] = np.where((df['ema'] > df['ema_slow']), 1, 0)
        df['rsi'] = self.calculate_rsi(df, 'bidclose')
        df['volume_ma'] = df['tickqty'].rolling(window=20).mean()
        df = self.apply_triggers_strategy(df, 'buy')
        df = self.apply_triggers_strategy(df, 'sell')
        self.evaluate_triggers_signals(df)
        return df

    def calculate_rsi(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Calcula el RSI estándar para una columna dada (14 periodos, método clásico).
        """
        rsi_window = 14
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_peaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta picos mínimos y máximos en los precios.
        """
        df['value1'] = 1
        df['peaks_min'] = df.iloc[signal.argrelextrema(df['bidclose'].values, np.less, order=30)[0]]['value1']
        df['peaks_max'] = df.iloc[signal.argrelextrema(df['bidclose'].values, np.greater, order=30)[0]]['value1']
        return df

    def calculate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la tendencia del precio usando picos y volumen.
        """
        df['trend'] = 0
        df['trend_line'] = df['bidclose']
        window_size = 7
        df['volume_ma'] = df['tickqty'].rolling(window=20).mean()
        df['high_low_range'] = df['bidhigh'] - df['bidlow']
        df['volatility'] = df['high_low_range'].rolling(window=14).mean()
        peak_min_indices = df[df['peaks_min'] == 1].index
        peak_max_indices = df[df['peaks_max'] == 1].index
        all_peaks = sorted(list(peak_min_indices) + list(peak_max_indices))
        for i in range(len(all_peaks) - window_size + 1):
            window_peaks = all_peaks[i:i + window_size]
            window_data = df.loc[window_peaks]
            if len(window_peaks) >= 2:
                x = np.arange(len(window_peaks))
                y = window_data['bidclose'].values
                slope, _ = np.polyfit(x, y, 1)
            else:
                slope = 0
            higher_highs = sum(1 for j in range(1, len(window_peaks)) if window_data.loc[window_peaks[j], 'bidhigh'] > window_data.loc[window_peaks[j-1], 'bidhigh'])
            higher_lows = sum(1 for j in range(1, len(window_peaks)) if window_data.loc[window_peaks[j], 'bidlow'] > window_data.loc[window_peaks[j-1], 'bidlow'])
            lower_highs = sum(1 for j in range(1, len(window_peaks)) if window_data.loc[window_peaks[j], 'bidhigh'] < window_data.loc[window_peaks[j-1], 'bidhigh'])
            lower_lows = sum(1 for j in range(1, len(window_peaks)) if window_data.loc[window_peaks[j], 'bidlow'] < window_data.loc[window_peaks[j-1], 'bidlow'])
            volume_increasing = window_data['tickqty'].mean() > window_data['volume_ma'].mean()
            current_volatility = window_data['volatility'].iloc[-1]
            avg_volatility = df['volatility'].mean()
            volatility_ok = current_volatility <= avg_volatility * 1.5
            if (higher_highs >= 4 and higher_lows >= 4 and slope > 0 and volume_increasing and volatility_ok):
                df.loc[window_peaks, 'trend'] = 1
                df.loc[window_peaks, 'trend_line'] = df.loc[window_peaks, 'bidlow'].rolling(window=2).min()
            elif (lower_highs >= 4 and lower_lows >= 4 and slope < 0 and volume_increasing and volatility_ok):
                df.loc[window_peaks, 'trend'] = -1
                df.loc[window_peaks, 'trend_line'] = df.loc[window_peaks, 'bidhigh'].rolling(window=2).max()
            else:
                df.loc[window_peaks, 'trend'] = 0
                df.loc[window_peaks, 'trend_line'] = (df.loc[window_peaks, 'bidhigh'] + df.loc[window_peaks, 'bidlow']) / 2
        df['trend_line'] = df['trend_line'].fillna(method='ffill')
        return df

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula el Average True Range (ATR) para medir volatilidad.
        """
        high = df['bidhigh']
        low = df['bidlow']
        close = df['bidclose']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def apply_triggers_strategy(self, df: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        """
        Aplica la estrategia de triggers para buy/sell.
        """
        SIGNAL_OPEN = 1
        SIGNAL_CLOSE = -1
        SIGNAL_NEUTRAL = 0
        signal_column = strategy_type
        df[signal_column] = SIGNAL_NEUTRAL
        open_condition = 'peaks_min' if strategy_type == 'buy' else 'peaks_max'
        close_condition = 'peaks_max' if strategy_type == 'buy' else 'peaks_min'
        df['atr'] = self.calculate_atr(df, period=14)
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        df['relative_volatility'] = df['atr'] / df['atr_ma']
        is_position_open = False
        for i in range(len(df)):
            current_volatility = df['relative_volatility'].iloc[i]
            current_rsi = df['rsi'].iloc[i]
            is_peak = df[open_condition].iloc[i] == 1
            if not is_position_open and is_peak:
                if 0.5 <= current_volatility <= 1.5:
                    if 25 < current_rsi < 75:
                        is_position_open = True
                        df.iloc[i, df.columns.get_loc(signal_column)] = SIGNAL_OPEN
                    else:
                        pass
                else:
                    pass
            elif is_position_open and df[close_condition].iloc[i] == 1:
                is_position_open = False
                df.iloc[i, df.columns.get_loc(signal_column)] = SIGNAL_CLOSE
            else:
                pass
        return df

    def evaluate_triggers_signals(self, df: pd.DataFrame):
        """
        Evalúa las señales generadas por los triggers y ejecuta operaciones si corresponde, excluyendo la última vela (en formación).
        """
        try:
            recent_rows = df.iloc[-7:-1]  # Excluye la última vela
            current_price = df['bidclose'].iloc[-2]  # Usar penúltima vela
            current_rsi = df['rsi'].iloc[-2]
            buy_signals = recent_rows[recent_rows['buy'] == 1]
            sell_signals = recent_rows[recent_rows['sell'] == 1]
            has_buy_signal = not buy_signals.empty
            has_sell_signal = not sell_signals.empty
            if has_buy_signal:
                if self.existingOperation(instrument=self.instrument, BuySell="S"):
                    self.CloseOperation(instrument=self.instrument, BuySell="S")
                if not self.existingOperation(instrument=self.instrument, BuySell="B"):
                    self.createEntryOrder(str_buy_sell="B")
                else:
                    pass
            if has_sell_signal:
                if self.existingOperation(instrument=self.instrument, BuySell="B"):
                    self.CloseOperation(instrument=self.instrument, BuySell="B")
                if not self.existingOperation(instrument=self.instrument, BuySell="S"):
                    self.createEntryOrder(str_buy_sell="S")
                else:
                    pass
            if not has_buy_signal and not has_sell_signal:
                pass
        except Exception as e:
            pass

    def existingOperation(self, instrument: str, BuySell: str) -> bool:
        """
        Verifica si existe una operación abierta para el instrumento y tipo Buy/Sell.
        """
        existOperation = False
        try:
            trades_table = self.connection.get_table(self.connection.TRADES)
            for trade_row in trades_table:
                if BuySell == trade_row.BuySell:
                    existOperation = True
            return existOperation
        except Exception as e:
            return existOperation

    def CloseOperation(self, instrument: str, BuySell: str):
        """
        Cierra una operación existente para el instrumento y tipo Buy/Sell.
        """
        try:
            accounts_response_reader = self.connection.get_table_reader(self.connection.ACCOUNTS)
            accountId = None
            for account in accounts_response_reader:
                accountId = account.account_id
            orders_table = self.connection.get_table(self.connection.TRADES)
            for trade in orders_table:
                buy_sell = ''
                if trade.instrument == instrument and trade.buy_sell == BuySell:
                    buy_sell = self.corepy.Constants.SELL if trade.buy_sell == self.corepy.Constants.BUY else self.corepy.Constants.BUY
                    if buy_sell != None:
                        request = self.connection.create_order_request(
                            order_type=self.corepy.Constants.Orders.TRUE_MARKET_CLOSE,
                            OFFER_ID=trade.offer_id,
                            ACCOUNT_ID=accountId,
                            BUY_SELL=buy_sell,
                            AMOUNT=trade.amount,
                            TRADE_ID=trade.trade_id
                        )
                        self.connection.send_request_async(request)
                        self._log_message(f"Operacion CERRADA: Instrumento={instrument}, Tipo={BuySell}, Monto={trade.amount}, TradeID={trade.trade_id}")
                else:
                    pass
        except Exception as e:
            self._log_message(f"Error al cerrar operacion: {e}", level='error')

    def createEntryOrder(self, str_buy_sell: str = None):
        """
        Crea una orden de entrada (compra o venta) para el instrumento configurado.
        """
        args = self.robotconnection.args
        common = self.robotconnection.common
        fxcorepy = self.robotconnection.fxcorepy
        str_instrument = self.instrument
        str_lots = args.lots
        str_account = args.account
        stop = args.stop
        peggedstop = args.peggedstop
        pegstoptype = args.pegstoptype
        limit = args.limit
        peggedlimit = args.peggedlimit
        peglimittype = args.peglimittype
        if peggedstop:
            if not pegstoptype:
                return
            if pegstoptype != 'O' and pegstoptype != 'M':
                return
            peggedstop = peggedstop.lower()
            if peggedstop != 'y':
                peggedstop = None
        if pegstoptype:
            pegstoptype = pegstoptype.upper()
        if peggedlimit:
            if not peglimittype:
                return
            if peglimittype != 'O' and peglimittype != 'M':
                return
            peggedlimit = peggedlimit.lower()
            if peggedlimit != 'y':
                peggedlimit = None
        if peglimittype:
            peglimittype = peglimittype.upper()
        try:
            account = common.get_account(self.connection, str_account)
            if not account:
                raise Exception("The account '{0}' is not valid".format(str_account))
            else:
                str_account = account.account_id
            offer = common.get_offer(self.connection, str_instrument)
            if offer is None:
                raise Exception("The instrument '{0}' is not valid".format(str_instrument))
            login_rules = self.connection.login_rules
            trading_settings_provider = login_rules.trading_settings_provider
            base_unit_size = trading_settings_provider.get_base_unit_size(str_instrument, account)
            amount = base_unit_size * str_lots
            entry = fxcorepy.Constants.Orders.TRUE_MARKET_OPEN
            if str_buy_sell == 'B':
                stopv = -stop
                limitv = limit
                str_buy_sell = fxcorepy.Constants.BUY
            else:
                stopv = stop
                limitv = -limit
                str_buy_sell = fxcorepy.Constants.SELL
            if peggedstop:
                if peggedlimit:
                    request = self.connection.create_order_request(
                        order_type=entry,
                        OFFER_ID=offer.offer_id,
                        ACCOUNT_ID=str_account,
                        BUY_SELL=str_buy_sell,
                        PEG_TYPE_STOP=pegstoptype,
                        PEG_OFFSET_STOP=stopv,
                        PEG_TYPE_LIMIT=peglimittype,
                        PEG_OFFSET_LIMIT=limitv,
                        AMOUNT=amount,
                    )
                else:
                    request = self.connection.create_order_request(
                        order_type=entry,
                        OFFER_ID=offer.offer_id,
                        ACCOUNT_ID=str_account,
                        BUY_SELL=str_buy_sell,
                        PEG_TYPE_STOP=pegstoptype,
                        PEG_OFFSET_STOP=stopv,
                        RATE_LIMIT=limit,
                        AMOUNT=amount,
                    )
            else:
                if peggedlimit:
                    request = self.connection.create_order_request(
                        order_type=entry,
                        OFFER_ID=offer.offer_id,
                        ACCOUNT_ID=str_account,
                        BUY_SELL=str_buy_sell,
                        RATE_STOP=stop,
                        PEG_TYPE_LIMIT=peglimittype,
                        PEG_OFFSET_LIMIT=limitv,
                        AMOUNT=amount,
                    )
                else:
                    request = self.connection.create_order_request(
                        order_type=entry,
                        OFFER_ID=offer.offer_id,
                        ACCOUNT_ID=str_account,
                        BUY_SELL=str_buy_sell,
                        AMOUNT=amount,
                        RATE_STOP=stop,
                        RATE_LIMIT=limit,
                    )
            self.connection.send_request_async(request)
            self._log_message(f"Operacion ABIERTA: Instrumento={str_instrument}, Tipo={'BUY' if str_buy_sell == fxcorepy.Constants.BUY else 'SELL'}, Monto={amount}, Stop={stop}, Limit={limit}")
        except Exception as e:
            self._log_message(f"Error al abrir operacion: {e}", level='error')