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
        Calcula y agrega solo los indicadores necesarios al DataFrame.
        """
        df = self.calculate_peaks(df)
        df = self.calculate_trend(df)
        df['rsi'] = self.calculate_rsi(df, 'bidclose')
        df = self.apply_triggers_strategy(df, 'buy')
        df = self.apply_triggers_strategy(df, 'sell')
        self.triggers_trades(df)
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
        df['peaks_min'] = df.iloc[signal.argrelextrema(df['bidclose'].values, np.less, order=5)[0]]['value1']
        df['peaks_max'] = df.iloc[signal.argrelextrema(df['bidclose'].values, np.greater, order=5)[0]]['value1']
        return df

    def calculate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la tendencia del precio usando picos y volumen - VERSIÓN ULTRA SIMPLIFICADA.
        """
        df['trend'] = 0
        df['trend_line'] = df['bidclose']
        
        # Calcular tendencia simple basada en EMA
        df['ema_short'] = df['bidclose'].ewm(span=10).mean()
        df['ema_long'] = df['bidclose'].ewm(span=20).mean()
        
        # Tendencia basada en EMA cruzada
        for i in range(1, len(df)):
            if df['ema_short'].iloc[i] > df['ema_long'].iloc[i] and df['ema_short'].iloc[i-1] <= df['ema_long'].iloc[i-1]:
                df.loc[df.index[i:], 'trend'] = 1  # Tendencia alcista
            elif df['ema_short'].iloc[i] < df['ema_long'].iloc[i] and df['ema_short'].iloc[i-1] >= df['ema_long'].iloc[i-1]:
                df.loc[df.index[i:], 'trend'] = -1  # Tendencia bajista
            # Si no hay cruce, mantener la tendencia anterior
        
        # Línea de tendencia simple
        df['trend_line'] = df['bidclose'].rolling(window=5).mean()
        
        return df

    def apply_triggers_strategy(self, df: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        """
        Aplica la estrategia de triggers para buy/sell incluyendo análisis de tendencia.
        """
        SIGNAL_OPEN = 1
        SIGNAL_CLOSE = -1
        SIGNAL_NEUTRAL = 0
        signal_column = strategy_type
        df[signal_column] = SIGNAL_NEUTRAL
        open_condition = 'peaks_min' if strategy_type == 'buy' else 'peaks_max'
        close_condition = 'peaks_max' if strategy_type == 'buy' else 'peaks_min'
        is_position_open = False
        for i in range(len(df)):
            current_rsi = df['rsi'].iloc[i]
            current_trend = df['trend'].iloc[i]
            is_peak = df[open_condition].iloc[i] == 1
            if not is_position_open and is_peak:
                # SOLO VERIFICAR RSI Y QUE SEA UN PICO
                if strategy_type == 'buy':
                    rsi_ok = 10 < current_rsi < 90
                else:
                    rsi_ok = 10 < current_rsi < 90
                if rsi_ok:
                    is_position_open = True
                    df.iloc[i, df.columns.get_loc(signal_column)] = SIGNAL_OPEN
                    self._log_message(
                        f"Señal {strategy_type.upper()} generada - "
                        f"Tendencia: {current_trend} ({'Alcista' if current_trend == 1 else 'Bajista' if current_trend == -1 else 'Lateral'}), "
                        f"RSI: {current_rsi:.2f}"
                    )
                else:
                    self._log_message(
                        f"Pico detectado pero RSI fuera de rango ({current_rsi:.2f}) para {strategy_type.upper()}"
                    )
            elif is_position_open and df[close_condition].iloc[i] == 1:
                is_position_open = False
                df.iloc[i, df.columns.get_loc(signal_column)] = SIGNAL_CLOSE
                self._log_message(f"Señal de CIERRE {strategy_type.upper()} generada en pico opuesto")
        return df

    def triggers_trades(self, df: pd.DataFrame):
        """
        Evalúa las señales generadas por los triggers y ejecuta operaciones si corresponde, excluyendo la última vela (en formación).
        """
        try:
            recent_rows = df.iloc[-7:-1]  # Excluye la última vela
            buy_signals = recent_rows[recent_rows['buy'] == 1]
            sell_signals = recent_rows[recent_rows['sell'] == 1]
            has_buy_signal = not buy_signals.empty
            has_sell_signal = not sell_signals.empty
            # Definir variables para logs
            if has_buy_signal:
                last_buy = buy_signals.iloc[-1]
                buy_fecha = last_buy['date']
                buy_price = last_buy['bidclose']
                buy_trend = last_buy['trend']
                buy_trend_desc = 'Alcista' if buy_trend == 1 else 'Bajista' if buy_trend == -1 else 'Lateral'
            if has_sell_signal:
                last_sell = sell_signals.iloc[-1]
                sell_fecha = last_sell['date']
                sell_price = last_sell['bidclose']
                sell_trend = last_sell['trend']
                sell_trend_desc = 'Alcista' if sell_trend == 1 else 'Bajista' if sell_trend == -1 else 'Lateral'
            if has_buy_signal:
                if self.existingOperation(instrument=self.instrument, BuySell="S"):
                    self._log_message(
                        f"\n[CIERRE SELL]\n  Motivo: Señal de COMPRA detectada\n  Fecha: {buy_fecha}\n  Precio: {buy_price}\n  Tendencia: {buy_trend} ({buy_trend_desc})\n  Instrumento: {self.instrument}\n  Timeframe: {self.timeframe}\n"
                    )
                    self.CloseOperation(instrument=self.instrument, BuySell="S")
                if not self.existingOperation(instrument=self.instrument, BuySell="B"):
                    self._log_message(
                        f"\n[APERTURA BUY]\n  Motivo: Señal de COMPRA detectada\n  Fecha: {buy_fecha}\n  Precio: {buy_price}\n  Tendencia: {buy_trend} ({buy_trend_desc})\n  Instrumento: {self.instrument}\n  Timeframe: {self.timeframe}\n"
                    )
                    self.createEntryOrder(str_buy_sell="B")
                else:
                    self._log_message(
                        f"\n[INFO]\n  Ya existe operación BUY abierta\n  Fecha: {buy_fecha}\n  Precio: {buy_price}\n  Tendencia: {buy_trend} ({buy_trend_desc})\n  Instrumento: {self.instrument}\n  Timeframe: {self.timeframe}\n"
                    )
            if has_sell_signal:
                if self.existingOperation(instrument=self.instrument, BuySell="B"):
                    self._log_message(
                        f"\n[CIERRE BUY]\n  Motivo: Señal de VENTA detectada\n  Fecha: {sell_fecha}\n  Precio: {sell_price}\n  Tendencia: {sell_trend} ({sell_trend_desc})\n  Instrumento: {self.instrument}\n  Timeframe: {self.timeframe}\n"
                    )
                    self.CloseOperation(instrument=self.instrument, BuySell="B")
                if not self.existingOperation(instrument=self.instrument, BuySell="S"):
                    self._log_message(
                        f"\n[APERTURA SELL]\n  Motivo: Señal de VENTA detectada\n  Fecha: {sell_fecha}\n  Precio: {sell_price}\n  Tendencia: {sell_trend} ({sell_trend_desc})\n  Instrumento: {self.instrument}\n  Timeframe: {self.timeframe}\n"
                    )
                    self.createEntryOrder(str_buy_sell="S")
                else:
                    self._log_message(
                        f"\n[INFO]\n  Ya existe operación SELL abierta\n  Fecha: {sell_fecha}\n  Precio: {sell_price}\n  Tendencia: {sell_trend} ({sell_trend_desc})\n  Instrumento: {self.instrument}\n  Timeframe: {self.timeframe}\n"
                    )
            if not has_buy_signal and not has_sell_signal:
                self._log_message(
                    f"\n[INFO]\n  No se detectaron señales de compra ni venta en las últimas velas\n  Instrumento: {self.instrument}\n  Timeframe: {self.timeframe}\n"
                )
        except Exception as e:
            self._log_message(f"Error en evaluate_triggers_signals: {e}", level='error')

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