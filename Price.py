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
from ConfigurationOperation import ConfigurationOperation

class RobotPrice:
    """
    Clase principal para manejo de precios, indicadores y operaciones de trading.
    """

    # Constantes de señal globales para la clase
    SIGNAL_BUY = 1
    SIGNAL_SELL = -1
    SIGNAL_NEUTRAL = 0

    # Constantes de tendencia globales para la clase
    TREND_UP = 1         # Tendencia alcista
    TREND_DOWN = -1      # Tendencia bajista
    TREND_FLAT = 0       # Sin cambio
    TREND_NA = np.nan    # No calculado

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

    def _create_log_handlers(self, log_file):
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        return file_handler, console_handler

    def _setup_logging(self):
        """
        Configura el sistema de logging para que cada instrumento tenga su propio archivo de log.
        """
        if not os.path.exists('logs'):
            os.makedirs('logs')
        self.logger = logging.getLogger(f'RobotPrice_{self.instrument}')
        self.logger.setLevel(logging.INFO)
        log_file = f'logs/robot_price_{self.instrument.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.log'
        if not self.logger.handlers:
            file_handler, console_handler = self._create_log_handlers(log_file)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _normalize_log_message(self, message: str) -> str:
        replacements = {
            '•': '*', 'ó': 'o', 'ñ': 'n', 'á': 'a', 'é': 'e', 'í': 'i', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ñ': 'N'
        }
        for old, new in replacements.items():
            message = message.replace(old, new)
        return message

    def _log_message(self, message: str, level: str = 'info'):
        if isinstance(message, str):
            message = message.encode('utf-8').decode('utf-8')
        message = self._normalize_log_message(message)
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
        # Verificar si existe operación abierta para el instrumento actual (BUY y SELL)
        exists_buy = self.existingOperation(instrument, 'B')
        exists_sell = self.existingOperation(instrument, 'S')
        print(f"DEBUG: Existe operación BUY para {instrument}: {exists_buy}")
        print(f"DEBUG: Existe operación SELL para {instrument}: {exists_sell}")
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

    def set_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula y agrega solo los indicadores necesarios al DataFrame (solo picos y confluencias).
        """
        df = self.calculate_peaks(df) 
        df = self.apply_triggers_strategy(df)
        self.triggers_trades_close(df)
        self.triggers_trades_open(df)
        return df




    def calculate_peaks(self, df: pd.DataFrame, order: int = 100, tolerance: int = None) -> pd.DataFrame:
        if tolerance is None:
            tolerance = ConfigurationOperation.tolerance_peaks
        df['value1'] = 1
        self._add_price_peaks(df, order)
        self._add_ema_peaks(df, order)
        self._add_ema_crosses(df)
        self._mark_trade_open_zone(df, order, tolerance)
        df.drop(['ema10_above_ema30', 'ema10_above_ema30_shift'], axis=1, inplace=True)
        return df

    def _add_price_peaks(self, df, order):
        peaks_min_idx = signal.argrelextrema(df['bidclose'].values, np.less, order=order)[0]
        peaks_max_idx = signal.argrelextrema(df['bidclose'].values, np.greater, order=order)[0]
        df['peaks_min'] = 0
        df['peaks_max'] = 0
        df.loc[peaks_min_idx, 'peaks_min'] = 1
        df.loc[peaks_max_idx, 'peaks_max'] = 1

    def _add_ema_peaks(self, df, order):
        df['ema_10'] = df['bidclose'].ewm(span=10, adjust=False).mean()
        df['ema_30'] = df['bidclose'].ewm(span=30, adjust=False).mean()
        peaks_ema10_min = signal.argrelextrema(df['ema_10'].values, np.less, order=order)[0]
        peaks_ema10_max = signal.argrelextrema(df['ema_10'].values, np.greater, order=order)[0]
        peaks_ema30_min = signal.argrelextrema(df['ema_30'].values, np.less, order=order)[0]
        peaks_ema30_max = signal.argrelextrema(df['ema_30'].values, np.greater, order=order)[0]
        df['peaks_min_ema_10'] = 0
        df['peaks_max_ema_10'] = 0
        df['peaks_min_ema_30'] = 0
        df['peaks_max_ema_30'] = 0
        df.loc[peaks_ema10_min, 'peaks_min_ema_10'] = 1
        df.loc[peaks_ema10_max, 'peaks_max_ema_10'] = 1
        df.loc[peaks_ema30_min, 'peaks_min_ema_30'] = 1
        df.loc[peaks_ema30_max, 'peaks_max_ema_30'] = 1

    def _add_ema_crosses(self, df):
        df['ema_cross_up'] = 0
        df['ema_cross_down'] = 0
        df['ema10_above_ema30'] = df['ema_10'] > df['ema_30']
        df['ema10_above_ema30_shift'] = df['ema10_above_ema30'].shift(1).fillna(False)
        df.loc[(df['ema10_above_ema30'] == True) & (df['ema10_above_ema30_shift'] == False), 'ema_cross_up'] = 1
        df.loc[(df['ema10_above_ema30'] == False) & (df['ema10_above_ema30_shift'] == True), 'ema_cross_down'] = 1

    def _mark_trade_open_zone(self, df, order, tolerance):
        df['trade_open_zone_buy'] = 0
        df['trade_open_zone_sell'] = 0
        peaks_min_idx = df.index[df['peaks_min'] == 1].tolist()
        peaks_max_idx = df.index[df['peaks_max'] == 1].tolist()
        peaks_ema10_min = df.index[df['peaks_min_ema_10'] == 1].tolist()
        peaks_ema10_max = df.index[df['peaks_max_ema_10'] == 1].tolist()
        peaks_ema30_min = df.index[df['peaks_min_ema_30'] == 1].tolist()
        peaks_ema30_max = df.index[df['peaks_max_ema_30'] == 1].tolist()
        for idx in df.index[df['ema_cross_up'] == 1]:
            min_price = [p for p in peaks_min_idx if abs(idx - p) <= tolerance]
            min_ema10 = [p for p in peaks_ema10_min if abs(idx - p) <= tolerance]
            min_ema30 = [p for p in peaks_ema30_min if abs(idx - p) <= tolerance]
            if min_price and min_ema10 and min_ema30:
                df.at[idx, 'trade_open_zone_buy'] = 1
        for idx in df.index[df['ema_cross_down'] == 1]:
            max_price = [p for p in peaks_max_idx if abs(idx - p) <= tolerance]
            max_ema10 = [p for p in peaks_ema10_max if abs(idx - p) <= tolerance]
            max_ema30 = [p for p in peaks_ema30_max if abs(idx - p) <= tolerance]
            if max_price and max_ema10 and max_ema30:
                df.at[idx, 'trade_open_zone_sell'] = 1

    def apply_triggers_strategy(self, df: pd.DataFrame, config=None) -> pd.DataFrame:
        if config is None:
            from ConfigurationOperation import ConfigurationOperation
            config = ConfigurationOperation()
        signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
        peaks_min_col = config.peaks_min_col if hasattr(config, 'peaks_min_col') else 'peaks_min'
        peaks_max_col = config.peaks_max_col if hasattr(config, 'peaks_max_col') else 'peaks_max'
        df[signal_col] = self.SIGNAL_NEUTRAL
        for i in range(len(df)):
            if df['trade_open_zone_buy'].iloc[i] == 1 and df[peaks_min_col].iloc[i] == 1:
                self._set_signal(df, i, signal_col, self.SIGNAL_BUY)
            elif df['trade_open_zone_sell'].iloc[i] == 1 and df[peaks_max_col].iloc[i] == 1:
                self._set_signal(df, i, signal_col, self.SIGNAL_SELL)
        return df

    def _set_signal(self, df, idx, signal_col, value):
        df.iloc[idx, df.columns.get_loc(signal_col)] = value

    def triggers_trades_open(self, df: pd.DataFrame, config=None):
        if config is None:
            from ConfigurationOperation import ConfigurationOperation
            config = ConfigurationOperation()
        signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
        recent_range = config.recent_range if hasattr(config, 'recent_range') else (-12, -8)
        try:
            recent_rows = df.iloc[recent_range[0]:recent_range[1]]
            self._handle_trade_signals(recent_rows, signal_col)
        except Exception as e:
            self._log_message(f"Error en triggers_trades_open: {e}", level='error')

    def _handle_trade_signals(self, recent_rows, signal_col):
        buy_signals = recent_rows[recent_rows[signal_col] == self.SIGNAL_BUY]
        sell_signals = recent_rows[recent_rows[signal_col] == self.SIGNAL_SELL]
        has_buy_signal = not buy_signals.empty
        has_sell_signal = not sell_signals.empty
        if has_buy_signal:
            last_buy = buy_signals.iloc[-1]
            self._process_buy_signal(last_buy)
        if has_sell_signal:
            last_sell = sell_signals.iloc[-1]
            self._process_sell_signal(last_sell)

    def _process_buy_signal(self, last_buy):
        buy_fecha = last_buy['date']
        buy_price = last_buy['bidclose']
        if self.existingOperation(instrument=self.instrument, BuySell="S"):
            self._log_message(
                f"[CIERRE SELL] Motivo: Señal de COMPRA detectada | Fecha: {buy_fecha} | Precio: {buy_price} | Instrumento: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="S")
        if not self.existingOperation(instrument=self.instrument, BuySell="B"):
            self._log_message(
                f"[APERTURA BUY] Motivo: Señal de COMPRA detectada | Fecha: {buy_fecha} | Precio: {buy_price} | Instrumento: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.createEntryOrder(str_buy_sell="B")
        else:
            self._log_message(
                f"[INFO] Ya existe operación BUY abierta | Fecha: {buy_fecha} | Precio: {buy_price} | Instrumento: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def _process_sell_signal(self, last_sell):
        sell_fecha = last_sell['date']
        sell_price = last_sell['bidclose']
        if self.existingOperation(instrument=self.instrument, BuySell="B"):
            self._log_message(
                f"[CIERRE BUY] Motivo: Señal de VENTA detectada | Fecha: {sell_fecha} | Precio: {sell_price} | Instrumento: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="B")
        if not self.existingOperation(instrument=self.instrument, BuySell="S"):
            self._log_message(
                f"[APERTURA SELL] Motivo: Señal de VENTA detectada | Fecha: {sell_fecha} | Precio: {sell_price} | Instrumento: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.createEntryOrder(str_buy_sell="S")
        else:
            self._log_message(
                f"[INFO] Ya existe operación SELL abierta | Fecha: {sell_fecha} | Precio: {sell_price} | Instrumento: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def triggers_trades_close(self, df: pd.DataFrame, config=None):
        if config is None:
            from ConfigurationOperation import ConfigurationOperation
            config = ConfigurationOperation()
        recent_close_range = config.recent_close_range if hasattr(config, 'recent_close_range') else (-7, -4)
        try:
            recent_rows = df.iloc[recent_close_range[0]:recent_close_range[1]]
            self._handle_close_signals(recent_rows)
        except Exception as e:
            self._log_message(f"Error en triggers_trades_close: {e}", level='error')

    def _handle_close_signals(self, recent_rows):
        peaks_max = recent_rows[recent_rows['peaks_max'] == 1]
        if not peaks_max.empty:
            last_peak = peaks_max.iloc[-1]
            self._close_buy_on_peak(last_peak)
        peaks_min = recent_rows[recent_rows['peaks_min'] == 1]
        if not peaks_min.empty:
            last_peak = peaks_min.iloc[-1]
            self._close_sell_on_peak(last_peak)

    def _close_buy_on_peak(self, last_peak):
        fecha = last_peak['date']
        price = last_peak['bidclose']
        if self.existingOperation(instrument=self.instrument, BuySell="B"):
            self._log_message(
                f"[CIERRE BUY] Motivo: Pico máximo detectado | Fecha: {fecha} | Precio: {price} | Instrumento: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="B")

    def _close_sell_on_peak(self, last_peak):
        fecha = last_peak['date']
        price = last_peak['bidclose']
        if self.existingOperation(instrument=self.instrument, BuySell="S"):
            self._log_message(
                f"[CIERRE SELL] Motivo: Pico mínimo detectado | Fecha: {fecha} | Precio: {price} | Instrumento: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="S")

    def existingOperation(self, instrument: str, BuySell: str) -> bool:
        existOperation = False
        try:
            trades_table = self.connection.get_table(self.connection.TRADES)
            try:
                for row in trades_table:
                    if getattr(row, 'instrument', None) == instrument and getattr(row, 'buy_sell', None) == BuySell:
                        existOperation = True
                        self._log_message(f"Operacion encontrada: Instrumento={instrument}, Tipo={BuySell}")
            except Exception as e:
                try:
                    size = trades_table.size() if callable(trades_table.size) else trades_table.size
                    for i in range(size):
                        row = trades_table.get_row(i)
                        if getattr(row, 'instrument', None) == instrument and getattr(row, 'buy_sell', None) == BuySell:
                            existOperation = True
                            self._log_message(f"Operacion encontrada: Instrumento={instrument}, Tipo={BuySell}")
                except Exception as e2:
                    pass
            if not existOperation:
                self._log_message(f"No existe operacion para Instrumento={instrument}, Tipo={BuySell}")
            return existOperation
        except Exception as e:
            self._log_message(f"Exception en existingOperation: {e}", level='error')
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
            fxcorepy = self.robotconnection.fxcorepy
            for trade in orders_table:
                buy_sell = ''
                if trade.instrument == instrument and trade.buy_sell == BuySell:
                    buy_sell = fxcorepy.Constants.SELL if trade.buy_sell == fxcorepy.Constants.BUY else fxcorepy.Constants.BUY
                    if buy_sell != None:
                        request = self.connection.create_order_request(
                            order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
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

    def createEntryOrder(self, str_buy_sell: str = None, trailing_stop_pips: int = 10, lock_in_pips: int = 5):
        """
        Crea una orden de entrada (compra o venta) para el instrumento configurado con trailing stop automático.
        
        Args:
            str_buy_sell: 'B' para compra, 'S' para venta
            trailing_stop_pips: Distancia del trailing stop en pips
            lock_in_pips: Ganancia mínima en pips para activar el trailing stop
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
            self._log_message(f"Operacion ABIERTA: Instrumento={str_instrument}, Tipo={'BUY' if str_buy_sell == fxcorepy.Constants.BUY else 'SELL'}, Monto={amount}, Stop={stop}, Limit={limit}, TrailingStop={trailing_stop_pips}pips, LockIn={lock_in_pips}pips")
            
            # Iniciar monitoreo del trailing stop
            self.start_trailing_stop_monitor(str_instrument, str_buy_sell, trailing_stop_pips, lock_in_pips)
            
        except Exception as e:
            self._log_message(f"Error al abrir operacion: {e}", level='error')

    def start_trailing_stop_monitor(self, instrument: str, buy_sell: str, trailing_stop_pips: int, lock_in_pips: int):
        """
        Inicia el monitoreo del trailing stop para una operación específica.
        """
        try:
            # Obtener el trade_id de la operación recién abierta
            trades_table = self.connection.get_table(self.connection.TRADES)
            trade_id = None
            open_price = None
            
            for trade in trades_table:
                if (getattr(trade, 'instrument', None) == instrument and 
                    getattr(trade, 'buy_sell', None) == buy_sell):
                    trade_id = getattr(trade, 'trade_id', None)
                    open_price = getattr(trade, 'open', None)
                    break
            
            if trade_id and open_price:
                self._log_message(f"Trailing stop iniciado: TradeID={trade_id}, PrecioApertura={open_price}, TrailingStop={trailing_stop_pips}pips, LockIn={lock_in_pips}pips")
                
                # Aquí podrías implementar un thread o timer para monitorear continuamente
                # Por ahora, solo registramos la información
                self.monitor_trailing_stop(trade_id, instrument, buy_sell, open_price, trailing_stop_pips, lock_in_pips)
            else:
                self._log_message(f"No se pudo obtener información del trade para trailing stop", level='warning')
                
        except Exception as e:
            self._log_message(f"Error al iniciar trailing stop monitor: {e}", level='error')

    def monitor_trailing_stop(self, trade_id: str, instrument: str, buy_sell: str, open_price: float, trailing_stop_pips: int, lock_in_pips: int):
        """
        Monitorea y actualiza el trailing stop para una operación.
        """
        try:
            pip_size = 0.01 if "JPY" in instrument else 0.0001
            current_price = self.get_latest_price(instrument, buy_sell)
            
            if current_price is None:
                return
            
            # Calcular ganancia actual en pips
            if buy_sell == "B":
                current_pips = (current_price - open_price) / pip_size
            else:
                current_pips = (open_price - current_price) / pip_size
            
            # Si alcanzamos la ganancia mínima para lock-in
            if current_pips >= lock_in_pips:
                # Calcular nuevo stop loss
                if buy_sell == "B":
                    new_stop_price = current_price - (trailing_stop_pips * pip_size)
                else:
                    new_stop_price = current_price + (trailing_stop_pips * pip_size)
                
                # Actualizar el stop loss
                self.update_trade_stop_loss(trade_id, new_stop_price, instrument, buy_sell)
                self._log_message(f"Trailing stop actualizado: TradeID={trade_id}, NuevoStop={new_stop_price}, PipsGanancia={current_pips:.1f}")
            
        except Exception as e:
            self._log_message(f"Error en monitor_trailing_stop: {e}", level='error')

    def update_trade_stop_loss(self, trade_id: str, new_stop_price: float, instrument: str, buy_sell: str):
        """
        Actualiza el stop loss de una operación específica.
        """
        try:
            # Obtener información de la cuenta
            accounts_response_reader = self.connection.get_table_reader(self.connection.ACCOUNTS)
            account_id = None
            for account in accounts_response_reader:
                account_id = account.account_id
                break
            if not account_id:
                self._log_message("No se pudo obtener account_id para actualizar stop loss", level='error')
                return
            fxcorepy = self.robotconnection.fxcorepy
            # Crear request para actualizar stop loss
            request = self.connection.create_order_request(
                order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                OFFER_ID=self.get_offer_id(instrument),
                ACCOUNT_ID=account_id,
                BUY_SELL=buy_sell,
                AMOUNT=self.get_trade_amount(trade_id),
                TRADE_ID=trade_id,
                RATE_STOP=new_stop_price
            )
            self.connection.send_request_async(request)
            self._log_message(f"Stop loss actualizado: TradeID={trade_id}, NuevoStop={new_stop_price}")
        except Exception as e:
            self._log_message(f"Error al actualizar stop loss: {e}", level='error')

    def get_offer_id(self, instrument: str) -> str:
        """
        Obtiene el offer_id para un instrumento.
        """
        try:
            offer = self.robotconnection.common.get_offer(self.connection, instrument)
            return offer.offer_id if offer else None
        except Exception as e:
            self._log_message(f"Error al obtener offer_id: {e}", level='error')
            return None

    def get_trade_amount(self, trade_id: str) -> float:
        """
        Obtiene el monto de una operación específica.
        """
        try:
            trades_table = self.connection.get_table(self.connection.TRADES)
            for trade in trades_table:
                if getattr(trade, 'trade_id', None) == trade_id:
                    return getattr(trade, 'amount', 0)
            return 0
        except Exception as e:
            self._log_message(f"Error al obtener trade amount: {e}", level='error')
            return 0
    
    def tradeHasAtLeast10Pips(self, instrument: str, BuySell: str) -> bool:
        """
        Verifica si existe una operación abierta para el instrumento y tipo Buy/Sell
        que haya ganado al menos 10 pips.
        """
        pip_size = 0.01 if "JPY" in instrument else 0.0001
        try:
            trades_table = self.connection.get_table(self.connection.TRADES)
            for row in trades_table:
                if getattr(row, 'instrument', None) == instrument and getattr(row, 'buy_sell', None) == BuySell:
                    open_price = getattr(row, 'open', None)
                    is_buy = getattr(row, 'isBuy', None)
                    # Try to get the current price (bid/ask) from your price data or offers table
                    # For simplicity, let's assume you have a method to get the latest price:
                    current_price = self.get_latest_price(instrument, BuySell)
                    if open_price is not None and current_price is not None:
                        if BuySell == "B":
                            pips = (current_price - open_price) / pip_size
                        else:
                            pips = (open_price - current_price) / pip_size
                        if pips >= 10:
                            self._log_message(f"Trade {BuySell} on {instrument} has {pips:.1f} pips (>=10)")
                            return True
            return False
        except Exception as e:
            self._log_message(f"Exception in tradeHasAtLeast10Pips: {e}", level='error')
            return False

    def get_latest_price(self, instrument: str, BuySell: str) -> float:
        """
        Obtiene el último precio para el instrumento.
        Puedes implementar esto usando tu DataFrame de precios o la tabla de ofertas.
        """
        # Ejemplo usando self.pricedata
        if self.pricedata is not None and not self.pricedata.empty:
            # Usar el último precio de cierre
            return float(self.pricedata['bidclose'].iloc[-1])
        # Si tienes acceso a la tabla de ofertas, puedes obtener el precio actual de ahí
        # Si no hay datos, retorna None
        return None
