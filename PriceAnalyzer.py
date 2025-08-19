import pandas as pd
import numpy as np
from scipy import signal
from datetime import datetime
import pytz
import datetime as dt
import logging
import os
from ConfigurationOperation import TradingConfig

try:
    from ConnectionFxcm import RobotConnection
    ROBOT_CONNECTION_AVAILABLE = True
except ImportError:
    ROBOT_CONNECTION_AVAILABLE = False
    print("Warning: RobotConnection not available - forex trading features will be disabled")


class PriceAnalyzer:
    
    SIGNAL_BUY = 1
    SIGNAL_SELL = -1
    SIGNAL_NEUTRAL = 0

    def __init__(self, days: int, instrument: str, timeframe: str):
        self.instrument = instrument
        self.timeframe = timeframe
        self.pricedata = None
        self.days = days
        self.trade_log_file = os.path.join('logs', 'triggers_trades_open.csv')
        self._last_signal_info = None  # holds context for logging after order open
        
        if ROBOT_CONNECTION_AVAILABLE:
            self.robotconnection = RobotConnection()
            self.connection = self.robotconnection.getConnection()
        else:
            self.robotconnection = None
            self.connection = None
            
        self._setup_logging()
        self._ensure_trade_log_file()

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
        if not os.path.exists('logs'):
            os.makedirs('logs')
        self.logger = logging.getLogger(f'PriceAnalyzer_{self.instrument}')
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

    def _ensure_trade_log_file(self):
        try:
            if not os.path.exists('logs'):
                os.makedirs('logs')
            if not os.path.exists(self.trade_log_file):
                import csv
                with open(self.trade_log_file, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'instrument', 'timeframe', 'action', 'side', 'price', 'signal_date', 'details'
                    ])
        except Exception as e:
            self._log_message(f"Error ensuring trade log file: {e}", level='error')

    def _append_trade_log(self, action: str, side: str, price: float = None, signal_date: str = '', details: str = ''):
        try:
            import csv
            europe_london = pytz.timezone('Europe/London')
            ts = datetime.now(europe_london).isoformat()
            with open(self.trade_log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts,
                    self.instrument,
                    self.timeframe,
                    action,
                    side,
                    price if price is not None else '',
                    signal_date if signal_date is not None else '',
                    details
                ])
        except Exception as e:
            self._log_message(f"Error appending to trade log: {e}", level='error')

    def get_price_data(self, instrument: str, timeframe: str, days: int, connection) -> pd.DataFrame:
        try:
            europe_London_datetime = datetime.now(pytz.timezone('Europe/London'))
            date_from = europe_London_datetime - dt.timedelta(days=days)
            date_to = europe_London_datetime
            
            history = connection.get_history(instrument, timeframe, date_from, date_to)
               
            pricedata = pd.DataFrame(history, columns=["Date", "BidOpen", "BidHigh", "BidLow", "BidClose", "Volume"])
           
            if pricedata.empty:
                self._log_message(f"Empty DataFrame created for {instrument} {timeframe}", level='error')
                return pd.DataFrame()
            
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
            
           
            return df
            
        except Exception as e:
            self._log_message(f"Error in get_price_data for {instrument} {timeframe}: {str(e)}", level='error')
            import traceback
            self._log_message(f"Traceback: {traceback.format_exc()}", level='error')
            return pd.DataFrame()

    def save_price_data_file(self, pricedata: pd.DataFrame):
        fileName = self.instrument.replace("/", "_") + "_" + self.timeframe + ".csv"
        pricedata.to_csv(fileName)

    def get_latest_price(self, instrument: str, BuySell: str) -> float:
        if self.pricedata is not None and not self.pricedata.empty:
            return float(self.pricedata['bidclose'].iloc[-1])
        return None

    def set_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
               
            # Calcular peaks
            df = self.calculate_peaks(df)
            
            # Calcular medianas (precio, min, max)
            df = self._add_median_indicators(df)
            
            # Calcular tendencia basada en medianas
            df = self._calculate_trend_indicators(df)
      
            self._log_message(f"All indicators calculated successfully for {len(df)} rows: peaks, medians, trend indicators (based on medians), EMAs (50,80)")
            return df
            
        except Exception as e:
            self._log_message(f"Error setting indicators: {str(e)}", 'error')
            return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend direction based on medianas vs EMAs comparison"""
        try:
            # Calculate EMAs
            df['ema50'] = df['bidclose'].ewm(span=50).mean()
            df['ema80'] = df['bidclose'].ewm(span=80).mean()
            
            # Calculate trend based on medianas vs EMAs
            def tendencia(row):
                # Verificar que las medianas y EMAs existan
                if (pd.isna(row['bidclose_median']) or pd.isna(row['bidopen_median']) or 
                    pd.isna(row['ema50']) or pd.isna(row['ema80'])):
                    return 'FLAT'
                
                # BULL: ambas medianas deben ser mayores que las EMAs
                if (row['bidclose_median'] > row['ema50'] and row['bidclose_median'] > row['ema80'] and
                    row['bidopen_median'] > row['ema50'] and row['bidopen_median'] > row['ema80']):
                    return 'BULL'
                # BEAR: ambas medianas deben ser menores que las EMAs
                elif (row['bidclose_median'] < row['ema50'] and row['bidclose_median'] < row['ema80'] and
                      row['bidopen_median'] < row['ema50'] and row['bidopen_median'] < row['ema80']):
                    return 'BEAR'
                # Si no cumple ninguna condición → FLAT (plano)
                else:
                    return 'FLAT'
            
            df['trend'] = df.apply(tendencia, axis=1)
            
            # Contar tendencias para logging
            bull_count = (df['trend'] == 'BULL').sum()
            bear_count = (df['trend'] == 'BEAR').sum()
            flat_count = (df['trend'] == 'FLAT').sum()
            
            self._log_message(f"Trend indicators calculated: BULL={bull_count}, BEAR={bear_count}, FLAT={flat_count} based on medianas > EMAs (BULL) or medianas < EMAs (BEAR)")
            return df
            
        except Exception as e:
            self._log_message(f"Error calculating trend indicators: {e}", level='error')
            return df

    def calculate_peaks(self, df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
        self._add_price_peaks(df, order)
        return df

    def _add_price_peaks(self, df, order):        
            df['peaks_min'] = 0
            df['peaks_max'] = 0

            peaks_min_idx = signal.argrelextrema(df['bidclose'].values, np.less, order=order)[0]
            peaks_max_idx = signal.argrelextrema(df['bidclose'].values, np.greater, order=order)[0]
            
            df.loc[peaks_min_idx, 'peaks_min'] = 1
            df.loc[peaks_max_idx, 'peaks_max'] = 1

    def _add_median_indicators(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """Add median indicators (bidclose_median, bidopen_median, min_low_median, max_high_median) and near median zones"""
        try:
            # Calcular medianas móviles
            df['bidclose_median'] = df['bidclose'].rolling(window=window, center=True).median()
            df['bidopen_median'] = df['bidopen'].rolling(window=window, center=True).median()
            df['min_low_median'] = df['bidlow'].rolling(window=window, center=True).median()
            df['max_high_median'] = df['bidhigh'].rolling(window=window, center=True).median()
                        
            # Rellenar valores NaN siguiendo la tendencia usando interpolación
            df['bidclose_median'] = self._fill_na_with_trend(df['bidclose_median'])
            df['bidopen_median'] = self._fill_na_with_trend(df['bidopen_median'])
            df['min_low_median'] = self._fill_na_with_trend(df['min_low_median'])
            df['max_high_median'] = self._fill_na_with_trend(df['max_high_median'])
            
            self._log_message(f"Median indicators calculated: bidclose_median, bidopen_median, min_low_median, max_high_median, near_median_zones (window: {window})")
            
            return df
            
        except Exception as e:
            self._log_message(f"Error calculating median indicators: {e}", level='error')
            return df
    
    def _fill_na_with_trend(self, series: pd.Series) -> pd.Series:
        """Fill NaN values following the trend using linear interpolation and trend analysis"""
        try:
            # Si no hay valores NaN, retornar la serie original
            if not series.isna().any():
                return series
            
            # Crear una copia para trabajar
            filled_series = series.copy()
            
            # Encontrar todos los índices con valores NaN
            nan_indices = filled_series[filled_series.isna()].index
            
            for nan_idx in nan_indices:
                # Buscar el valor válido más cercano antes del NaN
                before_nan = filled_series.loc[:nan_idx].dropna()
                after_nan = filled_series.loc[nan_idx:].dropna()
                
                if len(before_nan) > 0 and len(after_nan) > 0:
                    # Tenemos valores antes y después, usar interpolación lineal
                    before_val = before_nan.iloc[-1]
                    after_val = after_nan.iloc[0]
                    before_idx = before_nan.index[-1]
                    after_idx = after_nan.index[0]
                    
                    # Calcular la pendiente entre los dos puntos válidos
                    if after_idx != before_idx:
                        slope = (after_val - before_val) / (after_idx - before_idx)
                        # Interpolar usando la pendiente
                        filled_series.loc[nan_idx] = before_val + slope * (nan_idx - before_idx)
                    else:
                        # Si los índices son iguales, usar el promedio
                        filled_series.loc[nan_idx] = (before_val + after_val) / 2
                
                elif len(before_nan) > 0:
                    # Solo tenemos valores antes, usar el último valor
                    filled_series.loc[nan_idx] = before_nan.iloc[-1]
                
                elif len(after_nan) > 0:
                    # Solo tenemos valores después, usar el primer valor
                    filled_series.loc[nan_idx] = after_nan.iloc[0]
                
                else:
                    # No hay valores válidos, usar 0 como fallback
                    filled_series.loc[nan_idx] = 0
            
            return filled_series
            
        except Exception as e:
            self._log_message(f"Error in _fill_na_with_trend: {e}", level='error')
            # En caso de error, usar el método original
            return series.fillna(method='ffill').fillna(method='bfill')
    

    

    

    

    

    





        
    def set_signals_to_trades(self, df: pd.DataFrame, config=None) -> pd.DataFrame:
        if config is None:
            from ConfigurationOperation import TradingConfig
            config = TradingConfig()
        
        signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
        df[signal_col] = self.SIGNAL_NEUTRAL
                
        buy_signals = 0
        sell_signals = 0
        
        for i in range(len(df)):
            # === CONDICIONES DE COMPRA (SIMPLIFICADAS) ===
            # Peak mínimo + Tendencia alcista
            buy_peak = df['peaks_min'].iloc[i] == 1
            bullish_trend = df['trend'].iloc[i] == 'BULL'
            
            # Generar señal si hay peak Y tendencia alcista
            if buy_peak and bullish_trend:
                self._set_signal(df, i, signal_col, self.SIGNAL_BUY)
                buy_signals += 1
            
            # === CONDICIONES DE VENTA (SIMPLIFICADAS) ===
            # Peak máximo + Tendencia bajista
            sell_peak = df['peaks_max'].iloc[i] == 1
            bearish_trend = df['trend'].iloc[i] == 'BEAR'
            
            # Generar señal si hay peak Y tendencia bajista
            if sell_peak and bearish_trend:
                self._set_signal(df, i, signal_col, self.SIGNAL_SELL)
                sell_signals += 1
        
        # Since signals are already filtered by trend, valid_signal equals signal
        df['valid_signal'] = df[signal_col]        
        self._log_message(f"Generated {buy_signals} buy signals and {sell_signals} sell signals - peaks with trend validation")
        return df

    def _set_signal(self, df, idx, signal_col, value):
        df.iloc[idx, df.columns.get_loc(signal_col)] = value

    def triggers_trades_open(self, df: pd.DataFrame, config=None):
        """Simplified trade opening - processes last 7 candles but ignores last 2"""
        try:
            # Initialize config
            if config is None:
                from ConfigurationOperation import TradingConfig
                config = TradingConfig()
            
            signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
            
            # Get the last 7 candles but ignore the last 2 (process candles 3-7 from the end)
            last_7_candles = df.tail(7)
            validation_candles = last_7_candles.head(5)  # Exclude last 2 candles
            
            self._log_message(f"Processing signals in last 7 candles (excluding last 2) for trade execution")
            
            # Check for signals in the validation period (candles 3-7 from the end)
            buy_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_BUY]
            sell_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_SELL]
            
            self._log_message(f"Found {len(buy_signals)} buy signals, {len(sell_signals)} sell signals in validation period (candles 3-7)")
            
            # Process signals immediately
            if not buy_signals.empty:
                latest_buy = buy_signals.iloc[-1]
                self._log_message(f"BUY signal detected - opening trade from candle {latest_buy['date']}")
                self._process_buy_signal(latest_buy)
            elif not sell_signals.empty:
                latest_sell = sell_signals.iloc[-1]
                self._log_message(f"SELL signal detected - opening trade from candle {latest_sell['date']}")
                self._process_sell_signal(latest_sell)
            else:
                self._log_message("No signals found in validation period (candles 3-7)")
            
        except Exception as e:
            self._log_message(f"Error in triggers_trades_open: {e}", level='error')

    def _process_buy_signal(self, last_buy):
        try:
            buy_date = last_buy['date']
            buy_price = last_buy['bidclose']
            
            self._log_message(f"Processing BUY signal - Date: {buy_date}, Price: {buy_price}")
            
            self._close_existing_sell_operations(buy_date, buy_price)
            
            self._open_buy_operation(buy_date, buy_price)
            
        except Exception as e:
            self._log_message(f"Error processing BUY signal: {e}", level='error')

    def _close_existing_sell_operations(self, signal_date, signal_price):
        if self.existingOperation(instrument=self.instrument, BuySell="S"):
            self._log_message(
                f"[CLOSE SELL] Reason: BUY signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="S")
        else:
            self._log_message(f"No existing SELL operations to close")

    def _open_buy_operation(self, signal_date, signal_price):
        if not self.existingOperation(instrument=self.instrument, BuySell="B"):
            self._log_message(
                f"[OPEN BUY] Reason: BUY signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            # store context so createEntryOrder can log after order submission
            self._last_signal_info = {
                'side': 'B',
                'signal_date': str(signal_date),
                'price': float(signal_price) if signal_price is not None else None,
                'details': 'BUY signal'
            }
            self.createEntryOrder(str_buy_sell="B")
        else:
            self._log_message(
                f"[INFO] BUY operation already exists | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def _process_sell_signal(self, last_sell):
        try:
            sell_date = last_sell['date']
            sell_price = last_sell['bidclose']
            
            self._log_message(f"Processing SELL signal - Date: {sell_date}, Price: {sell_price}")
            
            self._close_existing_buy_operations(sell_date, sell_price)
            
            self._open_sell_operation(sell_date, sell_price)
            
        except Exception as e:
            self._log_message(f"Error processing SELL signal: {e}", level='error')

    def _close_existing_buy_operations(self, signal_date, signal_price):
        if self.existingOperation(instrument=self.instrument, BuySell="B"):
            self._log_message(
                f"[CLOSE BUY] Reason: SELL signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="B")
        else:
            self._log_message(f"No existing BUY operations to close")

    def _open_sell_operation(self, signal_date, signal_price):
        if not self.existingOperation(instrument=self.instrument, BuySell="S"):
            self._log_message(
                f"[OPEN SELL] Reason: SELL signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            # store context so createEntryOrder can log after order submission
            self._last_signal_info = {
                'side': 'S',
                'signal_date': str(signal_date),
                'price': float(signal_price) if signal_price is not None else None,
                'details': 'SELL signal'
            }
            self.createEntryOrder(str_buy_sell="S")
        else:
            self._log_message(
                f"[INFO] SELL operation already exists | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def triggers_trades_close(self, df: pd.DataFrame):
        """Process trade closing signals from peaks"""
        try:
            # Validate data
            if df is None or df.empty:
                self._log_message("No data available for trade closing analysis", level='warning')
                return
            
            # Process only recent range to avoid repetitive closes
            recent_range = 10  # Last 10 candles
            recent_df = df.tail(recent_range) if len(df) > recent_range else df
            
            self._log_message(f"Processing close signals from recent range ({len(recent_df)} candles)")
            
            # Find peak signals for closing
            peaks_max = recent_df[recent_df['peaks_max'] == 1]
            peaks_min = recent_df[recent_df['peaks_min'] == 1]
            
            # Close BUY operations on maximum peaks
            if not peaks_max.empty:
                last_peak = peaks_max.iloc[-1]
                if self.existingOperation(instrument=self.instrument, BuySell="B"):
                    self._log_message(
                        f"[CLOSE BUY] Maximum peak detected | Date: {last_peak['date']} | "
                        f"Price: {last_peak['bidclose']} | Instrument: {self.instrument}"
                    )
                    self.CloseOperation(instrument=self.instrument, BuySell="B")
            
            # Close SELL operations on minimum peaks
            if not peaks_min.empty:
                last_peak = peaks_min.iloc[-1]
                if self.existingOperation(instrument=self.instrument, BuySell="S"):
                    self._log_message(
                        f"[CLOSE SELL] Minimum peak detected | Date: {last_peak['date']} | "
                        f"Price: {last_peak['bidclose']} | Instrument: {self.instrument}"
                    )
                    self.CloseOperation(instrument=self.instrument, BuySell="S")
            
        except Exception as e:
            self._log_message(f"Error in triggers_trades_close: {e}", level='error')

    def existingOperation(self, instrument: str, BuySell: str) -> bool:
        existOperation = False
        try:
            # Check if connection is available
            if not ROBOT_CONNECTION_AVAILABLE or self.connection is None:
                self._log_message(f"Connection not available - assuming no operation exists", level='warning')
                return False
            
            trades_table = self.connection.get_table(self.connection.TRADES)
            try:
                for row in trades_table:
                    if getattr(row, 'instrument', None) == instrument and getattr(row, 'buy_sell', None) == BuySell:
                        existOperation = True
                        self._log_message(f"Operation found: Instrument={instrument}, Type={BuySell}")
            except Exception as e:
                try:
                    size = trades_table.size() if callable(trades_table.size) else trades_table.size
                    for i in range(size):
                        row = trades_table.get_row(i)
                        if getattr(row, 'instrument', None) == instrument and getattr(row, 'buy_sell', None) == BuySell:
                            existOperation = True
                            self._log_message(f"Operation found: Instrument={instrument}, Type={BuySell}")
                except Exception as e2:
                    self._log_message(f"Error accessing trades table: {e2}", level='error')
            
            if not existOperation:
                self._log_message(f"No operation exists for Instrument={instrument}, Type={BuySell}")
            return existOperation
        except Exception as e:
            self._log_message(f"Exception in existingOperation: {e}", level='error')
            return False  # Return False on error to be safe

    def CloseOperation(self, instrument: str, BuySell: str):
        try:
            accounts_response_reader = self.connection.get_table_reader(self.connection.ACCOUNTS)
            accountId = None
            for account in accounts_response_reader:
                accountId = account.account_id
            
            orders_table = self.connection.get_table(self.connection.TRADES)
            fxcorepy = self.robotconnection.fxcorepy
            
            for trade in orders_table:
                if trade.instrument == instrument and trade.buy_sell == BuySell:
                    buy_sell = fxcorepy.Constants.SELL if trade.buy_sell == fxcorepy.Constants.BUY else fxcorepy.Constants.BUY
                    if buy_sell is not None:
                        request = self.connection.create_order_request(
                            order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                            OFFER_ID=trade.offer_id,
                            ACCOUNT_ID=accountId,
                            BUY_SELL=buy_sell,
                            AMOUNT=trade.amount,
                            TRADE_ID=trade.trade_id
                        )
                        self.connection.send_request_async(request)
                        self._log_message(f"Operation CLOSED: Instrument={instrument}, Type={BuySell}, Amount={trade.amount}, TradeID={trade.trade_id}")
                        # log close to CSV
                        latest_price = self.get_latest_price(instrument=instrument, BuySell=BuySell)
                        self._append_trade_log(action='CLOSE', side=BuySell, price=latest_price, signal_date='', details=f'TradeID={trade.trade_id}')
        except Exception as e:
            self._log_message(f"Error closing operation: {e}", level='error')

    def createEntryOrder(self, str_buy_sell: str = None):
        try:
            # Check if connection is available
            if not ROBOT_CONNECTION_AVAILABLE or self.robotconnection is None:
                self._log_message("Robot connection not available - cannot create entry order", level='error')
                return
            
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
                if not pegstoptype or pegstoptype not in ['O', 'M']:
                    return
                peggedstop = peggedstop.lower()
                if peggedstop != 'y':
                    peggedstop = None
            if pegstoptype:
                pegstoptype = pegstoptype.upper()
            
            if peggedlimit:
                if not peglimittype or peglimittype not in ['O', 'M']:
                    return
                peggedlimit = peggedlimit.lower()
                if peggedlimit != 'y':
                    peggedlimit = None
            if peglimittype:
                peglimittype = peglimittype.upper()
            
            try:
                account = common.get_account(self.connection, str_account)
                if not account:
                    raise Exception(f"The account '{str_account}' is not valid")
                str_account = account.account_id
                
                offer = common.get_offer(self.connection, str_instrument)
                if offer is None:
                    raise Exception(f"The instrument '{str_instrument}' is not valid")
                
                login_rules = self.connection.login_rules
                trading_settings_provider = login_rules.trading_settings_provider
                base_unit_size = trading_settings_provider.get_base_unit_size(str_instrument, account)
                amount = base_unit_size * str_lots
                
                entry = fxcorepy.Constants.Orders.TRUE_MARKET_OPEN
                original_side = 'B' if str_buy_sell == 'B' else 'S'
                if original_side == 'B':
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
                self._log_message(f"Operation OPENED: Instrument={str_instrument}, Type={'BUY' if str_buy_sell == fxcorepy.Constants.BUY else 'SELL'}, Amount={amount}, Stop={stop}, Limit={limit}")
                # log open to CSV after order submission
                if self._last_signal_info is not None:
                    self._append_trade_log(
                        action='OPEN',
                        side=self._last_signal_info.get('side', original_side),
                        price=self._last_signal_info.get('price'),
                        signal_date=self._last_signal_info.get('signal_date', ''),
                        details=self._last_signal_info.get('details', '')
                    )
                    self._last_signal_info = None
                else:
                    # Fallback log without signal context
                    latest_price = self.get_latest_price(instrument=str_instrument, BuySell=original_side)
                    self._append_trade_log(action='OPEN', side=original_side, price=latest_price, signal_date='', details='No signal context')
                
            except Exception as e:
                self._log_message(f"Error opening operation: {e}", level='error')
                
        except Exception as e:
            self._log_message(f"Error in createEntryOrder: {e}", level='error')

    def get_offer_id(self, instrument: str) -> str:
        try:
            offer = self.robotconnection.common.get_offer(self.connection, instrument)
            return offer.offer_id if offer else None
        except Exception as e:
            self._log_message(f"Error getting offer_id: {e}", level='error')
            return None

    def get_trade_amount(self, trade_id: str) -> float:
        try:
            trades_table = self.connection.get_table(self.connection.TRADES)
            for trade in trades_table:
                if getattr(trade, 'trade_id', None) == trade_id:
                    return getattr(trade, 'amount', 0)
            return 0
        except Exception as e:
            self._log_message(f"Error getting trade amount: {e}", level='error')
            return 0 

