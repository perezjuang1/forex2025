import pandas as pd
import numpy as np
from scipy import signal
from datetime import datetime
import pytz
import datetime as dt
import logging
import os
from ConfigurationOperation import ConfigurationOperation

try:
    from ConnectionFxcm import RobotConnection
    ROBOT_CONNECTION_AVAILABLE = True
except ImportError:
    ROBOT_CONNECTION_AVAILABLE = False
    print("Warning: RobotConnection not available - forex trading features will be disabled")


class Price:
    
    SIGNAL_BUY = 1
    SIGNAL_SELL = -1
    SIGNAL_NEUTRAL = 0

    def __init__(self, days: int, instrument: str, timeframe: str):
        self.instrument = instrument
        self.timeframe = timeframe
        self.pricedata = None
        self.days = days
        
        if ROBOT_CONNECTION_AVAILABLE:
            self.robotconnection = RobotConnection()
            self.connection = self.robotconnection.getConnection()
        else:
            self.robotconnection = None
            self.connection = None
            
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
        if not os.path.exists('logs'):
            os.makedirs('logs')
        self.logger = logging.getLogger(f'Price_{self.instrument}')
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
        try:
            exists_buy = self.existingOperation(instrument, 'B')
            exists_sell = self.existingOperation(instrument, 'S')
            print(f"DEBUG: BUY operation exists for {instrument}: {exists_buy}")
            print(f"DEBUG: SELL operation exists for {instrument}: {exists_sell}")
            
            europe_London_datetime = datetime.now(pytz.timezone('Europe/London'))
            date_from = europe_London_datetime - dt.timedelta(days=days)
            date_to = europe_London_datetime
            
            history = connection.get_history(instrument, timeframe, date_from, date_to)
            
            if history is None or len(history) == 0:
                self._log_message(f"No historical data received for {instrument} {timeframe}", level='error')
                return pd.DataFrame()
            
            current_unit, _ = connection.parse_timeframe(timeframe)
            
            self._log_message(
                f"PRICE DATA: Unit={current_unit}, Timeframe={timeframe}, Instrument={instrument}, Date={europe_London_datetime}, Rows={len(history)}"
            )
            
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
            
            self.pricedata = self.set_indicators(df)
            
            if self.pricedata is None or self.pricedata.empty:
                self._log_message(f"Failed to process indicators for {instrument} {timeframe}", level='error')
                return pd.DataFrame()
            
            self.save_price_data_file(self.pricedata)
            return self.pricedata
            
        except Exception as e:
            self._log_message(f"Error in get_price_data for {instrument} {timeframe}: {str(e)}", level='error')
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
            self._ensure_basic_columns(df)
            
            df = self.add_median_and_trend(df)
            
            df = self.calculate_peaks(df)
            
            df = self._mark_trading_zones(df)
            
            df = self.apply_triggers_strategy(df)
            
            self._log_message(f"All indicators set successfully for {len(df)} rows")
            return df
            
        except Exception as e:
            self._log_message(f"Error setting indicators: {str(e)}", 'error')
            return df

    def _ensure_basic_columns(self, df):
        required_columns = {
            'median_bidclose': 0,
            'median_trend': 'flat',
            'peaks_min': 0,
            'peaks_max': 0,
            'trade_open_zone_buy': 0,
            'trade_open_zone_sell': 0,
            'trade_trend_zone_buy': 0,
            'trade_trend_zone_sell': 0,
            'tolerance_zone_buy': 0,
            'tolerance_zone_sell': 0,
            'signal': self.SIGNAL_NEUTRAL
        }
        
        for col, default_value in required_columns.items():
            if col not in df.columns:
                df[col] = default_value
                self._log_message(f"Created missing column: {col}", level='warning')

    def add_median_and_trend(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        df['median_bidclose'] = df['bidclose'].rolling(window=window, center=True, min_periods=1).median()
        
        df['short_trend'] = df['bidclose'].rolling(window=50, center=True, min_periods=1).mean()
        
        raw_trend = np.where(
            df['median_bidclose'].diff() > 0, 'going up',
            np.where(df['median_bidclose'].diff() < 0, 'going down', 'flat')
        )
        
        df['momentum'] = df['bidclose'].diff(5)
        
        trend_filled = []
        last_trend = None
        for i, t in enumerate(raw_trend):
            if t == 'flat' and last_trend is not None:
                if i > 0 and df['momentum'].iloc[i] > 0 and last_trend == 'going down':
                    trend_filled.append('going up')
                elif i > 0 and df['momentum'].iloc[i] < 0 and last_trend == 'going up':
                    trend_filled.append('going down')
                else:
                    trend_filled.append(last_trend)
            else:
                trend_filled.append(t)
                if t != 'flat':
                    last_trend = t
        
        df['median_trend'] = trend_filled
        return df

    def calculate_peaks(self, df: pd.DataFrame, order: int = 50) -> pd.DataFrame:
        self._add_price_peaks(df, order)
        return df

    def _add_price_peaks(self, df, order):
        try:
            optimal_order = self._calculate_optimal_order(df, order)
            
            optimal_order = int(optimal_order)
            
            peaks_min_idx = signal.argrelextrema(df['bidclose'].values, np.less, order=optimal_order)[0]
            peaks_max_idx = signal.argrelextrema(df['bidclose'].values, np.greater, order=optimal_order)[0]
            
            df['peaks_min'] = 0
            df['peaks_max'] = 0
            
            df.loc[peaks_min_idx, 'peaks_min'] = 1
            df.loc[peaks_max_idx, 'peaks_max'] = 1
            
            self._log_message(f"Peaks calculated with adaptive order: {optimal_order} (original: {order})")
            
        except Exception as e:
            self._log_message(f"Error calculating peaks: {e}", level='error')
            df['peaks_min'] = 0
            df['peaks_max'] = 0

    def _calculate_optimal_order(self, df, base_order):
        try:
            data_length = len(df)
            
            if data_length < 500:
                optimal_order = max(5, base_order // 4)
            elif data_length < 1000:
                optimal_order = max(8, base_order // 3)
            elif data_length > 3000:
                optimal_order = min(50, base_order)
            else:
                optimal_order = base_order // 2
            
            optimal_order = int(max(3, min(optimal_order, data_length // 8)))
            
            self._log_message(f"Data length: {data_length}, Optimal order: {optimal_order}")
            return optimal_order
            
        except Exception as e:
            self._log_message(f"Error in optimal order calculation: {e}", level='error')
            return int(base_order // 2)

    def _mark_trading_zones(self, df):
        if 'peaks_min' not in df.columns:
            df['peaks_min'] = 0
        if 'peaks_max' not in df.columns:
            df['peaks_max'] = 0
        if 'median_trend' not in df.columns:
            df['median_trend'] = 'flat'
        if 'trade_open_zone_buy' not in df.columns:
            df['trade_open_zone_buy'] = 0
        if 'trade_open_zone_sell' not in df.columns:
            df['trade_open_zone_sell'] = 0
        if 'trade_trend_zone_buy' not in df.columns:
            df['trade_trend_zone_buy'] = 0
        if 'trade_trend_zone_sell' not in df.columns:
            df['trade_trend_zone_sell'] = 0

        config = ConfigurationOperation()
        tolerance = getattr(config, 'tolerance_peaks', 10)

        df['trade_zone_buy'] = 0
        df['trade_zone_sell'] = 0
        df['tolerance_zone_buy'] = 0
        df['tolerance_zone_sell'] = 0

        df.loc[df['median_trend'].values == 'going up', 'trade_trend_zone_buy'] = 1
        df.loc[df['median_trend'].values == 'going down', 'trade_trend_zone_sell'] = 1

        buy_points = df[
            (df['peaks_min'].values == 1) & 
            (df['median_trend'].values == 'going up')
        ].index

        sell_points = df[
            (df['peaks_max'].values == 1) & 
            (df['median_trend'].values == 'going down')
        ].index

        for idx in buy_points:
            start_idx = max(0, idx - tolerance)
            end_idx = min(len(df), idx + tolerance + 1)
            df.loc[start_idx:end_idx, 'tolerance_zone_buy'] = 1
            df.loc[idx, 'trade_zone_buy'] = 1

        for idx in sell_points:
            start_idx = max(0, idx - tolerance)
            end_idx = min(len(df), idx + tolerance + 1)
            df.loc[start_idx:end_idx, 'tolerance_zone_sell'] = 1
            df.loc[idx, 'trade_zone_sell'] = 1

        self._log_message(f"Trading zones marked with tolerance: {tolerance} candles around condition points")
        self._log_message(f"BUY signals (mínimos en tendencia alcista): {len(buy_points)}")
        self._log_message(f"SELL signals (máximos en tendencia bajista): {len(sell_points)}")
        
        self._log_message(f"DEBUG: trade_zone_buy column exists: {'trade_zone_buy' in df.columns}")
        self._log_message(f"DEBUG: trade_zone_sell column exists: {'trade_zone_sell' in df.columns}")
        self._log_message(f"DEBUG: trade_zone_buy values > 0: {(df['trade_zone_buy'] > 0).sum()}")
        self._log_message(f"DEBUG: trade_zone_sell values > 0: {(df['trade_zone_sell'] > 0).sum()}")

        recent_data = df.tail(100)
        
        if len(recent_data) >= 20:
            recent_momentum = recent_data['bidclose'].diff(5).tail(10).mean()
            current_trend = df['median_trend'].iloc[-1]
            
            if recent_momentum > 0 and current_trend == 'going down':
                df.loc[df.index[-10:], 'trade_trend_zone_buy'] = 1
                self._log_message("Early uptrend detection in recent data")
            elif recent_momentum < 0 and current_trend == 'going up':
                df.loc[df.index[-10:], 'trade_trend_zone_sell'] = 1
                self._log_message("Early downtrend detection in recent data")

        return df

    def apply_triggers_strategy(self, df: pd.DataFrame, config=None) -> pd.DataFrame:
        if config is None:
            from ConfigurationOperation import ConfigurationOperation
            config = ConfigurationOperation()
        signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
        df[signal_col] = self.SIGNAL_NEUTRAL
        
        self._log_message(f"DEBUG: Applying triggers strategy - trade_zone_buy exists: {'trade_zone_buy' in df.columns}")
        self._log_message(f"DEBUG: Applying triggers strategy - trade_zone_sell exists: {'trade_zone_sell' in df.columns}")
        
        buy_signals = 0
        sell_signals = 0
        
        for i in range(len(df)):
            if df['trade_zone_buy'].iloc[i] == 1:
                self._set_signal(df, i, signal_col, self.SIGNAL_BUY)
                buy_signals += 1
            elif df['trade_zone_sell'].iloc[i] == 1:
                self._set_signal(df, i, signal_col, self.SIGNAL_SELL)
                sell_signals += 1
        
        self._log_message(f"DEBUG: Generated {buy_signals} buy signals and {sell_signals} sell signals")
        return df

    def _set_signal(self, df, idx, signal_col, value):
        df.iloc[idx, df.columns.get_loc(signal_col)] = value

    def triggers_trades(self, df: pd.DataFrame, config=None):
        try:
            config = self._initialize_trade_config(config)
            if not self._validate_data_for_trading(df):
                return
            
            if not self._check_tolerance_zone_conditions(df):
                return
            
            self._process_trade_signals(df, config)
            
        except Exception as e:
            self._log_message(f"Error in triggers_trades: {e}", level='error')

    def _initialize_trade_config(self, config):
        if config is None:
            from ConfigurationOperation import ConfigurationOperation
            config = ConfigurationOperation()
        return config

    def _validate_data_for_trading(self, df):
        if df is None or df.empty:
            self._log_message("No data available for trading analysis", level='warning')
            return False
        
        if len(df) < 2:
            self._log_message("Insufficient data for trading analysis - need at least 2 candles", level='warning')
            return False
        
        return True

    def _check_tolerance_zone_conditions(self, df):
        penultimate_candle = df.iloc[-2]
        
        self._log_message(
            f"Analyzing penultimate candle - Date: {penultimate_candle.get('date', 'N/A')}, "
            f"Price: {penultimate_candle.get('bidclose', 'N/A')}"
        )
        
        outside_tolerance = self._is_candle_outside_tolerance_zones(penultimate_candle)
        
        if outside_tolerance:
            self._log_message("Tolerance zone conditions met - proceeding with signal processing")
            return True
        else:
            self._log_message("Tolerance zone conditions not met - skipping signal processing")
            return False

    def _process_trade_signals(self, df, config):
        signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
        
        recent_range = (-9, -1)
        
        if hasattr(config, 'recent_range'):
            recent_range = config.recent_range
        
        recent_rows = df.iloc[recent_range[0]:recent_range[1]]
        
        self._log_message(
            f"Processing signals from range {recent_range} "
            f"(excluding last 8 candles) "
            f"with {len(recent_rows)} candles"
        )
        
        self._handle_trade_signals(recent_rows, signal_col)

    def _is_candle_outside_tolerance_zones(self, candle):
        outside_buy_tolerance = candle.get('tolerance_zone_buy', 0) == 0
        outside_sell_tolerance = candle.get('tolerance_zone_sell', 0) == 0
        
        self._log_message(f"Tolerance check - Buy zone: {candle.get('tolerance_zone_buy', 0)}, Sell zone: {candle.get('tolerance_zone_sell', 0)}")
        
        return outside_buy_tolerance and outside_sell_tolerance

    def _handle_trade_signals(self, recent_rows, signal_col):
        buy_signals = recent_rows[recent_rows[signal_col] == self.SIGNAL_BUY]
        sell_signals = recent_rows[recent_rows[signal_col] == self.SIGNAL_SELL]
        
        self._log_message(f"Signal analysis - Buy signals: {len(buy_signals)}, Sell signals: {len(sell_signals)}")
        
        if not buy_signals.empty:
            last_buy = buy_signals.iloc[-1]
            self._log_message(f"Processing BUY signal at date: {last_buy.get('date', 'N/A')}")
            self._process_buy_signal(last_buy)
        
        if not sell_signals.empty:
            last_sell = sell_signals.iloc[-1]
            self._log_message(f"Processing SELL signal at date: {last_sell.get('date', 'N/A')}")
            self._process_sell_signal(last_sell)

    def _process_buy_signal(self, last_buy):
        buy_date = last_buy['date']
        buy_price = last_buy['bidclose']
        
        self._log_message(f"Processing BUY signal - Date: {buy_date}, Price: {buy_price}")
        
        self._close_existing_sell_operations(buy_date, buy_price)
        
        self._open_buy_operation(buy_date, buy_price)

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
            self.createEntryOrder(str_buy_sell="B")
        else:
            self._log_message(
                f"[INFO] BUY operation already exists | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def _process_sell_signal(self, last_sell):
        sell_date = last_sell['date']
        sell_price = last_sell['bidclose']
        
        self._log_message(f"Processing SELL signal - Date: {sell_date}, Price: {sell_price}")
        
        self._close_existing_buy_operations(sell_date, sell_price)
        
        self._open_sell_operation(sell_date, sell_price)

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
            self.createEntryOrder(str_buy_sell="S")
        else:
            self._log_message(
                f"[INFO] SELL operation already exists | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def triggers_trades_close(self, df: pd.DataFrame, config=None):
        if df is None or df.empty:
            self._log_message("No data available for trade closing analysis", level='warning')
            return
        
        if config is None:
            from ConfigurationOperation import ConfigurationOperation
            config = ConfigurationOperation()
        
        recent_close_range = config.recent_close_range if hasattr(config, 'recent_close_range') else (-7, -4)
        
        try:
            if len(df) < abs(recent_close_range[0]):
                self._log_message(f"Insufficient data for closing analysis. Need {abs(recent_close_range[0])} rows, have {len(df)}", level='warning')
                return
            
            recent_rows = df.iloc[recent_close_range[0]:recent_close_range[1]]
            self._handle_close_signals(recent_rows)
            
        except Exception as e:
            self._log_message(f"Error in triggers_trades_close: {e}", level='error')

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
        date = last_peak['date']
        price = last_peak['bidclose']
        
        if self.existingOperation(instrument=self.instrument, BuySell="B"):
            self._log_message(
                f"[CLOSE BUY] Reason: Maximum peak detected | Date: {date} | Price: {price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="B")

    def _close_sell_on_peak(self, last_peak):
        date = last_peak['date']
        price = last_peak['bidclose']
        
        if self.existingOperation(instrument=self.instrument, BuySell="S"):
            self._log_message(
                f"[CLOSE SELL] Reason: Minimum peak detected | Date: {date} | Price: {price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
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
                    pass
            
            if not existOperation:
                self._log_message(f"No operation exists for Instrument={instrument}, Type={BuySell}")
            return existOperation
        except Exception as e:
            self._log_message(f"Exception in existingOperation: {e}", level='error')
            return existOperation

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
        except Exception as e:
            self._log_message(f"Error closing operation: {e}", level='error')

    def createEntryOrder(self, str_buy_sell: str = None):
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
            self._log_message(f"Operation OPENED: Instrument={str_instrument}, Type={'BUY' if str_buy_sell == fxcorepy.Constants.BUY else 'SELL'}, Amount={amount}, Stop={stop}, Limit={limit}")
            
        except Exception as e:
            self._log_message(f"Error opening operation: {e}", level='error')

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


