import pandas as pd
import numpy as np
from scipy import signal
from datetime import datetime
import pytz
import datetime as dt
import logging
import os
import time
from TradingConfiguration import TradingConfig

try:
    from FxcmConnection import RobotConnection
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
        
        self._setup_logging()
        self._ensure_trade_log_file()
        
        if ROBOT_CONNECTION_AVAILABLE:
            self.robotconnection = RobotConnection(instrument=self.instrument)
            self.connection = self.robotconnection.getConnection()
        else:
            self.robotconnection = None
            self.connection = None

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
            os.makedirs('logs', exist_ok=True)
        self.logger = logging.getLogger(f'PriceAnalyzer_{self.instrument}')
        self.logger.setLevel(logging.WARNING)  # Only warnings and errors
        log_file = f'logs/robot_price_{self.instrument.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.log'
        if not self.logger.handlers:
            file_handler, _ = self._create_log_handlers(log_file)
            self.logger.addHandler(file_handler)

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

    def _reconnect_if_needed(self):
        """Intenta reconectar con FXCM si es necesario"""
        try:
            if hasattr(self, 'robotconnection'):
                self._log_message("Intentando reconexión con FXCM...", level='warning')
                self.connection = self.robotconnection.getConnection(force_new=True)
                self._log_message("Reconexión exitosa", level='info')
                return True
        except Exception as e:
            self._log_message(f"Error en reconexión: {str(e)}", level='error')
        return False

    def get_price_data(self, instrument: str, timeframe: str, days: int, connection) -> pd.DataFrame:
        max_retries = 3
        base_wait_time = 2  # segundos
        
        for attempt in range(max_retries):
            try:
                
                europe_london_datetime = datetime.now(pytz.timezone('Europe/London'))
                date_from = europe_london_datetime - dt.timedelta(days=days)
                date_to = europe_london_datetime
                
                history = connection.get_history(instrument, timeframe, date_from, date_to)
                   
                pricedata = pd.DataFrame(history, columns=["Date", "BidOpen", "BidHigh", "BidLow", "BidClose", "Volume"])
               
                if pricedata.empty:
                    self._log_message(f"Empty DataFrame created for {instrument} {timeframe} - No data available", level='error')
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
                
                # Only log data issues, not successes
                
                return df
                
            except Exception as e:
                error_msg = f"Error in get_price_data for {instrument} {timeframe}: {str(e)}"
                
                # Verifica si el error está relacionado con la sesión o scope
                if "unsupported scope" in str(e).lower() or "session" in str(e).lower():
                    self._log_message(f"Intento {attempt + 1}/{max_retries}: Error de sesión/scope detectado", level='warning')
                    if self._reconnect_if_needed():
                        wait_time = base_wait_time * (2 ** attempt)  # Espera exponencial
                        self._log_message(f"Esperando {wait_time} segundos antes de reintentar...", level='info')
                        time.sleep(wait_time)
                        continue
                    else:
                        # Si la reconexión falla, espera y reintenta de todos modos
                        wait_time = base_wait_time * (2 ** attempt)
                        self._log_message(f"Reconexión falló, esperando {wait_time} segundos antes de reintentar...", level='warning')
                        time.sleep(wait_time)
                        continue
                
                # Si es el último intento o no es un error de sesión, registra y retorna
                if attempt == max_retries - 1:
                    self._log_message(error_msg, level='error')
                    import traceback
                    self._log_message(f"Traceback: {traceback.format_exc()}", level='error')
                    return pd.DataFrame()
                else:
                    # Si no es el último intento y no es un error de sesión, espera antes de reintentar
                    wait_time = base_wait_time * (2 ** attempt)
                    self._log_message(f"Esperando {wait_time} segundos antes de reintentar...", level='warning')
                    time.sleep(wait_time)
                    continue
        
        # Si llegamos aquí, todos los intentos fallaron
        self._log_message(f"Todos los {max_retries} intentos fallaron para obtener datos de {instrument} {timeframe}", level='error')
        return pd.DataFrame()

    def save_price_data_file(self, pricedata: pd.DataFrame):
        file_name = os.path.join('data', self.instrument.replace("/", "_") + "_" + self.timeframe + ".csv")
        pricedata.to_csv(file_name)

    def get_latest_price(self) -> float:
        if self.pricedata is not None and not self.pricedata.empty:
            return float(self.pricedata['bidclose'].iloc[-1])
        return 0.0

    def set_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate essential indicators for signal generation"""
        try:
            # Core EMAs for trend analysis
            df = self.calculate_ema_200(df)
            df = self.calculate_ema_100(df)
            df = self.calculate_ema_80(df)            
            df = self.calculate_peaks(df)
            
            # Channel analysis
            df = self.calculate_channel_medians(df)
            df = self.detect_channel_breakout(df)
            
            return df
        except Exception as e:
            self._log_message(f"Error setting indicators: {str(e)}", 'error')
            return df


    def calculate_peaks(self, df: pd.DataFrame, order: int = 30) -> pd.DataFrame:
        self._add_price_peaks(df, order)
        return df

    def calculate_channel_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate median max and median min using EMAs as dynamic channels"""
        if df is None or df.empty:
            return df
        
        # Use EMA 200 as median max (upper channel)
        df['median_max'] = df['ema_200']
        
        # Use EMA 80 as median min (lower channel) 
        df['median_min'] = df['ema_80']
        
        # Calculate channel width in pips
        df['channel_width_pips'] = self._calculate_pips_distance(df['median_max'], df['median_min'])
        
        return df

    def calculate_pips_distance(self, price1: float, price2: float, instrument: str = None) -> float:
        """Calculate distance between two prices in pips"""
        if instrument is None:
            instrument = self.instrument
            
        # Determine pip value based on instrument
        if 'JPY' in instrument:
            pip_value = 0.01  # For JPY pairs, 1 pip = 0.01
        else:
            pip_value = 0.0001  # For major pairs, 1 pip = 0.0001
            
        distance = abs(price1 - price2)
        pips = distance / pip_value
        return round(pips, 1)

    def _calculate_pips_distance(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Calculate pips distance between two price series"""
        if series1 is None or series2 is None:
            return pd.Series()
            
        # Determine pip value based on instrument
        if 'JPY' in self.instrument:
            pip_value = 0.01
        else:
            pip_value = 0.0001
            
        distance = abs(series1 - series2)
        pips = distance / pip_value
        return pips.round(1)

    def detect_channel_breakout(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect channel breakouts for trading signals"""
        if df is None or df.empty or 'median_max' not in df.columns or 'median_min' not in df.columns:
            return df
            
        df['channel_breakout_up'] = 0
        df['channel_breakout_down'] = 0
        
        # Detect upward breakout (price breaks above median_max)
        df.loc[df['bidclose'] > df['median_max'], 'channel_breakout_up'] = 1
        
        # Detect downward breakout (price breaks below median_min)
        df.loc[df['bidclose'] < df['median_min'], 'channel_breakout_down'] = 1
        
        return df

    def get_peak_distance_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze distances between peaks in pips"""
        if df is None or df.empty:
            return {}
            
        analysis = {}
        
        # Get min peaks
        min_peaks = df[df['peaks_min'] == 1]['bidclose'].values
        max_peaks = df[df['peaks_max'] == 1]['bidclose'].values
        
        if len(min_peaks) > 1:
            min_distances = []
            for i in range(1, len(min_peaks)):
                dist = self.calculate_pips_distance(min_peaks[i-1], min_peaks[i])
                min_distances.append(dist)
            analysis['min_peak_distances'] = min_distances
            analysis['avg_min_distance'] = round(np.mean(min_distances), 1) if min_distances else 0
            
        if len(max_peaks) > 1:
            max_distances = []
            for i in range(1, len(max_peaks)):
                dist = self.calculate_pips_distance(max_peaks[i-1], max_peaks[i])
                max_distances.append(dist)
            analysis['max_peak_distances'] = max_distances
            analysis['avg_max_distance'] = round(np.mean(max_distances), 1) if max_distances else 0
            
        # Calculate channel width statistics
        if 'channel_width_pips' in df.columns:
            channel_widths = df['channel_width_pips'].dropna()
            if len(channel_widths) > 0:
                analysis['avg_channel_width'] = round(channel_widths.mean(), 1)
                analysis['max_channel_width'] = round(channel_widths.max(), 1)
                analysis['min_channel_width'] = round(channel_widths.min(), 1)
        
        return analysis

    def calculate_ema_200(self, df: pd.DataFrame, period: int = 200) -> pd.DataFrame:
        """Calculate Exponential Moving Average of 200 periods on bidclose"""
        if df is None or df.empty or 'bidclose' not in df.columns:
            return df
        
        # Initialize EMA column
        df['ema_200'] = np.nan
        
        # Calculate EMA using pandas ewm (exponentially weighted moving average)
        df['ema_200'] = df['bidclose'].ewm(span=period, adjust=False).mean()
        
        return df

    def calculate_ema_100(self, df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
        """Calculate Exponential Moving Average of 100 periods on bidclose"""
        if df is None or df.empty or 'bidclose' not in df.columns:
            return df
        
        # Initialize EMA column
        df['ema_100'] = np.nan
        
        # Calculate EMA using pandas ewm (exponentially weighted moving average)
        df['ema_100'] = df['bidclose'].ewm(span=period, adjust=False).mean()
        
        return df

    def calculate_ema_80(self, df: pd.DataFrame, period: int = 80) -> pd.DataFrame:
        """Calculate Exponential Moving Average of 80 periods on bidclose"""
        if df is None or df.empty or 'bidclose' not in df.columns:
            return df
        
        # Initialize EMA column
        df['ema_80'] = np.nan
        
        # Calculate EMA using pandas ewm (exponentially weighted moving average)
        df['ema_80'] = df['bidclose'].ewm(span=period, adjust=False).mean()
        
        return df



    def _add_price_peaks(self, df, order):        
            df['peaks_min'] = 0
            df['peaks_max'] = 0

            peaks_min_idx = signal.argrelextrema(df['bidclose'].values, np.less, order=order)[0]
            peaks_max_idx = signal.argrelextrema(df['bidclose'].values, np.greater, order=order)[0]
            
            df.loc[peaks_min_idx, 'peaks_min'] = 1
            df.loc[peaks_max_idx, 'peaks_max'] = 1

            
    def set_signals_to_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on peaks, EMA alignment, and channel strategy"""
        if df is None or df.empty:
            return df
            
        df['signal'] = self.SIGNAL_NEUTRAL
        buy_signals = 0
        sell_signals = 0
        filtered_buy = 0
        filtered_sell = 0
        channel_signals = 0
        
        # Check if required columns exist
        has_ema_200 = 'ema_200' in df.columns
        has_ema_100 = 'ema_100' in df.columns
        has_ema_80 = 'ema_80' in df.columns
        has_all_emas = has_ema_200 and has_ema_100 and has_ema_80
        has_channel = 'median_max' in df.columns and 'median_min' in df.columns
        
        for i in range(len(df)):
            current_price = df['bidclose'].iloc[i]
            signal_generated = False
            
            # CHANNEL STRATEGY: Breakout signals
            if has_channel and not signal_generated:
                median_max = df['median_max'].iloc[i]
                median_min = df['median_min'].iloc[i]
                
                if not any(pd.isna([median_max, median_min])):
                    # BUY signal: Price breaks above median_max (upper channel)
                    if current_price > median_max:
                        df.at[i, 'signal'] = self.SIGNAL_BUY
                        buy_signals += 1
                        channel_signals += 1
                        signal_generated = True
                        candle_date = df['date'].iloc[i] if 'date' in df.columns else i
                        channel_width = self.calculate_pips_distance(median_max, median_min)
                        self._log_message(f"CHANNEL BUY: Date={candle_date} Price={current_price:.5f} Channel Width={channel_width:.1f} pips")
                    
                    # SELL signal: Price breaks below median_min (lower channel)
                    elif current_price < median_min:
                        df.at[i, 'signal'] = self.SIGNAL_SELL
                        sell_signals += 1
                        channel_signals += 1
                        signal_generated = True
                        candle_date = df['date'].iloc[i] if 'date' in df.columns else i
                        channel_width = self.calculate_pips_distance(median_max, median_min)
                        self._log_message(f"CHANNEL SELL: Date={candle_date} Price={current_price:.5f} Channel Width={channel_width:.1f} pips")
            
            # PEAK STRATEGY: Traditional peak-based signals (if no channel signal)
            if not signal_generated:
                # BUY signals at LOW peaks
                if df['peaks_min'].iloc[i] == 1:
                    signal_valid = True
                    
                    # EMA alignment check for BUY signals
                    if has_all_emas:
                        ema_200 = df['ema_200'].iloc[i]
                        ema_100 = df['ema_100'].iloc[i]
                        ema_80 = df['ema_80'].iloc[i]
                        
                        if not any(pd.isna([ema_200, ema_100, ema_80])):
                            # BUY: require bullish EMA alignment
                            if not (ema_80 > ema_100 > ema_200):
                                signal_valid = False
                                filtered_buy += 1
                    
                    # Generate BUY signal if all checks pass
                    if signal_valid:
                        df.at[i, 'signal'] = self.SIGNAL_BUY
                        buy_signals += 1
                        candle_date = df['date'].iloc[i] if 'date' in df.columns else i
                        self._log_message(f"PEAK BUY: Date={candle_date} Price={current_price:.5f}")
                    else:
                        filtered_buy += 1
                        
                # SELL signals at HIGH peaks
                elif df['peaks_max'].iloc[i] == 1:
                    signal_valid = True
                    
                    # EMA alignment check for SELL signals
                    if has_all_emas:
                        ema_200 = df['ema_200'].iloc[i]
                        ema_100 = df['ema_100'].iloc[i]
                        ema_80 = df['ema_80'].iloc[i]
                        
                        if not any(pd.isna([ema_200, ema_100, ema_80])):
                            # SELL: require bearish EMA alignment
                            if not (ema_80 < ema_100 < ema_200):
                                signal_valid = False
                                filtered_sell += 1
                    
                    # Generate SELL signal if all checks pass
                    if signal_valid:
                        df.at[i, 'signal'] = self.SIGNAL_SELL
                        sell_signals += 1
                        candle_date = df['date'].iloc[i] if 'date' in df.columns else i
                        self._log_message(f"PEAK SELL: Date={candle_date} Price={current_price:.5f}")
                    else:
                        filtered_sell += 1
                
        df['valid_signal'] = df['signal']
        
        # Log signal statistics
        total_signals = buy_signals + sell_signals
        if has_channel:
            self._log_message(f"Channel + Peak signals: buy={buy_signals} sell={sell_signals} (channel={channel_signals}, filtered: {filtered_buy} buy, {filtered_sell} sell) instrument={self.instrument} timeframe={self.timeframe}")
        elif has_all_emas:
            self._log_message(f"EMA alignment filtered signals: buy={buy_signals} sell={sell_signals} (filtered: {filtered_buy} buy, {filtered_sell} sell) instrument={self.instrument} timeframe={self.timeframe}")
        else:
            self._log_message(f"Peak signals: buy={buy_signals} sell={sell_signals} instrument={self.instrument} timeframe={self.timeframe}")
        
        return df

    def _set_signal(self, df, idx, signal_col, value):
        df.iloc[idx, df.columns.get_loc(signal_col)] = value

    def triggers_trades_open(self, df: pd.DataFrame, config=None):
        """Enhanced trade opening with channel strategy - processes last 7 candles but ignores last 2"""
        try:
            # Initialize config
            if config is None:
                from TradingConfiguration import TradingConfig
                config = TradingConfig()
            
            signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
            
            # Get the last 7 candles but ignore the last 2 (process candles 3-7 from the end)
            last_7_candles = df.tail(7)
            validation_candles = last_7_candles.head(5)  # Exclude last 2 candles
            
            self._log_message("Processing signals in last 7 candles (excluding last 2) for trade execution")
            
            # Check for signals in the validation period (candles 3-7 from the end)
            buy_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_BUY]
            sell_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_SELL]
            
            # Analyze channel conditions for signal validation
            channel_analysis = self._analyze_channel_conditions(validation_candles)
            
            self._log_message(f"Found {len(buy_signals)} buy signals, {len(sell_signals)} sell signals in validation period (candles 3-7)")
            self._log_message(f"Channel analysis: width={channel_analysis.get('avg_width', 'N/A')} pips, trend={channel_analysis.get('trend', 'N/A')}")
            
            # Process signals with channel validation
            if not buy_signals.empty:
                latest_buy = buy_signals.iloc[-1]
                if self._validate_channel_signal(latest_buy, 'BUY', channel_analysis):
                    self._log_message(f"CHANNEL BUY signal validated - opening trade from candle {latest_buy['date']}")
                    self._process_buy_signal(latest_buy)
                else:
                    self._log_message(f"BUY signal filtered by channel conditions")
            elif not sell_signals.empty:
                latest_sell = sell_signals.iloc[-1]
                if self._validate_channel_signal(latest_sell, 'SELL', channel_analysis):
                    self._log_message(f"CHANNEL SELL signal validated - opening trade from candle {latest_sell['date']}")
                    self._process_sell_signal(latest_sell)
                else:
                    self._log_message(f"SELL signal filtered by channel conditions")
            else:
                self._log_message("No signals found in validation period (candles 3-7)")
            
        except Exception as e:
            self._log_message(f"Error in triggers_trades_open: {e}", level='error')

    def _analyze_channel_conditions(self, df: pd.DataFrame) -> dict:
        """Analyze current channel conditions for signal validation"""
        analysis = {
            'avg_width': 0,
            'trend': 'neutral',
            'price_position': 'middle',
            'channel_volatility': 'low'
        }
        
        try:
            if 'median_max' in df.columns and 'median_min' in df.columns:
                # Calculate average channel width
                channel_widths = df['channel_width_pips'].dropna()
                if len(channel_widths) > 0:
                    analysis['avg_width'] = round(channel_widths.mean(), 1)
                
                # Determine trend based on EMA alignment
                if 'ema_80' in df.columns and 'ema_100' in df.columns and 'ema_200' in df.columns:
                    latest_ema_80 = df['ema_80'].iloc[-1]
                    latest_ema_100 = df['ema_100'].iloc[-1]
                    latest_ema_200 = df['ema_200'].iloc[-1]
                    
                    if not any(pd.isna([latest_ema_80, latest_ema_100, latest_ema_200])):
                        if latest_ema_80 > latest_ema_100 > latest_ema_200:
                            analysis['trend'] = 'bullish'
                        elif latest_ema_80 < latest_ema_100 < latest_ema_200:
                            analysis['trend'] = 'bearish'
                
                # Determine price position within channel
                if 'bidclose' in df.columns:
                    latest_price = df['bidclose'].iloc[-1]
                    latest_median_max = df['median_max'].iloc[-1]
                    latest_median_min = df['median_min'].iloc[-1]
                    
                    if not any(pd.isna([latest_price, latest_median_max, latest_median_min])):
                        channel_range = latest_median_max - latest_median_min
                        if channel_range > 0:
                            price_position = (latest_price - latest_median_min) / channel_range
                            if price_position > 0.7:
                                analysis['price_position'] = 'upper'
                            elif price_position < 0.3:
                                analysis['price_position'] = 'lower'
                            else:
                                analysis['price_position'] = 'middle'
                
                # Determine channel volatility
                if analysis['avg_width'] > 20:
                    analysis['channel_volatility'] = 'high'
                elif analysis['avg_width'] > 10:
                    analysis['channel_volatility'] = 'medium'
                else:
                    analysis['channel_volatility'] = 'low'
                    
        except Exception as e:
            self._log_message(f"Error analyzing channel conditions: {e}", level='error')
        
        return analysis

    def _validate_channel_signal(self, signal_data, signal_type: str, channel_analysis: dict) -> bool:
        """Validate trading signal based on channel conditions"""
        try:
            # Basic validation - always allow if channel analysis is not available
            if not channel_analysis or channel_analysis.get('avg_width', 0) == 0:
                return True
            
            # Channel width validation - avoid trading in very narrow channels
            if channel_analysis.get('avg_width', 0) < 5:
                self._log_message(f"Signal filtered: Channel too narrow ({channel_analysis.get('avg_width', 0)} pips)")
                return False
            
            # Price position validation
            price_position = channel_analysis.get('price_position', 'middle')
            trend = channel_analysis.get('trend', 'neutral')
            
            if signal_type == 'BUY':
                # For BUY signals, prefer when price is in lower part of channel or trend is bullish
                if price_position == 'upper' and trend == 'bearish':
                    self._log_message(f"BUY signal filtered: Price in upper channel with bearish trend")
                    return False
                    
            elif signal_type == 'SELL':
                # For SELL signals, prefer when price is in upper part of channel or trend is bearish
                if price_position == 'lower' and trend == 'bullish':
                    self._log_message(f"SELL signal filtered: Price in lower channel with bullish trend")
                    return False
            
            # High volatility validation - be more selective in high volatility
            if channel_analysis.get('channel_volatility') == 'high':
                if signal_type == 'BUY' and trend != 'bullish':
                    self._log_message(f"BUY signal filtered: High volatility without bullish trend")
                    return False
                elif signal_type == 'SELL' and trend != 'bearish':
                    self._log_message(f"SELL signal filtered: High volatility without bearish trend")
                    return False
            
            return True
            
        except Exception as e:
            self._log_message(f"Error validating channel signal: {e}", level='error')
            return True  # Allow signal if validation fails

    def _process_buy_signal(self, last_buy):
        try:
            buy_date = last_buy['date']
            buy_price = last_buy['bidclose']
            
            # Clear console log when processing real signal
            print(f"🟢 {self.instrument}: SEÑAL BUY procesada → Precio: {buy_price} | Fecha: {buy_date}")
            self._log_message(f"Processing BUY signal - Date: {buy_date}, Price: {buy_price}")
            
            self._open_buy_operation(buy_date, buy_price)
            
        except Exception as e:
            print(f"❌ {self.instrument}: ERROR en señal BUY → {str(e)}")
            self._log_message(f"Error processing BUY signal: {e}", level='error')

    def _open_buy_operation(self, signal_date, signal_price):
        if not self.existingOperation(instrument=self.instrument, BuySell="B"):
            print(f"📈 {self.instrument}: ABRIENDO OPERACIÓN BUY → Precio: {signal_price}")
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
            print(f"ℹ️ {self.instrument}: Operación BUY ya existe → No se abre nueva")
            self._log_message(
                f"[INFO] BUY operation already exists | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def _process_sell_signal(self, last_sell):
        try:
            sell_date = last_sell['date']
            sell_price = last_sell['bidclose']
            
            # Clear console log when processing real signal
            print(f"🔴 {self.instrument}: SEÑAL SELL procesada → Precio: {sell_price} | Fecha: {sell_date}")
            self._log_message(f"Processing SELL signal - Date: {sell_date}, Price: {sell_price}")
            
            self._open_sell_operation(sell_date, sell_price)
            
        except Exception as e:
            print(f"❌ {self.instrument}: ERROR en señal SELL → {str(e)}")
            self._log_message(f"Error processing SELL signal: {e}", level='error')

    def _open_sell_operation(self, signal_date, signal_price):
        if not self.existingOperation(instrument=self.instrument, BuySell="S"):
            print(f"📉 {self.instrument}: ABRIENDO OPERACIÓN SELL → Precio: {signal_price}")
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
            print(f"ℹ️ {self.instrument}: Operación SELL ya existe → No se abre nueva")
            self._log_message(
                f"[INFO] SELL operation already exists | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def triggers_trades_close(self, df: pd.DataFrame, config=None):
        """Enhanced trade closing with channel strategy - processes peaks and channel conditions in last 7 candles but ignores last 2"""
        try:
            # Initialize config
            if config is None:
                from TradingConfiguration import TradingConfig
                config = TradingConfig()
            
            peaks_max_col = config.peaks_max_col if hasattr(config, 'peaks_max_col') else 'peaks_max'
            peaks_min_col = config.peaks_min_col if hasattr(config, 'peaks_min_col') else 'peaks_min'
            
            # Get the last 7 candles but ignore the last 2 (process candles 3-7 from the end)
            last_7_candles = df.tail(7)
            validation_candles = last_7_candles.head(5)  # Exclude last 2 candles
            
            self._log_message("Processing close signals and peaks in last 7 candles (excluding last 2) for trade closing")
            
            # Check for peaks in the validation period (candles 3-7 from the end)
            peaks_max = validation_candles[validation_candles[peaks_max_col] == 1] if peaks_max_col in validation_candles.columns else pd.DataFrame()
            peaks_min = validation_candles[validation_candles[peaks_min_col] == 1] if peaks_min_col in validation_candles.columns else pd.DataFrame()
            
            # Check for channel breakout signals
            channel_breakout_up = validation_candles[validation_candles.get('channel_breakout_up', 0) == 1] if 'channel_breakout_up' in validation_candles.columns else pd.DataFrame()
            channel_breakout_down = validation_candles[validation_candles.get('channel_breakout_down', 0) == 1] if 'channel_breakout_down' in validation_candles.columns else pd.DataFrame()
            
            self._log_message(f"Found {len(peaks_max)} max peaks, {len(peaks_min)} min peaks, {len(channel_breakout_up)} channel breakouts up, {len(channel_breakout_down)} channel breakouts down for closing")
            
            # Close BUY operations when peaks_max is detected or channel breakout down
            if not peaks_max.empty or not channel_breakout_down.empty:
                if self.existingOperation(instrument=self.instrument, BuySell="B"):
                    if not peaks_max.empty:
                        latest_peak_max = peaks_max.iloc[-1]
                        self._log_message(f"PEAKS MAX detected - closing BUY operations from candle {latest_peak_max['date']}")
                        self._process_close_buy_peak_signal(latest_peak_max, "PEAKS_MAX")
                    elif not channel_breakout_down.empty:
                        latest_breakout = channel_breakout_down.iloc[-1]
                        self._log_message(f"CHANNEL BREAKOUT DOWN detected - closing BUY operations from candle {latest_breakout['date']}")
                        self._process_close_buy_peak_signal(latest_breakout, "CHANNEL_BREAKOUT_DOWN")
            
            # Close SELL operations when peaks_min is detected or channel breakout up
            if not peaks_min.empty or not channel_breakout_up.empty:
                if self.existingOperation(instrument=self.instrument, BuySell="S"):
                    if not peaks_min.empty:
                        latest_peak_min = peaks_min.iloc[-1]
                        self._log_message(f"PEAKS MIN detected - closing SELL operations from candle {latest_peak_min['date']}")
                        self._process_close_sell_peak_signal(latest_peak_min, "PEAKS_MIN")
                    elif not channel_breakout_up.empty:
                        latest_breakout = channel_breakout_up.iloc[-1]
                        self._log_message(f"CHANNEL BREAKOUT UP detected - closing SELL operations from candle {latest_breakout['date']}")
                        self._process_close_sell_peak_signal(latest_breakout, "CHANNEL_BREAKOUT_UP")
            
            # Check for channel reversal signals (price returns to opposite side of channel)
            self._check_channel_reversal_signals(validation_candles)
            
            if peaks_max.empty and peaks_min.empty and channel_breakout_up.empty and channel_breakout_down.empty:
                self._log_message("No close signals found in validation period (candles 3-7)")
            
        except Exception as e:
            self._log_message(f"Error in triggers_trades_close: {e}", level='error')

    def _check_channel_reversal_signals(self, df: pd.DataFrame):
        """Check for channel reversal signals to close trades"""
        try:
            if 'median_max' not in df.columns or 'median_min' not in df.columns or 'bidclose' not in df.columns:
                return
            
            # Get latest price and channel levels
            latest_price = df['bidclose'].iloc[-1]
            latest_median_max = df['median_max'].iloc[-1]
            latest_median_min = df['median_min'].iloc[-1]
            
            if any(pd.isna([latest_price, latest_median_max, latest_median_min])):
                return
            
            # Check if price has returned to the middle of the channel (reversal signal)
            channel_middle = (latest_median_max + latest_median_min) / 2
            channel_range = latest_median_max - latest_median_min
            
            if channel_range > 0:
                # If price is close to channel middle, consider closing trades
                distance_to_middle = abs(latest_price - channel_middle)
                middle_threshold = channel_range * 0.2  # Within 20% of channel middle
                
                if distance_to_middle <= middle_threshold:
                    # Close BUY operations if price returns to middle from upper channel
                    if self.existingOperation(instrument=self.instrument, BuySell="B"):
                        self._log_message(f"CHANNEL REVERSAL detected - closing BUY operations (price near channel middle)")
                        self._process_close_buy_peak_signal(df.iloc[-1], "CHANNEL_REVERSAL")
                    
                    # Close SELL operations if price returns to middle from lower channel
                    if self.existingOperation(instrument=self.instrument, BuySell="S"):
                        self._log_message(f"CHANNEL REVERSAL detected - closing SELL operations (price near channel middle)")
                        self._process_close_sell_peak_signal(df.iloc[-1], "CHANNEL_REVERSAL")
                        
        except Exception as e:
            self._log_message(f"Error checking channel reversal signals: {e}", level='error')


    def _process_close_buy_peak_signal(self, peak_data, peak_type):
        """Process closing of BUY operations when PEAKS MAX is detected"""
        try:
            peak_date = peak_data.get('date', 'N/A')
            peak_price = peak_data.get('bidclose', None)
            
            self._log_message(f"Processing CLOSE BUY peak signal - Date: {peak_date}, Price: {peak_price}, Peak: {peak_type}")
            
            # Close existing BUY operations
            self._log_message(
                f"[CLOSE BUY] Reason: {peak_type} detected | Date: {peak_date} | "
                f"Price: {peak_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="B")
            
        except Exception as e:
            self._log_message(f"Error processing close BUY peak signal: {e}", level='error')

    def _process_close_sell_peak_signal(self, peak_data, peak_type):
        """Process closing of SELL operations when PEAKS MIN is detected"""
        try:
            peak_date = peak_data.get('date', 'N/A')
            peak_price = peak_data.get('bidclose', None)
            
            self._log_message(f"Processing CLOSE SELL peak signal - Date: {peak_date}, Price: {peak_price}, Peak: {peak_type}")
            
            # Close existing SELL operations
            self._log_message(
                f"[CLOSE SELL] Reason: {peak_type} detected | Date: {peak_date} | "
                f"Price: {peak_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="S")
            
        except Exception as e:
            self._log_message(f"Error processing close SELL peak signal: {e}", level='error')

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
                        latest_price = self.get_latest_price()
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
                    latest_price = self.get_latest_price()
                    self._append_trade_log(action='OPEN', side=original_side, price=latest_price, signal_date='', details='No signal context')
                
            except Exception as e:
                self._log_message(f"Error opening operation: {e}", level='error')
                
        except Exception as e:
            self._log_message(f"Error in createEntryOrder: {e}", level='error')

    def list_available_instruments(self):
        """List all available instruments from FXCM"""
        try:
            if not ROBOT_CONNECTION_AVAILABLE or self.connection is None:
                self._log_message("Connection not available for listing instruments", level='error')
                return []
            
            # Get offers table
            offers_table = self.connection.get_table(self.connection.OFFERS)
            instruments = []
            
            if offers_table:
                self._log_message(f"Found {len(offers_table)} available instruments")
                for offer in offers_table:
                    instrument_name = offer.instrument
                    instruments.append(instrument_name)
                    # Only log forex pairs to reduce noise
                    if '/' in instrument_name:
                        self._log_message(f"Available forex pair: {instrument_name}")
            else:
                self._log_message("No offers table available", level='error')
                
            return instruments
        except Exception as e:
            self._log_message(f"Error listing instruments: {e}", level='error')
            return []


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

