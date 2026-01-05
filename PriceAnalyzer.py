import pandas as pd
import numpy as np
from scipy import signal
from datetime import datetime
import pytz
import datetime as dt
import logging
import os
import time
import threading
from config import TradingConfig
from MarketConditionAnalyzer import MarketConditionAnalyzer

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
    
    # Shared lock for FXCM API calls to prevent concurrent access issues
    _fxcm_lock = threading.Lock()

    def __init__(self, days: int, instrument: str, timeframe: str):
        self.instrument = instrument
        self.timeframe = timeframe
        self.pricedata = None
        self.days = days
        self.trade_log_file = os.path.join('logs', 'triggers_trades_open.csv')
        self._last_signal_info = None  # holds context for logging after order open
        
        # Track last processed signal to avoid duplicate signals
        self._last_processed_signal_date = None
        self._last_processed_signal_side = None
        
        # Track last close time for cooldown period (5 minutes = 300 seconds)
        self._last_close_time = {}
        self._cooldown_period_seconds = 300  # 5 minutes cooldown after closing
        
        # Initialize market condition analyzer with config values
        self._init_market_condition_analyzer()
        
        self._setup_logging()
        self._ensure_trade_log_file()
        
        if ROBOT_CONNECTION_AVAILABLE:
            self.robotconnection = RobotConnection()
            self.connection = self.robotconnection.getConnection()
            # List available instruments on connection
            self._log_message("Listing available instruments from FXCM:")
            available_instruments = self.list_available_instruments()
            if available_instruments:
                self._log_message(f"Found {len(available_instruments)} available instruments")
        else:
            self.robotconnection = None
            self.connection = None

    def _init_market_condition_analyzer(self):
        """Initialize market condition analyzer with config parameters."""
        try:
            self.market_filter_enabled = TradingConfig.get_market_filter_enabled()
            
            if self.market_filter_enabled:
                self.market_analyzer = MarketConditionAnalyzer(
                    window_liquidity=TradingConfig.get_window_liquidity(),
                    window_volatility=TradingConfig.get_window_volatility(),
                    window_movement=TradingConfig.get_window_movement()
                )
                
                # Set thresholds from config
                self.market_analyzer.min_liquidity_percentile = TradingConfig.get_min_liquidity_percentile()
                self.market_analyzer.min_atr_percentile = TradingConfig.get_min_atr_percentile()
                self.market_analyzer.min_movement_frequency = TradingConfig.get_min_movement_frequency()
                self.market_analyzer.min_range_pips = TradingConfig.get_min_range_pips()
            else:
                self.market_analyzer = None
        except Exception as e:
            self.market_filter_enabled = False
            self.market_analyzer = None
            # Will log later when logger is set up

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
        """Get price data with retry logic and thread-safe access to FXCM API"""
        max_retries = 3
        retry_delay = 0.5  # Start with 0.5 seconds
        
        for attempt in range(max_retries):
            try:
                europe_London_datetime = datetime.now(pytz.timezone('Europe/London'))
                date_from = europe_London_datetime - dt.timedelta(days=days)
                date_to = europe_London_datetime
                
                # Use lock to serialize FXCM API calls and prevent "Quotes storage is busy" errors
                with PriceAnalyzer._fxcm_lock:
                    history = connection.get_history(instrument, timeframe, date_from, date_to)
                   
                pricedata = pd.DataFrame(history, columns=["Date", "BidOpen", "BidHigh", "BidLow", "BidClose", "Volume"])
               
                if pricedata.empty:
                    self._log_message(f"Empty DataFrame created for {instrument} {timeframe}", level='warning')
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
                error_msg = str(e)
                is_locked_error = 'Quotes storage is busy' in error_msg or 'Locked' in error_msg
                
                if is_locked_error and attempt < max_retries - 1:
                    # Exponential backoff for locked errors (even with lock, sometimes FXCM needs time)
                    wait_time = retry_delay * (2 ** attempt)
                    self._log_message(f"Quotes storage busy for {instrument} {timeframe}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})", level='warning')
                    time.sleep(wait_time)
                    continue
                elif not is_locked_error:
                    # For non-locked errors, don't retry
                    self._log_message(f"Error in get_price_data for {instrument} {timeframe}: {error_msg}", level='error')
                    if attempt == max_retries - 1:
                        import traceback
                        self._log_message(f"Traceback: {traceback.format_exc()}", level='error')
                    return pd.DataFrame()
                else:
                    self._log_message(f"Error in get_price_data for {instrument} {timeframe}: {error_msg}", level='error')
                    if attempt == max_retries - 1:
                        import traceback
                        self._log_message(f"Traceback: {traceback.format_exc()}", level='error')
                    return pd.DataFrame()
        
        return pd.DataFrame()

    def save_price_data_file(self, pricedata: pd.DataFrame):
        fileName = os.path.join('data', self.instrument.replace("/", "_") + "_" + self.timeframe + ".csv")
        pricedata.to_csv(fileName)

    def get_latest_price(self, instrument: str, BuySell: str) -> float:
        if self.pricedata is not None and not self.pricedata.empty:
            return float(self.pricedata['bidclose'].iloc[-1])
        return None

    def set_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with validation"""
        try:
            # Validate DataFrame is not empty and has required columns
            if df is None or df.empty:
                self._log_message("Cannot calculate indicators: DataFrame is empty", level='warning')
                return df
            
            if 'bidclose' not in df.columns:
                self._log_message("Cannot calculate indicators: missing 'bidclose' column", level='error')
                return df
            
            # Calculate peaks
            df = self.calculate_peaks(df)
            # Calculate medians
            df = self.calculate_medians(df)
            # Calculate EMA 200
            df = self.calculate_ema_200(df)
            
            # Calculate advanced indicators for improved strategy
            df = self.calculate_rsi(df, period=14)
            df = self.calculate_macd(df)
            df = self.calculate_atr(df, period=14)
            df = self.calculate_ema_fast_slow(df, fast=12, slow=26)
            
            # Calculate market condition metrics if filter is enabled
            if self.market_filter_enabled and self.market_analyzer is not None:
                try:
                    df = self.market_analyzer.analyze_market_conditions(df)
                    self._log_message(f"Market condition analysis completed: liquidity, volatility, movement frequency, spread metrics added")
                except Exception as e:
                    self._log_message(f"Error in market condition analysis: {str(e)}", level='warning')
            
            self._log_message(f"All indicators calculated successfully for {len(df)} rows: peaks, medians, EMA200, RSI, MACD, ATR, EMA fast/slow")
            return df
        except Exception as e:
            self._log_message(f"Error setting indicators: {str(e)}", level='error')
            return df


    def calculate_peaks(self, df: pd.DataFrame, order: int = 50) -> pd.DataFrame:
        self._add_price_peaks(df, order)
        return df

    def calculate_medians(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """Calculate rolling medians for bidhigh, bidlow, bidclose, bidopen"""
        if df is None or df.empty:
            return df
        
        # Initialize median columns
        df['median_high'] = np.nan
        df['median_low'] = np.nan
        df['median_close'] = np.nan
        df['median_open'] = np.nan
        
        # Calculate rolling medians
        if 'bidhigh' in df.columns:
            df['median_high'] = df['bidhigh'].rolling(window=window, min_periods=1).median()
        
        if 'bidlow' in df.columns:
            df['median_low'] = df['bidlow'].rolling(window=window, min_periods=1).median()
        
        if 'bidclose' in df.columns:
            df['median_close'] = df['bidclose'].rolling(window=window, min_periods=1).median()
        
        if 'bidopen' in df.columns:
            df['median_open'] = df['bidopen'].rolling(window=window, min_periods=1).median()
        
        return df

    def calculate_ema_200(self, df: pd.DataFrame, period: int = 200) -> pd.DataFrame:
        """Calculate Exponential Moving Average of 200 periods on bidclose"""
        if df is None or df.empty or 'bidclose' not in df.columns:
            return df
        
        # Initialize EMA column
        df['ema_200'] = np.nan
        
        # Calculate EMA using pandas ewm (exponentially weighted moving average)
        df['ema_200'] = df['bidclose'].ewm(span=period, adjust=False).mean()
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI)"""
        if df is None or df.empty or 'bidclose' not in df.columns:
            return df
        
        delta = df['bidclose'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # Default to neutral
        
        return df
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if df is None or df.empty or 'bidclose' not in df.columns:
            return df
        
        # Calculate fast and slow EMAs
        ema_fast = df['bidclose'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['bidclose'].ewm(span=slow, adjust=False).mean()
        
        # MACD line
        df['macd'] = ema_fast - ema_slow
        
        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        
        # Histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range (ATR)"""
        if df is None or df.empty:
            return df
        
        if not all(col in df.columns for col in ['bidhigh', 'bidlow', 'bidclose']):
            df['atr'] = 0.0
            return df
        
        high_low = df['bidhigh'] - df['bidlow']
        high_close = abs(df['bidhigh'] - df['bidclose'].shift(1))
        low_close = abs(df['bidlow'] - df['bidclose'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period, min_periods=1).mean()
        df['atr'] = df['atr'].fillna(0.0)
        
        return df
    
    def calculate_ema_fast_slow(self, df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
        """Calculate fast and slow EMAs for trend confirmation"""
        if df is None or df.empty or 'bidclose' not in df.columns:
            return df
        
        df['ema_fast'] = df['bidclose'].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df['bidclose'].ewm(span=slow, adjust=False).mean()
        
        return df

    def _add_price_peaks(self, df, order):        
            df['peaks_min'] = 0
            df['peaks_max'] = 0

            peaks_min_idx = signal.argrelextrema(df['bidclose'].values, np.less, order=order)[0]
            peaks_max_idx = signal.argrelextrema(df['bidclose'].values, np.greater, order=order)[0]
            
            df.loc[peaks_min_idx, 'peaks_min'] = 1
            df.loc[peaks_max_idx, 'peaks_max'] = 1

            
    def set_signals_to_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate improved signals using multiple technical indicators for better profitability"""
        if df is None or df.empty:
            return df
            
        df['signal'] = self.SIGNAL_NEUTRAL
        buy_signals = 0
        sell_signals = 0
        filtered_buy = 0
        filtered_sell = 0
        filtered_market_conditions = 0
        
        # Check if required columns exist
        has_medians = all(col in df.columns for col in ['median_high', 'median_low', 'median_close', 'median_open'])
        has_ema = 'ema_200' in df.columns
        has_rsi = 'rsi' in df.columns
        has_macd = 'macd' in df.columns and 'macd_signal' in df.columns
        has_ema_fast_slow = 'ema_fast' in df.columns and 'ema_slow' in df.columns
        has_market_conditions = 'is_good_condition' in df.columns if self.market_filter_enabled else False
        
        for i in range(len(df)):
            # Improved BUY signal logic: More flexible with confirmations
            if df['peaks_min'].iloc[i] == 1:  # Potential BUY signal at LOW peak
                confirmations = 0
                max_confirmations = 0
                
                # 1. Trend confirmation: Price should be above EMA 200 (uptrend) - STRONG signal
                if has_ema:
                    ema_value = df['ema_200'].iloc[i]
                    current_price = df['bidclose'].iloc[i]
                    if not pd.isna(ema_value) and not pd.isna(current_price):
                        max_confirmations += 1
                        if current_price > ema_value:
                            confirmations += 1
                        # Don't invalidate if price is close to EMA (within 0.1%)
                        elif abs(current_price - ema_value) / ema_value < 0.001:
                            confirmations += 0.5
                
                # 2. EMA Fast/Slow confirmation: Fast EMA above Slow EMA (bullish)
                if has_ema_fast_slow:
                    ema_fast = df['ema_fast'].iloc[i]
                    ema_slow = df['ema_slow'].iloc[i]
                    if not pd.isna(ema_fast) and not pd.isna(ema_slow):
                        max_confirmations += 1
                        if ema_fast > ema_slow:
                            confirmations += 1
                        # Allow if they're close (within 0.05%)
                        elif abs(ema_fast - ema_slow) / ema_slow < 0.0005:
                            confirmations += 0.5
                
                # 3. RSI confirmation: More flexible RSI ranges
                if has_rsi:
                    rsi = df['rsi'].iloc[i]
                    if not pd.isna(rsi):
                        max_confirmations += 1
                        # Accept wider range: 20-60 for BUY (oversold to neutral)
                        if 20 <= rsi <= 60:
                            confirmations += 1
                        # Very oversold (10-20) still acceptable but less ideal
                        elif 10 <= rsi < 20:
                            confirmations += 0.5
                
                # 4. MACD confirmation: More flexible MACD conditions
                if has_macd:
                    macd = df['macd'].iloc[i]
                    macd_signal = df['macd_signal'].iloc[i]
                    macd_hist = df['macd_hist'].iloc[i]
                    if not pd.isna(macd) and not pd.isna(macd_signal):
                        max_confirmations += 1
                        # MACD above signal OR crossing above OR histogram positive
                        if macd > macd_signal or (macd_hist > 0 and i > 0 and df['macd_hist'].iloc[i-1] <= 0):
                            confirmations += 1
                        # Allow if MACD is close to signal (within 10% of signal value)
                        elif abs(macd - macd_signal) / (abs(macd_signal) + 1e-10) < 0.1:
                            confirmations += 0.5
                
                # 5. Median confirmation: Medians should support uptrend (optional)
                if has_medians and has_ema:
                    ema_value = df['ema_200'].iloc[i]
                    median_close = df['median_close'].iloc[i]
                    if not pd.isna(ema_value) and not pd.isna(median_close):
                        max_confirmations += 1
                        if median_close > ema_value:
                            confirmations += 1
                
                # Require at least 2 confirmations (reduced from 3) OR 1.5 if we have fewer indicators
                min_confirmations = max(1.5, max_confirmations * 0.4)  # At least 40% of available confirmations
                if confirmations >= min_confirmations:
                    # Market condition filter - improved threshold (0.35 instead of 0.2)
                    if has_market_conditions:
                        market_quality = df.get('market_quality_score', pd.Series([0.5] * len(df))).iloc[i]
                        if not pd.isna(market_quality) and market_quality < 0.35:  # Block poor conditions (increased from 0.2)
                            filtered_market_conditions += 1
                            continue
                    
                    df.at[i, 'signal'] = self.SIGNAL_BUY
                    buy_signals += 1
                else:
                    filtered_buy += 1
                    
            # Improved SELL signal logic: More flexible with confirmations
            elif df['peaks_max'].iloc[i] == 1:  # Potential SELL signal at HIGH peak
                confirmations = 0
                max_confirmations = 0
                
                # 1. Trend confirmation: Price should be below EMA 200 (downtrend) - STRONG signal
                if has_ema:
                    ema_value = df['ema_200'].iloc[i]
                    current_price = df['bidclose'].iloc[i]
                    if not pd.isna(ema_value) and not pd.isna(current_price):
                        max_confirmations += 1
                        if current_price < ema_value:
                            confirmations += 1
                        # Don't invalidate if price is close to EMA (within 0.1%)
                        elif abs(current_price - ema_value) / ema_value < 0.001:
                            confirmations += 0.5
                
                # 2. EMA Fast/Slow confirmation: Fast EMA below Slow EMA (bearish)
                if has_ema_fast_slow:
                    ema_fast = df['ema_fast'].iloc[i]
                    ema_slow = df['ema_slow'].iloc[i]
                    if not pd.isna(ema_fast) and not pd.isna(ema_slow):
                        max_confirmations += 1
                        if ema_fast < ema_slow:
                            confirmations += 1
                        # Allow if they're close (within 0.05%)
                        elif abs(ema_fast - ema_slow) / ema_slow < 0.0005:
                            confirmations += 0.5
                
                # 3. RSI confirmation: More flexible RSI ranges
                if has_rsi:
                    rsi = df['rsi'].iloc[i]
                    if not pd.isna(rsi):
                        max_confirmations += 1
                        # Accept wider range: 40-80 for SELL (neutral to overbought)
                        if 40 <= rsi <= 80:
                            confirmations += 1
                        # Very overbought (80-90) still acceptable but less ideal
                        elif 80 < rsi <= 90:
                            confirmations += 0.5
                
                # 4. MACD confirmation: More flexible MACD conditions
                if has_macd:
                    macd = df['macd'].iloc[i]
                    macd_signal = df['macd_signal'].iloc[i]
                    macd_hist = df['macd_hist'].iloc[i]
                    if not pd.isna(macd) and not pd.isna(macd_signal):
                        max_confirmations += 1
                        # MACD below signal OR crossing below OR histogram negative
                        if macd < macd_signal or (macd_hist < 0 and i > 0 and df['macd_hist'].iloc[i-1] >= 0):
                            confirmations += 1
                        # Allow if MACD is close to signal (within 10% of signal value)
                        elif abs(macd - macd_signal) / (abs(macd_signal) + 1e-10) < 0.1:
                            confirmations += 0.5
                
                # 5. Median confirmation: Medians should support downtrend (optional)
                if has_medians and has_ema:
                    ema_value = df['ema_200'].iloc[i]
                    median_close = df['median_close'].iloc[i]
                    if not pd.isna(ema_value) and not pd.isna(median_close):
                        max_confirmations += 1
                        if median_close < ema_value:
                            confirmations += 1
                
                # Require at least 2 confirmations (reduced from 3) OR 1.5 if we have fewer indicators
                min_confirmations = max(1.5, max_confirmations * 0.4)  # At least 40% of available confirmations
                if confirmations >= min_confirmations:
                    # Market condition filter - improved threshold (0.35 instead of 0.2)
                    if has_market_conditions:
                        market_quality = df.get('market_quality_score', pd.Series([0.5] * len(df))).iloc[i]
                        if not pd.isna(market_quality) and market_quality < 0.35:  # Block poor conditions (increased from 0.2)
                            filtered_market_conditions += 1
                            continue
                    
                    df.at[i, 'signal'] = self.SIGNAL_SELL
                    sell_signals += 1
                else:
                    filtered_sell += 1
                
        df['valid_signal'] = df['signal']
        
        # Log results
        filter_msg = ""
        if has_medians and has_ema:
            filter_msg += f" (filtered: {filtered_buy} buy, {filtered_sell} sell)"
        if has_market_conditions and filtered_market_conditions > 0:
            filter_msg += f" (filtered by market conditions: {filtered_market_conditions})"
        
        self._log_message(f"Improved signals generated: buy={buy_signals} sell={sell_signals}{filter_msg} instrument={self.instrument} timeframe={self.timeframe}")
        
        return df

    def _set_signal(self, df, idx, signal_col, value):
        df.iloc[idx, df.columns.get_loc(signal_col)] = value

    def triggers_trades_open(self, df: pd.DataFrame, config=None):
        """Simplified trade opening - processes last 7 candles but ignores last 2"""
        try:
            # Validate DataFrame
            if df is None or df.empty:
                self._log_message("Cannot process trade signals: DataFrame is empty", level='warning')
                return
            
            # Initialize config
            if config is None:
                from config import TradingConfig
                config = TradingConfig()
            
            signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
            
            # Check if signal column exists
            if signal_col not in df.columns:
                self._log_message(f"Cannot process trade signals: missing '{signal_col}' column", level='warning')
                return
            
            # Get the last 7 candles but ignore the last 2 (process candles 3-7 from the end)
            if len(df) < 7:
                self._log_message(f"Not enough data for signal processing: {len(df)} rows (need at least 7)", level='warning')
                return
            
            last_7_candles = df.tail(7)
            validation_candles = last_7_candles.head(5)  # Exclude last 2 candles
            
            self._log_message(f"Processing signals in last 7 candles (excluding last 2) for trade execution")
            
            # Check for signals in the validation period (candles 3-7 from the end)
            buy_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_BUY]
            sell_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_SELL]
            
            self._log_message(f"Found {len(buy_signals)} buy signals, {len(sell_signals)} sell signals in validation period (candles 3-7)")
            
            # Process signals immediately, but avoid duplicate signals
            if not buy_signals.empty:
                latest_buy = buy_signals.iloc[-1]
                signal_date = latest_buy['date']
                
                # Check if this is a duplicate signal (same signal_date and side as last processed)
                if self._last_processed_signal_date == signal_date and self._last_processed_signal_side == 'B':
                    self._log_message(f"BUY signal with date {signal_date} already processed, skipping duplicate")
                else:
                    self._log_message(f"BUY signal detected - opening trade from candle {signal_date}")
                    self._process_buy_signal(latest_buy)
                    self._last_processed_signal_date = signal_date
                    self._last_processed_signal_side = 'B'
            elif not sell_signals.empty:
                latest_sell = sell_signals.iloc[-1]
                signal_date = latest_sell['date']
                
                # Check if this is a duplicate signal (same signal_date and side as last processed)
                if self._last_processed_signal_date == signal_date and self._last_processed_signal_side == 'S':
                    self._log_message(f"SELL signal with date {signal_date} already processed, skipping duplicate")
                else:
                    self._log_message(f"SELL signal detected - opening trade from candle {signal_date}")
                    self._process_sell_signal(latest_sell)
                    self._last_processed_signal_date = signal_date
                    self._last_processed_signal_side = 'S'
            else:
                self._log_message("No signals found in validation period (candles 3-7)")
            
        except Exception as e:
            self._log_message(f"Error in triggers_trades_open: {e}", level='error')

    def _process_buy_signal(self, last_buy):
        try:
            buy_date = last_buy['date']
            buy_price = last_buy['bidclose']
            
            # Additional market condition check before opening (even if signal passed initial filter)
            # This provides a double-check right before opening the trade
            if hasattr(self, 'market_filter_enabled') and self.market_filter_enabled:
                market_quality = last_buy.get('market_quality_score', None)
                if market_quality is not None and not pd.isna(market_quality):
                    if market_quality < 0.35:  # Same threshold as signal generation
                        self._log_message(
                            f"[SKIP BUY] Market conditions too poor (quality={market_quality:.3f} < 0.35) | "
                            f"Date: {buy_date} | Price: {buy_price}"
                        )
                        return
            
            # Calculate dynamic stop loss and take profit based on ATR
            atr_value = last_buy.get('atr', None)
            dynamic_stop = None
            dynamic_limit = None
            
            if atr_value is not None and not pd.isna(atr_value) and atr_value > 0:
                # Stop loss: 1.5x ATR below entry
                # Take profit: 3x ATR above entry (2:1 risk/reward ratio)
                # Convert ATR to pips (assuming 4 decimal places for most forex pairs)
                atr_pips = atr_value * 10000  # Convert to pips
                dynamic_stop = int(max(5, atr_pips * 1.5))  # Minimum 5 pips
                dynamic_limit = int(dynamic_stop * 2)  # 2:1 risk/reward
                self._log_message(f"Dynamic SL/TP calculated: ATR={atr_value:.6f} ({atr_pips:.1f} pips), Stop={dynamic_stop} pips, Limit={dynamic_limit} pips")
            else:
                # Fallback to config values
                from config import TradingConfig
                dynamic_stop = TradingConfig.get_stop()
                dynamic_limit = TradingConfig.get_limit()
                self._log_message(f"Using config SL/TP: Stop={dynamic_stop} pips, Limit={dynamic_limit} pips")
            
            self._log_message(f"Processing BUY signal - Date: {buy_date}, Price: {buy_price}")
            
            self._close_existing_sell_operations(buy_date, buy_price)
            
            self._open_buy_operation(buy_date, buy_price, dynamic_stop, dynamic_limit)
            
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

    def _open_buy_operation(self, signal_date, signal_price, dynamic_stop=None, dynamic_limit=None):
        # Check cooldown period
        if 'B' in self._last_close_time:
            time_since_close = (datetime.now(pytz.timezone('Europe/London')) - self._last_close_time['B']).total_seconds()
            if time_since_close < self._cooldown_period_seconds:
                self._log_message(
                    f"[SKIP BUY] Cooldown period active: {time_since_close:.1f}s / {self._cooldown_period_seconds}s | "
                    f"Date: {signal_date} | Price: {signal_price}"
                )
                return
        
        if not self.existingOperation(instrument=self.instrument, BuySell="B"):
            self._log_message(
                f"[OPEN BUY] Reason: BUY signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe} | "
                f"Stop: {dynamic_stop} pips | Limit: {dynamic_limit} pips"
            )
            # store context so createEntryOrder can log after order submission
            self._last_signal_info = {
                'side': 'B',
                'signal_date': str(signal_date),
                'price': float(signal_price) if signal_price is not None else None,
                'details': 'BUY signal',
                'dynamic_stop': dynamic_stop,
                'dynamic_limit': dynamic_limit
            }
            self.createEntryOrder(str_buy_sell="B", dynamic_stop=dynamic_stop, dynamic_limit=dynamic_limit)
        else:
            self._log_message(
                f"[INFO] BUY operation already exists | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def _process_sell_signal(self, last_sell):
        try:
            sell_date = last_sell['date']
            sell_price = last_sell['bidclose']
            
            # Additional market condition check before opening (even if signal passed initial filter)
            # This provides a double-check right before opening the trade
            if hasattr(self, 'market_filter_enabled') and self.market_filter_enabled:
                market_quality = last_sell.get('market_quality_score', None)
                if market_quality is not None and not pd.isna(market_quality):
                    if market_quality < 0.35:  # Same threshold as signal generation
                        self._log_message(
                            f"[SKIP SELL] Market conditions too poor (quality={market_quality:.3f} < 0.35) | "
                            f"Date: {sell_date} | Price: {sell_price}"
                        )
                        return
            
            # Calculate dynamic stop loss and take profit based on ATR
            atr_value = last_sell.get('atr', None)
            dynamic_stop = None
            dynamic_limit = None
            
            if atr_value is not None and not pd.isna(atr_value) and atr_value > 0:
                # Stop loss: 1.5x ATR above entry
                # Take profit: 3x ATR below entry (2:1 risk/reward ratio)
                # Convert ATR to pips (assuming 4 decimal places for most forex pairs)
                atr_pips = atr_value * 10000  # Convert to pips
                dynamic_stop = int(max(5, atr_pips * 1.5))  # Minimum 5 pips
                dynamic_limit = int(dynamic_stop * 2)  # 2:1 risk/reward
                self._log_message(f"Dynamic SL/TP calculated: ATR={atr_value:.6f} ({atr_pips:.1f} pips), Stop={dynamic_stop} pips, Limit={dynamic_limit} pips")
            else:
                # Fallback to config values
                from config import TradingConfig
                dynamic_stop = TradingConfig.get_stop()
                dynamic_limit = TradingConfig.get_limit()
                self._log_message(f"Using config SL/TP: Stop={dynamic_stop} pips, Limit={dynamic_limit} pips")
            
            self._log_message(f"Processing SELL signal - Date: {sell_date}, Price: {sell_price}")
            
            self._close_existing_buy_operations(sell_date, sell_price)
            
            self._open_sell_operation(sell_date, sell_price, dynamic_stop, dynamic_limit)
            
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

    def _open_sell_operation(self, signal_date, signal_price, dynamic_stop=None, dynamic_limit=None):
        # Check cooldown period
        if 'S' in self._last_close_time:
            time_since_close = (datetime.now(pytz.timezone('Europe/London')) - self._last_close_time['S']).total_seconds()
            if time_since_close < self._cooldown_period_seconds:
                self._log_message(
                    f"[SKIP SELL] Cooldown period active: {time_since_close:.1f}s / {self._cooldown_period_seconds}s | "
                    f"Date: {signal_date} | Price: {signal_price}"
                )
                return
        
        if not self.existingOperation(instrument=self.instrument, BuySell="S"):
            self._log_message(
                f"[OPEN SELL] Reason: SELL signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe} | "
                f"Stop: {dynamic_stop} pips | Limit: {dynamic_limit} pips"
            )
            # store context so createEntryOrder can log after order submission
            self._last_signal_info = {
                'side': 'S',
                'signal_date': str(signal_date),
                'price': float(signal_price) if signal_price is not None else None,
                'details': 'SELL signal',
                'dynamic_stop': dynamic_stop,
                'dynamic_limit': dynamic_limit
            }
            self.createEntryOrder(str_buy_sell="S", dynamic_stop=dynamic_stop, dynamic_limit=dynamic_limit)
        else:
            self._log_message(
                f"[INFO] SELL operation already exists | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )

    def triggers_trades_close(self, df: pd.DataFrame, config=None):
        """Simplified trade closing - processes last 7 candles but ignores last 2"""
        try:
            # Validate DataFrame
            if df is None or df.empty:
                self._log_message("Cannot process close signals: DataFrame is empty", level='warning')
                return
            
            # Initialize config
            if config is None:
                from config import TradingConfig
                config = TradingConfig()
            
            signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
            
            # Check if signal column exists
            if signal_col not in df.columns:
                self._log_message(f"Cannot process close signals: missing '{signal_col}' column", level='warning')
                return
            
            # Get the last 7 candles but ignore the last 2 (process candles 3-7 from the end)
            if len(df) < 7:
                self._log_message(f"Not enough data for close signal processing: {len(df)} rows (need at least 7)", level='warning')
                return
            
            last_7_candles = df.tail(7)
            validation_candles = last_7_candles.head(5)  # Exclude last 2 candles
            
            self._log_message(f"Processing close signals in last 7 candles (excluding last 2) for trade closing")
            
            # Check for opposite signals in the validation period (candles 3-7 from the end)
            buy_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_BUY]
            sell_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_SELL]
            
            self._log_message(f"Found {len(buy_signals)} buy signals, {len(sell_signals)} sell signals for closing in validation period (candles 3-7)")
            
            # Close opposite positions when signals change
            if not buy_signals.empty:
                # BUY signal detected - close any existing SELL operations
                if self.existingOperation(instrument=self.instrument, BuySell="S"):
                    latest_buy = buy_signals.iloc[-1]
                    self._log_message(f"BUY signal detected - closing SELL operations from candle {latest_buy['date']}")
                    self._process_close_sell_signal(latest_buy)
                    
            if not sell_signals.empty:
                # SELL signal detected - close any existing BUY operations  
                if self.existingOperation(instrument=self.instrument, BuySell="B"):
                    latest_sell = sell_signals.iloc[-1]
                    self._log_message(f"SELL signal detected - closing BUY operations from candle {latest_sell['date']}")
                    self._process_close_buy_signal(latest_sell)
            
            if buy_signals.empty and sell_signals.empty:
                self._log_message("No close signals found in validation period (candles 3-7)")
            
        except Exception as e:
            self._log_message(f"Error in triggers_trades_close: {e}", level='error')

    def _process_close_buy_signal(self, signal_data):
        """Process closing of BUY operations when SELL signal is detected"""
        try:
            signal_date = signal_data.get('date', 'N/A')
            signal_price = signal_data.get('bidclose', None)
            
            self._log_message(f"Processing CLOSE BUY signal - Date: {signal_date}, Price: {signal_price}")
            
            # Close existing BUY operations
            self._log_message(
                f"[CLOSE BUY] Reason: SELL signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="B")
            
        except Exception as e:
            self._log_message(f"Error processing close BUY signal: {e}", level='error')

    def _process_close_sell_signal(self, signal_data):
        """Process closing of SELL operations when BUY signal is detected"""
        try:
            signal_date = signal_data.get('date', 'N/A')
            signal_price = signal_data.get('bidclose', None)
            
            self._log_message(f"Processing CLOSE SELL signal - Date: {signal_date}, Price: {signal_price}")
            
            # Close existing SELL operations
            self._log_message(
                f"[CLOSE SELL] Reason: BUY signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="S")
            
        except Exception as e:
            self._log_message(f"Error processing close SELL signal: {e}", level='error')

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
                        # Track close time for cooldown period
                        self._last_close_time[BuySell] = datetime.now(pytz.timezone('Europe/London'))
                        # Reset last processed signal to allow new signals after cooldown
                        self._last_processed_signal_date = None
                        self._last_processed_signal_side = None
                        # log close to CSV
                        latest_price = self.get_latest_price(instrument=instrument, BuySell=BuySell)
                        self._append_trade_log(action='CLOSE', side=BuySell, price=latest_price, signal_date='', details=f'TradeID={trade.trade_id}')
        except Exception as e:
            self._log_message(f"Error closing operation: {e}", level='error')

    def createEntryOrder(self, str_buy_sell: str = None, dynamic_stop=None, dynamic_limit=None):
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
            
            # Use dynamic stop/limit if provided, otherwise use config values
            stop = dynamic_stop if dynamic_stop is not None else args.stop
            limit = dynamic_limit if dynamic_limit is not None else args.limit
            
            peggedstop = args.peggedstop
            pegstoptype = args.pegstoptype
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

