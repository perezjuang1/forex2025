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
from MarketHoursChecker import MarketHoursChecker

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
        self.market_checker = MarketHoursChecker()  # Initialize market hours checker
        self._suspension_state = {'is_suspended': False, 'next_check_time': None, 'reason': ''}
        
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
            file_handler, console_handler = self._create_log_handlers(log_file)
            self.logger.addHandler(file_handler)
            # Remove console handler to reduce console spam
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

    def check_instrument_availability(self) -> dict:
        """Check if the instrument is available for trading using market hours checker"""
        try:
            current_time = datetime.now(pytz.timezone('UTC'))
            market_status = self.market_checker.is_market_open(self.instrument, current_time)
            recommendation = self.market_checker.get_trading_recommendation(self.instrument, current_time)
            
            # Only log if there's an issue
            if not market_status['is_available']:
                self._log_message(f"{self.instrument}: {market_status['status']} - {market_status['reason']}", level='warning')
            
            return {
                'is_available': market_status['is_available'],
                'market_status': market_status,
                'recommendation': recommendation,
                'should_suspend': not market_status['is_available'] or recommendation['recommendation'] == 'wait'
            }
        except Exception as e:
            self._log_message(f"Error checking instrument availability: {e}", level='error')
            return {
                'is_available': False,
                'market_status': {'status': 'error', 'reason': str(e)},
                'recommendation': {'recommendation': 'wait'},
                'should_suspend': True
            }

    def handle_instrument_suspension(self, availability_info: dict):
        """Handle suspension when instrument is not available"""
        if availability_info['should_suspend']:
            market_status = availability_info['market_status']
            next_open = market_status.get('next_open')
            
            if next_open:
                if isinstance(next_open, str):
                    try:
                        next_open_dt = datetime.fromisoformat(next_open.replace('Z', '+00:00'))
                    except Exception:
                        next_open_dt = datetime.now(pytz.timezone('UTC')) + dt.timedelta(hours=1)
                else:
                    next_open_dt = next_open
                
                self._suspension_state = {
                    'is_suspended': True,
                    'next_check_time': next_open_dt,
                    'reason': market_status.get('reason', 'Market not available')
                }
                
                wait_minutes = max(1, int((next_open_dt - datetime.now(pytz.timezone('UTC'))).total_seconds() / 60))
                
                self._log_message(f"[SUSPEND] {self.instrument} suspended until {next_open_dt.strftime('%H:%M')} UTC", level='warning')
                
                return True
            else:
                # No specific reopen time, check again in 30 minutes
                next_check = datetime.now(pytz.timezone('UTC')) + dt.timedelta(minutes=30)
                self._suspension_state = {
                    'is_suspended': True,
                    'next_check_time': next_check,
                    'reason': market_status.get('reason', 'Market not available')
                }
                
                self._log_message(f"[SUSPEND] {self.instrument} suspended - checking again at {next_check.strftime('%H:%M')} UTC", level='warning')
                return True
        
        return False

    def is_instrument_suspended(self) -> bool:
        """Check if instrument is currently suspended"""
        if not self._suspension_state['is_suspended']:
            return False
        
        current_time = datetime.now(pytz.timezone('UTC'))
        next_check = self._suspension_state['next_check_time']
        
        if next_check and current_time >= next_check:
            # Time to check if market is available again
            availability_info = self.check_instrument_availability()
            
            if not availability_info['should_suspend']:
                # Market is available again
                self._suspension_state = {'is_suspended': False, 'next_check_time': None, 'reason': ''}
                print(f"✅ {self.instrument}: REACTIVADO → Opera normal")
                return False
            else:
                # Still not available, extend suspension
                self.handle_instrument_suspension(availability_info)
                return True
        
        return True

    def wait_for_instrument_availability(self):
        """Wait until instrument becomes available or until next check time"""
        if not self._suspension_state['is_suspended']:
            return
        
        current_time = datetime.now(pytz.timezone('UTC'))
        next_check = self._suspension_state['next_check_time']
        
        if next_check:
            wait_seconds = max(0, (next_check - current_time).total_seconds())
            if wait_seconds > 0:
                wait_minutes = wait_seconds / 60
                # Only log long waits
                if wait_minutes > 60:
                    self._log_message(f"[WAIT] {self.instrument} waiting {wait_minutes/60:.1f} hours", level='warning')
                
                # Sleep in chunks to avoid blocking too long
                chunk_size = min(300, wait_seconds)  # Max 5 minutes chunks
                while wait_seconds > 0:
                    sleep_time = min(chunk_size, wait_seconds)
                    time.sleep(sleep_time)
                    wait_seconds -= sleep_time
                    
                    # Log progress only for very long waits
                    if wait_seconds > 3600:  # Only log every hour for waits > 1 hour
                        remaining_hours = wait_seconds / 3600
                        self._log_message(f"[WAIT] {self.instrument} still waiting {remaining_hours:.1f}h", level='warning')
        else:
            # No specific time, wait 30 minutes
            time.sleep(1800)  # 30 minutes

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
                # Check if instrument is available before attempting to get data
                if hasattr(self, 'market_checker'):
                    availability_info = self.check_instrument_availability()
                    if availability_info['should_suspend']:
                        self._log_message(f"Instrument {instrument} not available for data retrieval - {availability_info['market_status']['reason']}", level='warning')
                        # Still try to get data but with awareness of market status
                
                europe_london_datetime = datetime.now(pytz.timezone('Europe/London'))
                date_from = europe_london_datetime - dt.timedelta(days=days)
                date_to = europe_london_datetime
                
                history = connection.get_history(instrument, timeframe, date_from, date_to)
                   
                pricedata = pd.DataFrame(history, columns=["Date", "BidOpen", "BidHigh", "BidLow", "BidClose", "Volume"])
               
                if pricedata.empty:
                    self._log_message(f"Empty DataFrame created for {instrument} {timeframe} - Market may be closed or instrument unavailable", level='error')
                    
                    # If we have market checker, provide more context
                    # Market status already logged in availability check
                    
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
                
                # Check if this might be a market availability issue
                if "not available" in str(e).lower() or "closed" in str(e).lower():
                    if hasattr(self, 'market_checker'):
                        market_status = self.market_checker.is_market_open(instrument)
                        if not market_status['is_available']:
                            error_msg += f" - Market appears to be closed: {market_status['reason']}"
                            # Handle suspension for unavailable market
                            availability_info = {
                                'should_suspend': True,
                                'market_status': market_status,
                                'recommendation': {'recommendation': 'wait'}
                            }
                            self.handle_instrument_suspension(availability_info)
                
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

    def get_latest_price(self, instrument: str, BuySell: str) -> float:
        if self.pricedata is not None and not self.pricedata.empty:
            return float(self.pricedata['bidclose'].iloc[-1])
        return None

    def set_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Calculate peaks
            df = self.calculate_peaks(df)
            # Calculate medians
            df = self.calculate_medians(df)
            # Calculate EMAs
            df = self.calculate_ema_200(df)
            df = self.calculate_ema_100(df)
            df = self.calculate_ema_80(df)
            # Calculate ATR and filters
            df = self.calculate_atr_and_filters(df)
            # Indicators calculated successfully - no need to log
            return df
        except Exception as e:
            self._log_message(f"Error setting indicators: {str(e)}", 'error')
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
            df['median_high'] = df['bidhigh'].rolling(window=window, min_periods=10).median()
        
        if 'bidlow' in df.columns:
            df['median_low'] = df['bidlow'].rolling(window=window, min_periods=10).median()
        
        if 'bidclose' in df.columns:
            df['median_close'] = df['bidclose'].rolling(window=window, min_periods=10).median()
        
        if 'bidopen' in df.columns:
            df['median_open'] = df['bidopen'].rolling(window=window, min_periods=10).median()
        
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

    def calculate_atr_and_filters(self, df: pd.DataFrame, atr_period: int = 14, slope_period: int = 5) -> pd.DataFrame:
        """Calculate ATR and trading filters to improve signal quality"""
        if df is None or df.empty:
            return df

        # Calculate True Range and ATR
        df['prev_close'] = df['bidclose'].shift(1)
        tr1 = df['bidhigh'] - df['bidlow']
        tr2 = (df['bidhigh'] - df['prev_close']).abs()
        tr3 = (df['bidlow'] - df['prev_close']).abs()
        df['true_range'] = np.nanmax(np.vstack([tr1.values, tr2.values, tr3.values]).T, axis=1)
        df['atr_14'] = pd.Series(df['true_range']).rolling(window=atr_period, min_periods=5).mean()
        
        # SIMPLE LIQUIDITY FILTER - Add basic liquidity detection
        df = self._add_simple_liquidity_filter(df)

        # EMA 200 slope filter (less restrictive)
        if 'ema_200' in df.columns:
            df['ema_200_slope'] = ((df['ema_200'] - df['ema_200'].shift(slope_period)) / 
                                   df['ema_200'].shift(slope_period) * 100).fillna(0)
            # Slope direction: 1=up, -1=down, 0=flat (more restrictive threshold)
            slope_threshold = 0.01  # 0.01% minimum slope (was 0.005%)
            df['ema_200_trend'] = np.where(df['ema_200_slope'] > slope_threshold, 1,
                                          np.where(df['ema_200_slope'] < -slope_threshold, -1, 0))

        # Distance to EMA 200 filter (less restrictive)
        if 'ema_200' in df.columns:
            df['distance_to_ema200'] = abs(df['bidclose'] - df['ema_200'])
            # Minimum distance = max(0.0002, 0.3*ATR14) - more permissive
            df['min_distance_required'] = np.maximum(0.0002, 0.3 * df['atr_14'])
            df['far_from_ema200'] = df['distance_to_ema200'] > df['min_distance_required']

        # EMA compression filter (less restrictive)
        if all(col in df.columns for col in ['ema_80', 'ema_100', 'ema_200']):
            ema_max = np.maximum.reduce([df['ema_80'], df['ema_100'], df['ema_200']])
            ema_min = np.minimum.reduce([df['ema_80'], df['ema_100'], df['ema_200']])
            df['ema_spread'] = ema_max - ema_min
            # EMAs not compressed if spread > 0.20*ATR14 (was 0.15)
            df['emas_not_compressed'] = df['ema_spread'] > (0.20 * df['atr_14'])

        # Advanced filters for signal quality
        
        # 1. RSI Momentum Filter (14 periods)
        delta = df['bidclose'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # RSI momentum conditions
        df['rsi_not_overbought'] = df['rsi_14'] < 70  # For BUY signals
        df['rsi_not_oversold'] = df['rsi_14'] > 30    # For SELL signals
        
        # 2. Volume/Range Minimum Filter
        df['candle_range'] = df['bidhigh'] - df['bidlow']
        df['candle_body'] = abs(df['bidclose'] - df['bidopen'])
        df['body_ratio'] = df['candle_body'] / df['candle_range'].replace(0, np.nan)
        
        # Range and body filters (more permissive)
        if 'atr_14' in df.columns:
            df['significant_range'] = df['candle_range'] > (0.5 * df['atr_14'])  # Was 0.8, now 0.5
            df['significant_body'] = df['body_ratio'] > 0.4  # Was 0.6, now 0.4
        else:
            df['significant_range'] = True
            df['significant_body'] = True
        
        # 3. Stack Confirmation Filter
        # Price position relative to EMA stack
        if all(col in df.columns for col in ['ema_80', 'ema_100', 'ema_200']):
            # For BUY: price should be above EMA 80, and EMAs in bullish order
            df['price_above_ema80'] = df['bidclose'] > df['ema_80']
            df['price_below_ema80'] = df['bidclose'] < df['ema_80']
            
            # EMA stack order (bullish/bearish)
            df['ema_stack_bullish'] = (df['ema_80'] > df['ema_100']) & (df['ema_100'] > df['ema_200'])
            df['ema_stack_bearish'] = (df['ema_80'] < df['ema_100']) & (df['ema_100'] < df['ema_200'])
            
            # Combined stack confirmation
            df['stack_confirms_buy'] = df['price_above_ema80'] & df['ema_stack_bullish']
            df['stack_confirms_sell'] = df['price_below_ema80'] & df['ema_stack_bearish']
        else:
            df['stack_confirms_buy'] = True
            df['stack_confirms_sell'] = True
        
        # 4. Probability Scoring System (0-100)
        df['signal_score'] = 0
        
        # Base score for EMA alignment (30 points)
        if all(col in df.columns for col in ['ema_80', 'ema_100', 'ema_200']):
            df.loc[df['ema_stack_bullish'], 'signal_score'] += 15  # Bullish stack
            df.loc[df['ema_stack_bearish'], 'signal_score'] += 15  # Bearish stack
            
            # Price position bonus (15 points)
            df.loc[df['price_above_ema80'] & df['ema_stack_bullish'], 'signal_score'] += 15
            df.loc[df['price_below_ema80'] & df['ema_stack_bearish'], 'signal_score'] += 15
        
        # EMA 200 slope score (20 points)
        if 'ema_200_trend' in df.columns:
            df.loc[df['ema_200_trend'] == 1, 'signal_score'] += 20  # Bullish slope
            df.loc[df['ema_200_trend'] == -1, 'signal_score'] += 20  # Bearish slope
            df.loc[df['ema_200_trend'] == 0, 'signal_score'] += 10   # Flat (partial)
        
        # Distance and compression (15 points each)
        if 'far_from_ema200' in df.columns:
            df.loc[df['far_from_ema200'], 'signal_score'] += 15
        if 'emas_not_compressed' in df.columns:
            df.loc[df['emas_not_compressed'], 'signal_score'] += 15
        
        # RSI momentum (10 points)
        df.loc[df['rsi_not_overbought'], 'signal_score'] += 5   # For BUY potential
        df.loc[df['rsi_not_oversold'], 'signal_score'] += 5    # For SELL potential
        
        # Range and body quality (15 points)
        df.loc[df['significant_range'], 'signal_score'] += 8
        df.loc[df['significant_body'], 'signal_score'] += 7
        
        # High probability threshold (more restrictive)
        df['high_probability'] = df['signal_score'] >= 60  # Was 50, now 60
        
        # Clean up temporary columns
        df = df.drop(columns=['prev_close'], errors='ignore')
        
        return df

    def _add_simple_liquidity_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple liquidity filter based on time sessions and basic price behavior"""
        if df is None or df.empty:
            return df
        
        try:
            # Convert date to datetime for hour extraction
            df['datetime'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
            df['hour_utc'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            
            # Define trading sessions (UTC times)
            # London: 8-16 UTC, New York: 13-21 UTC, Tokyo: 0-8 UTC
            df['london_session'] = df['hour_utc'].between(8, 16)
            df['newyork_session'] = df['hour_utc'].between(13, 21)
            df['tokyo_session'] = df['hour_utc'].between(0, 8)
            
            # High liquidity = overlap periods or major sessions
            df['session_overlap'] = df['london_session'] & df['newyork_session']  # 13-16 UTC overlap
            df['major_session'] = df['london_session'] | df['newyork_session']
            
            # Low liquidity periods (simple detection)
            df['weekend_approach'] = (df['datetime'].dt.dayofweek == 4) & (df['hour_utc'] >= 20)  # Friday 20+ UTC
            df['weekend_start'] = df['datetime'].dt.dayofweek.isin([5, 6])  # Saturday, Sunday
            df['asian_low_period'] = df['hour_utc'].between(2, 6) & (~df['tokyo_session'])  # Dead hours
            
            # Simple gap detection (indicates low liquidity)
            if 'atr_14' in df.columns:
                df['price_gap'] = abs(df['bidopen'] - df['bidclose'].shift(1))
                df['significant_gap'] = df['price_gap'] > (df['atr_14'] * 0.3)  # Gap > 30% of ATR
                df['gap_count'] = df['significant_gap'].rolling(window=10).sum()  # Gaps in last 10 candles
            else:
                df['significant_gap'] = False
                df['gap_count'] = 0
            
            # Final liquidity assessment (SIMPLE)
            df['low_liquidity_period'] = (
                df['weekend_approach'] | 
                df['weekend_start'] | 
                df['asian_low_period'] | 
                (df['gap_count'] >= 3)  # Too many gaps recently
            )
            
            df['high_liquidity_period'] = (
                df['session_overlap'] |  # Best time: London-NY overlap
                (df['major_session'] & ~df['low_liquidity_period'])  # Major sessions without issues
            )
            
            # Liquidity score (0-100, simple)
            df['liquidity_score'] = np.where(
                df['high_liquidity_period'], 80,  # High liquidity
                np.where(df['low_liquidity_period'], 20, 50)  # Low or medium liquidity
            )
            
            # Trading permission flag
            df['liquidity_allows_trading'] = ~df['low_liquidity_period']
            
            # Clean up temporary columns
            df = df.drop(columns=['datetime'], errors='ignore')
            
            return df
            
        except Exception as e:
            self._log_message(f"Error in simple liquidity filter: {e}", level='error')
            # Fallback: allow all trading if filter fails
            df['liquidity_allows_trading'] = True
            df['liquidity_score'] = 50
            df['low_liquidity_period'] = False
            df['high_liquidity_period'] = False
            return df

    def _add_price_peaks(self, df, order):        
            df['peaks_min'] = 0
            df['peaks_max'] = 0

            peaks_min_idx = signal.argrelextrema(df['bidclose'].values, np.less, order=order)[0]
            peaks_max_idx = signal.argrelextrema(df['bidclose'].values, np.greater, order=order)[0]
            
            df.loc[peaks_min_idx, 'peaks_min'] = 1
            df.loc[peaks_max_idx, 'peaks_max'] = 1

            
    def set_signals_to_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on peaks with median/EMA filter"""
        if df is None or df.empty:
            return df
            
        df['signal'] = self.SIGNAL_NEUTRAL
        buy_signals = 0
        sell_signals = 0
        filtered_buy = 0
        filtered_sell = 0
        
        # Check if required columns exist
        has_medians = all(col in df.columns for col in ['median_high', 'median_low', 'median_close', 'median_open'])
        has_ema_200 = 'ema_200' in df.columns
        has_ema_100 = 'ema_100' in df.columns
        has_ema_80 = 'ema_80' in df.columns
        has_all_emas = has_ema_200 and has_ema_100 and has_ema_80
        
        for i in range(len(df)):
            # Peak-based signals with median/EMA filter
            if df['peaks_min'].iloc[i] == 1:  # Potential BUY signal at LOW peak
                signal_valid = True
                filter_details = []
                
                # Enhanced EMA alignment filter for BUY signals
                if has_all_emas:
                    # BUY: require bullish EMA alignment (shorter EMAs above longer EMAs)
                    ema_200 = df['ema_200'].iloc[i]
                    ema_100 = df['ema_100'].iloc[i]
                    ema_80 = df['ema_80'].iloc[i]
                    
                    if not any(pd.isna([ema_200, ema_100, ema_80])):
                        # For BUY: EMA 80 > EMA 100 > EMA 200 (bullish alignment)
                        ema_alignment_bullish = ema_80 > ema_100 > ema_200
                        if not ema_alignment_bullish:
                            signal_valid = False
                            filtered_buy += 1
                            filter_details.append(f"EMA_ALIGNMENT_FAIL: 80={ema_80:.5f} 100={ema_100:.5f} 200={ema_200:.5f}")
                        else:
                            filter_details.append("EMA_ALIGNMENT_PASS")
                
                # Advanced Quality Filters for BUY signals
                if signal_valid:
                    signal_score = df.get('signal_score', pd.Series([0])).iloc[i] if 'signal_score' in df.columns else 0
                    
                    # 1. RSI Momentum Filter
                    rsi_ok = df.get('rsi_not_overbought', pd.Series([True])).iloc[i] if 'rsi_not_overbought' in df.columns else True
                    rsi_value = df.get('rsi_14', pd.Series([50])).iloc[i] if 'rsi_14' in df.columns else 50
                    
                    # 2. Range/Body Quality Filter  
                    range_ok = df.get('significant_range', pd.Series([True])).iloc[i] if 'significant_range' in df.columns else True
                    body_ok = df.get('significant_body', pd.Series([True])).iloc[i] if 'significant_body' in df.columns else True
                    body_ratio = df.get('body_ratio', pd.Series([1])).iloc[i] if 'body_ratio' in df.columns else 1
                    
                    # 3. Stack Confirmation Filter
                    stack_ok = df.get('stack_confirms_buy', pd.Series([True])).iloc[i] if 'stack_confirms_buy' in df.columns else True
                    price_above_ema80 = df.get('price_above_ema80', pd.Series([True])).iloc[i] if 'price_above_ema80' in df.columns else True
                    
                    # 4. High Probability Filter (Primary Gate)
                    high_prob = df.get('high_probability', pd.Series([True])).iloc[i] if 'high_probability' in df.columns else True
                    
                    # 5. SIMPLE LIQUIDITY FILTER (NEW)
                    liquidity_ok = df.get('liquidity_allows_trading', pd.Series([True])).iloc[i] if 'liquidity_allows_trading' in df.columns else True
                    liquidity_score = df.get('liquidity_score', pd.Series([50])).iloc[i] if 'liquidity_score' in df.columns else 50
                    is_high_liquidity = df.get('high_liquidity_period', pd.Series([False])).iloc[i] if 'high_liquidity_period' in df.columns else False
                    
                    # LIQUIDITY CHECK - Block trading in low liquidity periods
                    if not liquidity_ok:
                        signal_valid = False
                        filtered_buy += 1
                        filter_details.append(f"LIQUIDITY_BLOCKED: score={liquidity_score}")
                    else:
                        # Mandatory filters (more lenient - only 2 of 3 must pass)
                        mandatory_count = sum([rsi_ok, stack_ok, high_prob])
                        mandatory_pass = mandatory_count >= 2  # At least 2 of 3
                        
                        # Quality filters (at least 1 of 2 must pass)
                        quality_pass = range_ok or body_ok
                        
                        if not mandatory_pass or not quality_pass:
                            signal_valid = False
                            filtered_buy += 1
                            
                            # Detailed logging
                            filter_details.append(f"SCORE: {signal_score}/100")
                            if not rsi_ok:
                                filter_details.append(f"RSI_FAIL: {rsi_value:.1f} (overbought)")
                            if not stack_ok:
                                filter_details.append(f"STACK_FAIL: price_above_ema80={price_above_ema80}")
                            if not high_prob:
                                filter_details.append(f"LOW_PROBABILITY: score={signal_score}<50")
                            if not quality_pass:
                                filter_details.append(f"QUALITY_FAIL: range={range_ok} body={body_ok} ratio={body_ratio:.2f}")
                            filter_details.append(f"MANDATORY: {mandatory_count}/3 passed")
                        else:
                            # Add liquidity info to successful signals
                            liquidity_status = "HIGH_LIQ" if is_high_liquidity else "MED_LIQ"
                            filter_details.append(f"ALL_PASS: score={signal_score}/100 rsi={rsi_value:.1f} stack={stack_ok} quality={quality_pass} mandatory={mandatory_count}/3 liquidity={liquidity_status}({liquidity_score})")
                
                # Original median filter (with EMA 200 as fallback)
                elif has_medians and has_ema_200:
                    # BUY: medians should be ABOVE EMA 200 (bullish condition)
                    ema_value = df['ema_200'].iloc[i]
                    median_close = df['median_close'].iloc[i]
                    median_high = df['median_high'].iloc[i]
                    median_low = df['median_low'].iloc[i]
                    
                    if not pd.isna(ema_value) and not pd.isna(median_close):
                        # Check if majority of medians are above EMA
                        medians_above_ema = 0
                        if not pd.isna(median_high) and median_high > ema_value:
                            medians_above_ema += 1
                        if not pd.isna(median_low) and median_low > ema_value:
                            medians_above_ema += 1
                        if not pd.isna(median_close) and median_close > ema_value:
                            medians_above_ema += 1
                        
                        # BUY only if at least 2 of 3 main medians are above EMA
                        if medians_above_ema < 2:
                            signal_valid = False
                            filtered_buy += 1
                
                if signal_valid:
                    df.at[i, 'signal'] = self.SIGNAL_BUY
                    buy_signals += 1
                    # Log successful BUY signal with filter details
                    candle_date = df['date'].iloc[i] if 'date' in df.columns else i
                    price = df['bidclose'].iloc[i]
                    self._log_message(f"BUY SIGNAL GENERATED: Date={candle_date} Price={price:.5f} Filters=[{', '.join(filter_details)}]")
                elif len(filter_details) > 0:
                    # Log filtered out BUY signal
                    candle_date = df['date'].iloc[i] if 'date' in df.columns else i
                    price = df['bidclose'].iloc[i]
                    self._log_message(f"BUY SIGNAL FILTERED: Date={candle_date} Price={price:.5f} Reason=[{', '.join(filter_details)}]")
                    
            elif df['peaks_max'].iloc[i] == 1:  # Potential SELL signal at HIGH peak
                signal_valid = True
                filter_details = []
                
                # Enhanced EMA alignment filter for SELL signals
                if has_all_emas:
                    # SELL: require bearish EMA alignment (shorter EMAs below longer EMAs)
                    ema_200 = df['ema_200'].iloc[i]
                    ema_100 = df['ema_100'].iloc[i]
                    ema_80 = df['ema_80'].iloc[i]
                    
                    if not any(pd.isna([ema_200, ema_100, ema_80])):
                        # For SELL: EMA 80 < EMA 100 < EMA 200 (bearish alignment)
                        ema_alignment_bearish = ema_80 < ema_100 < ema_200
                        if not ema_alignment_bearish:
                            signal_valid = False
                            filtered_sell += 1
                            filter_details.append(f"EMA_ALIGNMENT_FAIL: 80={ema_80:.5f} 100={ema_100:.5f} 200={ema_200:.5f}")
                        else:
                            filter_details.append("EMA_ALIGNMENT_PASS")
                
                # Advanced Quality Filters for SELL signals
                if signal_valid:
                    signal_score = df.get('signal_score', pd.Series([0])).iloc[i] if 'signal_score' in df.columns else 0
                    
                    # 1. RSI Momentum Filter
                    rsi_ok = df.get('rsi_not_oversold', pd.Series([True])).iloc[i] if 'rsi_not_oversold' in df.columns else True
                    rsi_value = df.get('rsi_14', pd.Series([50])).iloc[i] if 'rsi_14' in df.columns else 50
                    
                    # 2. Range/Body Quality Filter  
                    range_ok = df.get('significant_range', pd.Series([True])).iloc[i] if 'significant_range' in df.columns else True
                    body_ok = df.get('significant_body', pd.Series([True])).iloc[i] if 'significant_body' in df.columns else True
                    body_ratio = df.get('body_ratio', pd.Series([1])).iloc[i] if 'body_ratio' in df.columns else 1
                    
                    # 3. Stack Confirmation Filter
                    stack_ok = df.get('stack_confirms_sell', pd.Series([True])).iloc[i] if 'stack_confirms_sell' in df.columns else True
                    price_below_ema80 = df.get('price_below_ema80', pd.Series([True])).iloc[i] if 'price_below_ema80' in df.columns else True
                    
                    # 4. High Probability Filter (Primary Gate)
                    high_prob = df.get('high_probability', pd.Series([True])).iloc[i] if 'high_probability' in df.columns else True
                    
                    # 5. SIMPLE LIQUIDITY FILTER (NEW)
                    liquidity_ok = df.get('liquidity_allows_trading', pd.Series([True])).iloc[i] if 'liquidity_allows_trading' in df.columns else True
                    liquidity_score = df.get('liquidity_score', pd.Series([50])).iloc[i] if 'liquidity_score' in df.columns else 50
                    is_high_liquidity = df.get('high_liquidity_period', pd.Series([False])).iloc[i] if 'high_liquidity_period' in df.columns else False
                    
                    # LIQUIDITY CHECK - Block trading in low liquidity periods
                    if not liquidity_ok:
                        signal_valid = False
                        filtered_sell += 1
                        filter_details.append(f"LIQUIDITY_BLOCKED: score={liquidity_score}")
                    else:
                        # Mandatory filters (more lenient - only 2 of 3 must pass)
                        mandatory_count = sum([rsi_ok, stack_ok, high_prob])
                        mandatory_pass = mandatory_count >= 2  # At least 2 of 3
                        
                        # Quality filters (at least 1 of 2 must pass)
                        quality_pass = range_ok or body_ok
                        
                        if not mandatory_pass or not quality_pass:
                            signal_valid = False
                            filtered_sell += 1
                            
                            # Detailed logging
                            filter_details.append(f"SCORE: {signal_score}/100")
                            if not rsi_ok:
                                filter_details.append(f"RSI_FAIL: {rsi_value:.1f} (oversold)")
                            if not stack_ok:
                                filter_details.append(f"STACK_FAIL: price_below_ema80={price_below_ema80}")
                            if not high_prob:
                                filter_details.append(f"LOW_PROBABILITY: score={signal_score}<50")
                            if not quality_pass:
                                filter_details.append(f"QUALITY_FAIL: range={range_ok} body={body_ok} ratio={body_ratio:.2f}")
                            filter_details.append(f"MANDATORY: {mandatory_count}/3 passed")
                        else:
                            # Add liquidity info to successful signals
                            liquidity_status = "HIGH_LIQ" if is_high_liquidity else "MED_LIQ"
                            filter_details.append(f"ALL_PASS: score={signal_score}/100 rsi={rsi_value:.1f} stack={stack_ok} quality={quality_pass} mandatory={mandatory_count}/3 liquidity={liquidity_status}({liquidity_score})")
                
                # Original median filter (with EMA 200 as fallback)
                elif has_medians and has_ema_200:
                    # SELL: medians should be BELOW EMA 200 (bearish condition)
                    ema_value = df['ema_200'].iloc[i]
                    median_close = df['median_close'].iloc[i]
                    median_high = df['median_high'].iloc[i]
                    median_low = df['median_low'].iloc[i]
                    
                    if not pd.isna(ema_value) and not pd.isna(median_close):
                        # Check if majority of medians are below EMA
                        medians_below_ema = 0
                        if not pd.isna(median_high) and median_high < ema_value:
                            medians_below_ema += 1
                        if not pd.isna(median_low) and median_low < ema_value:
                            medians_below_ema += 1
                        if not pd.isna(median_close) and median_close < ema_value:
                            medians_below_ema += 1
                        
                        # SELL only if at least 2 of 3 main medians are below EMA
                        if medians_below_ema < 2:
                            signal_valid = False
                            filtered_sell += 1
                
                if signal_valid:
                    df.at[i, 'signal'] = self.SIGNAL_SELL
                    sell_signals += 1
                    # Log successful SELL signal with filter details
                    candle_date = df['date'].iloc[i] if 'date' in df.columns else i
                    price = df['bidclose'].iloc[i]
                    self._log_message(f"SELL SIGNAL GENERATED: Date={candle_date} Price={price:.5f} Filters=[{', '.join(filter_details)}]")
                elif len(filter_details) > 0:
                    # Log filtered out SELL signal
                    candle_date = df['date'].iloc[i] if 'date' in df.columns else i
                    price = df['bidclose'].iloc[i]
                    self._log_message(f"SELL SIGNAL FILTERED: Date={candle_date} Price={price:.5f} Reason=[{', '.join(filter_details)}]")
                
        df['valid_signal'] = df['signal']
        
        # Calculate liquidity statistics
        liquidity_stats = ""
        if 'liquidity_allows_trading' in df.columns:
            total_candles = len(df)
            low_liq_candles = (~df['liquidity_allows_trading']).sum()
            high_liq_candles = df.get('high_liquidity_period', pd.Series([False])).sum()
            liquidity_stats = f" liquidity_blocked={low_liq_candles}/{total_candles} high_liq={high_liq_candles}/{total_candles}"
        
        if has_all_emas:
            self._log_message(f"EMA alignment filtered signals generated: buy={buy_signals} sell={sell_signals} (filtered: {filtered_buy} buy, {filtered_sell} sell){liquidity_stats} instrument={self.instrument} timeframe={self.timeframe}")
        elif has_medians and has_ema_200:
            self._log_message(f"Median filtered signals generated: buy={buy_signals} sell={sell_signals} (filtered: {filtered_buy} buy, {filtered_sell} sell){liquidity_stats} instrument={self.instrument} timeframe={self.timeframe}")
        else:
            self._log_message(f"Peak signals generated: buy={buy_signals} sell={sell_signals}{liquidity_stats} instrument={self.instrument} timeframe={self.timeframe}")
        
        return df

    def _set_signal(self, df, idx, signal_col, value):
        df.iloc[idx, df.columns.get_loc(signal_col)] = value

    def triggers_trades_open(self, df: pd.DataFrame, config=None):
        """Simplified trade opening - processes last 7 candles but ignores last 2"""
        try:
            # Initialize config
            if config is None:
                from TradingConfiguration import TradingConfig
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
            
            # Clear console log when processing real signal
            print(f"🟢 {self.instrument}: SEÑAL BUY procesada → Precio: {buy_price} | Fecha: {buy_date}")
            self._log_message(f"Processing BUY signal - Date: {buy_date}, Price: {buy_price}")
            
            self._close_existing_sell_operations(buy_date, buy_price)
            
            self._open_buy_operation(buy_date, buy_price)
            
        except Exception as e:
            print(f"❌ {self.instrument}: ERROR en señal BUY → {str(e)}")
            self._log_message(f"Error processing BUY signal: {e}", level='error')

    def _close_existing_sell_operations(self, signal_date, signal_price):
        if self.existingOperation(instrument=self.instrument, BuySell="S"):
            print(f"❌ {self.instrument}: CERRANDO OPERACIÓN SELL → Señal BUY detectada | Precio: {signal_price}")
            self._log_message(
                f"[CLOSE SELL] Reason: BUY signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="S")
        else:
            self._log_message(f"No existing SELL operations to close")

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
            
            self._close_existing_buy_operations(sell_date, sell_price)
            
            self._open_sell_operation(sell_date, sell_price)
            
        except Exception as e:
            print(f"❌ {self.instrument}: ERROR en señal SELL → {str(e)}")
            self._log_message(f"Error processing SELL signal: {e}", level='error')

    def _close_existing_buy_operations(self, signal_date, signal_price):
        if self.existingOperation(instrument=self.instrument, BuySell="B"):
            print(f"❌ {self.instrument}: CERRANDO OPERACIÓN BUY → Señal SELL detectada | Precio: {signal_price}")
            self._log_message(
                f"[CLOSE BUY] Reason: SELL signal detected | Date: {signal_date} | "
                f"Price: {signal_price} | Instrument: {self.instrument} | Timeframe: {self.timeframe}"
            )
            self.CloseOperation(instrument=self.instrument, BuySell="B")
        else:
            self._log_message(f"No existing BUY operations to close")

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
        """Enhanced trade closing - processes signals and peaks in last 7 candles but ignores last 2"""
        try:
            # Initialize config
            if config is None:
                from TradingConfiguration import TradingConfig
                config = TradingConfig()
            
            signal_col = config.signal_col if hasattr(config, 'signal_col') else 'signal'
            peaks_max_col = config.peaks_max_col if hasattr(config, 'peaks_max_col') else 'peaks_max'
            peaks_min_col = config.peaks_min_col if hasattr(config, 'peaks_min_col') else 'peaks_min'
            
            # Get the last 7 candles but ignore the last 2 (process candles 3-7 from the end)
            last_7_candles = df.tail(7)
            validation_candles = last_7_candles.head(5)  # Exclude last 2 candles
            
            self._log_message(f"Processing close signals and peaks in last 7 candles (excluding last 2) for trade closing")
            
            # Check for peaks in the validation period (candles 3-7 from the end)
            peaks_max = validation_candles[validation_candles[peaks_max_col] == 1] if peaks_max_col in validation_candles.columns else pd.DataFrame()
            peaks_min = validation_candles[validation_candles[peaks_min_col] == 1] if peaks_min_col in validation_candles.columns else pd.DataFrame()
            
            # Check for opposite signals in the validation period (candles 3-7 from the end)
            buy_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_BUY]
            sell_signals = validation_candles[validation_candles[signal_col] == self.SIGNAL_SELL]
            
            self._log_message(f"Found {len(buy_signals)} buy signals, {len(sell_signals)} sell signals, {len(peaks_max)} max peaks, {len(peaks_min)} min peaks for closing in validation period (candles 3-7)")
            
            # Close BUY operations when peaks_max is detected (price at maximum)
            if not peaks_max.empty:
                if self.existingOperation(instrument=self.instrument, BuySell="B"):
                    latest_peak_max = peaks_max.iloc[-1]
                    self._log_message(f"PEAKS MAX detected - closing BUY operations from candle {latest_peak_max['date']}")
                    self._process_close_buy_peak_signal(latest_peak_max, "PEAKS_MAX")
            
            # Close SELL operations when peaks_min is detected (price at minimum)
            if not peaks_min.empty:
                if self.existingOperation(instrument=self.instrument, BuySell="S"):
                    latest_peak_min = peaks_min.iloc[-1]
                    self._log_message(f"PEAKS MIN detected - closing SELL operations from candle {latest_peak_min['date']}")
                    self._process_close_sell_peak_signal(latest_peak_min, "PEAKS_MIN")
            
            # Close opposite positions when signals change (existing logic)
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
            
            if buy_signals.empty and sell_signals.empty and peaks_max.empty and peaks_min.empty:
                self._log_message("No close signals or peaks found in validation period (candles 3-7)")
            
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

