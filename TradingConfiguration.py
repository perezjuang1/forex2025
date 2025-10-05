import datetime as dt
from datetime import datetime
import time
import pytz

class TradingConfig:
    """
    Centralized configuration class for the forex trading system.
    
    This class contains all configuration parameters for:
    - Trading strategy parameters
    - Connection settings
    - Trading parameters
    - Instrument lists
    """
    
    # ============================================================================
    # TRADING STRATEGY PARAMETERS
    # ============================================================================
    
    # Column names for signal processing
    peaks_min_col = 'peaks_min'
    peaks_max_col = 'peaks_max'
    signal_col = 'signal'  # Column name for trading signals
    
    # ============================================================================
    # CONNECTION SETTINGS
    # ============================================================================
    
    # FXCM connection parameters
    userid = "U10D2471243"
    password = "Fkr5q"
    url = "http://www.fxcorporate.com/Hosts.jsp"
    connectiontype = "Demo"
    
    # Session and account settings
    account = None
    
    # ============================================================================
    # TRADING PARAMETERS
    # ============================================================================
    
    # Default instrument and timeframe
    timeframe = "m1"  # Available periods: 'm1', 'm5', 'm15', 'm30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'D1', 'W1', 'M1'
    
    # Position sizing and risk management
    lots = 3  # Default lot size for trading
    stop = 5   # Stop loss in pips (reduced from 10 to avoid FXCM limits)
    limit = 15  # Take profit in pips (reduced proportionally)
    
    # JPY pairs specific settings (very conservative)
    jpy_stop = 2   # Stop loss for JPY pairs (2 pips)
    jpy_limit = 6  # Take profit for JPY pairs (6 pips)
    
    # Data retrieval settings
    days = 5  # Number of days of historical data to retrieve
    
    # ============================================================================
    # INSTRUMENT CONFIGURATION - TRADING 24/7
    # ============================================================================
    
    # Simplified instrument list - trading 24/7 (no market hours restrictions)
    trading_instruments = [
        "EUR/USD",    # Major pair - high liquidity
        "GBP/USD",    # Major pair - high liquidity
        "EUR/GBP",    # Cross pair - good volatility
        "AUD/USD",    # Commodity currency
        "NZD/USD",    # Commodity currency
        "USD/CAD",    # Major pair - commodity currency
        "USD/CHF"     # Safe haven pair
    ]
    
    def __init__(self):
        """
        Initialize the configuration with current London time settings.
        """
        # Set date range based on London timezone
        europe_london_datetime = datetime.now(pytz.timezone('Europe/London'))
        self.date_from = europe_london_datetime - dt.timedelta(days=self.days)
        self.date_to = europe_london_datetime
    
    @classmethod
    def get_timeframe(cls) -> str:
        """Get the current timeframe setting"""
        return cls.timeframe
    
    @classmethod
    def get_trading_instruments(cls) -> list:
        """Get the list of trading instruments"""
        return cls.trading_instruments.copy()
    
    @classmethod
    def get_lot_size(cls) -> int:
        """Get the lot size for trading"""
        return cls.lots
    
    @classmethod
    def get_stop_loss(cls, instrument: str = None) -> int:
        """Get stop loss in pips (JPY pairs use different values)"""
        if instrument and 'JPY' in instrument:
            return cls.jpy_stop
        return cls.stop
    
    @classmethod
    def get_take_profit(cls, instrument: str = None) -> int:
        """Get take profit in pips (JPY pairs use different values)"""
        if instrument and 'JPY' in instrument:
            return cls.jpy_limit
        return cls.limit
    
    @classmethod
    def is_trading_instrument(cls, instrument: str) -> bool:
        """Check if an instrument is in the trading list"""
        return instrument in cls.trading_instruments
    
    @classmethod
    def get_instrument_count(cls) -> int:
        """Get the total number of trading instruments"""
        return len(cls.trading_instruments)
    
    def get_date_range(self) -> tuple:
        """Get the date range for data retrieval"""
        return self.date_from, self.date_to
    
    def update_date_range(self, days: int = None):
        """Update the date range for data retrieval"""
        if days:
            self.days = days
        europe_london_datetime = datetime.now(pytz.timezone('Europe/London'))
        self.date_from = europe_london_datetime - dt.timedelta(days=self.days)
        self.date_to = europe_london_datetime