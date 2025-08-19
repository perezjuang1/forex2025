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
    signal_col = 'signal'
    open_zone_col = 'trade_open_zone'
    peaks_min_col = 'peaks_min'
    peaks_max_col = 'peaks_max'
    
    # Peak detection tolerance
    tolerance_peaks = 10
    
    # ============================================================================
    # CONNECTION SETTINGS
    # ============================================================================
    
    # FXCM connection parameters
    userid = "U10D2470792"
    password = "i4Cea"
    url = "http://www.fxcorporate.com/Hosts.jsp"
    connectiontype = "Demo"
    
    # Session and account settings
    session = None
    pin = None
    account = None
    
    # ============================================================================
    # TRADING PARAMETERS
    # ============================================================================
    
    # Default instrument and timeframe
    instrument_symbol = "EUR/USD"
    timeframe = "m1"  # Available periods: 'm1', 'm5', 'm15', 'm30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'D1', 'W1', 'M1'
    
    # Position sizing and risk management
    lots = 3
    stop = 6
    limit = 15
    
    # Pegged order settings
    peggedstop = 'Y'
    peggedlimit = 'Y'
    pegstoptype = 'M'
    peglimittype = 'M'
    
    # Date format and time settings
    dateFormat = '%m.%d.%Y %H:%M:%S'
    date_from = None
    date_to = None
    days = 4
    
    # ============================================================================
    # INSTRUMENT CONFIGURATION
    # ============================================================================
    
    # List of instruments to trade
    instruments = ["EUR/USD", "GBP/USD", "EUR/JPY", "AUD/JPY", "EUR/CAD", "USD/JPY", "GBP/JPY", "USD/CHF"]
    
    def __init__(self):
        """
        Initialize the configuration with current London time settings.
        """
        # Set date range based on London timezone
        europe_London_datetime = datetime.now(pytz.timezone('Europe/London'))
        self.date_from = europe_London_datetime - dt.timedelta(days=self.days)
        self.date_to = europe_London_datetime
    
    @classmethod
    def get_timeframe(cls) -> str:
        """Get the default timeframe."""
        return cls.timeframe
    
    @classmethod
    def get_instruments(cls) -> list:
        """Get the list of trading instruments."""
        return cls.instruments.copy()
    
    @classmethod
    def get_trading_params(cls) -> dict:
        """Get trading parameters as a dictionary."""
        return {
            'lots': cls.lots,
            'stop': cls.stop,
            'limit': cls.limit,
            'peggedstop': cls.peggedstop,
            'peggedlimit': cls.peggedlimit,
            'pegstoptype': cls.pegstoptype,
            'peglimittype': cls.peglimittype
        }
    
    @classmethod
    def get_connection_params(cls) -> dict:
        """Get connection parameters as a dictionary."""
        return {
            'userid': cls.userid,
            'password': cls.password,
            'url': cls.url,
            'connectiontype': cls.connectiontype
        }
    
    @classmethod
    def get_strategy_params(cls) -> dict:
        """Get strategy parameters as a dictionary."""
        return {
            'signal_col': cls.signal_col,
            'open_zone_col': cls.open_zone_col,
            'peaks_min_col': cls.peaks_min_col,
            'peaks_max_col': cls.peaks_max_col,
            'tolerance_peaks': cls.tolerance_peaks
        }

