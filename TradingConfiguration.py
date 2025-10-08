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
    
    # Peak detection parameters
    peak_order = 30  # Order parameter for peak detection (lower = more peaks, higher = fewer peaks)
    
    # Median adjustment percentages for high/low
    median_high_upper_pct = 0.0015  # +0.15% by default (~15 pips) for bidhigh/bidlow
    median_low_lower_pct = 0.0015   # -0.15% by default (~15 pips)
    
    # Median adjustment percentages for close (smaller than high/low)
    median_close_upper_pct = 0.0008  # +0.08% by default (~8 pips) for bidclose
    median_close_lower_pct = 0.0008  # -0.08% by default (~8 pips)
    
    # Instrument-specific median adjustments (optional)
    # If instrument not in dict, uses default percentages above
    instrument_median_adjustments = {
        "EUR/USD": {
            "upper": 0.001, "lower": 0.001,           # 0.10% for high/low (~10 pips)
            "close_upper": 0.0004, "close_lower": 0.0004  # 0.04% for close (~4 pips)
        },
        "GBP/USD": {
            "upper": 0.0015, "lower": 0.0015,           # 0.15% for high/low (~15 pips, more volatile)
            "close_upper": 0.0005, "close_lower": 0.0005  # 0.05% for close (~5 pips)
        },
        "USD/JPY": {
            "upper": 0.002, "lower": 0.002,           # 0.20% for high/low (~20 pips, higher price)
            "close_upper": 0.0006, "close_lower": 0.0006  # 0.06% for close (~6 pips)
        }
    }
    
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
    lots = 1  # Default lot size for trading
    stop = 10  # Stop loss in pips
    limit = 30  # Take profit in pips
    
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
    instruments = ["EUR/USD", "GBP/USD", "USD/JPY"]
    
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
            'pegstoptype': cls.pegstoptype,
            'peggedlimit': cls.peggedlimit,
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
            'peaks_min_col': cls.peaks_min_col,
            'peaks_max_col': cls.peaks_max_col,
            'median_high_upper_pct': cls.median_high_upper_pct,
            'median_low_lower_pct': cls.median_low_lower_pct
        }
    
    @classmethod
    def get_median_adjustments(cls, instrument: str) -> dict:
        """Get median adjustment percentages for a specific instrument.
        
        Args:
            instrument: Trading instrument (e.g., 'EUR/USD')
            
        Returns:
            Dictionary with 'upper' and 'lower' percentage adjustments
        """
        return cls.instrument_median_adjustments.get(
            instrument,
            {"upper": cls.median_high_upper_pct, "lower": cls.median_low_lower_pct}
        )

