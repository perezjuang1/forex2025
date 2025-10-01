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
    
    # Peak detection tolerance
        # Eliminados open_zone_col y tolerance_peaks
    
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
    lots = 2  # Default lot size for trading
    stop = 10  # Stop loss in pips
    limit = 20  # Take profit in pips
    
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
    
    # List of instruments to trade - organized in two groups with different schedules
    instruments = [
        "EUR/USD",    # Major pair - high liquidity
        "GBP/USD",    # Major pair - cable
        "USD/JPY",    # Major pair - yen crosses
        "EUR/JPY",    # Recommended - EUR strength + JPY flows
        "EUR/GBP",    # EUR cross - very liquid, tight spreads
        "AUD/USD",    # Commodity currency - good volatility
        "USD/CAD",    # Oil-sensitive - stable trends
        "NZD/USD",    # Commodity currency - Pacific session
        "USD/CHF",    # Safe haven pair - European session
        "GBP/JPY"     # Volatile cross - multiple sessions
    ]
    
    # Group A: Asian/Pacific optimized (5 instruments)
    # Best trading hours: 21:00-08:00 UTC (Sydney/Tokyo sessions)
    group_a_instruments = [
        "AUD/USD",    # Primary: Sydney session
        "NZD/USD",    # Primary: Sydney session  
        "USD/JPY",    # Primary: Tokyo session
        "EUR/JPY",    # Cross: Tokyo/London overlap
        "GBP/JPY"     # Cross: Tokyo/London overlap
    ]
    
    # Group B: European/American optimized (5 instruments)  
    # Best trading hours: 07:00-21:00 UTC (London/New York sessions)
    group_b_instruments = [
        "EUR/USD",    # Primary: London/New York overlap
        "GBP/USD",    # Primary: London/New York overlap
        "EUR/GBP",    # Primary: London session
        "USD/CAD",    # Primary: New York session
        "USD/CHF"     # Primary: London/New York sessions
    ]
    
    # Trading schedule configuration
    trading_schedules = {
        'group_a': {
            'name': 'Asian/Pacific Group',
            'instruments': group_a_instruments,
            'optimal_hours_utc': {'start': 21, 'end': 8},  # 21:00 UTC to 08:00 UTC next day
            'description': 'Optimized for Sydney and Tokyo sessions'
        },
        'group_b': {
            'name': 'European/American Group', 
            'instruments': group_b_instruments,
            'optimal_hours_utc': {'start': 7, 'end': 21},  # 07:00 UTC to 21:00 UTC
            'description': 'Optimized for London and New York sessions'
        }
    }
    
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
            'peaks_max_col': cls.peaks_max_col
        }
    
    @classmethod
    def get_trading_schedules(cls) -> dict:
        """Get trading schedules configuration."""
        return cls.trading_schedules.copy()
    
    @classmethod
    def get_group_instruments(cls, group_name: str) -> list:
        """Get instruments for a specific group."""
        if group_name in cls.trading_schedules:
            return cls.trading_schedules[group_name]['instruments'].copy()
        return []
    
    @classmethod
    def get_instrument_group(cls, instrument: str) -> str:
        """Get the group name for a specific instrument."""
        for group_name, group_config in cls.trading_schedules.items():
            if instrument in group_config['instruments']:
                return group_name
        return 'unknown'
    
    @classmethod
    def is_optimal_trading_time(cls, instrument: str, check_time: datetime = None) -> dict:
        """
        Check if current time is optimal for trading a specific instrument.
        
        Args:
            instrument: Currency pair to check
            check_time: Time to check (default: current UTC time)
            
        Returns:
            Dict with optimization status and details
        """
        if check_time is None:
            check_time = datetime.now(pytz.timezone('UTC'))
        elif check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=pytz.timezone('UTC'))
        
        group_name = cls.get_instrument_group(instrument)
        if group_name == 'unknown':
            return {
                'is_optimal': False,
                'group': 'unknown',
                'reason': f'Instrument {instrument} not found in any trading group'
            }
        
        group_config = cls.trading_schedules[group_name]
        optimal_hours = group_config['optimal_hours_utc']
        current_hour = check_time.hour
        
        start_hour = optimal_hours['start']
        end_hour = optimal_hours['end']
        
        # Handle time ranges that cross midnight
        if start_hour > end_hour:  # e.g., 21:00 to 08:00 (next day)
            is_optimal = current_hour >= start_hour or current_hour < end_hour
        else:  # e.g., 07:00 to 21:00 (same day)
            is_optimal = start_hour <= current_hour < end_hour
        
        return {
            'is_optimal': is_optimal,
            'group': group_name,
            'group_name': group_config['name'],
            'optimal_hours': f"{start_hour:02d}:00-{end_hour:02d}:00 UTC",
            'current_hour': f"{current_hour:02d}:00 UTC",
            'description': group_config['description'],
            'reason': f"{'Optimal' if is_optimal else 'Non-optimal'} trading time for {group_config['name']}"
        }
    
    @classmethod
    def get_active_instruments_for_time(cls, check_time: datetime = None) -> dict:
        """
        Get instruments that should be actively trading at a specific time.
        
        Returns:
            Dict with active and inactive instruments
        """
        if check_time is None:
            check_time = datetime.now(pytz.timezone('UTC'))
        
        active_instruments = []
        inactive_instruments = []
        
        for instrument in cls.instruments:
            optimization_status = cls.is_optimal_trading_time(instrument, check_time)
            if optimization_status['is_optimal']:
                active_instruments.append({
                    'instrument': instrument,
                    'group': optimization_status['group'],
                    'group_name': optimization_status['group_name']
                })
            else:
                inactive_instruments.append({
                    'instrument': instrument,
                    'group': optimization_status['group'],
                    'group_name': optimization_status['group_name'],
                    'reason': optimization_status['reason']
                })
        
        return {
            'check_time': check_time.isoformat(),
            'active_instruments': active_instruments,
            'inactive_instruments': inactive_instruments,
            'active_count': len(active_instruments),
            'inactive_count': len(inactive_instruments)
        }

