import datetime as dt
from datetime import datetime
import os
import configparser
import pytz
from typing import List, Dict, Optional


class TradingConfig:
    """
    Centralized configuration class for the forex trading system.
    
    This class loads configuration from a config.ini file and provides
    access to all configuration parameters for:
    - Trading strategy parameters
    - Connection settings
    - Trading parameters
    - Instrument lists
    
    Best practices implemented:
    - Configuration loaded from external file
    - Type safety with proper conversions
    - Default values for missing configuration
    - Environment variable support for sensitive data
    - Singleton-like pattern with class-level initialization
    """
    
    _config = None
    _initialized = False
    
    # Default values (used if config file is missing or values are not found)
    _defaults = {
        'peaks_min_col': 'peaks_min',
        'peaks_max_col': 'peaks_max',
        'signal_col': 'signal',
        'userid': '',
        'password': '',
        'url': 'http://www.fxcorporate.com/Hosts.jsp',
        'connectiontype': 'Demo',
        'timeframe': 'm1',
        'lots': 1,
        'stop': 10,
        'limit': 30,
        'peggedstop': 'Y',
        'peggedlimit': 'Y',
        'pegstoptype': 'M',
        'peglimittype': 'M',
        'date_format': '%m.%d.%Y %H:%M:%S',
        'days': 4,
        'instruments': ['EUR/USD', 'GBP/USD', 'USD/JPY']
    }
    
    @classmethod
    def _load_config(cls, config_path: str = 'config.ini') -> configparser.ConfigParser:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            ConfigParser object with loaded configuration
        """
        config = configparser.ConfigParser()
        
        # Check if config file exists
        if not os.path.exists(config_path):
            print(f"[WARNING] Configuration file '{config_path}' not found. Using default values.")
            return config
        
        try:
            config.read(config_path, encoding='utf-8')
        except Exception as e:
            print(f"[ERROR] Failed to load configuration file '{config_path}': {e}. Using default values.")
        
        return config
    
    @classmethod
    def _get_config_value(cls, section: str, key: str, default=None, value_type=str):
        """
        Get a configuration value with type conversion and default fallback.
        
        Args:
            section: Configuration section name
            key: Configuration key name
            default: Default value if not found
            value_type: Type to convert the value to (str, int, float, bool)
            
        Returns:
            Configuration value converted to the specified type
        """
        if cls._config is None:
            cls._config = cls._load_config()
        
        try:
            if cls._config.has_section(section) and cls._config.has_option(section, key):
                # Use raw=True to avoid interpolation issues with % characters (e.g., date formats)
                value = cls._config.get(section, key, raw=True)
                
                # Type conversion
                if value_type == int:
                    return int(value)
                elif value_type == float:
                    return float(value)
                elif value_type == bool:
                    return cls._config.getboolean(section, key)
                elif value_type == list:
                    # Handle comma-separated lists
                    return [item.strip() for item in value.split(',') if item.strip()]
                else:
                    return value
        except (ValueError, configparser.NoOptionError, configparser.NoSectionError) as e:
            print(f"[WARNING] Error reading config [{section}]{key}: {e}. Using default: {default}")
        
        # Return default or fallback to _defaults
        if default is not None:
            return default
        return cls._defaults.get(key, None)
    
    @classmethod
    def _get_env_or_config(cls, env_var: str, section: str, key: str, default=None) -> str:
        """
        Get value from environment variable first, then from config file.
        Useful for sensitive data like passwords.
        
        Args:
            env_var: Environment variable name
            section: Configuration section name
            key: Configuration key name
            default: Default value if not found
            
        Returns:
            Value from environment variable or config file
        """
        # Check environment variable first
        value = os.getenv(env_var)
        if value:
            return value
        
        # Fall back to config file
        return cls._get_config_value(section, key, default)
    
    @classmethod
    def _initialize(cls):
        """Initialize configuration on first access."""
        if not cls._initialized:
            cls._config = cls._load_config()
            cls._initialized = True
    
    # ============================================================================
    # TRADING STRATEGY PARAMETERS
    # ============================================================================
    
    @classmethod
    def get_peaks_min_col(cls) -> str:
        """Get the column name for minimum peaks."""
        cls._initialize()
        return cls._get_config_value('TRADING_STRATEGY', 'peaks_min_col', cls._defaults['peaks_min_col'])
    
    @classmethod
    def get_peaks_max_col(cls) -> str:
        """Get the column name for maximum peaks."""
        cls._initialize()
        return cls._get_config_value('TRADING_STRATEGY', 'peaks_max_col', cls._defaults['peaks_max_col'])
    
    @classmethod
    def get_signal_col(cls) -> str:
        """Get the column name for trading signals."""
        cls._initialize()
        return cls._get_config_value('TRADING_STRATEGY', 'signal_col', cls._defaults['signal_col'])
    
    # Class-level attributes for backward compatibility (lazy-loaded)
    @classmethod
    def _get_peaks_min_col(cls) -> str:
        return cls.get_peaks_min_col()
    
    @classmethod
    def _get_peaks_max_col(cls) -> str:
        return cls.get_peaks_max_col()
    
    @classmethod
    def _get_signal_col(cls) -> str:
        return cls.get_signal_col()
    
    # ============================================================================
    # CONNECTION SETTINGS
    # ============================================================================
    
    @classmethod
    def get_userid(cls) -> str:
        """Get FXCM user ID from environment variable or config file."""
        cls._initialize()
        return cls._get_env_or_config('FXCM_USERID', 'CONNECTION', 'userid', cls._defaults['userid'])
    
    @classmethod
    def get_password(cls) -> str:
        """Get FXCM password from environment variable or config file."""
        cls._initialize()
        return cls._get_env_or_config('FXCM_PASSWORD', 'CONNECTION', 'password', cls._defaults['password'])
    
    @classmethod
    def get_url(cls) -> str:
        """Get FXCM connection URL."""
        cls._initialize()
        return cls._get_config_value('CONNECTION', 'url', cls._defaults['url'])
    
    @classmethod
    def get_connectiontype(cls) -> str:
        """Get FXCM connection type (Demo/Live)."""
        cls._initialize()
        return cls._get_config_value('CONNECTION', 'connectiontype', cls._defaults['connectiontype'])
    
    # Class-level accessors for backward compatibility
    @classmethod
    def _get_userid(cls) -> str:
        return cls.get_userid()
    
    @classmethod
    def _get_password(cls) -> str:
        return cls.get_password()
    
    @classmethod
    def _get_url(cls) -> str:
        return cls.get_url()
    
    @classmethod
    def _get_connectiontype(cls) -> str:
        return cls.get_connectiontype()
    
    # Session and account settings
    account = None
    
    # ============================================================================
    # TRADING PARAMETERS
    # ============================================================================
    
    @classmethod
    def get_timeframe(cls) -> str:
        """Get the default timeframe."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'timeframe', cls._defaults['timeframe'])
    
    @classmethod
    def get_lots(cls) -> int:
        """Get the default lot size."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'lots', cls._defaults['lots'], int)
    
    @classmethod
    def get_stop(cls) -> int:
        """Get the stop loss in pips."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'stop', cls._defaults['stop'], int)
    
    @classmethod
    def get_limit(cls) -> int:
        """Get the take profit in pips."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'limit', cls._defaults['limit'], int)
    
    @classmethod
    def get_peggedstop(cls) -> str:
        """Get pegged stop setting."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'peggedstop', cls._defaults['peggedstop'])
    
    @classmethod
    def get_peggedlimit(cls) -> str:
        """Get pegged limit setting."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'peggedlimit', cls._defaults['peggedlimit'])
    
    @classmethod
    def get_pegstoptype(cls) -> str:
        """Get pegged stop type."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'pegstoptype', cls._defaults['pegstoptype'])
    
    @classmethod
    def get_peglimittype(cls) -> str:
        """Get pegged limit type."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'peglimittype', cls._defaults['peglimittype'])
    
    @classmethod
    def get_date_format(cls) -> str:
        """Get the date format string."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'date_format', cls._defaults['date_format'])
    
    @classmethod
    def get_days(cls) -> int:
        """Get the number of days for historical data."""
        cls._initialize()
        return cls._get_config_value('TRADING_PARAMETERS', 'days', cls._defaults['days'], int)
    
    # Class-level accessors for backward compatibility
    @classmethod
    def _get_timeframe(cls) -> str:
        return cls.get_timeframe()
    
    @classmethod
    def _get_lots(cls) -> int:
        return cls.get_lots()
    
    @classmethod
    def _get_stop(cls) -> int:
        return cls.get_stop()
    
    @classmethod
    def _get_limit(cls) -> int:
        return cls.get_limit()
    
    @classmethod
    def _get_peggedstop(cls) -> str:
        return cls.get_peggedstop()
    
    @classmethod
    def _get_peggedlimit(cls) -> str:
        return cls.get_peggedlimit()
    
    @classmethod
    def _get_pegstoptype(cls) -> str:
        return cls.get_pegstoptype()
    
    @classmethod
    def _get_peglimittype(cls) -> str:
        return cls.get_peglimittype()
    
    @classmethod
    def _get_dateFormat(cls) -> str:
        return cls.get_date_format()
    
    @classmethod
    def _get_days(cls) -> int:
        return cls.get_days()
    
    # ============================================================================
    # INSTRUMENT CONFIGURATION
    # ============================================================================
    
    @classmethod
    def get_instruments(cls) -> List[str]:
        """Get the list of trading instruments."""
        cls._initialize()
        instruments = cls._get_config_value('INSTRUMENTS', 'instruments', cls._defaults['instruments'], list)
        if isinstance(instruments, list):
            return instruments.copy()
        return instruments
    
    # Class-level accessor for backward compatibility
    @classmethod
    def _get_instruments(cls) -> List[str]:
        return cls.get_instruments()
    
    def __init__(self):
        """
        Initialize the configuration with current London time settings.
        All configuration values are loaded and set as instance attributes
        for backward compatibility with existing code.
        """
        TradingConfig._initialize()
        
        # Load all configuration values as instance attributes for backward compatibility
        self.peaks_min_col = TradingConfig.get_peaks_min_col()
        self.peaks_max_col = TradingConfig.get_peaks_max_col()
        self.signal_col = TradingConfig.get_signal_col()
        
        self.userid = TradingConfig.get_userid()
        self.password = TradingConfig.get_password()
        self.url = TradingConfig.get_url()
        self.connectiontype = TradingConfig.get_connectiontype()
        self.account = None
        
        self.timeframe = TradingConfig.get_timeframe()
        self.lots = TradingConfig.get_lots()
        self.stop = TradingConfig.get_stop()
        self.limit = TradingConfig.get_limit()
        self.peggedstop = TradingConfig.get_peggedstop()
        self.peggedlimit = TradingConfig.get_peggedlimit()
        self.pegstoptype = TradingConfig.get_pegstoptype()
        self.peglimittype = TradingConfig.get_peglimittype()
        self.dateFormat = TradingConfig.get_date_format()
        self.days = TradingConfig.get_days()
        
        self.instruments = TradingConfig.get_instruments()
        
        # Set date range based on London timezone
        europe_london_datetime = datetime.now(pytz.timezone('Europe/London'))
        self.date_from = europe_london_datetime - dt.timedelta(days=self.days)
        self.date_to = europe_london_datetime
    
    @classmethod
    def get_trading_params(cls) -> Dict[str, any]:
        """Get trading parameters as a dictionary."""
        cls._initialize()
        return {
            'lots': cls.get_lots(),
            'stop': cls.get_stop(),
            'limit': cls.get_limit(),
            'peggedstop': cls.get_peggedstop(),
            'pegstoptype': cls.get_pegstoptype(),
            'peggedlimit': cls.get_peggedlimit(),
            'peglimittype': cls.get_peglimittype()
        }
    
    @classmethod
    def get_connection_params(cls) -> Dict[str, str]:
        """Get connection parameters as a dictionary."""
        cls._initialize()
        return {
            'userid': cls.get_userid(),
            'password': cls.get_password(),
            'url': cls.get_url(),
            'connectiontype': cls.get_connectiontype()
        }
    
    @classmethod
    def get_strategy_params(cls) -> Dict[str, str]:
        """Get strategy parameters as a dictionary."""
        cls._initialize()
        return {
            'peaks_min_col': cls.get_peaks_min_col(),
            'peaks_max_col': cls.get_peaks_max_col()
        }
    
    @classmethod
    def reload_config(cls, config_path: str = 'config.ini'):
        """
        Reload configuration from file. Useful for runtime configuration updates.
        
        Args:
            config_path: Path to the configuration file
        """
        cls._config = cls._load_config(config_path)
        cls._initialized = True
        print(f"[INFO] Configuration reloaded from '{config_path}'")
    
    # ============================================================================
    # MARKET CONDITION PARAMETERS
    # ============================================================================
    
    @classmethod
    def get_market_filter_enabled(cls) -> bool:
        """Get whether market condition filtering is enabled."""
        cls._initialize()
        return cls._get_config_value('MARKET_CONDITIONS', 'enable_market_filter', True, bool)
    
    @classmethod
    def get_min_liquidity_percentile(cls) -> int:
        """Get minimum liquidity percentile threshold."""
        cls._initialize()
        return cls._get_config_value('MARKET_CONDITIONS', 'min_liquidity_percentile', 30, int)
    
    @classmethod
    def get_min_atr_percentile(cls) -> int:
        """Get minimum ATR percentile threshold."""
        cls._initialize()
        return cls._get_config_value('MARKET_CONDITIONS', 'min_atr_percentile', 25, int)
    
    @classmethod
    def get_min_movement_frequency(cls) -> float:
        """Get minimum movement frequency threshold."""
        cls._initialize()
        return cls._get_config_value('MARKET_CONDITIONS', 'min_movement_frequency', 0.3, float)
    
    @classmethod
    def get_min_range_pips(cls) -> float:
        """Get minimum range in pips."""
        cls._initialize()
        return cls._get_config_value('MARKET_CONDITIONS', 'min_range_pips', 0.0002, float)
    
    @classmethod
    def get_window_liquidity(cls) -> int:
        """Get liquidity analysis window size."""
        cls._initialize()
        return cls._get_config_value('MARKET_CONDITIONS', 'window_liquidity', 20, int)
    
    @classmethod
    def get_window_volatility(cls) -> int:
        """Get volatility analysis window size."""
        cls._initialize()
        return cls._get_config_value('MARKET_CONDITIONS', 'window_volatility', 20, int)
    
    @classmethod
    def get_window_movement(cls) -> int:
        """Get movement frequency analysis window size."""
        cls._initialize()
        return cls._get_config_value('MARKET_CONDITIONS', 'window_movement', 10, int)


