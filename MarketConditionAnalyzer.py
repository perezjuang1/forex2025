import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from config import TradingConfig


class MarketConditionAnalyzer:
    """
    Analyzes market conditions to identify low liquidity, low volatility,
    and low movement frequency periods. These conditions are typically
    associated with poor trading opportunities and increased risk.
    """
    
    def __init__(self, window_liquidity: int = 20, window_volatility: int = 20, window_movement: int = 10):
        """
        Initialize the market condition analyzer.
        
        Args:
            window_liquidity: Rolling window size for liquidity analysis
            window_volatility: Rolling window size for volatility (ATR) calculation
            window_movement: Rolling window size for movement frequency analysis
        """
        self.window_liquidity = window_liquidity
        self.window_volatility = window_volatility
        self.window_movement = window_movement
        
        # Default thresholds (can be overridden via config)
        self.min_liquidity_percentile = 30  # Bottom 30% of volume = low liquidity
        self.min_atr_percentile = 25  # Bottom 25% of volatility = low volatility
        self.min_movement_frequency = 0.3  # At least 30% of candles should have significant movement
        self.min_range_pips = 0.0002  # Minimum range for EUR/USD (2 pips equivalent)
        
    def calculate_liquidity_metric(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate liquidity based on tick quantity (volume proxy).
        Lower tick quantity indicates lower liquidity.
        
        Args:
            df: DataFrame with 'tickqty' column
            
        Returns:
            Series with liquidity scores (0-1, where 1 = highest liquidity)
        """
        if 'tickqty' not in df.columns:
            return pd.Series([1.0] * len(df), index=df.index)
        
        # Use rolling median to smooth out spikes
        rolling_median = df['tickqty'].rolling(window=self.window_liquidity, min_periods=5).median()
        
        # Normalize to 0-1 scale based on percentile ranks
        liquidity_score = df['tickqty'].rolling(window=self.window_liquidity, min_periods=5).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )
        
        return liquidity_score.fillna(0.5)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Average True Range (ATR) as volatility measure.
        ATR measures the average price movement over a period.
        
        Args:
            df: DataFrame with 'bidhigh', 'bidlow', 'bidclose' columns
            period: Period for ATR calculation (defaults to window_volatility)
            
        Returns:
            Series with ATR values
        """
        if period is None:
            period = self.window_volatility
            
        if not all(col in df.columns for col in ['bidhigh', 'bidlow', 'bidclose']):
            return pd.Series([0.0] * len(df), index=df.index)
        
        # Calculate True Range components
        high_low = df['bidhigh'] - df['bidlow']
        high_close_prev = abs(df['bidhigh'] - df['bidclose'].shift(1))
        low_close_prev = abs(df['bidlow'] - df['bidclose'].shift(1))
        
        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # ATR is the moving average of True Range
        atr = true_range.rolling(window=period, min_periods=5).mean()
        
        return atr.bfill().fillna(0.0)
    
    def calculate_volatility_metric(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility score based on ATR percentile.
        Lower ATR indicates lower volatility and potentially poor conditions.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with volatility scores (0-1, where 1 = highest volatility)
        """
        atr = self.calculate_atr(df)
        
        # Calculate percentile rank of ATR in rolling window
        volatility_score = atr.rolling(window=self.window_volatility, min_periods=5).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )
        
        return volatility_score.fillna(0.5)
    
    def calculate_movement_frequency(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the frequency of significant price movements.
        Low frequency indicates choppy/sideways markets with little directional movement.
        
        Args:
            df: DataFrame with 'bidhigh', 'bidlow', 'bidclose' columns
            
        Returns:
            Series with movement frequency scores (0-1)
        """
        if not all(col in df.columns for col in ['bidhigh', 'bidlow']):
            return pd.Series([1.0] * len(df), index=df.index)
        
        # Calculate range (high - low) for each candle
        candle_range = df['bidhigh'] - df['bidlow']
        
        # Consider candles with range above minimum threshold as "moving"
        moving_candles = (candle_range >= self.min_range_pips).astype(int)
        
        # Calculate rolling percentage of moving candles
        movement_frequency = moving_candles.rolling(
            window=self.window_movement, 
            min_periods=3
        ).mean()
        
        return movement_frequency.fillna(0.5)
    
    def calculate_spread_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate spread proxy based on candle ranges.
        In low liquidity, spreads widen, leading to larger high-low ranges
        relative to price movement. This metric captures that.
        
        Args:
            df: DataFrame with 'bidhigh', 'bidlow', 'bidclose' columns
            
        Returns:
            Series with spread scores (0-1, where 1 = tighter spreads)
        """
        if not all(col in df.columns for col in ['bidhigh', 'bidlow', 'bidclose']):
            return pd.Series([1.0] * len(df), index=df.index)
        
        # Calculate range
        candle_range = df['bidhigh'] - df['bidlow']
        
        # Calculate price change magnitude
        price_change = abs(df['bidclose'].diff())
        
        # In good conditions, price change should be significant relative to range
        # In poor conditions (wide spreads), range is large but price doesn't move much
        spread_ratio = price_change / (candle_range + 1e-10)  # Avoid division by zero
        
        # Normalize to 0-1 scale
        spread_score = spread_ratio.rolling(window=self.window_movement, min_periods=3).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )
        
        return spread_score.fillna(0.5)
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive market condition analysis.
        
        Args:
            df: DataFrame with price data (bidhigh, bidlow, bidclose, tickqty)
            
        Returns:
            DataFrame with additional columns:
            - liquidity_score: 0-1 score for liquidity
            - volatility_score: 0-1 score for volatility
            - movement_frequency: 0-1 score for movement frequency
            - spread_score: 0-1 score for spread conditions
            - market_quality_score: Combined quality score (0-1)
            - is_good_condition: Boolean indicating if conditions are suitable for trading
        """
        if df is None or df.empty:
            return df
        
        result_df = df.copy()
        
        # Calculate individual metrics
        result_df['liquidity_score'] = self.calculate_liquidity_metric(df)
        result_df['volatility_score'] = self.calculate_volatility_metric(df)
        result_df['movement_frequency'] = self.calculate_movement_frequency(df)
        result_df['spread_score'] = self.calculate_spread_proxy(df)
        
        # Calculate combined market quality score (weighted average)
        # All metrics are equally important
        result_df['market_quality_score'] = (
            result_df['liquidity_score'] * 0.3 +
            result_df['volatility_score'] * 0.3 +
            result_df['movement_frequency'] * 0.2 +
            result_df['spread_score'] * 0.2
        )
        
        # Determine if conditions are good for trading
        # Conditions are good if all individual metrics pass their thresholds
        min_quality_threshold = 0.4  # Combined score threshold
        result_df['is_good_condition'] = (
            (result_df['liquidity_score'] >= (self.min_liquidity_percentile / 100.0)) &
            (result_df['volatility_score'] >= (self.min_atr_percentile / 100.0)) &
            (result_df['movement_frequency'] >= self.min_movement_frequency) &
            (result_df['market_quality_score'] >= min_quality_threshold)
        )
        
        return result_df
    
    def is_condition_suitable(self, df: pd.DataFrame, index: int = -1) -> Tuple[bool, Dict[str, float]]:
        """
        Check if market conditions at a specific index are suitable for trading.
        
        Args:
            df: DataFrame with market condition metrics
            index: Row index to check (default -1 for last row)
            
        Returns:
            Tuple of (is_suitable, metrics_dict)
        """
        if df is None or df.empty or 'is_good_condition' not in df.columns:
            # If analysis hasn't been done, assume conditions are suitable
            return True, {}
        
        row = df.iloc[index] if abs(index) <= len(df) else df.iloc[-1]
        
        metrics = {
            'liquidity_score': float(row.get('liquidity_score', 0.5)),
            'volatility_score': float(row.get('volatility_score', 0.5)),
            'movement_frequency': float(row.get('movement_frequency', 0.5)),
            'spread_score': float(row.get('spread_score', 0.5)),
            'market_quality_score': float(row.get('market_quality_score', 0.5))
        }
        
        is_suitable = bool(row.get('is_good_condition', True))
        
        return is_suitable, metrics


