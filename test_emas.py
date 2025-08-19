#!/usr/bin/env python3
"""
Test script to verify EMA50 and EMA80 are working correctly
"""

import pandas as pd
import numpy as np
from PriceAnalyzer import PriceAnalyzer

def test_ema_calculation():
    """Test that EMAs are calculated correctly"""
    
    print("="*60)
    print("TESTING EMA50 AND EMA80 CALCULATION")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_rows = 100
    
    # Generate sample price data
    base_price = 1.2000
    dates = [f"20241201{i:02d}0000" for i in range(n_rows)]
    
    # Create realistic price movements
    price_changes = np.random.normal(0, 0.0005, n_rows)
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] + change)
    
    # Create DataFrame with OHLC data
    df = pd.DataFrame({
        'date': dates,
        'bidhigh': [p + 0.0002 for p in prices],
        'bidlow': [p - 0.0002 for p in prices],
        'bidclose': prices,
        'bidopen': [p + np.random.normal(0, 0.0001) for p in prices],
        'tickqty': np.random.randint(100, 1000, n_rows)
    })
    
    print(f"Created test DataFrame with {len(df)} rows")
    print(f"Price range: {df['bidclose'].min():.6f} - {df['bidclose'].max():.6f}")
    
    # Initialize PriceAnalyzer
    analyzer = PriceAnalyzer(days=30, instrument="EUR/USD", timeframe="m1")
    
    # Calculate indicators
    df_with_indicators = analyzer.set_indicators(df)
    
    print(f"\nDataFrame columns after indicators: {list(df_with_indicators.columns)}")
    
    # Check if EMAs were calculated
    if 'ema50' in df_with_indicators.columns:
        print(f"✅ EMA50 calculated successfully")
        print(f"   First 5 values: {df_with_indicators['ema50'].head().tolist()}")
        print(f"   Last 5 values: {df_with_indicators['ema50'].tail().tolist()}")
        print(f"   NaN count: {df_with_indicators['ema50'].isna().sum()}")
    else:
        print("❌ EMA50 not found in DataFrame")
    
    if 'ema80' in df_with_indicators.columns:
        print(f"✅ EMA80 calculated successfully")
        print(f"   First 5 values: {df_with_indicators['ema80'].head().tolist()}")
        print(f"   Last 5 values: {df_with_indicators['ema80'].tail().tolist()}")
        print(f"   NaN count: {df_with_indicators['ema80'].isna().sum()}")
    else:
        print("❌ EMA80 not found in DataFrame")
    
    # Verify EMA calculation manually
    if 'ema50' in df_with_indicators.columns and 'ema80' in df_with_indicators.columns:
        # Manual EMA calculation for verification
        manual_ema50 = df_with_indicators['bidclose'].ewm(span=50).mean()
        manual_ema80 = df_with_indicators['bidclose'].ewm(span=80).mean()
        
        # Compare with calculated values
        ema50_diff = abs(df_with_indicators['ema50'] - manual_ema50).max()
        ema80_diff = abs(df_with_indicators['ema80'] - manual_ema80).max()
        
        print(f"\nVerification:")
        print(f"   EMA50 max difference: {ema50_diff:.10f}")
        print(f"   EMA80 max difference: {ema80_diff:.10f}")
        
        if ema50_diff < 1e-10 and ema80_diff < 1e-10:
            print("✅ EMAs calculated correctly!")
        else:
            print("❌ EMAs calculation mismatch detected")
    
    # Check trend calculation
    if 'trend' in df_with_indicators.columns:
        print(f"\nTrend calculation:")
        print(f"   BULL count: {(df_with_indicators['trend'] == 'BULL').sum()}")
        print(f"   BEAR count: {(df_with_indicators['trend'] == 'BEAR').sum()}")
        print(f"   FLAT count: {(df_with_indicators['trend'] == 'FLAT').sum()}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_ema_calculation()
