#!/usr/bin/env python3
"""
Simple test script to debug EMA calculation
"""

import pandas as pd
import numpy as np

def test_simple_ema():
    """Simple EMA test"""
    
    print("="*60)
    print("SIMPLE EMA TEST")
    print("="*60)
    
    # Create simple data
    np.random.seed(42)
    n_rows = 100
    
    # Generate sample price data
    base_price = 1.2000
    prices = [base_price]
    for i in range(1, n_rows):
        change = np.random.normal(0, 0.0005)
        prices.append(prices[-1] + change)
    
    # Create DataFrame
    df = pd.DataFrame({
        'bidclose': prices,
        'bidopen': [p + np.random.normal(0, 0.0001) for p in prices],
        'bidhigh': [p + 0.0002 for p in prices],
        'bidlow': [p - 0.0002 for p in prices]
    })
    
    print(f"Created DataFrame with {len(df)} rows")
    
    # Calculate EMAs manually
    df['ema50'] = df['bidclose'].ewm(span=50).mean()
    df['ema80'] = df['bidclose'].ewm(span=80).mean()
    
    print(f"EMAs calculated manually:")
    print(f"   EMA50 first 5: {df['ema50'].head().tolist()}")
    print(f"   EMA50 last 5: {df['ema50'].tail().tolist()}")
    print(f"   EMA80 first 5: {df['ema80'].head().tolist()}")
    print(f"   EMA80 last 5: {df['ema80'].tail().tolist()}")
    
    # Check for NaN values
    print(f"\nNaN counts:")
    print(f"   EMA50 NaN: {df['ema50'].isna().sum()}")
    print(f"   EMA80 NaN: {df['ema80'].isna().sum()}")
    
    # Verify calculation
    manual_ema50 = df['bidclose'].ewm(span=50).mean()
    manual_ema80 = df['bidclose'].ewm(span=80).mean()
    
    diff50 = abs(df['ema50'] - manual_ema50).max()
    diff80 = abs(df['ema80'] - manual_ema80).max()
    
    print(f"\nVerification:")
    print(f"   EMA50 max diff: {diff50:.10f}")
    print(f"   EMA80 max diff: {diff80:.10f}")
    
    if diff50 < 1e-10 and diff80 < 1e-10:
        print("✅ EMAs calculated correctly!")
    else:
        print("❌ EMA calculation mismatch")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_simple_ema()
