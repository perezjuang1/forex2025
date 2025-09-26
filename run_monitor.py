#!/usr/bin/env python3
"""
Independent Trade Monitor Runner

This script runs only the trade monitor without the main trading system.
Use this when you want to monitor existing trades without opening new ones.

The monitor will:
- Check open trades every minute
- Track pip gains/losses 
- Record highest pip gains achieved
- Automatically close trades when pips start declining from peak
- Log all activity to CSV and log files

Usage:
    python run_monitor.py
"""

import sys
import os
import time
import datetime as dt

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from TradeMonitor import TradeMonitor


def main():
    """Main function to run the trade monitor"""
    print("=" * 60)
    print("          INDEPENDENT TRADE MONITOR")
    print("=" * 60)
    print(f"Started at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Features:")
    print("- Monitors all open trades every minute")
    print("- Tracks pip gains and maximum pip peaks")
    print("- Auto-closes trades when pips decline from peak")
    print("- Logs all activity to files")
    print()
    print("Files created:")
    print("- logs/trade_monitor_YYYYMMDD.log (detailed logs)")
    print("- logs/trade_monitor.csv (monitoring data)")
    print()
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)
    print()
    
    try:
        # Create and start the monitor
        monitor = TradeMonitor()
        
        # Check initial status
        print("[MONITOR] Checking connection and initial trades...")
        time.sleep(2)
        
        # Start monitoring
        monitor.start_monitoring()
        
    except KeyboardInterrupt:
        print("\n[MONITOR] Monitoring stopped by user")
    except Exception as e:
        print(f"\n[MONITOR] Error: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("\n[MONITOR] Trade Monitor shutdown complete")


if __name__ == "__main__":
    main()

