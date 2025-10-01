from PriceAnalyzer import PriceAnalyzer
import datetime as dt
import time
import numpy as np
import traceback
import threading
import multiprocessing
from TradingConfiguration import TradingConfig

class TradingSystem:
    def __init__(self, days, timeframe=None, instrument="EUR/USD"):
        if timeframe is None:
            timeframe = TradingConfig.get_timeframe()
        self.timeframe = timeframe
        self.instrument = instrument
        self.days = days
        
        # Initialize PriceAnalyzer object which handles connection internally
        self.priceAnalyzer = PriceAnalyzer(days, self.instrument, self.timeframe)
        
        # Get connection from Price object to avoid duplication
        self.connection = self.priceAnalyzer.connection
        self.robotconnection = self.priceAnalyzer.robotconnection

    def __del__(self):
        print('Object gets destroyed')
        if hasattr(self, 'connection') and self.connection:
            self.connection.logout()

    def start_trade_monitor(self):    
        self.operation_detection(timeframe=self.timeframe)  
        while True:            
            currenttime = dt.datetime.now()  
            if self.timeframe == "m1" and currenttime.second == 0:
                self.operation_detection(timeframe=self.timeframe)    
                time.sleep(1)                    
            elif self.timeframe == "m5" and currenttime.second == 0 and currenttime.minute % 5 == 0:
                self.operation_detection(timeframe=self.timeframe)
                time.sleep(240)
            elif self.timeframe == "m15" and currenttime.second == 0 and currenttime.minute % 15 == 0:
                self.operation_detection(timeframe=self.timeframe)
                time.sleep(840)
            elif self.timeframe == "m30" and currenttime.second == 0 and currenttime.minute % 30 == 0:
                self.operation_detection(timeframe=self.timeframe)
                time.sleep(1740)
            elif self.timeframe == "H1" and currenttime.second == 0 and currenttime.minute == 0:
                self.operation_detection(timeframe=self.timeframe)
                time.sleep(3540)
            time.sleep(1)

    def operation_detection(self, timeframe): 
        try:
            # Check if instrument is suspended
            if self.priceAnalyzer.is_instrument_suspended():
                print(f"❌ {self.instrument}: SUSPENDIDO → Esperando reactivación")
                self.priceAnalyzer.wait_for_instrument_availability()
                return
            
            # Check instrument availability before processing
            availability_info = self.priceAnalyzer.check_instrument_availability()
            if availability_info['should_suspend']:
                market_status = availability_info['market_status']
                next_open = market_status.get('next_open', 'Unknown')
                if next_open and hasattr(next_open, 'strftime'):
                    next_time = next_open.strftime('%H:%M UTC')
                elif isinstance(next_open, str):
                    next_time = next_open
                else:
                    next_time = "Unknown"
                print(f"❌ {self.instrument}: No disponible → SUSPENDIDO hasta {next_time}")
                self.priceAnalyzer.handle_instrument_suspension(availability_info)
                return
            
            # Show that instrument is operating normally
            print(f"✅ {self.instrument}: Disponible → Opera normal")
            
            # Process trading data
            df = self.priceAnalyzer.get_price_data(instrument=self.instrument, timeframe=timeframe, days=self.days, connection=self.connection)
            df = self.priceAnalyzer.set_indicators(df)
            df = self.priceAnalyzer.set_signals_to_trades(df)      
            self.priceAnalyzer.triggers_trades_open(df)
            self.priceAnalyzer.triggers_trades_close(df)
            
            # Save price data with indicators if not empty
            if not df.empty:
                self.priceAnalyzer.save_price_data_file(df)
        except Exception as e:
            print(f"❌ {self.instrument}: ERROR → {str(e)}")
            print(traceback.format_exc())

def run_trading_for_instrument(instrument):
    """Run the trading system for a specific instrument"""
    try:
        print(f"STARTING {instrument}")
        trading = TradingSystem(days=7, instrument=instrument)
        trading.start_trade_monitor()
    except Exception as e:
        print(f"FATAL ERROR {instrument}: Restarting in 20s...")
        print(traceback.format_exc())
        time.sleep(20)

def run_visualizer_for_instrument(instrument):
    """Run the visualizer for a specific instrument"""
    try:
        from TradingVisualizer import run_single_visualizer
        run_single_visualizer()
    except Exception as e:
        print(f"Error running visualizer: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    from TradingConfiguration import TradingConfig
    instruments = TradingConfig.get_instruments()
    schedules = TradingConfig.get_trading_schedules()
    current_status = TradingConfig.get_active_instruments_for_time()
    
    print("Starting Trading and Plotting System...")
    print(f"Total instruments: {len(instruments)}")
    print("\n=== TRADING GROUPS CONFIGURATION ===")
    
    for group_key, group_config in schedules.items():
        print(f"\n{group_config['name']} ({group_key}):")
        print(f"  Description: {group_config['description']}")
        print(f"  Trading hours: {group_config['optimal_hours_utc']['start']:02d}:00-{group_config['optimal_hours_utc']['end']:02d}:00 UTC")
        print(f"  Instruments ({len(group_config['instruments'])}):")
        for inst in group_config['instruments']:
            print(f"    - {inst}")
    
    print(f"\n=== CURRENT STATUS (UTC: {dt.datetime.now().strftime('%H:%M')}) ===")
    if current_status['active_instruments']:
        print(f"ACTIVE INSTRUMENTS ({current_status['active_count']}):")
        for item in current_status['active_instruments']:
            print(f"  ✓ {item['instrument']} ({item['group_name']})")
    
    if current_status['inactive_instruments']:
        print(f"\nINACTIVE INSTRUMENTS ({current_status['inactive_count']}):")
        for item in current_status['inactive_instruments']:
            print(f"  - {item['instrument']} ({item['group_name']})")
    
    print(f"\nAll instruments: {instruments}")
    
    # Show initial availability status
    print(f"\n=== ESTADO INICIAL DE INSTRUMENTOS ===")
    from MarketHoursChecker import MarketHoursChecker
    market_checker = MarketHoursChecker()
    
    for instrument in instruments:
        market_status = market_checker.is_market_open(instrument)
        if market_status['is_available']:
            print(f"✅ {instrument}: Disponible → Opera normal")
        else:
            next_open = market_status.get('next_open', 'Unknown')
            if next_open and hasattr(next_open, 'strftime'):
                next_time = next_open.strftime('%H:%M UTC')
            elif isinstance(next_open, str):
                next_time = next_open
            else:
                next_time = "Unknown"
            print(f"❌ {instrument}: No disponible → SUSPENDIDO hasta {next_time}")
    
    # Ask user if they want to start the trade monitor
    start_monitor = input("\nStart independent trade monitor? (y/n): ").lower().strip()
    monitor_process = None
    
    if start_monitor == 'y':
        try:
            from TradeMonitorLauncher import start_independent_monitor
            monitor_process = start_independent_monitor()
            print("Trade Monitor started successfully!")
        except Exception as e:
            print(f"Failed to start Trade Monitor: {e}")
    
    # Start trading threads
    trading_threads = []
    for instrument in instruments:
        t = threading.Thread(target=run_trading_for_instrument, args=(instrument,))
        t.daemon = True  # Make threads daemon so they exit when main program exits
        t.start()
        trading_threads.append(t)
        print(f"Started trading thread for {instrument}")
    
    # Start single visualizer process
    print("Starting single window visualizer...")
    visualizer_process = multiprocessing.Process(target=run_visualizer_for_instrument, args=(None,))
    visualizer_process.start()
    
    try:
        # Wait for visualizer process to complete
        visualizer_process.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Terminate visualizer process
        if visualizer_process.is_alive():
            visualizer_process.terminate()
            visualizer_process.join()
        # Terminate monitor process if running
        if monitor_process and monitor_process.is_alive():
            monitor_process.terminate()
            monitor_process.join()
            print("Trade Monitor terminated.")
        print("All processes terminated.") 