from FxcmConnection import RobotConnection
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
            print(f"[LOG] Starting operation_detection - timeframe: {timeframe} - {dt.datetime.now()}")
            df = self.priceAnalyzer.get_price_data(instrument=self.instrument, timeframe=timeframe, days=self.days, connection=self.connection)
            df = self.priceAnalyzer.set_indicators(df)
            df = self.priceAnalyzer.set_signals_to_trades(df)      
            self.priceAnalyzer.triggers_trades_open(df)
            self.priceAnalyzer.triggers_trades_close(df)
            
            # Save price data with indicators if not empty
            if not df.empty:
                self.priceAnalyzer.save_price_data_file(df)
        except Exception as e:
            print(f"Exception in operation_detection: {str(e)}")
            print(traceback.format_exc())

def run_trading_for_instrument(instrument):
    """Run the trading system for a specific instrument"""
    try:
        print(f"[LOG] Starting trading for {instrument} - {dt.datetime.now()}")
        trading = TradingSystem(days=7, instrument=instrument)
        trading.start_trade_monitor()
    except Exception as e:
        print(f"Fatal error occurred for {instrument}. Restarting Trading session...")
        print(traceback.format_exc())
        print("20 seconds to restart...")
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
    
    print("Starting Trading and Plotting System...")
    print(f"Instruments: {instruments}")
    
    # Ask user if they want to start the trade monitor
    start_monitor = input("Start independent trade monitor? (y/n): ").lower().strip()
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