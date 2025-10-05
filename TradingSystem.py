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
        last_restart_hour = dt.datetime.now().hour
        
        while True:            
            currenttime = dt.datetime.now()
            current_hour = currenttime.hour
            
            # Check if we need to restart (new hour)
            if current_hour != last_restart_hour:
                print(f"🔄 {self.instrument}: REINICIANDO SISTEMA → Nueva hora detectada ({currenttime.strftime('%H:%M')})")
                self._restart_system()
                last_restart_hour = current_hour
                continue
            
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

    def _restart_system(self):
        """Restart the trading system by reinitializing components"""
        try:
            print(f"🔄 {self.instrument}: Iniciando reinicio del sistema...")
            
            # Close existing connection
            if hasattr(self, 'connection') and self.connection:
                try:
                    self.connection.logout()
                    print(f"✅ {self.instrument}: Conexión cerrada")
                except:
                    pass
            
            # Reinitialize PriceAnalyzer
            self.priceAnalyzer = PriceAnalyzer(self.days, self.instrument, self.timeframe)
            self.connection = self.priceAnalyzer.connection
            self.robotconnection = self.priceAnalyzer.robotconnection
            
            print(f"✅ {self.instrument}: Sistema reiniciado exitosamente")
            
        except Exception as e:
            print(f"❌ {self.instrument}: ERROR en reinicio → {str(e)}")
            print(traceback.format_exc())

    def operation_detection(self, timeframe): 
        try:
            # Process trading data (24/7 trading)
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
    except Exception:
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
    instruments = TradingConfig.get_trading_instruments()
    
    print("Starting Trading and Plotting System...")
    print(f"Total instruments: {len(instruments)}")
    print(f"Instruments ({len(instruments)}):")
    for inst in instruments:
        print(f"  - {inst}")
    
    print(f"\n=== CURRENT STATUS (UTC: {dt.datetime.now().strftime('%H:%M')}) ===")
    
    # Show initial status
    print("\n=== ESTADO INICIAL DE INSTRUMENTOS ===")
    for instrument in instruments:
        print(f"✅ {instrument}: Listo para operar (24/7)")
    
    # Start trade monitor automatically
    print("\n[MONITOR LAUNCHER] Launching Independent Trade Monitor...")
    monitor_process = None
    
    try:
        from TradeMonitorLauncher import start_independent_monitor
        monitor_process = start_independent_monitor()
        print("[MONITOR LAUNCHER] Trade Monitor started with PID:", monitor_process.pid)
        print("Trade Monitor started automatically!")
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