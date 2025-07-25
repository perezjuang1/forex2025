from ConnectionFxcm import RobotConnection
from Price import Price
import datetime as dt
import time
import numpy as np
import traceback
import threading
import multiprocessing
from ConfigurationOperation import ConfigurationOperation

class Trading:
    def __init__(self, days, timeframe=None, instrument="EUR/USD"):
        if timeframe is None:
            timeframe = ConfigurationOperation.timeframe
        self.timeframe = timeframe
        self.instrument = instrument
        self.days = days
        
        # Initialize Price object which handles connection internally
        self._robot_price = Price(days, self.instrument, self.timeframe)
        
        # Get connection from Price object to avoid duplication
        self.connection = self._robot_price.connection
        self.robotconnection = self._robot_price.robotconnection

    def __del__(self):
        print('Object gets destroyed')
        if self.connection:
            self.connection.logout()

    def start_trade_monitor(self):    
        self.operation_detection(timeframe=self.timeframe)  
        while True:            
            currenttime = dt.datetime.now()  
            if self.timeframe == "m1" and currenttime.second == 0:
                self.operation_detection( timeframe=self.timeframe)    
                time.sleep(1)                    
            elif self.timeframe == "m5" and currenttime.second == 0 and currenttime.minute % 5 == 0:
                self.operation_detection( timeframe=self.timeframe)
                time.sleep(240)
            elif self.timeframe == "m15" and currenttime.second == 0 and currenttime.minute % 15 == 0:
                self.operation_detection( timeframe=self.timeframe)
                time.sleep(840)
            elif self.timeframe == "m30" and currenttime.second == 0 and currenttime.minute % 30 == 0:
                self.operation_detection( timeframe=self.timeframe)
                time.sleep(1740)
            elif self.timeframe == "H1" and currenttime.second == 0 and currenttime.minute == 0:
                self.operation_detection( timeframe=self.timeframe)
                time.sleep(3540)
            time.sleep(1)

    def operation_detection(self, timeframe): 
        try:
            print(f"[LOG] Iniciando operación_detection - timeframe: {timeframe} - " + str(dt.datetime.now())  )
            df = self._robot_price.get_price_data(instrument=self.instrument, timeframe=timeframe, days=self.days, connection=self.connection)
            df = self._robot_price.triggers_trades(df)
            df = self._robot_price.triggers_trades_close(df)
            print(df)
        except Exception as e:
            print("Exception: " + str(e))
            print(traceback.format_exc())
            raise 

def run_trading_for_instrument(instrument):
    while True:
        trading = None
        try:
            time.sleep(5)
            trading = Trading(days=7, instrument=instrument)
            trading.start_trade_monitor()
        except Exception as e:
            print(f"Fatal error occurred for {instrument}. Restarting Trading session...")
            print(traceback.format_exc())
            del trading
            print("20 seconds to restart...")
            time.sleep(20)

def run_plotter_for_instrument(instrument):
    """Run the plotter for a specific instrument"""
    try:
        from Plotter2025 import PlotConfig, ForexPlotter
        config = PlotConfig(
            instrument=instrument.replace("/", "_"),
            timeframe=ConfigurationOperation.timeframe
        )
        plotter = ForexPlotter(config)
        plotter.animate()
    except Exception as e:
        print(f"Error running plotter for {instrument}: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    from ConfigurationOperation import ConfigurationOperation
    instruments = ConfigurationOperation.instruments
    
    print("Starting Trading and Plotting System...")
    print(f"Instruments: {instruments}")
    
    # Start trading threads
    trading_threads = []
    for instrument in instruments:
        t = threading.Thread(target=run_trading_for_instrument, args=(instrument,))
        t.daemon = True  # Make threads daemon so they exit when main program exits
        t.start()
        trading_threads.append(t)
        print(f"Started trading thread for {instrument}")
    
    # Start plotting processes (separate processes for GUI)
    plotting_processes = []
    for instrument in instruments:
        p = multiprocessing.Process(target=run_plotter_for_instrument, args=(instrument,))
        p.start()
        plotting_processes.append(p)
        print(f"Started plotting process for {instrument}")
    
    try:
        # Wait for all processes to complete (they won't unless there's an error)
        for p in plotting_processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Terminate plotting processes
        for p in plotting_processes:
            if p.is_alive():
                p.terminate()
                p.join()
        print("All processes terminated.")

