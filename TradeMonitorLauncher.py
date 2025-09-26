import multiprocessing
import threading
import time
import datetime as dt
from TradeMonitor import run_trade_monitor


def run_monitor_process():
    """Run the trade monitor in a separate process"""
    try:
        print(f"[MONITOR LAUNCHER] Starting Trade Monitor Process at {dt.datetime.now()}")
        run_trade_monitor()
    except Exception as e:
        print(f"[MONITOR LAUNCHER] Error in monitor process: {e}")
        import traceback
        print(traceback.format_exc())


def start_independent_monitor():
    """Start the trade monitor as an independent process"""
    print("[MONITOR LAUNCHER] Launching Independent Trade Monitor...")
    
    # Create and start the monitor process
    monitor_process = multiprocessing.Process(target=run_monitor_process)
    monitor_process.daemon = True  # Make it daemon so it closes when main program exits
    monitor_process.start()
    
    print(f"[MONITOR LAUNCHER] Trade Monitor started with PID: {monitor_process.pid}")
    
    return monitor_process


if __name__ == "__main__":
    # Run as standalone application
    print("[MONITOR LAUNCHER] Starting Trade Monitor as standalone application...")
    monitor_process = start_independent_monitor()
    
    try:
        # Keep the main thread alive
        monitor_process.join()
    except KeyboardInterrupt:
        print("\n[MONITOR LAUNCHER] Shutting down...")
        if monitor_process.is_alive():
            monitor_process.terminate()
            monitor_process.join()
        print("[MONITOR LAUNCHER] Trade Monitor terminated.")

