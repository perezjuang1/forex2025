from ConnectionFxcm import RobotConnection
from Price import RobotPrice
import datetime as dt
import time
import numpy as np
import traceback

class Trading:
    def __init__(self, days, timeframe='m5', instrument="EUR/USD"):
        self.timeframe = timeframe
        self.instrument = instrument
        self.robotconnection = RobotConnection()
        self.connection = self.robotconnection.getConnection()
        self.corepy = self.robotconnection.getCorepy()
        self._robot_price = RobotPrice(days, self.instrument, self.timeframe)
        self.days = days

    def __del__(self):
        print('Object gets destroyed')
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
        except Exception as e:
            print("Exception: " + str(e))
            print(traceback.format_exc())
            raise 

# Example usage
if __name__ == "__main__":
    while True:
        trading = None
        try:
            time.sleep(5)  # Esperar antes de reiniciar
            trading = Trading(days=7, timeframe='m5', instrument="EUR/USD")
            trading.start_trade_monitor()
        except Exception as e:
            print("Fatal error occurred. Restarting Trading session...")
            print(traceback.format_exc())
            del trading  # Forzar destructor
            print("20 seconds to restart...")
            time.sleep(240)  # Esperar antes de reiniciar

