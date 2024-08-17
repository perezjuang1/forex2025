import threading
from Price import RobotPrice
from Plotter import SubplotAnimation
from TradeMonitor import TradeMonitor

def create_MonitorPrice(instrument,days,timeframe):
    robot = RobotPrice(days,instrument,timeframe)
    robot.setMonitorPriceData()
    
def create_trademonitor(instrument,timeframe,timeframe_sup,days):
    trade = TradeMonitor(instrument,timeframe,timeframe_sup,days)
    trade.startMonitor()

def create_plotter(instrument,timeframe,timeframe_sup,days):
    ani = SubplotAnimation(instrument=instrument, timeframe=timeframe,timeframe_sup=timeframe_sup,days=days)
    ani.plt.show()
    

if __name__ == "__main__":
        instrument="EUR/USD"
        timeframe = "m5"
        timeframe_sup = "m30"
        days = 5

        t1 = threading.Thread(target=create_MonitorPrice, name="timeFrame"+timeframe+instrument, args=(instrument,days,timeframe))
        t2 = threading.Thread(target=create_MonitorPrice, name="timeFrame"+timeframe_sup+instrument, args=(instrument,days,timeframe_sup))
        t1.start(), t2.start()

        monitor = threading.Thread(target=create_trademonitor, name="monitor"+instrument, args=(instrument,timeframe,timeframe_sup,days))
        monitor.start()

        ploter = threading.Thread(target=create_plotter, name="ploter"+instrument, args=(instrument,timeframe, timeframe_sup,days))
        ploter.start()

