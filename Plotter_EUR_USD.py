import configparser
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

config = configparser.ConfigParser()
config.read('RobotV5.ini')

# This function runs once at the beginning of the strategy to create price/indicator streams
time_frame_operations = config['timeframe']
timeframe = time_frame_operations['timeframe']

#symbol = time_frame_operations['symbol']
fileName = str(os.path.basename(__file__))
fileName = fileName.replace(".py", "")
fileName = fileName.replace("Plotter_", "")
symbol = fileName  # .replace("_", "/")


amount_value = 1
vallimit = 8
valstop = -15

plt.style.use('dark_background')

def readData():
    return pd.read_csv(symbol + '_m30.csv')

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        fig = plt.figure()

        self.axBase = fig.add_subplot(4, 1, 1)
        self.RSI = fig.add_subplot(4, 1, 2)
        self.VOLUM = fig.add_subplot(4, 1, 3)
        self.STO = fig.add_subplot(4, 1, 4)

        self.t = np.linspace(0, 80, 400)

        self.axBase.set_xlabel('Date')
        self.axBase.set_ylabel('Price Move' + symbol)
        self.linePrice = Line2D([], [],color='white')

        self.lineSMA200 = Line2D([], [], color='green')
        self.lineSMA400 = Line2D([], [], color='red')
        self.ema_res1 = Line2D([], [], color='orange')
        self.ema_res2 = Line2D([], [], color='red')
        self.ema_res3 = Line2D([], [], color='green')
        self.sellOpen, = self.axBase.plot([], [], 'v', color='green')
        self.sellbidclose, = self.axBase.plot([], [], '.', color='green')
        self.buyOpen, = self.axBase.plot([], [], '^', color='red')
        self.buybidclose, = self.axBase.plot([], [], '.', color='red')
        self.peaks_min, = self.axBase.plot([], [], '--', color='yellow')
        self.peaks_max, = self.axBase.plot([], [], '--', color='yellow')

        self.axBase.add_line(self.linePrice)
        self.axBase.add_line(self.lineSMA200)
        self.axBase.add_line(self.lineSMA400)
        self.axBase.add_line(self.ema_res1)
        self.axBase.add_line(self.ema_res2)
        self.axBase.add_line(self.ema_res3)
        self.axBase.add_line(self.sellOpen)
        self.axBase.add_line(self.sellbidclose)
        self.axBase.add_line(self.buyOpen)
        self.axBase.add_line(self.buybidclose)
        self.axBase.add_line(self.peaks_min)
        self.axBase.add_line(self.peaks_max)


        self.VOLUM.set_xlabel('VOLUM')
        self.VOLUM.set_ylabel('Volumen')
        self.VOLUMLineVolum = Line2D([], [], color='blue')
        self.VOLUMLinePromedio = Line2D([], [], color='orange')
        #self.VOLUMFast = Line2D([], [], color='green')
        #self.VOLUMLimit = Line2D([], [], color='red')

        self.VOLUM.add_line(self.VOLUMLineVolum)
        self.VOLUM.add_line(self.VOLUMLinePromedio)
        # self.VOLUM.add_line(self.VOLUMFast)
        # self.VOLUM.add_line(self.VOLUMLimit)

        self.RSI.set_xlabel('DATE')
        self.RSI.set_ylabel('RSI')
        self.lineRSI_INF = Line2D([], [], color='red')
        self.lineRSI_SUP = Line2D([], [], color='red')
        self.lineRSI_MED = Line2D([], [], color='white')
        self.lineRSI = Line2D([], [], color='orange')

        self.RSI.add_line(self.lineRSI_INF)
        self.RSI.add_line(self.lineRSI_SUP)
        self.RSI.add_line(self.lineRSI_MED)
        self.RSI.add_line(self.lineRSI)


        self.STO.set_xlabel('DATE')
        self.STO.set_ylabel('STO')
        self.STO_K = Line2D([], [], color='green')
        self.STO_D = Line2D([], [], color='red')
        self.sto_LimitSup = Line2D([], [], color='orange')
        self.sto_LImitInf = Line2D([], [], color='orange')

        self.STO.add_line(self.STO_K)
        self.STO.add_line(self.STO_D)
        self.STO.add_line(self.sto_LimitSup)
        self.STO.add_line(self.sto_LImitInf)

        animation.TimedAnimation.__init__(self, fig, interval=20000, blit=True)

    def _draw_frame(self, framedata):
        self.axBase.clear
        self.RSI.clear
        self.VOLUM.clear
        self.STO.clear

        pricedata = readData()

        # SMA
        x = pricedata['bidclose'].index
        self.linePrice.set_data(x, pricedata['bidclose'])
        self.lineSMA200.set_data(x, pricedata['ema'])
        self.lineSMA400.set_data(x, pricedata['ema_slow'])
        self.ema_res1.set_data(x, pricedata['ema_res1'])
        self.ema_res2.set_data(x, pricedata['ema_res2'])
        self.ema_res2.set_data(x, pricedata['ema_res3'])

        self.sellOpen.set_data(
            pricedata.loc[pricedata.zone_sell == 1.0].index, pricedata.bidclose[pricedata.zone_sell == 1.0])
        self.sellbidclose.set_data(
            pricedata.loc[pricedata.zone_sell == -1.0].index, pricedata.bidclose[pricedata.zone_sell == -1.0])

        self.buyOpen.set_data(
            pricedata.loc[pricedata.zone_buy == 1.0].index, pricedata.bidclose[pricedata.zone_buy == 1.0])
        self.buybidclose.set_data(
            pricedata.loc[pricedata.zone_buy == -1.0].index, pricedata.bidclose[pricedata.zone_buy == -1.0])

        self.peaks_min.set_data(
            pricedata.loc[pricedata.peaks_min == 1.0].index, pricedata.bidclose[pricedata.peaks_min == 1.0])
        self.peaks_max.set_data(
            pricedata.loc[pricedata.peaks_max == 1.0].index, pricedata.bidclose[pricedata.peaks_max == 1.0])
        # # Plot results
        #self.axPicsLinePriceHigh.set_data(x, pricedata['bidhigh'])
        #self.axPicsLinePriceLow.set_data(x, pricedata['bidlow'])

        # RSI
        pricedata['RSI_INF'] = 30
        pricedata['RSI_SUP'] = 70
        pricedata['RSI_MED'] = 50

        self.lineRSI_INF.set_data(x, pricedata['RSI_INF'])
        self.lineRSI_SUP.set_data(x, pricedata['RSI_SUP'])
        self.lineRSI_MED.set_data(x, pricedata['RSI_MED'])
        self.lineRSI.set_data(x, pricedata['rsi'])

        self.VOLUMLineVolum.set_data(x, pricedata['volumenPipsDiference'])
        self.VOLUMLinePromedio.set_data(x, pricedata['volumLimitOperation'])
        #self.VOLUMFast.set_data(x, pricedata['tickqtySMAFast'])
        #self.VOLUMLimit.set_data(x, pricedata['tickqtyLIMIT'])

        self.RSI.relim()
        self.RSI.autoscale_view()



        # STO
        pricedata['sto_LimitSup'] = 0.20
        pricedata['sto_LImitInf'] = 0.80
   
        self.STO_K.set_data(x, pricedata['sto_k'])
        self.STO_D.set_data(x, pricedata['sto_d'])
        self.sto_LimitSup.set_data(x, pricedata['sto_LimitSup'])
        self.sto_LImitInf.set_data(x, pricedata['sto_LImitInf'])

        self.STO.relim()
        self.STO.autoscale_view()


        self.axBase.relim()
        self.axBase.autoscale_view()

        self.VOLUM.relim()
        self.VOLUM.autoscale_view()

        self._drawn_artists = [self.linePrice, self.lineSMA200, self.lineSMA400, self.ema_res1, self.ema_res2, self.ema_res3, self.sellOpen, self.sellbidclose, self.buyOpen, self.buybidclose,self.peaks_min,self.peaks_max,
                               self.lineRSI_INF, self.lineRSI_SUP, self.lineRSI,self.lineRSI_MED,
                               self.VOLUMLineVolum, self.VOLUMLinePromedio,#self.VOLUMFast,self.VOLUMLimit,
                               self.STO_K, self.STO_D, self.sto_LimitSup,self.sto_LImitInf,
                               ]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = [self.linePrice, self.lineSMA200, self.lineSMA400, self.ema_res1, self.ema_res2, self.ema_res3, self.sellOpen, self.sellbidclose, self.buyOpen, self.buybidclose,self.peaks_min,self.peaks_max,
                 self.lineRSI_INF, self.lineRSI_SUP, self.lineRSI,self.lineRSI_MED,
                 self.VOLUMLineVolum, #self.VOLUMLinePromedio,self.VOLUMFast,self.VOLUMLimit,
                 self.STO_K, self.STO_D, self.sto_LimitSup,self.sto_LImitInf,
                 ]
        for l in lines:
            l.set_data([], [])


def start():
    ani = SubplotAnimation()
    # ani.save('test_sub.mp4')
    plt.show()


if __name__ == "__main__":
    start()
