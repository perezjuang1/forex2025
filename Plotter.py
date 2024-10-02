import configparser
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from Price import RobotPrice
from PriceUtils import PriceUtils

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self,instrument,timeframe,timeframe_sup,timeframe_sup2,days):
        plt.style.use('dark_background')
        self.plt = plt
        fig = self.plt.figure()
        self.plt.ion

        self.instrument = instrument
        self.timeframe = timeframe
        self.timeframe_sup = timeframe_sup
        self.timeframe_sup2 = timeframe_sup2
        self.days = days

        self.axBase = fig.add_subplot(1, 1, 1)

        self.t = np.linspace(0, 80, 400)

        self.axBase.set_xlabel('Date')
        self.axBase.set_ylabel('Price Move' + instrument)

        self.linePrice, = self.axBase.plot([], [], linestyle='dotted', color='gray', label='Price')
        #self.linePrice_Inf, =  self.axBase.plot([], [], marker = '.', color='green', label='Price Time Frame Inf')
        #self.linePrice_Sup, =  self.axBase.plot([], [], marker = '.', color='red',label='Price Time Frame Sup')
        self.lineEMA1, = self.axBase.plot([], [], linestyle='dotted', color='pink', label='moving_avg')   #self.axBase.plot([], [], color='orange', label='moving_avg')
        #self.lineEMA2, = self.axBase.plot([], [], color='gray', label='EMA')


        self.peaks_min_Inf, = self.axBase.plot([], [],  linestyle='dotted',  marker =  '^' , color='green')
        self.peaks_max_Inf, = self.axBase.plot([], [],  linestyle='dotted', marker = 'v', color='blue')
        self.peaks_min_Sup, = self.axBase.plot([], [], linestyle='-', marker =  '^', color='white')
        self.peaks_max_Sup, = self.axBase.plot([], [],linestyle='-',  marker = 'v', color='white')
        self.peaks_min_Sup2, = self.axBase.plot([], [], linestyle='-', marker =  '^', color='gray')
        self.peaks_max_Sup2, = self.axBase.plot([], [],linestyle='-',  marker = 'v', color='gray')

        self.sellOpen_Inf, = self.axBase.plot([], [], ',', color='green')
        self.sellClose_Inf, = self.axBase.plot([], [], ',', color='green')        
        self.buyOpen_Inf, = self.axBase.plot([], [], ',', color='red')
        self.buyClose_Inf, = self.axBase.plot([], [], ',', color='red')



        self.sellOpen_Sup, = self.axBase.plot([], [], ',', color='green')
        self.sellClose_Sup, = self.axBase.plot([], [], ',', color='green')

        self.buyOpen_Sup, = self.axBase.plot([], [], ',', color='red')
        self.buyClose_Sup, = self.axBase.plot([], [], ',', color='red')


        self.TriggerBuy, = self.axBase.plot([], [], '^', color='orange')
        self.TriggerSell, = self.axBase.plot([], [], 'v', color='orange')






        self.axBase.add_line(self.linePrice)
        #self.axBase.add_line(self.linePrice_Sup)
        #self.axBase.add_line(self.linePrice_Inf)
        self.axBase.add_line(self.lineEMA1)
        #self.axBase.add_line(self.lineEMA2)

        self.axBase.add_line(self.sellOpen_Inf)
        self.axBase.add_line(self.sellClose_Inf)
        self.axBase.add_line(self.buyOpen_Inf)
        self.axBase.add_line(self.buyClose_Inf)
        self.axBase.add_line(self.peaks_min_Inf)
        self.axBase.add_line(self.peaks_max_Inf)



        self.axBase.add_line(self.sellOpen_Sup)
        self.axBase.add_line(self.sellClose_Sup)

        self.axBase.add_line(self.buyClose_Sup)
        self.axBase.add_line(self.peaks_min_Sup)
        self.axBase.add_line(self.peaks_max_Sup)

        self.axBase.add_line(self.peaks_min_Sup2)
        self.axBase.add_line(self.peaks_max_Sup2)
        
        self.axBase.add_line(self.TriggerBuy)
        self.axBase.add_line(self.TriggerSell)
        #self.VOLUM.set_xlabel('VOLUM')
        #self.VOLUM.set_ylabel('Volumen')
        #self.VOLUMLineVolum = Line2D([], [], color='blue')
        #self.VOLUMLinePromedio = Line2D([], [], color='orange')
        #self.VOLUMFast = Line2D([], [], color='green')
        #self.VOLUMLimit = Line2D([], [], color='red')

        #self.VOLUM.add_line(self.VOLUMLineVolum)
        #self.VOLUM.add_line(self.VOLUMLinePromedio)
        # self.VOLUM.add_line(self.VOLUMFast)
        # self.VOLUM.add_line(self.VOLUMLimit)

        #self.RSI.set_xlabel('DATE')
        #self.RSI.set_ylabel('RSI')
        #self.lineRSI_INF = Line2D([], [], color='red')
        #self.lineRSI_SUP = Line2D([], [], color='red')
        #self.lineRSI_MED = Line2D([], [], color='white')
        #self.lineRSI = Line2D([], [], color='orange')

        #self.RSI.add_line(self.lineRSI_INF)
        #self.RSI.add_line(self.lineRSI_SUP)
        #self.RSI.add_line(self.lineRSI_MED)
        #self.RSI.add_line(self.lineRSI)


        #self.STO.set_xlabel('DATE')
        #self.STO.set_ylabel('STO')
        #self.STO_K = Line2D([], [], color='green')
        #self.STO_D = Line2D([], [], color='red')
        #self.sto_LimitSup = Line2D([], [], color='orange')
        #self.sto_LImitInf = Line2D([], [], color='orange')

        #self.STO.add_line(self.STO_K)
        #self.STO.add_line(self.STO_D)
        #self.STO.add_line(self.sto_LimitSup)
        #self.STO.add_line(self.sto_LImitInf)
        self.plt.legend()
        animation.TimedAnimation.__init__(self, fig, interval=20000, blit=True)
    
    
    def _draw_frame(self, framedata):
        self.axBase.clear
        #self.RSI.clear
        #self.VOLUM.clear
        #self.STO.clear


        pricedataConsolidated = PriceUtils().readPriceDataFileConsolidated(self.instrument, self.timeframe,  self.timeframe_sup)
        self.linePrice.set_data(pricedataConsolidated['date'].index, pricedataConsolidated['bidclose'])

        filtered_inf = pricedataConsolidated[pricedataConsolidated['timeframe'] == self.timeframe]
        #self.linePrice_Inf.set_data(filtered_inf['date'].index, filtered_inf['bidclose'])

        filtered_sup = pricedataConsolidated[pricedataConsolidated['timeframe'] == self.timeframe_sup]
        #self.linePrice_Sup.set_data(filtered_sup['date'].index, filtered_sup['bidclose'])


        filtered_sup2 = pricedataConsolidated[pricedataConsolidated['timeframe'] == self.timeframe_sup2]
        #self.linePrice_Sup.set_data(filtered_sup['date'].index, filtered_sup['bidclose'])


        #self.lineSMA200.set_data(x, pricedata['ema'])
        #self.lineSMA400.set_data(x, pricedata['ema_slow'])
        #self.ema_res1.set_data(x, pricedata['ema_res1'])
        #self.ema_res2.set_data(x, pricedata['ema_res2'])
        #self.ema_res2.set_data(x, pricedata['ema_res3'])


        #Draw Inferior Price
        self.sellOpen_Inf.set_data(filtered_inf.loc[filtered_inf.sell == 1.0].index, filtered_inf.bidclose[filtered_inf.sell == 1.0])
        self.sellClose_Inf.set_data(filtered_inf.loc[filtered_inf.sell == -1.0].index, filtered_inf.bidclose[filtered_inf.sell == -1.0])

        self.buyOpen_Inf.set_data(filtered_inf.loc[filtered_inf.buy == 1.0].index, filtered_inf.bidclose[filtered_inf.buy == 1.0])
        self.buyClose_Inf.set_data(filtered_inf.loc[filtered_inf.buy == -1.0].index, filtered_inf.bidclose[filtered_inf.buy == -1.0])

        self.peaks_min_Inf.set_data(filtered_inf.loc[filtered_inf.peaks_min == 1.0].index, filtered_inf.bidclose[filtered_inf.peaks_min == 1.0])
        self.peaks_max_Inf.set_data(filtered_inf.loc[filtered_inf.peaks_max == 1.0].index, filtered_inf.bidclose[filtered_inf.peaks_max == 1.0])
        
        #EMA
        self.lineEMA1.set_data(pricedataConsolidated['date'].index, pricedataConsolidated['moving_avg'])
        #self.lineEMA2.set_data(filtered_inf['date'].index, filtered_inf['ema_slow'])


        #Draw Superior Price
        self.sellOpen_Sup.set_data(filtered_sup.loc[filtered_sup.sell == 1.0].index, filtered_sup.bidclose[filtered_sup.sell == 1.0])
        self.sellClose_Sup.set_data(filtered_sup.loc[filtered_sup.sell == -1.0].index, filtered_sup.bidclose[filtered_sup.sell == -1.0])


        self.buyOpen_Sup.set_data(filtered_sup.loc[filtered_sup.buy == 1.0].index, filtered_sup.bidclose[filtered_sup.buy == 1.0])
        self.buyClose_Sup.set_data(filtered_sup.loc[filtered_sup.buy == -1.0].index, filtered_sup.bidclose[filtered_sup.buy == -1.0])

        self.peaks_min_Sup.set_data(filtered_sup.loc[filtered_sup.peaks_min == 1.0].index, filtered_sup.bidclose[filtered_sup.peaks_min == 1.0])
        self.peaks_max_Sup.set_data(filtered_sup.loc[filtered_sup.peaks_max == 1.0].index, filtered_sup.bidclose[filtered_sup.peaks_max == 1.0])

        self.peaks_min_Sup2.set_data(filtered_sup2.loc[filtered_sup2.peaks_min == 1.0].index, filtered_sup2.bidclose[filtered_sup2.peaks_min == 1.0])
        self.peaks_max_Sup2.set_data(filtered_sup2.loc[filtered_sup2.peaks_max == 1.0].index, filtered_sup2.bidclose[filtered_sup2.peaks_max == 1.0])


        self.TriggerSell.set_data(filtered_inf.loc[filtered_inf.TriggerSell == 1.0].index, filtered_inf.bidclose[filtered_inf.TriggerSell == 1.0])
        self.TriggerBuy.set_data(filtered_inf.loc[filtered_inf.TriggerBuy == 1.0].index, filtered_inf.bidclose[filtered_inf.TriggerBuy == 1.0])

        # # Plot results
        #self.axPicsLinePriceHigh.set_data(x, pricedata['bidhigh'])
        #self.axPicsLinePriceLow.set_data(x, pricedata['bidlow'])

        # RSI
        #pricedata['RSI_INF'] = 30
        #pricedata['RSI_SUP'] = 70
        #pricedata['RSI_MED'] = 50

        #self.lineRSI_INF.set_data(x, pricedata['RSI_INF'])
        #self.lineRSI_SUP.set_data(x, pricedata['RSI_SUP'])
        #self.lineRSI_MED.set_data(x, pricedata['RSI_MED'])
        #self.lineRSI.set_data(x, pricedata['rsi'])

        #self.VOLUMLineVolum.set_data(x, pricedata['volumenPipsDiference'])
        #self.VOLUMLinePromedio.set_data(x, pricedata['volumLimitOperation'])
        #self.VOLUMFast.set_data(x, pricedata['tickqtySMAFast'])
        #self.VOLUMLimit.set_data(x, pricedata['tickqtyLIMIT'])

        #self.RSI.relim()
        #self.RSI.autoscale_view()



        # STO
        #pricedata['sto_LimitSup'] = 0.20
        #pricedata['sto_LImitInf'] = 0.80
   
        #self.STO_K.set_data(x, pricedata['sto_k'])
        #self.STO_D.set_data(x, pricedata['sto_d'])
        #self.sto_LimitSup.set_data(x, pricedata['sto_LimitSup'])
        #self.sto_LImitInf.set_data(x, pricedata['sto_LImitInf'])

        #self.STO.relim()
        #self.STO.autoscale_view()


        #self.axBase.relim()
        #self.axBase.autoscale_view()

        #self.VOLUM.relim()
        #self.VOLUM.autoscale_view()

        self.axBase.relim()
        self.axBase.autoscale_view()

        self._drawn_artists = [self.linePrice, self.lineEMA1, # self.lineEMA2,#self.lineSMA400, self.ema_res1, self.ema_res2, self.ema_res3, 
                               #self.linePrice_Sup,
                               #self.linePrice_Inf,
              

                               self.sellOpen_Inf, self.sellClose_Inf, self.buyOpen_Inf, self.buyClose_Inf,self.peaks_min_Inf,self.peaks_max_Inf,
                               #self.lineRSI_INF, self.lineRSI_SUP, self.lineRSI,self.lineRSI_MED,
                               #self.VOLUMLineVolum, self.VOLUMLinePromedio,#self.VOLUMFast,self.VOLUMLimit,
                               #self.STO_K, self.STO_D, self.sto_LimitSup,self.sto_LImitInf,

                               self.sellOpen_Sup, self.sellClose_Sup, self.buyOpen_Sup, self.buyClose_Sup, self.peaks_min_Sup, self.peaks_max_Sup,
                                                                                                            self.peaks_min_Sup2, self.peaks_max_Sup2,
                                self.TriggerBuy, 
                                self.TriggerSell
                               ]
        plt.pause(2)
        
    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = [self.linePrice, self.lineEMA1, #self.lineEMA2, # self.lineSMA400, self.ema_res1, self.ema_res2, self.ema_res3, 
                 #self.linePrice_Sup,
                 #self.linePrice_Inf,
                 #self.sellOpen, self.sellbidclose, self.buyOpen, self.buybidclose,self.peaks_min,self.peaks_max,
                 #self.lineRSI_INF, self.lineRSI_SUP, self.lineRSI,self.lineRSI_MED,
                 #self.VOLUMLineVolum, self.VOLUMLinePromedio,#self.VOLUMFast,self.VOLUMLimit,
                 #self.STO_K, self.STO_D, self.sto_LimitSup,self.sto_LImitInf,


                                self.sellOpen_Sup,
                                self.sellClose_Sup,
                                self.buyOpen_Sup,
                                self.buyClose_Sup,
                                self.peaks_min_Sup,
                                self.peaks_max_Sup,
                                self.peaks_min_Sup2,
                                self.peaks_max_Sup2,
                                self.TriggerBuy, 
                                self.TriggerSell
                 ]
        for l in lines:
            l.set_data([], [])