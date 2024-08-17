from pytz import timezone 
import pandas as pd


class PriceUtils:
        def __init__(self):  
                self.pricedata = None      
        def readPriceDataFileConsolidated(self, instrument, timeframe, timeframe_sup):
                return pd.read_csv(instrument.replace("/", "_") + '_' + timeframe + "_" + timeframe_sup + ".csv")
