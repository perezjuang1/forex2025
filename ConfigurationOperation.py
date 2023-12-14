import datetime as dt
from datetime import datetime
from pytz import timezone 

from threading import Event
class ConfigurationOperation:
   
    userid = "U10D2460130"
    password = "Oze0i"
    url = "http://www.fxcorporate.com/Hosts.jsp"
    connectiontype = "Demo"
    instrument = 'EUR/CAD'
    session = None
    pin = None
    lots = 2
    stop = 8
    limit = 16
    account = None
    timeframe = "m1"  # Available periods :  'm1', 'm5', 'm15', 'm30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8','D1', 'W1', or 'M1'.
    quotes_count = 200000  #Number of Candles
    dateFormat = '%m.%d.%Y %H:%M:%S'
    date_from =  None
    date_to = None
    d_buy_sell = None # B(BUY) OR S(SELL)

    days = 1
    filename = None

    peggedstop = 'Y'
    peggedlimit = 'Y'

    pegstoptype = 'M'
    peglimittype = 'M'

    def __init__(self):   
        europe_London_datetime = datetime.now( timezone('Europe/London') )
        self.date_from =  europe_London_datetime - dt.timedelta(days=self.days)
        self.date_to = europe_London_datetime
        self.filename =  self.instrument.replace("/", "_")  + "_" + self.timeframe + ".csv"
