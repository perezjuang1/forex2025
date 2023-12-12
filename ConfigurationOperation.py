import datetime as dt
from datetime import datetime
from pytz import timezone 

from threading import Event
class ConfigurationOperation:
   
    userid = "U10D2460130"
    password = "Oze0i"
    url = "http://www.fxcorporate.com/Hosts.jsp"
    connectiontype = "Demo"
    currency = 'EUR/USD'
    session = None
    pin = None
    amount = 10
    stop = -10
    limit = 30
    account = None
    timeframe = "m30"  # Available periods :  'm1', 'm5', 'm15', 'm30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8','D1', 'W1', or 'M1'.
    quotes_count = 200000  #Number of Candles
    dateFormat = '%m.%d.%Y %H:%M:%S'
    date_from =  None
    date_to = None
    d = None # B(BUY) OR S(SELL)
    lots = 10
    days = 20
    filename = None
    # event = Event()

    def __init__(self):   
        europe_London_datetime = datetime.now( timezone('Europe/London') )
        self.date_from =  europe_London_datetime - dt.timedelta(days=self.days)
        self.date_to = europe_London_datetime
        self.filename =  self.currency.replace("/", "_")  + "_" + self.timeframe + ".csv"
