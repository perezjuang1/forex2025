import datetime as dt
from datetime import datetime
from backports.zoneinfo import ZoneInfo

from threading import Event
class ConfigurationOperation:
   
    userid = "U10D2470134"
    password = "Xvw0d"
    url = "http://www.fxcorporate.com/Hosts.jsp"
    connectiontype = "Demo"
    instrument_symbol = "EUR/USD"
    session = None
    pin = None
    lots = 10
    stop = 10
    limit = 30
    account = None
    timeframe = "m1"  # Available periods :  'm1', 'm5', 'm15', 'm30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8','D1', 'W1', or 'M1'.
    dateFormat = '%m.%d.%Y %H:%M:%S'
    date_from =  None
    date_to = None
    days = 4

    peggedstop = 'Y'
    peggedlimit = 'Y'

    pegstoptype = 'M'
    peglimittype = 'M'

    def __init__(self):   
        europe_London_datetime = datetime.now(ZoneInfo('Europe/London') )
        self.date_from =  europe_London_datetime - dt.timedelta(days=self.days)
        self.date_to = europe_London_datetime