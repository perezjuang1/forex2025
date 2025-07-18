import datetime as dt
from datetime import datetime
from backports.zoneinfo import ZoneInfo

class ConfigurationOperation:
    # Parámetros centralizados para la estrategia
    signal_col = 'signal'
    open_zone_col = 'trade_open_zone'
    peaks_min_col = 'peaks_min'
    peaks_max_col = 'peaks_max'
    recent_range = (-12, -8)  # Para triggers_trades_open
    recent_close_range = (-7, -4)  # Para triggers_trades_close
   
    userid = "U10D2470448"
    password = "2Rcha"
    url = "http://www.fxcorporate.com/Hosts.jsp"
    connectiontype = "Demo"
    instrument_symbol = "EUR/USD"
    session = None
    pin = None
    lots = 8
    stop = 10
    limit = 30
    account = None
    timeframe = "m1"  # Available periods :  'm1', 'm5', 'm15', 'm30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8','D1', 'W1', or 'M1'.
    dateFormat = '%m.%d.%Y %H:%M:%S'
    date_from =  None
    date_to = None
    days = 5

    peggedstop = 'Y'
    peggedlimit = 'Y'

    pegstoptype = 'M'
    peglimittype = 'M'

    # Centralizar la lista de instrumentos aquí
    instruments = ["EUR/USD", "GBP/USD", "EUR/JPY", "AUD/JPY", "EUR/CAD"]
    # Tolerancia global para picos
    tolerance_peaks = 20

    def __init__(self):   
        europe_London_datetime = datetime.now(ZoneInfo('Europe/London') )
        self.date_from =  europe_London_datetime - dt.timedelta(days=self.days)
        self.date_to = europe_London_datetime

