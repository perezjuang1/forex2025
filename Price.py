import pandas as pd


from datetime import datetime
from backports.zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import signal
import datetime as dt
from ConnectionFxcm import RobotConnection
import time

class RobotPrice: 
    def __init__(self):  
        self.pricedata = None            

    def __init__(self, days, instrument, timeframe):
        self.instrument = instrument
        self.timeframe = timeframe
        self.pricedata = None
        self.days = days
        self.robotconnection = RobotConnection()
        self.connection = self.robotconnection.getConnection()
        
    def existingOperation(self, instrument, BuySell):
        existOperation = False
        try:
            trades_table = self.connection.get_table(self.connection.TRADES)            
            for trade_row in trades_table:
                        if BuySell == trade_row.BuySell:
                            print("Existing Trade: " + str(trade_row.TradeID))
                            existOperation = True                  
            return existOperation
        except Exception as e:
                print("Exception: " + str(e))
                return existOperation
        
    def CloseOperation(self, instrument, BuySell):
        try:
            accounts_response_reader = self.connection.get_table_reader(self.connection.ACCOUNTS)
            accountId = None
            for account in accounts_response_reader:
                accountId = account.account_id
                print(accountId)    

            orders_table = self.connection.get_table(self.connection.TRADES)
            for trade in orders_table:
                buy_sell = ''
                if trade.instrument == instrument and trade.buy_sell == BuySell:   
                    buy_sell = self.corepy.Constants.SELL if trade.buy_sell == self.corepy.Constants.BUY else self.corepy.Constants.BUY                
                    print('Closing ' + str(buy_sell))
                    if buy_sell != None:
                        request = self.connection.create_order_request(
                            order_type=self.corepy.Constants.Orders.TRUE_MARKET_CLOSE,
                            OFFER_ID=trade.offer_id,
                            ACCOUNT_ID=accountId,
                            BUY_SELL=buy_sell,
                            AMOUNT=trade.amount,
                            TRADE_ID=trade.trade_id
                        )
                        self.connection.send_request_async(request)
                        print('Request Sended')
                        print('Closed ' + str(buy_sell))
                else:
                    print('Trade is not the same ' + str(buy_sell) + ' != ' + str(trade.BuySell))
        except Exception as e:
            print("Exception: " + str(e))

    def createEntryOrder(self, str_buy_sell=None):
        args = self.robotconnection.args
        common = self.robotconnection.common
        fxcorepy = self.robotconnection.fxcorepy

        str_instrument = self.instrument
        str_lots = args.lots
        str_account = args.account
        stop = args.stop
        peggedstop = args.peggedstop
        pegstoptype = args.pegstoptype
        limit = args.limit
        peggedlimit = args.peggedlimit
        peglimittype = args.peglimittype

        if peggedstop:
            if not pegstoptype:
                print('pegstoptype must be specified')
                return
            if pegstoptype != 'O' and pegstoptype != 'M':
                print('pegstoptype is invalid. "O" or "M" only.')
                return
            peggedstop = peggedstop.lower()
            if peggedstop != 'y':
                peggedstop = None

        if pegstoptype:
            pegstoptype = pegstoptype.upper()

        if peggedlimit:
            if not peglimittype:
                print('peglimittype must be specified')
                return
            if peglimittype != 'O' and peglimittype != 'M':
                print('peglimittype is invalid. "O" or "M" only.')
                return
            peggedlimit = peggedlimit.lower()
            if peggedlimit != 'y':
                peggedlimit = None

        if peglimittype:
            peglimittype = peglimittype.upper()

        try:
            account = common.get_account(self.connection, str_account)
            if not account:
                raise Exception("The account '{0}' is not valid".format(str_account))
            else:
                str_account = account.account_id
                print("AccountID='{0}'".format(str_account))

            offer = common.get_offer(self.connection, str_instrument)
            if offer is None:
                raise Exception("The instrument '{0}' is not valid".format(str_instrument))

            login_rules = self.connection.login_rules
            trading_settings_provider = login_rules.trading_settings_provider
            base_unit_size = trading_settings_provider.get_base_unit_size(str_instrument, account)
            amount = base_unit_size * str_lots
            entry = fxcorepy.Constants.Orders.TRUE_MARKET_OPEN

            if str_buy_sell == 'B':
                stopv = -stop
                limitv = limit
                str_buy_sell = fxcorepy.Constants.BUY
            else:
                stopv = stop
                limitv = -limit
                str_buy_sell = fxcorepy.Constants.SELL

            if peggedstop:
                if peggedlimit:
                    request = self.connection.create_order_request(
                        order_type=entry,
                        OFFER_ID=offer.offer_id,
                        ACCOUNT_ID=str_account,
                        BUY_SELL=str_buy_sell,
                        PEG_TYPE_STOP=pegstoptype,
                        PEG_OFFSET_STOP=stopv,
                        PEG_TYPE_LIMIT=peglimittype,
                        PEG_OFFSET_LIMIT=limitv,
                        AMOUNT=amount,
                    )
                else:
                    request = self.connection.create_order_request(
                        order_type=entry,
                        OFFER_ID=offer.offer_id,
                        ACCOUNT_ID=str_account,
                        BUY_SELL=str_buy_sell,
                        PEG_TYPE_STOP=pegstoptype,
                        PEG_OFFSET_STOP=stopv,
                        RATE_LIMIT=limit,
                        AMOUNT=amount,
                    )
            else:
                if peggedlimit:
                    request = self.connection.create_order_request(
                        order_type=entry,
                        OFFER_ID=offer.offer_id,
                        ACCOUNT_ID=str_account,
                        BUY_SELL=str_buy_sell,
                        RATE_STOP=stop,
                        PEG_TYPE_LIMIT=peglimittype,
                        PEG_OFFSET_LIMIT=limitv,
                        AMOUNT=amount,
                    )
                else:
                    request = self.connection.create_order_request(
                        order_type=entry,
                        OFFER_ID=offer.offer_id,
                        ACCOUNT_ID=str_account,
                        BUY_SELL=str_buy_sell,
                        AMOUNT=amount,
                        RATE_STOP=stop,
                        RATE_LIMIT=limit,
                    )
            self.connection.send_request_async(request)
        except Exception as e:
            print(e)

    def __del__(self):
        print('Object gets destroyed')
        self.connection.logout()

    def savePriceDataFile(self, pricedata):
        fileName = self.instrument.replace("/", "_") + "_" + self.timeframe + ".csv"                
        pricedata.to_csv(fileName)
        
    def savePriceDataFileConsolidated(self, pricedata, timeframe, timeframe_sup):
        fileName = self.instrument.replace("/", "_") + "_" + timeframe + "_" + timeframe_sup + ".csv"   
        pricedata.to_csv(fileName)

    def readData(self, instrument, timeframe):
        return pd.read_csv(instrument.replace("/", "_") + '_' + timeframe + '.csv')

    def calculate_price_median(self, df: pd.DataFrame) -> pd.DataFrame:
        # Refine the calculation to make it slightly more sensitive
        df['price_median'] = (
            df['bidclose'] * 0.35 +  # Slightly higher weight to bidclose
            df['bidopen'] * 0.25 +  # Moderate weight to bidopen
            df['bidhigh'] * 0.2 +  # Lower weight to bidhigh
            df['bidlow'] * 0.2    # Equal weight to bidhigh and bidlow
        )

        return df

    def calculate_rsi(self, df: pd.DataFrame, column: str) -> pd.Series:

        # Determine RSI window based on timeframe
        if self.timeframe in ["m1", "m5"]:
            rsi_window = 7  # Shorter window for lower timeframes
        elif self.timeframe in ["m15", "m30"]:
            rsi_window = 14  # Standard window for medium timeframes
        elif self.timeframe in ["H1", "H4"]:
            rsi_window = 21  # Longer window for higher timeframes
        else:
            rsi_window = 14  # Default fallback

        # Calculate RSI
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def apply_triggers_strategy(self, df: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
        # Constants for signal values
        SIGNAL_OPEN = 1
        SIGNAL_CLOSE = -1
        SIGNAL_NEUTRAL = 0
        
        signal_column = strategy_type
        df[signal_column] = SIGNAL_NEUTRAL
        
        open_condition = 'peaks_min' if strategy_type == 'buy' else 'peaks_max'
        close_condition = 'peaks_max' if strategy_type == 'buy' else 'peaks_min'
        
        open_signals = df[open_condition] == 1
        close_signals = df[close_condition] == 1
        
        is_position_open = False
        
        for i in range(len(df)):
            if not is_position_open and open_signals.iloc[i]:
                # Apply filters for buy signals
                if strategy_type == 'buy':
                    # Verificar clímax de volumen para compras
                    #climax_condition = (
                    #    df['volume_climax'].iloc[i] == 1 and 
                    #    df['climax_type'].iloc[i] == -1  # Clímax de venta (señal de compra)
                    #)
                    
                    #if (#df['rsi'].iloc[i] < 70 and     # RSI no sobrecomprado
                    #    df.loc[i, 'MediaPositionBuy'] == 1 and  # EMA rápida por encima de la lenta
                        #df.loc[i, 'regression_trend'] == "bullish" 
                        #(climax_condition and df['tickqty'].iloc[i] > df['volume_ma'].iloc[i])  # Volumen confirmando
                    #    climax_condition
                    
                    #    ):
                        is_position_open = True
                        df.iloc[i, df.columns.get_loc(signal_column)] = SIGNAL_OPEN
                        
                # Apply filters for sell signals
                else:
                    # Verificar clímax de volumen para ventas
                    #climax_condition = (
                    #    df['volume_climax'].iloc[i] == 1 and 
                    #    df['climax_type'].iloc[i] == 1  # Clímax de compra (señal de venta)
                    #)
                    
                    #if (df['rsi'].iloc[i] > 30 and      # RSI no sobrevendido
                    #    df['peaks_max'].iloc[i] == 1 #and
                    #    df.loc[i, 'MediaPositionSell'] == 1 and
                        #df.loc[i, 'regression_trend'] == "bearish"   # EMA rápida por debajo de la lenta
                        #(climax_condition and df['tickqty'].iloc[i] > df['volume_ma'].iloc[i])  # Volumen confirmando
                        
                    #    ):
                            is_position_open = True
                            df.iloc[i, df.columns.get_loc(signal_column)] = SIGNAL_OPEN
                        
            elif is_position_open and close_signals.iloc[i]:
                # Cerrar posición si hay clímax de volumen contrario
                if strategy_type == 'buy':
                    #if (#df['volume_climax'].iloc[i] == 1 and 
                    #    df['climax_type'].iloc[i] == 1  # Clímax de compra (señal de venta)
                        is_position_open = False
                        df.iloc[i, df.columns.get_loc(signal_column)] = SIGNAL_CLOSE
                else:
                    #if (df['volume_climax'].iloc[i] == 1 and 
                    #    df['climax_type'].iloc[i] == -1):  # Clímax de venta (señal de compra)
                        is_position_open = False
                        df.iloc[i, df.columns.get_loc(signal_column)] = SIGNAL_CLOSE
                
        return df

    def evaluate_triggers_signals(self, df):
        # Selecciona las filas de interés: de la -6 a la -2 (excluyendo las dos últimas)
        recent_rows = df.iloc[-6:-4]

        # Señales
        buy_signal = (recent_rows['buy'] == 1).any()
        sell_signal = (recent_rows['sell'] == 1).any()

        # Execute trades based on signals
        if buy_signal:
            if self.existingOperation(instrument=self.instrument, BuySell="S"):
                self.CloseOperation(instrument=self.instrument, BuySell="S")
            if not self.existingOperation(instrument=self.instrument, BuySell="B"):
                self.createEntryOrder(str_buy_sell="B")
                
        if sell_signal:
            if self.existingOperation(instrument=self.instrument, BuySell="B"):
                self.CloseOperation(instrument=self.instrument, BuySell="B")
            if not self.existingOperation(instrument=self.instrument, BuySell="S"):
                self.createEntryOrder(str_buy_sell="S")

    def calculate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        df['trend'] = 0
        df['trend_line'] = df['bidclose']  # Initialize trend_line with bidclose prices
        window_size = 7  # Increased window size for more confirmation
        
        # Calculate volume moving average for volume confirmation
        df['volume_ma'] = df['tickqty'].rolling(window=20).mean()
        
        # Calculate volatility using ATR-like measure
        df['high_low_range'] = df['bidhigh'] - df['bidlow']
        df['volatility'] = df['high_low_range'].rolling(window=14).mean()
        
        # Get indices of peaks
        peak_min_indices = df[df['peaks_min'] == 1].index
        peak_max_indices = df[df['peaks_max'] == 1].index
        
        # Combine and sort all peaks
        all_peaks = sorted(list(peak_min_indices) + list(peak_max_indices))
        
        for i in range(len(all_peaks) - window_size + 1):
            window_peaks = all_peaks[i:i + window_size]
            window_data = df.loc[window_peaks]
            
            # Calculate trend slope
            if len(window_peaks) >= 2:
                x = np.arange(len(window_peaks))
                y = window_data['bidclose'].values
                slope, _ = np.polyfit(x, y, 1)
            else:
                slope = 0
            
            # Count higher highs and higher lows for uptrend
            higher_highs = sum(1 for j in range(1, len(window_peaks)) 
                             if window_data.loc[window_peaks[j], 'bidhigh'] > 
                                window_data.loc[window_peaks[j-1], 'bidhigh'])
            
            higher_lows = sum(1 for j in range(1, len(window_peaks)) 
                            if window_data.loc[window_peaks[j], 'bidlow'] > 
                               window_data.loc[window_peaks[j-1], 'bidlow'])
            
            # Count lower highs and lower lows for downtrend
            lower_highs = sum(1 for j in range(1, len(window_peaks)) 
                            if window_data.loc[window_peaks[j], 'bidhigh'] < 
                               window_data.loc[window_peaks[j-1], 'bidhigh'])
            
            lower_lows = sum(1 for j in range(1, len(window_peaks)) 
                           if window_data.loc[window_peaks[j], 'bidlow'] < 
                              window_data.loc[window_peaks[j-1], 'bidlow'])
            
            # Volume confirmation
            volume_increasing = window_data['tickqty'].mean() > window_data['volume_ma'].mean()
            
            # Volatility check
            current_volatility = window_data['volatility'].iloc[-1]
            avg_volatility = df['volatility'].mean()
            volatility_ok = current_volatility <= avg_volatility * 1.5  # Allow 50% more volatility than average
            
            # Determine trend for this window with stricter conditions
            if (higher_highs >= 4 and higher_lows >= 4 and  # Increased required number of higher highs/lows
                slope > 0 and  # Positive slope required
                volume_increasing and  # Volume confirmation
                volatility_ok):  # Volatility check
                df.loc[window_peaks, 'trend'] = 1  # Uptrend
                # For uptrend, trend_line follows the higher lows
                df.loc[window_peaks, 'trend_line'] = df.loc[window_peaks, 'bidlow'].rolling(window=2).min()
            elif (lower_highs >= 4 and lower_lows >= 4 and  # Increased required number of lower highs/lows
                  slope < 0 and  # Negative slope required
                  volume_increasing and  # Volume confirmation
                  volatility_ok):  # Volatility check
                df.loc[window_peaks, 'trend'] = -1  # Downtrend
                # For downtrend, trend_line follows the lower highs
                df.loc[window_peaks, 'trend_line'] = df.loc[window_peaks, 'bidhigh'].rolling(window=2).max()
            else:
                df.loc[window_peaks, 'trend'] = 0  # Ranging
                # For ranging, trend_line follows the middle of the range
                df.loc[window_peaks, 'trend_line'] = (df.loc[window_peaks, 'bidhigh'] + df.loc[window_peaks, 'bidlow']) / 2
        
        # Forward fill the trend_line values to create a continuous line
        df['trend_line'] = df['trend_line'].fillna(method='ffill')
        
        return df

    def calculate_peaks(self, df: pd.DataFrame) -> pd.DataFrame:
        df['value1'] = 1
        df['peaks_min'] = df.iloc[signal.argrelextrema(df['bidclose'].values, np.less, order=30)[0]]['value1']
        df['peaks_max'] = df.iloc[signal.argrelextrema(df['bidclose'].values, np.greater, order=30)[0]]['value1']
        return df

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR) for volatility measurement."""
        high = df['bidhigh']
        low = df['bidlow']
        close = df['bidclose']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

    def detect_peaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect price peaks using ATR-based adaptive order."""
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Clean the data first
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=['bidclose', 'bidhigh', 'bidlow'])
            
            # Calculate ATR for volatility measurement
            atr = self.calculate_atr(df, period=14)
            
            # Handle any NaN values in ATR
            atr = atr.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate adaptive order based on ATR
            # Normalize ATR to price level and handle division by zero
            atr_pct = atr / df['bidclose'].replace(0, np.nan)
            atr_pct = atr_pct.fillna(0)
            
            # Convert to order size (20-40 range) with safe conversion
            peak_order = (atr_pct * 1000).clip(20, 40)
            peak_order = peak_order.fillna(30)  # Default to 30 if any NaN remains
            
            # Convert peak_order to a single integer value for the entire series
            # Use the median value to represent the overall volatility
            order_value = int(peak_order.median())
            
            # Detect peaks with the fixed order value
            min_peaks = signal.argrelextrema(df['bidclose'].values, np.less, order=order_value)
            max_peaks = signal.argrelextrema(df['bidclose'].values, np.greater, order=order_value)
            
            # Initialize peak columns with zeros
            df['peaks_min'] = 0
            df['peaks_max'] = 0
            
            # Set peaks only where we have valid indices
            if len(min_peaks[0]) > 0:
                df.loc[min_peaks[0], 'peaks_min'] = 1
            if len(max_peaks[0]) > 0:
                df.loc[max_peaks[0], 'peaks_max'] = 1
            
            return df
            
        except Exception as e:
            print(f"Error in detect_peaks: {str(e)}")
            # Return a safe fallback
            df['peaks_min'] = 0
            df['peaks_max'] = 0
            return df

    def setIndicators(self, df):
        # Calcular peaks primero
        df = self.calculate_peaks(df)
        
        # Calcular trend primero
        df = self.calculate_trend(df)

        # Calcular EMAs
        df['ema'] = df['bidclose'].ewm(span=50).mean()
        df['ema_slow'] = df['bidclose'].ewm(span=100).mean()

        df['MediaPositionSell'] = np.where( (df['ema'] < df['ema_slow']), 1, 0 )
        df['MediaPositionBuy'] = np.where( (df['ema'] > df['ema_slow']), 1, 0)

        # Calcular RSI
        df['rsi'] = self.calculate_rsi(df, 'bidclose')

        # Calcular volumen promedio móvil
        df['volume_ma'] = df['tickqty'].rolling(window=20).mean()
        
        # Calcular clímax de volumen
        df['volume_climax'] = 0
        df['climax_type'] = 0
        
        # Detectar clímax de volumen
        volume_threshold = df['volume_ma'] * 2
        price_change = df['bidclose'].pct_change()
        
        # Marcar clímax de volumen
        for i in range(1, len(df)):
            if df['tickqty'].iloc[i] > volume_threshold.iloc[i]:
                df.loc[df.index[i], 'volume_climax'] = 1
                # Determinar tipo de clímax
                if price_change.iloc[i] > 0:
                    df.loc[df.index[i], 'climax_type'] = 1  # Clímax de compra
                else:
                    df.loc[df.index[i], 'climax_type'] = -1  # Clímax de venta

        # Calcular líneas de regresión por secciones
        df = self.calculate_section_regressions(df)

        # Aplicar estrategias
        df = self.apply_triggers_strategy(df, 'buy')
        df = self.apply_triggers_strategy(df, 'sell')

        # Evaluar señales
        self.evaluate_triggers_signals(df)

        return df

    def getPriceData(self, instrument, timeframe, days, connection):
        europe_London_datetime = datetime.now(ZoneInfo('Europe/London'))  # Uso de ZoneInfo
        date_from = europe_London_datetime - dt.timedelta(days=days)
        date_to = europe_London_datetime

        history = connection.get_history(instrument, timeframe, date_from, date_to)
        current_unit, _ = connection.parse_timeframe(timeframe)
        print("Price Data Received..." + str(current_unit) + " " + str(timeframe) + " " + str(instrument) + " " + str(europe_London_datetime))

        pricedata = pd.DataFrame(history, columns=["Date", "BidOpen", "BidHigh", "BidLow", "BidClose", "Volume"])

        d = {
            'date': pricedata['Date'],
            'bidhigh': pricedata['BidHigh'],
            'bidlow': pricedata['BidLow'],
            'bidclose': pricedata['BidClose'],
            'bidopen': pricedata['BidOpen'],
            'tickqty': pricedata['Volume']
        }

        df = pd.DataFrame(data=d)
        df['timeframe'] = timeframe
        df['date'] = df['date'].astype(str).str.replace('-', '').str.replace(':', '').str.replace(' ', '').str[:-2]
        df['date'] = df['date'].apply(lambda x: int(x))
        self.pricedata = self.setIndicators(df)
        self.savePriceDataFile(self.pricedata)
        return self.pricedata


    def setMonitorPriceData(self):
        self.getPriceData(instrument=self.instrument, timeframe=self.timeframe, days=self.days,connection=self.connection)
        while True:
            currenttime = dt.datetime.now()  
            if self.timeframe == "m1" and currenttime.second == 0:
                self.getPriceData(instrument=self.instrument, timeframe=self.timeframe, days=self.days,connection=self.connection)
                time.sleep(1)                    
            elif self.timeframe == "m5" and currenttime.second == 0 and currenttime.minute % 5 == 0:
                self.getPriceData(instrument=self.instrument, timeframe=self.timeframe, days=self.days,connection=self.connection)
                time.sleep(240)
            elif self.timeframe == "m15" and currenttime.second == 0 and currenttime.minute % 15 == 0:
                self.getPriceData(instrument=self.instrument, timeframe=self.timeframe, days=self.days,connection=self.connection)
                time.sleep(840)
            elif self.timeframe == "m30" and currenttime.second == 0 and currenttime.minute % 30 == 0:
                self.getPriceData(instrument=self.instrument, timeframe=self.timeframe, days=self.days,connection=self.connection)
                time.sleep(1740)
            elif self.timeframe == "H1" and currenttime.second == 0 and currenttime.minute == 0:
                self.getPriceData(instrument=self.instrument, timeframe=self.timeframe, days=self.days,connection=self.connection)
                time.sleep(3540)
            time.sleep(1)

    def detect_volume_climax(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta clímax de volumen y sus diferentes tipos
        """
        # Calcular el volumen promedio móvil
        df['volume_ma'] = df['tickqty'].rolling(window=20).mean()
        
        # Calcular el cambio de precio
        df['price_change'] = df['bidclose'].pct_change()
        
        # Inicializar columnas para clímax
        df['volume_climax'] = 0
        df['climax_type'] = 0  # 1: compra, -1: venta
        
        # Umbral para considerar clímax (2 veces el volumen promedio)
        volume_threshold = df['volume_ma'] * 2
        
        for i in range(1, len(df)):
            # Detectar clímax de compra
            if (df['tickqty'].iloc[i] > volume_threshold.iloc[i] and 
                df['price_change'].iloc[i] > 0):
                df.loc[df.index[i], 'volume_climax'] = 1
                df.loc[df.index[i], 'climax_type'] = 1
                
            # Detectar clímax de venta
            elif (df['tickqty'].iloc[i] > volume_threshold.iloc[i] and 
                  df['price_change'].iloc[i] < 0):
                df.loc[df.index[i], 'volume_climax'] = 1
                df.loc[df.index[i], 'climax_type'] = -1
        
        return df

    def analyze_climax_implications(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza las implicaciones del clímax de volumen
        """
        # Inicializar columna para señales
        df['climax_signal'] = 0
        
        for i in range(1, len(df)):
            if df['volume_climax'].iloc[i] == 1:
                # Clímax de compra - posible señal de venta
                if df['climax_type'].iloc[i] == 1:
                    df.loc[df.index[i], 'climax_signal'] = -1
                    
                # Clímax de venta - posible señal de compra
                elif df['climax_type'].iloc[i] == -1:
                    df.loc[df.index[i], 'climax_signal'] = 1
        
        return df

    def calculate_section_regressions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula líneas de regresión para 15 secciones del DataFrame.
        Retorna un DataFrame con las líneas de regresión calculadas, la sección a la que pertenece cada punto
        y el tipo de tendencia de la regresión (alcista, bajista o neutra).
        """
        # Crear una copia del DataFrame para no modificar el original
        df = df.copy()
        
        # Calcular el tamaño de cada sección
        section_size = len(df) // 15
        
        # Inicializar columnas para las líneas de regresión
        df['regression_line_1'] = np.nan
        df['regression_line_2'] = np.nan
        df['regression_line_3'] = np.nan
        df['regression_line_4'] = np.nan
        df['regression_line_5'] = np.nan
        df['regression_line_6'] = np.nan
        df['regression_line_7'] = np.nan
        df['regression_line_8'] = np.nan
        df['regression_line_9'] = np.nan
        df['regression_line_10'] = np.nan
        df['regression_line_11'] = np.nan
        df['regression_line_12'] = np.nan
        df['regression_line_13'] = np.nan
        df['regression_line_14'] = np.nan
        df['regression_line_15'] = np.nan
        
        # Inicializar columnas para identificar la sección y la tendencia
        df['regression_section'] = 0
        df['regression_trend'] = 'neutral'  # 'bullish', 'bearish', 'neutral'
        
        # Calcular regresión para cada sección
        for i in range(15):
            start_idx = i * section_size
            end_idx = start_idx + section_size if i < 14 else len(df)
            
            # Obtener la sección actual
            section = df.iloc[start_idx:end_idx]
            
            # Calcular la regresión lineal
            x = np.arange(len(section))
            y = section['bidclose'].values
            
            # Usar polyfit para calcular la línea de regresión
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calcular los valores de la línea de regresión
            regression_line = slope * x + intercept
            
            # Asignar los valores al DataFrame original
            df.iloc[start_idx:end_idx, df.columns.get_loc(f'regression_line_{i+1}')] = regression_line
            
            # Marcar la sección a la que pertenece cada punto
            df.iloc[start_idx:end_idx, df.columns.get_loc('regression_section')] = i + 1
            
            # Calcular el rango de precios en la sección
            price_range = section['bidclose'].max() - section['bidclose'].min()
            
            # Calcular el umbral dinámico basado en el rango de precios
            # El umbral será el 0.1% del rango de precios
            slope_threshold = price_range * 0.001
            
            # Determinar la tendencia basada en la pendiente
            if abs(slope) < slope_threshold:
                trend = 'neutral'
            else:
                trend = 'bullish' if slope > 0 else 'bearish'
                
            # Asignar la tendencia a todos los puntos de la sección
            df.iloc[start_idx:end_idx, df.columns.get_loc('regression_trend')] = trend
        
        return df