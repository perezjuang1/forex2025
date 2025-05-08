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
        
    @staticmethod
    def readPriceDataFileConsolidated(self, timeframe, timeframe_sup):
        return pd.read_csv(self.instrument.replace("/", "_") + '_' + timeframe + "_" + timeframe_sup + ".csv")

    def getPricesConsolidated(self, instrument, timeframe, timeframe_sup, timeframe_sup2):
        pricedata_inf = self.readData(instrument, timeframe)
        pricedata_sup = self.readData(instrument, timeframe_sup)        
        pricedata_sup2 = self.readData(instrument, timeframe_sup2)
        pricedata = pd.concat([pricedata_sup2 , pricedata_sup, pricedata_inf], ignore_index=True)
        pricedata = pricedata.sort_values(by='date').reset_index(drop=True)
        return pricedata     
        
    def calculate_linear_regression(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Calculate linear regression for a given column in the DataFrame."""
        if len(df) > 1:  # Ensure there are enough data points
            x = np.arange(len(df))  # Use index as x-axis
            y = df[column]
            coeffs = np.polyfit(x, y, 1)  # Linear regression (degree 1)
            return coeffs[0] * x + coeffs[1]  # y = mx + b
        return pd.Series(np.nan, index=df.index)  # Return NaN if not enough data

    def apply_sell_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the sell strategy to the DataFrame, considering the 100-period EMA and deviation zones."""
        df['sell'] = 0
        operationActive = False
        for index, row in df.iterrows():
            # Only consider sell operations below the 100-period EMA and near a deviation zone
            if not operationActive and df.loc[index, 'peaks_max'] == 1 and df.loc[index, 'deviation_zone'] == 1:
                operationActive = True
                df.loc[index, 'sell'] = 1  # Open sell operation
            # Only consider buy operations above the 100-period EMA and near a deviation zone
            elif operationActive and df.loc[index, 'peaks_min'] == 1:
                df.loc[index, 'sell'] = -1  # Close sell operation
                operationActive = False
        return df

    def apply_buy_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the buy strategy to the DataFrame, considering the 100-period EMA and deviation zones."""
        df['buy'] = 0
        operationActive = False
        for index, row in df.iterrows():
            # Only consider buy operations above the 100-period EMA and near a deviation zone
            if not operationActive and df.loc[index, 'peaks_min'] == 1 and df.loc[index, 'deviation_zone'] == 1:
                operationActive = True
                df.loc[index, 'buy'] = 1  # Open buy operation
            # Only consider sell operations below the 100-period EMA and near a deviation zone
            elif operationActive and df.loc[index, 'peaks_max'] == 1 :
                df.loc[index, 'buy'] = -1  # Close buy operation
                operationActive = False
        return df

    def calculate_rsi(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Calculate the RSI for a given column in the DataFrame."""
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

    def calculate_price_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate a slightly more sensitive median price by refining the calculation."""
        required_columns = ['bidclose', 'bidopen', 'bidhigh', 'bidlow']

        # Ensure all required columns are present
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

        # Refine the calculation to make it slightly more sensitive
        df['price_median'] = (
            df['bidclose'] * 0.35 +  # Slightly higher weight to bidclose
            df['bidopen'] * 0.25 +  # Moderate weight to bidopen
            df['bidhigh'] * 0.2 +  # Lower weight to bidhigh
            df['bidlow'] * 0.2    # Equal weight to bidhigh and bidlow
        )

        return df

    def identify_price_deviation_zones(self, df: pd.DataFrame, threshold: float = 0.0005) -> pd.DataFrame:
        """Identify zones where the price deviates significantly from its median and mark them."""
        df['deviation_zone'] = 0

        # Adjust the threshold to make detection more sensitive
        df['deviation_zone'] = (
            (df['bidclose'] - df['price_median']).abs() > threshold
        ).astype(int)

        # Add a condition to detect zones where the price is consistently above or below the median
        df['deviation_zone'] |= (
            (df['bidclose'] > df['price_median'] * 1.0002) | (df['bidclose'] < df['price_median'] * 0.9998)
        ).astype(int)

        return df

    def evaluate_trading_signals(self, df):
        """Evaluate trading signals and execute operations based on peaks and deviation zones."""
        triggers = {
            "S": any((df.loc[index, 'peaks_max'] == 1 and df.loc[index, 'deviation_zone'] == 1) for index in df.tail(4).index),
            "B": any((df.loc[index, 'peaks_min'] == 1 and df.loc[index, 'deviation_zone'] == 1) for index in df.tail(4).index),
        }

        for buy_sell, trigger in triggers.items():
            if trigger:
                opposite = "B" if buy_sell == "S" else "S"
                if self.existingOperation(instrument=self.instrument, BuySell=opposite):
                    self.CloseOperation(instrument=self.instrument, BuySell=opposite)
                print(f"{'SELL' if buy_sell == 'S' else 'BUY'} OPERATION!")
                if not self.existingOperation(instrument=self.instrument, BuySell=buy_sell):
                    self.createEntryOrder(str_buy_sell=buy_sell)

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
        df['value1'] = 1
        # Find local peaks
        df['ema'] = df['bidclose'].ewm(span=3).mean()
        df['ema_slow'] = df['bidclose'].ewm(span=30).mean()
        df['ema_100'] = df['bidclose'].ewm(span=80).mean()  # Add 100-period EMA

        # Use new peak detection method
        df = self.detect_peaks(df)

        # Add RSI indicator
        df['rsi'] = self.calculate_rsi(df, 'bidclose')

        # Add linear regression for bidclose prices
        df['price_regression'] = self.calculate_linear_regression(df, 'bidclose')
                
        # Calculate median price
        df = self.calculate_price_median(df)
                
        # Identify price deviation zones
        df = self.identify_price_deviation_zones(df)
                
        # Apply sell and buy strategies
        df = self.apply_sell_strategy(df)
        df = self.apply_buy_strategy(df)

        # Evaluate trading signals after applying strategies
        self.evaluate_trading_signals(df)

        return df

    def getPriceData(self, instrument, timeframe, days, connection):
        europe_London_datetime = datetime.now(ZoneInfo('Europe/London'))  # Uso de ZoneInfo
        date_from = europe_London_datetime - dt.timedelta(days=days)
        date_to = europe_London_datetime
        try:
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
        except Exception as e:
            print("Exception: " + str(e))

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