import datetime as dt
import time


from ConnectionFxcm import RobotConnection
from Price import RobotPrice

class TradeMonitor:
    def __init__(self,instrument,timeframe,timeframe_sup,timeframe_sup2,days):
        self.timeframe = timeframe
        self.instrument = instrument
        self.timeframe_sup = timeframe_sup
        self.timeframe_sup2 = timeframe_sup2
        self.robotconnection = RobotConnection()
        self.connection = self.robotconnection.getConnection()
        self.corepy = self.robotconnection.getCorepy()
        self.robotPrice = RobotPrice(days, self.instrument, self.timeframe)


    def __del__(self):
        print('Object gets destroyed')
        self.connection.logout()

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



    def CloseOperation(self,instrument,BuySell):
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

    def createEntryOrder(self,str_buy_sell=None):
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
                    raise Exception(
                        "The account '{0}' is not valid".format(str_account))
                else:
                    str_account = account.account_id
                    print("AccountID='{0}'".format(str_account))

                offer = common.get_offer(self.connection, str_instrument)
                if offer is None:
                    raise Exception(
                        "The instrument '{0}' is not valid".format(str_instrument))

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
                            AMOUNT=amount,)
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
                                        AMOUNT=amount,)
                                else:
                                    request = self.connection.create_order_request(
                                        order_type=entry,
                                        OFFER_ID=offer.offer_id,
                                        ACCOUNT_ID=str_account,
                                        BUY_SELL=str_buy_sell,
                                        AMOUNT=amount,
                                        RATE_STOP=stop,
                                        RATE_LIMIT=limit,)
                self.connection.send_request_async(request)
            except Exception as e:
                print(e)



    def operationDetection(self,timeframe,timeframe_sup,timeframe_sup2): 
        time.sleep(5)       
        df = self.robotPrice.getPricesConsolidated(instrument=self.instrument, timeframe=timeframe, timeframe_sup=timeframe_sup,timeframe_sup2=timeframe_sup2)
        size = len(df.index)
        df['TriggerSell'] = 0
        df['TriggerBuy'] = 0
        #m30
        for indexTimeSup2, row in df.iterrows():
                
                if df.loc[indexTimeSup2, 'timeframe'] == timeframe_sup2 and df.loc[indexTimeSup2, 'peaks_max'] == 1: 
                    peakcount = 0
                    #m15
                    for indexTimeSup, row in  df.iloc[indexTimeSup2:].iterrows():
                        if df.loc[indexTimeSup, 'timeframe'] == timeframe_sup and df.loc[indexTimeSup, 'peaks_max'] == 1:

                            #m5
                            for indexTimeInf, row in df.iloc[indexTimeSup:].iterrows():
                                    if df.loc[indexTimeInf, 'timeframe'] == timeframe and df.loc[indexTimeInf, 'peaks_max'] == 1 :
                                        peakcount = peakcount + 1
                                        if peakcount == 1:    
                                                df.loc[indexTimeInf, 'TriggerSell'] = 1
                                                break 


                if df.loc[indexTimeSup2, 'timeframe'] == timeframe_sup2 and df.loc[indexTimeSup2, 'peaks_min'] == 1:                                  
                    peakcount = 0
                    #m15
                    for indexTimeSup, row in  df.iloc[indexTimeSup2:].iterrows():
                            if df.loc[indexTimeSup, 'timeframe'] == timeframe_sup and df.loc[indexTimeSup, 'peaks_min'] == 1 :
                               
                                #m5
                                for indexTimeInf, row in df.iloc[indexTimeSup:].iterrows():  
                                    if df.loc[indexTimeInf, 'timeframe'] == timeframe and df.loc[indexTimeInf, 'peaks_min'] == 1 :
                                        peakcount = peakcount + 1
                                        if peakcount == 1:
                                                df.loc[indexTimeInf, 'TriggerBuy'] = 1
                                                break 
        









        #Open Operation


        TriggerSell = False
        TriggerBuy = False
        for index, row in df.tail(6).iterrows():
             if df.loc[index, 'TriggerSell']  == 1:
                          TriggerSell = True
                          df.loc[index, 'TriggerSell'].to_csv("selloperacion.csv")
             if df.loc[index, 'TriggerBuy']  == 1:
                          TriggerBuy = True
                          df.loc[index, 'TriggerBuy'].to_csv("buyperacion.csv")
             

        if TriggerSell == True:
            if self.existingOperation(instrument=self.instrument, BuySell= "B"):
                self.CloseOperation(instrument=self.instrument,BuySell = "B")
        
            print("	  SELL OPERATION! ")
            if self.existingOperation(instrument=self.instrument, BuySell= "S")  != True:
                self.createEntryOrder(str_buy_sell="S")


        if TriggerBuy == True:
            if self.existingOperation(instrument=self.instrument, BuySell= "S"):
                self.CloseOperation(instrument=self.instrument,BuySell = "S")
        
            print("	  BUY OPERATION! ")
            if self.existingOperation(instrument=self.instrument, BuySell= "B")  != True:
                self.createEntryOrder(str_buy_sell="B")


        self.robotPrice.savePriceDataFileConsolidated(pricedata=df, timeframe=timeframe, timeframe_sup=timeframe_sup)


    def startMonitor(self):    
        self.operationDetection( timeframe=self.timeframe, timeframe_sup=self.timeframe_sup,timeframe_sup2=self.timeframe_sup2)        
        while True:            
            currenttime = dt.datetime.now()  
            if self.timeframe == "m1" and currenttime.second == 0:
                self.operationDetection( timeframe=self.timeframe, timeframe_sup=self.timeframe_sup,timeframe_sup2=self.timeframe_sup2)    
                time.sleep(1)                    
            elif self.timeframe == "m5" and currenttime.second == 0 and currenttime.minute % 5 == 0:
                self.operationDetection( timeframe=self.timeframe, timeframe_sup=self.timeframe_sup,timeframe_sup2=self.timeframe_sup2)
                time.sleep(240)
            elif self.timeframe == "m15" and currenttime.second == 0 and currenttime.minute % 15 == 0:

                self.operationDetection( timeframe=self.timeframe, timeframe_sup=self.timeframe_sup,timeframe_sup2=self.timeframe_sup2)
                time.sleep(840)
            elif self.timeframe == "m30" and currenttime.second == 0 and currenttime.minute % 30 == 0:

                self.operationDetection( timeframe=self.timeframe, timeframe_sup=self.timeframe_sup,timeframe_sup2=self.timeframe_sup2)
                time.sleep(1740)
            elif self.timeframe == "H1" and currenttime.second == 0 and currenttime.minute == 0:
                self.operationDetection( timeframe=self.timeframe, timeframe_sup=self.timeframe_sup,timeframe_sup2=self.timeframe_sup2)
                time.sleep(3540)
            time.sleep(1)