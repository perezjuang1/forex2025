import datetime as dt
import os
import time
import numpy as np
import pandas as pd
import pyti.stochastic as sto
from pyti.relative_strength_index import relative_strength_index as rsi
from scipy import signal
from time import sleep
from forexconnect import fxcorepy, ForexConnect, Common
from OrdersMonitor import OrdersMonitor
from threading import Event
import ClosePosition as positions

from ConfigurationOperation import ConfigurationOperation
ArgsConf = ConfigurationOperation()

fileName = str(os.path.basename(__file__))
fileName = fileName.replace(".py", "")
fileName = fileName.replace("Real_Time_", "")
symbol = fileName.replace("_", "/")
file = symbol.replace("/", "_") + "_" + ArgsConf.str_timeframe + ".csv"

DOWNWARD_TREND = 'DOWNWARD-TREND'
UPWARD_TREND = 'UPWARD-TREND'


def createOperation(args):
    
    with ForexConnect() as fx:
        fx.login(args.l, args.p, args.u, args.c, args.session,  args.pin, None)
        try:
            account = Common.get_account(fx, args.account)
            if not account:
                raise Exception(
                    "The account '{0}' is not valid".format(args.account))

            else:
                args.account = account.account_id
                print("AccountID='{0}'".format(args.account))

            offer = Common.get_offer(fx, args.i)

            if offer is None:
                raise Exception(
                    "The instrument '{0}' is not valid".format(args.i))


            login_rules = fx.login_rules

            trading_settings_provider = login_rules.trading_settings_provider

            base_unit_size = trading_settings_provider.get_base_unit_size(args.i, account)

            amount = base_unit_size * args.lots

            entry = fxcorepy.Constants.Orders.TRUE_MARKET_OPEN

            request = fx.create_order_request(
                                order_type=entry,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=args.account,
                                BUY_SELL=args.d,
                                AMOUNT=amount,
                                #RATE_STOP=stop,
                                #RATE_LIMIT=limit,
                            )
            orders_monitor = OrdersMonitor()
            orders_table = fx.get_table(ForexConnect.ORDERS)
            orders_listener = Common.subscribe_table_updates(orders_table, on_add_callback=orders_monitor.on_added_order)

            try:
                resp = fx.send_request(request)
                order_id = resp.order_id

            except Exception as e:
                print(e)
                orders_listener.unsubscribe()

            else:
                # Waiting for an order to appear or timeout (default 30)
                order_row = orders_monitor.wait(30, order_id)
                if order_row is None:
                    print("Response waiting timeout expired.\n")
                else:
                    print("The order has been added. OrderID={0:s}, "
                          "Type={1:s}, BuySell={2:s}, Rate={3:.5f}, TimeInForce={4:s}".format(
                        order_row.order_id, order_row.type, order_row.buy_sell, order_row.rate,
                        order_row.time_in_force))
                orders_listener.unsubscribe()
        except Exception as e:
            print(e)


def getLatestPriceData(Conf):
    with ForexConnect() as fx:
        try:
            fx.login(Conf.l, Conf.p, Conf.u, Conf.c, Conf.session, Conf.pin, None)
            print("Accounts:")
            accounts_response_reader = fx.get_table_reader(fx.ACCOUNTS)
            for account in accounts_response_reader:
                print("{0:s}".format(account.account_id))
                Conf.account = account.account_id
            print("Requesting Price Data ")
            history = fx.get_history(Conf.i, Conf.str_timeframe, Conf.date_from, Conf.date_to, Conf.quotes_count)
            current_unit, _ = ForexConnect.parse_timeframe(Conf.str_timeframe)
            result = type(history)
            print(result)
            print("Price Data Received...")
            print(current_unit)
            return pd.DataFrame(history, columns = ["Date", "BidOpen", "BidHigh", "BidLow", "BidClose", "Volume"]) , Conf
        except Exception as e:
            print(e)
            fx.logout()



def estimate_b0_b1(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    Sumatoria_xy = np.sum((x-m_x)*(y-m_y))
    Sumatoria_xx = np.sum(x*(x-m_x))
    b_1 = Sumatoria_xy / Sumatoria_xx
    b_0 = m_y - b_1*m_x
    return(b_0, b_1)



def StrategyHeartBeat():
    print("Starting.....") 
    Args = ConfigurationOperation()
    pricedata, ArgsUPDATED = getLatestPriceData( Args )
    Update(pricedata, ArgsUPDATED)
    #validateTendencyInf(Args)
    while True:
        Args = ConfigurationOperation()
        currenttime = dt.datetime.now()
        if Args.str_timeframe == "m1" and currenttime.second == 0:
            pricedata, Args = getLatestPriceData( Args )
            Update(pricedata,Args)
            time.sleep(1)
        elif Args.str_timeframe == "m5" and currenttime.second == 0 and currenttime.minute % 5 == 0:
            pricedata, Args = getLatestPriceData( Args )
            Update(pricedata,Args)
            time.sleep(240)
        elif Args.str_timeframe == "m15" and currenttime.second == 0 and currenttime.minute % 15 == 0:
            pricedata, Args = getLatestPriceData( Args )
            Update(pricedata,Args)
            time.sleep(840)
        elif Args.str_timeframe == "m30" and currenttime.second == 0 and currenttime.minute % 30 == 0:
            pricedata, Args = getLatestPriceData( Args )
            Update(pricedata,Args)
            time.sleep(1740)
        elif Args.str_timeframe == "H1" and currenttime.second == 0 and currenttime.minute == 0:
            pricedata, Args = getLatestPriceData( Args )
            Update(pricedata,Args)
            time.sleep(3540)
        time.sleep(1)




def Update(pricedata, args):
    print(str(dt.datetime.now()) + " " + args.str_timeframe + " Bar Closed " + symbol)

    d = {'bidhigh': pricedata['BidHigh'],
         'bidlow': pricedata['BidLow'],
         'bidclose': pricedata['BidClose'],
         'bidopen': pricedata['BidOpen'], 
         'tickqty': pricedata['Volume']         
         }
    
    df = pd.DataFrame(data=d)
    df = df.assign(row_count=range(len(df)))
    df.index = df['row_count'].values

    # HMA fast and slow calculation
    df['ema'] = df['bidclose'].ewm(span=100).mean()
    df['ema_slow'] = df['bidclose'].ewm(span=100).mean()
    df['ema_res1'] = df['bidclose'].ewm(span=100).mean()
    df['ema_res2'] = df['bidclose'].ewm(span=100).mean()
    df['ema_res3'] = df['bidclose'].ewm(span=100).mean()

    df['rsi'] = rsi(df['bidclose'], 15)
    df['sto_k'] = sto.percent_k(df['bidclose'], 10)
    df['sto_d'] = sto.percent_d(df['bidclose'], 10)
    df['sto_k'] = df['sto_k'].ewm(span=10).mean()
    df['sto_d'] = df['sto_d'].ewm(span=10).mean()

    # Medias Strategy
    #Sell
    df['MediaSell'] = np.where( (df['ema_slow'] < df['ema_res1']), 1, 0 )
    df['MediaBuy'] = np.where( (df['ema_slow'] > df['ema_res1']), 1, 0)
    df['MediaTriggerSell'] = df['MediaSell'].diff()
    df['MediaTriggerBuy'] = df['MediaBuy'].diff()


    df['value1'] = 1
    # Find local peaks
    df['peaks_min'] = df.iloc[signal.argrelextrema(df['bidclose'].values,np.less_equal, order=10)[0]]['value1']
    df['peaks_max'] = df.iloc[signal.argrelextrema(df['bidclose'].values,np.greater_equal,order=10)[0]]['value1']


    #Find Tendencies Pics Min
    trend = "NOT FINDED"
    for index, row in df.iterrows():        
        if df.loc[index, 'peaks_min'] == 1:
            iRV = index
            while (iRV > 1):
                iRV = iRV - 1
                if df.loc[iRV, 'peaks_min'] == 1:
                    if df.loc[index, 'bidclose'] == df.loc[iRV, 'bidclose']:
                        trend = trend
                    if df.loc[index, 'bidclose'] < df.loc[iRV, 'bidclose']:
                       trend = DOWNWARD_TREND
                       iRV = 0
                    else:
                        trend = UPWARD_TREND
                        iRV = 0
        df.loc[index, 'trend_min'] = trend

    #Find Tendencies Pics Min
    trend = "NOT FINDED"
    for index, row in df.iterrows():        
        if df.loc[index, 'peaks_max'] == 1:
            iRV = index
            while (iRV > 1):
                iRV = iRV - 1
                if df.loc[iRV, 'peaks_max'] == 1:
                    if df.loc[index, 'bidclose'] == df.loc[iRV, 'bidclose']:
                        trend = trend
                    if df.loc[index, 'bidclose'] < df.loc[iRV, 'bidclose']:
                       trend = DOWNWARD_TREND
                       iRV = 0
                    else:
                        trend = UPWARD_TREND
                        iRV = 0
        df.loc[index, 'trend_max'] = trend




    #Find Difference in PIPS AND VOLUMEN
    for index, row in df.iterrows():        
        if df.loc[index, 'peaks_max'] == 1:
            iRV = index
            while (iRV > 1):
                iRV = iRV - 1
                if df.loc[iRV, 'peaks_min'] == 1:
                      df.loc[index, 'volumenPipsDiference'] =  ( abs ( df.loc[index, 'bidclose'] - df.loc[iRV, 'bidclose'] ) )
                      iRV = 1

    for index, row in df.iterrows():        
        if df.loc[index, 'peaks_min'] == 1:
            iRV = index
            while (iRV > 1):
                iRV = iRV - 1
                if df.loc[iRV, 'peaks_max'] == 1:
                      df.loc[index, 'volumenPipsDiference'] = ( abs ( df.loc[index, 'bidclose'] - df.loc[iRV, 'bidclose'] ) )
                      iRV = 1



    df['volumenPipsDiference'] = df['volumenPipsDiference'].fillna(method="ffill")
    df['volumLimitOperation'] = 0.0006
    df['volumEnableOperation'] = np.where( (df['volumenPipsDiference'] >= df['volumLimitOperation']) , 1, 0)

    #Find Price after pic if in correct position
    df['priceInPositionSell'] = 0
    df['priceInPositionBuy'] = 0
    for index, row in df.iterrows():
        try:   #and df.loc[index + 5, 'bidclose'] < df.loc[index + 5, 'ema']
               #  and df.loc[index + 5, 'bidclose'] > df.loc[index + 5, 'ema']   and 
             if ( df.loc[index, 'peaks_max'] == 1 
                 #and df.loc[index, 'bidclose'] < df.loc[index, 'ema_res1']
                 #and df.loc[index, 'volumEnableOperation'] == 1 
                 and df.loc[index, 'trend_max'] == DOWNWARD_TREND 
                 #and df.loc[index, 'trend_min'] == DOWNWARD_TREND
                 #and df.loc[index, 'ema'] < df.loc[index, 'ema_slow']
                 #and df.loc[index, 'bidclose'] < df.loc[index, 'ema']
                 ):  
                    df.loc[index, 'priceInPositionSell'] = 1
             elif (  df.loc[index, 'peaks_min'] == 1
                 #  and df.loc[index, 'bidclose'] > df.loc[index, 'ema_res1'] 
                   #and df.loc[index, 'volumEnableOperation'] == 1 
                  # and df.loc[index, 'trend_max'] == UPWARD_TREND 
                  and df.loc[index, 'trend_min'] == UPWARD_TREND
                 #and df.loc[index, 'ema'] > df.loc[index, 'ema_slow']
                 #and df.loc[index, 'bidclose'] > df.loc[index, 'ema']
                 ): 
                    df.loc[index, 'priceInPositionBuy'] = 1
        except:
            print("peaks: In Validation")


    df['sell'] = np.where( (df['priceInPositionSell'] == 1) , 1, 0)
    
    # Close Strategy Operation Sell
    operationActive = False
    for index, row in df.iterrows():
        if df.loc[index, 'sell'] == 1:
            operationActive = True
        if operationActive == True:
            df.loc[index, 'sell'] = 1
        if df.loc[index, 'peaks_min'] == 1:
            operationActive = False



    df['zone_sell'] = df['sell'].diff()

    if df['zone_sell'][len(df) - 7] == -1:
            positions.CloseOperation(args)


    if df['zone_sell'][len(df) - 7] == 1:
        args.d = "S"
        createOperation(args)

    # ***********************************************************
    # * Estrategy  BUY
    # ***********************************************************
    df['buy'] = np.where( (df['priceInPositionBuy'] == 1) , 1, 0)


    # Close Strategy Operation Sell
    operationActive = False
    for index, row in df.iterrows():
        if df.loc[index, 'buy'] == 1:
            operationActive = True
        if operationActive == True:
            df.loc[index, 'buy'] = 1
        if (df.loc[index, 'peaks_max'] == 1 and df.loc[index, 'trend_max'] == UPWARD_TREND):
           operationActive = False



    df['zone_buy'] = df['buy'].diff()

    if df['zone_buy'][len(df) - 7] == -1:
        positions.CloseOperation(args)


    if df['zone_buy'][len(df) - 4] == 1:
        print("	  BUY SIGNAL! ")
        args.d = "B"
        createOperation(args)

    df.to_csv(args.filename)
    print(str(dt.datetime.now()) + " " + args.str_timeframe +  " Update Function Completed. " + symbol + "\n")
    #args.d = "S"
    #createOperation(args)


StrategyHeartBeat()  # Run strategy
