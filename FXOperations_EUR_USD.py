import os
import time
import numpy as np
import pandas as pd
import pyti.stochastic as sto
from pyti.relative_strength_index import relative_strength_index as rsi
from scipy import signal
from time import sleep
from forexconnect import fxcorepy, ForexConnect, Common
import os
from forexconnect import fxcorepy, ForexConnect
from ConfigurationOperation import ConfigurationOperation
import datetime as dt
from forexconnect import fxcorepy, ForexConnect, Common
import Probabilidades.RegrsionLineal2 as regresionlineal2

args = ConfigurationOperation()


fileName = str(os.path.basename(__file__))
fileName = fileName.replace(".py", "")
fileName = fileName.replace("FXOperations_", "")
instrument_symbol = fileName.replace("_", "/")
file = instrument_symbol.replace("/", "_") + "_" + args.timeframe + ".csv"

DOWNWARD_TREND = 'DOWNWARD-TREND'
UPWARD_TREND = 'UPWARD-TREND'

#======================================================
# FXM Functions


def session_status_changed(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
    print("Trading session status: " + str(status))




def CloseOperation(instrument,BuySell):
    with ForexConnect() as fx:
        try:
            fx.login(args.userid, args.password, args.url, session_status_callback=session_status_changed)
            accounts_response_reader = fx.get_table_reader(fx.ACCOUNTS)
            accountId = None
            for account in accounts_response_reader:
                accountId = account.account_id
            print(accountId)            
            orders_table = fx.get_table(ForexConnect.TRADES)
            for trade in orders_table:
                if trade.instrument == instrument and trade.buy_sell == BuySell:   
                    buy_sell = fxcorepy.Constants.SELL if trade.buy_sell == fxcorepy.Constants.BUY else fxcorepy.Constants.BUY                
                    print('Closing ' + str(buy_sell))
                    if buy_sell != None:
                        request = fx.create_order_request(
                            order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                            OFFER_ID=trade.offer_id,
                            ACCOUNT_ID=accountId,
                            BUY_SELL=buy_sell,
                            AMOUNT=trade.amount,
                            TRADE_ID=trade.trade_id
                            )
                        fx.send_request(request)
                        print('Request Sended')
                    print('Closed ' + str(buy_sell))
                else:
                     print('Trade is not the same ' + str(buy_sell) + ' != ' + str(trade.BuySell))
            fx.logout()
        except Exception as e:
            print("Exception: " + str(e))


def existingOperation(instrument, BuySell):
    existOperation = False
    with ForexConnect() as fx:
        try:
            fx.login(args.userid, args.password, args.url, session_status_callback=session_status_changed)
            trades_table = fx.get_table(ForexConnect.TRADES)            
            for trade_row in trades_table:
                    if BuySell == trade_row.BuySell and trade_row.instrument == instrument:                       
                        print("Existing Trade: " + str(trade_row.TradeID))
                        existOperation = True                  
            fx.logout()
            return existOperation
        except Exception as e:
            print("Exception: " + str(e))
            return existOperation


def getLatestPriceData():
    global args
    with ForexConnect() as fx:
        try:
            fx.login(args.userid, args.password, args.url, session_status_callback=session_status_changed)
            args = ConfigurationOperation()
            history = fx.get_history(instrument_symbol, args.timeframe, args.date_from, args.date_to)
            current_unit, _ = ForexConnect.parse_timeframe(args.timeframe)
            print("Price Data Received...")
            print(current_unit)
            fx.logout()
            return pd.DataFrame(history, columns = ["Date", "BidOpen", "BidHigh", "BidLow", "BidClose", "Volume"])
        except Exception as e:
            print("Exception: " + str(e))


def createEntryOrder(str_instrument, str_buy_sell=None):
    str_user_id = args.userid
    str_password = args.password
    str_url = args.url
    str_connection = args.connectiontype
    str_session_id = args.session
    str_pin = args.pin
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

    with ForexConnect() as fx:
        fx.login(str_user_id, str_password, str_url, str_connection, str_session_id,
                 str_pin, session_status_changed)
        
        try:
            account = Common.get_account(fx, str_account)
            if not account:
                raise Exception(
                    "The account '{0}' is not valid".format(str_account))
            else:
                str_account = account.account_id
                print("AccountID='{0}'".format(str_account))

            offer = Common.get_offer(fx, str_instrument)
            if offer is None:
                raise Exception(
                    "The instrument '{0}' is not valid".format(str_instrument))

            login_rules = fx.login_rules
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
                    request = fx.create_order_request(
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
                    request = fx.create_order_request(
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
                                request = fx.create_order_request(
                                    order_type=entry,
                                    OFFER_ID=offer.offer_id,
                                    ACCOUNT_ID=str_account,
                                    BUY_SELL=str_buy_sell,
                                    RATE_STOP=stop,
                                    PEG_TYPE_LIMIT=peglimittype,
                                    PEG_OFFSET_LIMIT=limitv,
                                    AMOUNT=amount,)
                            else:
                                request = fx.create_order_request(
                                    order_type=entry,
                                    OFFER_ID=offer.offer_id,
                                    ACCOUNT_ID=str_account,
                                    BUY_SELL=str_buy_sell,
                                    AMOUNT=amount,
                                    RATE_STOP=stop,
                                    RATE_LIMIT=limit,)
            fx.send_request(request)
            fx.logout()
        except Exception as e:
            print(e)


#===========================================================
# Trading Logic

def estimate_b0_b1(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    Sumatoria_xy = np.sum((x-m_x)*(y-m_y))
    Sumatoria_xx = np.sum(x*(x-m_x))
    b_1 = Sumatoria_xy / Sumatoria_xx
    b_0 = m_y - b_1*m_x
    return(b_0, b_1)


#===========================================================
# MAIN Logic
def StrategyHeartBeat():
    print("Starting.....") 
    pricedata = getLatestPriceData()
    Update(pricedata)
    print("Started..... Estrategy " + str(args.timeframe)) 
    while True:
        currenttime = dt.datetime.now()
        if args.timeframe == "m1" and currenttime.second == 0:
            pricedata = getLatestPriceData( )
            Update(pricedata)
            time.sleep(1)
        elif args.timeframe == "m5" and currenttime.second == 0 and currenttime.minute % 5 == 0:
            pricedata = getLatestPriceData( )
            Update(pricedata)
            time.sleep(240)
        elif args.timeframe == "m15" and currenttime.second == 0 and currenttime.minute % 15 == 0:
            pricedata = getLatestPriceData( )
            Update(pricedata)
            time.sleep(840)
        elif args.timeframe == "m30" and currenttime.second == 0 and currenttime.minute % 30 == 0:
            pricedata = getLatestPriceData( )
            Update(pricedata)
            time.sleep(1740)
        elif args.timeframe == "H1" and currenttime.second == 0 and currenttime.minute == 0:
            pricedata = getLatestPriceData( )
            Update(pricedata)
            time.sleep(3540)
        time.sleep(1)




def Update(pricedata):
    print("===============================================")
    print("===========       Start       =================")
    print("===============================================")
    print(str(dt.datetime.now()) + " TimeFrame: " + args.timeframe + " Instrument:  " + instrument_symbol)

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
    df['ema'] = df['bidclose'].ewm(span=30).mean()
    df['ema_slow'] = df['bidclose'].ewm(span=30).mean()
    df['ema_res1'] = df['bidclose'].ewm(span=30).mean()
    df['ema_res2'] = df['bidclose'].ewm(span=30).mean()
    df['ema_res3'] = df['bidclose'].ewm(span=30).mean()

    df['rsi'] = rsi(df['bidclose'], 14)
    df['sto_k'] = sto.percent_k(df['bidclose'], 10)
    df['sto_d'] = sto.percent_d(df['bidclose'], 10)
    df['sto_k'] = df['sto_k'].ewm(span=10).mean()
    df['sto_d'] = df['sto_d'].ewm(span=10).mean()

    # Medias Strategy
    #Sell
    df['MediaSell'] = np.where( (df['ema'] < df['ema_res1']), 1, 0 )
    df['MediaBuy'] = np.where( (df['ema'] > df['ema_res1']), 1, 0)
    df['MediaTriggerSell'] = df['MediaSell'].diff()
    df['MediaTriggerBuy'] = df['MediaBuy'].diff()


    df['value1'] = 1
    # Find local peaks
    df['peaks_min'] = df.iloc[signal.argrelextrema(df['ema_res1'].values,np.less,order=30)[0]]['value1']
    df['peaks_max'] = df.iloc[signal.argrelextrema(df['ema_res1'].values,np.greater,order=30)[0]]['value1']


    # ***********************************************************
    # *  Regresion al precio de cierre las velas ================
    # ***********************************************************
    df['x'] = np.arange(len(df['bidclose']))

    # ************* Calcular la poscion Relativa Y
    for index, row in df.iterrows():
        df.loc[index, 'y'] = int('{:.5f}'.format((df.loc[index, 'bidclose'])).replace('.', ''))
        max_value = max(np.array(df['y'].values))
        min_value = min(np.array(df['y'].values))

    for index, row in df.iterrows():
        value = df.loc[index, 'y'] - min_value
        NewPricePosition = ((value * 100) / max_value) * 100
        df.loc[index, 'y'] = NewPricePosition

    # ***********  Calcular la poscion Relativa X
    max_value = max(np.array(df['x'].values))
    min_value = min(np.array(df['x'].values))
    for index, row in df.iterrows():
        value = df.loc[index, 'x'] - min_value
        NewPricePosition = ((value * 100) / max_value)
        df.loc[index, 'x'] = NewPricePosition

    regresionLineal_xx = np.array(df['x'].values)
    regresionLineal_yy = np.array(df['y'].values)

    regresionLineal_bb = regresionlineal2.estimate_b0_b1(regresionLineal_xx, regresionLineal_yy)
    y_pred_sup = regresionLineal_bb[0] + regresionLineal_bb[1] * regresionLineal_xx
    df['y_pred'] = y_pred_sup

    if df.iloc[len(df) - 1]['y_pred'] < \
            df.iloc[1]['y_pred'] and \
            df.iloc[len(df) - 1]['y_pred'] < \
            df.iloc[1]['y_pred']:
        lv_Tendency = "Bajista"
    elif df.iloc[len(df) - 1]['y_pred'] > \
            df.iloc[1]['y_pred'] and \
            df.iloc[len(df) - 1]['y_pred'] > \
            df.iloc[1]['y_pred']:
        lv_Tendency = "Alcista"





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




    #Find Tendencies Pics Max
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



    df['volumenPipsDiference'] = 1#df['volumenPipsDiference'].fillna(method="ffill")
    df['volumLimitOperation'] = 0.0006
    df['volumEnableOperation'] = 1#np.where( (df['volumenPipsDiference'] >= df['volumLimitOperation']) , 1, 0)



    #Find Price after pic if in correct position
    df['priceInPositionSell'] = 0
    df['priceInPositionBuy'] = 0
    for index, row in df.iterrows():
        try:   #and df.loc[index + 5, 'bidclose'] < df.loc[index + 5, 'ema']
               #  and df.loc[index + 5, 'bidclose'] > df.loc[index + 5, 'ema']   and 
             if ( #df.loc[index, 'MediaTriggerSell'] == 1 #and                 
                 df.loc[index, 'peaks_max'] == 1
                 #and df.loc[index, 'bidclose'] < df.loc[index, 'ema_res1']
                 #and df.loc[index, 'volumEnableOperation'] == 1 
                 and df.loc[index, 'trend_max'] == DOWNWARD_TREND 
                 #and df.loc[index, 'trend_min'] == DOWNWARD_TREND
                 #and df.loc[index, 'ema'] < df.loc[index, 'ema_slow']
                 #and df.loc[index, 'rsi'] > 50
                 #and df.loc[index, 'bidclose'] < df.loc[index, 'ema']
                 ):  
                            #iRV = 0
                            #while (iRV <= 5):
                                #iRV = iRV + 1
                                #poscheck = index - iRV
                                #if df.loc[poscheck, 'peaks_max'] == 1:
                                    df.loc[index, 'priceInPositionSell'] = 1
             elif (  #df.loc[index, 'MediaTriggerBuy'] == 1 #and                  
                     df.loc[index, 'peaks_min'] == 1
                   # and  df.loc[index, 'bidclose'] > df.loc[index, 'ema_res1'] 
                   #and df.loc[index, 'volumEnableOperation'] == 1 
                  #and df.loc[index, 'trend_max'] == UPWARD_TREND 
                  and df.loc[index, 'trend_min'] == UPWARD_TREND
                  #and df.loc[index, 'ema'] > df.loc[index, 'ema_slow']
                  #and df.loc[index, 'rsi'] < 50
                 #and df.loc[index, 'bidclose'] > df.loc[index, 'ema']
                 ): 
                        #iRV = 0
                        #while (iRV <= 5):
                        #    iRV = iRV + 1
                        #    poscheck = index - iRV
                        #    if df.loc[poscheck, 'peaks_min'] == 1:
                                df.loc[index, 'priceInPositionBuy'] = 1
        except:
            print("peaks: In Validation")


    df['sell'] = np.where( (df['priceInPositionSell'] == 1) , 1, 0)
    df['buy'] = np.where( (df['priceInPositionBuy'] == 1) , 1, 0)
    # ***********************************************************
    # * Closs Operation First
    # ***********************************************************
    # Close Strategy Operation Sell
    operationActive = False
    for index, row in df.iterrows():
        if df.loc[index, 'sell'] == 1:
            operationActive = True
        if operationActive == True:
            df.loc[index, 'sell'] = 1
        if df.loc[index, 'peaks_min'] == 1:
            operationActive = False

    # Close Strategy Operation Sell
    operationActive = False
    for index, row in df.iterrows():
        if df.loc[index, 'buy'] == 1:
            operationActive = True
        if operationActive == True:
            df.loc[index, 'buy'] = 1
        if (df.loc[index, 'peaks_max'] == 1):
           operationActive = False

    #Close Operation
    df['zone_sell'] = df['sell'].diff()
    if df['zone_sell'][len(df) - 6] == -1:
        if existingOperation(instrument=instrument_symbol, BuySell= "S"):
            CloseOperation(instrument=instrument_symbol,BuySell = "S")


    # Close  Operation
    df['zone_buy'] = df['buy'].diff()
    if df['zone_buy'][len(df) - 6] == -1:
        if existingOperation(instrument=instrument_symbol, BuySell= "B"):
             CloseOperation(instrument=instrument_symbol,BuySell= "B")

    # ***********************************************************
    # * Estrategy  SELL
    # ***********************************************************

    #Open Operation
    if df['zone_sell'][len(df) - 6] == 1:
        print("	  SELL SIGNAL! ")
        if existingOperation(instrument=instrument_symbol, BuySell= "B"):
             CloseOperation(instrument=instrument_symbol,BuySell= "B")
        if existingOperation(instrument=instrument_symbol, BuySell= "S")  != True:
            createEntryOrder(str_instrument=instrument_symbol,str_buy_sell="S")


    # ***********************************************************
    # * Estrategy  BUY
    # ***********************************************************


    # Open Operation
    if df['zone_buy'][len(df) - 6] == 1:
        print("	  BUY SIGNAL! ")
        if existingOperation(instrument=instrument_symbol, BuySell= "S"):
            CloseOperation(instrument=instrument_symbol,BuySell = "S")
        if existingOperation(instrument=instrument_symbol, BuySell= "B") != True:
            createEntryOrder(str_instrument=instrument_symbol,str_buy_sell="B")
    
    # Log Operation =================================================
            
    print(  df[['peaks_max','sell','zone_sell','peaks_min','buy','zone_buy', 'trend_min', 'trend_max']].tail(7)  ) 

    df.to_csv(file)
    print(str(dt.datetime.now()) + " " + args.timeframe +  " Update Function Completed. " + instrument_symbol + "\n")

StrategyHeartBeat()  # Run strategy