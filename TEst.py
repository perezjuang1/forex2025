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
import os
from urllib.request import urlopen
import datetime
from forexconnect import fxcorepy, ForexConnect
from ConfigurationOperation import ConfigurationOperation

ArgsConf = ConfigurationOperation()
fx = None

def session_status_changed(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
    print("Trading session status: " + str(status))

def getHystory():
    with ForexConnect() as fx:
        try:
            fx.login(ArgsConf.userid, ArgsConf.password, ArgsConf.url, session_status_callback=session_status_changed)
            history = fx.get_history(ArgsConf.currency, ArgsConf.timeframe, ArgsConf.date_from, ArgsConf.date_to)
            current_unit, _ = ForexConnect.parse_timeframe(ArgsConf.timeframe)
            result = type(history)
            print(result)
            print("Price Data Received...")
            print(current_unit)
            fx.logout()
            return pd.DataFrame(history, columns = ["Date", "BidOpen", "BidHigh", "BidLow", "BidClose", "Volume"])
        except Exception as e:
            print("Exception: " + str(e))


def createEntryOrder()
      args = parse_args()
    str_user_id = args.l
    str_password = args.p
    str_url = args.u
    str_connection = args.c
    str_session_id = args.session
    str_pin = args.pin
    str_instrument = args.i
    str_buy_sell = args.d
    str_rate = args.r
    str_lots = args.lots
    str_account = args.account
    stop = args.stop
    peggedstop = args.peggedstop
    pegstoptype = args.pegstoptype
    limit = args.limit
    peggedlimit = args.peggedlimit
    peglimittype = args.peglimittype
    event = Event()

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
                 str_pin, common_samples.session_status_changed)

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

            base_unit_size = trading_settings_provider.get_base_unit_size(
                str_instrument, account)

            amount = base_unit_size * str_lots

            entry = fxcorepy.Constants.Orders.ENTRY

            if str_buy_sell == 'B':
                stopv = -stop
                limitv = limit
            else:
                stopv = stop
                limitv = -limit

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
                        RATE=str_rate,
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
                        AMOUNT=amount,
                        RATE=str_rate,
                    )
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
                        AMOUNT=amount,
                        RATE=str_rate,
                    )
                else:
                    request = fx.create_order_request(
                        order_type=entry,
                        OFFER_ID=offer.offer_id,
                        ACCOUNT_ID=str_account,
                        BUY_SELL=str_buy_sell,
                        AMOUNT=amount,
                        RATE_STOP=stop,
                        RATE_LIMIT=limit,
                        RATE=str_rate,
                    )

            orders_monitor = OrdersMonitor()

            orders_table = fx.get_table(ForexConnect.ORDERS)
            orders_listener = Common.subscribe_table_updates(orders_table,
                                                             on_add_callback=orders_monitor.on_added_order)

            try:
                resp = fx.send_request(request)
                order_id = resp.order_id

            except Exception as e:
                common_samples.print_exception(e)
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
            common_samples.print_exception(e)
        try:
            fx.logout()
        except Exception as e:
            common_samples.print_exception(e)


print("Login")
getHystory()

