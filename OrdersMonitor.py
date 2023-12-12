from threading import Event
class OrdersMonitor:
    def __init__(self):
        self.__order_id = None
        self.__orders = {}
        self.__event = Event()

    def on_added_order(self, _, __, order_row):
        order_id = order_row.order_id
        self.__orders[order_id] = order_row
        if self.__order_id == order_id:
            self.__event.set()

    def wait(self, time, order_id):
        self.__order_id = order_id

        order_row = self.find_order(order_id)
        if order_row is not None:
            return order_row

        self.__event.wait(time)

        return self.find_order(order_id)

    def find_order(self, order_id):
        if order_id in self.__orders:
            return self.__orders[order_id]
        else:
            return None

    def reset(self):
        self.__order_id = None
        self.__orders.clear()
        self.__event.clear()