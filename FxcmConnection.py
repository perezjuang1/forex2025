from TradingConfiguration import TradingConfig

from forexconnect import fxcorepy, ForexConnect, Common

class RobotConnection:
    def __init__(self):
        self.args = TradingConfig()
        self.common = Common()
        self.fxcorepy = fxcorepy

    @staticmethod
    def session_status_changed(session: fxcorepy.O2GSession,
                          status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
        print('Trading session status: {0}'.format(status))


    def getConnection(self):
                forexConnect = ForexConnect()
                forexConnect.login(self.args.userid, self.args.password, self.args.url, session_status_callback=self.session_status_changed)
                return forexConnect
    
    def getCorepy(self):
          return fxcorepy
                
