from ConfigurationOperation import ConfigurationOperation

from forexconnect import fxcorepy, ForexConnect, Common

class RobotConnection:
    def __init__(self):
        self.args = ConfigurationOperation()
        self.common = Common()
        self.fxcorepy = fxcorepy

    @staticmethod
    def session_status_changed(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
        print("Trading session status: " + str(status))


    def getConnection(self):
                forexConnect = ForexConnect()
                forexConnect.login(self.args.userid, self.args.password, self.args.url, session_status_callback=self.session_status_changed)
                return forexConnect
                
