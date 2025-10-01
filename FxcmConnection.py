from TradingConfiguration import TradingConfig

from forexconnect import fxcorepy, ForexConnect, Common

class RobotConnection:
    def __init__(self, instrument=None):
        self.args = TradingConfig()
        self.common = Common()
        self.fxcorepy = fxcorepy
        self.instrument = instrument or "Unknown"

    def session_status_changed(self, session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
        print(f"Trading session status: {str(status)} - {self.instrument}")

    def getConnection(self, force_new=False):
        """Obtiene una conexión a FXCM, con opción de forzar una nueva"""
        try:
            if hasattr(self, '_connection') and not force_new:
                # Intenta usar la conexión existente
                return self._connection
            
            # Crear nueva conexión
            forexConnect = ForexConnect()
            forexConnect.login(
                self.args.userid,
                self.args.password,
                self.args.url,
                session_status_callback=self.session_status_changed
            )
            
            # Guardar y retornar la nueva conexión
            self._connection = forexConnect
            return forexConnect
            
        except Exception as e:
            print(f"Error en conexión FXCM: {str(e)} - {self.instrument}")
            raise
    
    def getCorepy(self):
          return fxcorepy
                
