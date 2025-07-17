import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional

from dataclasses import dataclass
import warnings
import os
from datetime import datetime
import numpy as np
import multiprocessing
from ConfigurationOperation import ConfigurationOperation

# Ignore specific matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Matplotlib.*")

@dataclass
class PlotConfig:
    """Configuration for the plot"""
    instrument: str
    timeframe: str

class ForexPlotter:
    """Class to handle forex data plotting with animation"""
    
    def __init__(self, config: PlotConfig):
        """Initialize the plotter with configuration"""
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title(f"{self.config.instrument} - Forex Plotter")
        self.setup_plot()
        self.setup_lines()
        self.data = None
        self.last_update = 0
        self.update_interval = 20  # seconds
        # Eliminada la variable self.hourly_regression_segments
        
    def setup_plot(self) -> None:
        """Setup the basic plot configuration"""
        plt.style.use('dark_background')
        self.ax.set_xlabel('Date', color='white')
        self.ax.set_ylabel(f'Price Move {self.config.instrument}', color='white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.tick_params(colors='white')
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#1a1a1a')
        
    def setup_lines(self) -> None:
        """Initialize all plot lines"""
        # Price line
        self.price_line, = self.ax.plot([], [], linestyle='dotted', color='#00ff00', label='Price')
        # Peak markers
        self.peaks_min_inf, = self.ax.plot([], [], linestyle='dotted', marker='o', color='#00ccff', label='Min Peaks')
        self.peaks_max_inf, = self.ax.plot([], [], linestyle='dotted', marker='o', color='orange', label='Max Peaks')
        # Trigger markers
        self.trigger_buy, = self.ax.plot([], [], 'D', color='white', label='Buy Trigger', zorder=30)
        self.trigger_sell, = self.ax.plot([], [], 'd', color='blue', label='Sell Trigger', zorder=30)
        # EMA 10 y EMA 30 y sus picos
        self.ema_10_line, = self.ax.plot([], [], '-', color='yellow', linewidth=2, label='EMA 10', zorder=2)
        self.peaks_min_ema_10_line, = self.ax.plot([], [], 'x', color='cyan', label='Min EMA 10')
        self.peaks_max_ema_10_line, = self.ax.plot([], [], 'x', color='magenta', label='Max EMA 10')
        self.ema_30_line, = self.ax.plot([], [], '-', color='orange', linewidth=2, label='EMA 30', zorder=2)
        self.peaks_min_ema_30_line, = self.ax.plot([], [], 'x', color='lime', label='Min EMA 30')
        self.peaks_max_ema_30_line, = self.ax.plot([], [], 'x', color='red', label='Max EMA 30')
        # Trade open zones
        self.trade_open_zone_min_line, = self.ax.plot([], [], '*', color='lime', markersize=18, label='Trade Open Buy (Confluence)', zorder=1)
        self.trade_open_zone_max_line, = self.ax.plot([], [], '*', color='red', markersize=18, label='Trade Open Sell (Confluence)', zorder=1)
        # Leyenda
        handles, labels = self.ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        # Remove 'Última Min Trend' and 'Última Max Trend' from legend if present
        unique.pop('Última Min Trend', None)
        unique.pop('Última Max Trend', None)
        self.ax.legend(unique.values(), unique.keys(), facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
        
    def load_data(self) -> pd.DataFrame:
        """Load and process the forex data"""
        try:
            # Construct the file path
            file_name = f"{self.config.instrument}_{self.config.timeframe}.csv"
            print(f"Loading data from: {file_name}")
            
            # Check if file exists
            if not os.path.exists(file_name):
                print(f"Error: File {file_name} not found in current directory: {os.getcwd()}")
                return pd.DataFrame()
            
            # Read the CSV file
            df = pd.read_csv(file_name)
            print(f"Loaded {len(df)} rows")
            
            # Convert date column (acepta cualquier formato reconocible por pandas)
            df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
            
            # Create numeric index for plotting
            df['index'] = range(len(df))
            df.set_index('index', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()
            
    def update_plot(self, frame: int) -> List[plt.Line2D]:
        """Update the plot for each animation frame"""
        current_time = datetime.now().timestamp()
        
        # Only reload data if enough time has passed
        if self.data is None or (current_time - self.last_update) > self.update_interval:
            self.data = self.load_data()
            self.last_update = current_time
        
        if self.data.empty:
            print("No data available to plot")
            return []
            
        try:
            # Get current view limits
            xlim = self.ax.get_xlim()
            
            # Ensure xlim is within valid range
            xlim = (max(0, int(xlim[0])), min(len(self.data) - 1, int(xlim[1])))
            
            # Filter data based on current view
            mask = (self.data.index >= xlim[0]) & (self.data.index <= xlim[1])
            df_view = self.data[mask]
            
            if df_view.empty:
                print("No data in the current view range")
                return []

            # Price line
            self.price_line.set_data(df_view.index, df_view['bidclose'])
            # EMA 10
            self.ema_10_line.set_data(df_view.index, df_view['ema_10'])
            # EMA 30
            self.ema_30_line.set_data(df_view.index, df_view['ema_30'])
            # Peaks
            self.peaks_min_inf.set_data(df_view[df_view['peaks_min'] == 1.0].index, df_view[df_view['peaks_min'] == 1.0]['bidclose'])
            self.peaks_max_inf.set_data(df_view[df_view['peaks_max'] == 1.0].index, df_view[df_view['peaks_max'] == 1.0]['bidclose'])
            # Triggers
            self.trigger_buy.set_data(df_view[df_view['signal'] == 1.0].index, df_view[df_view['signal'] == 1.0]['bidclose'])
            self.trigger_sell.set_data(df_view[df_view['signal'] == -1.0].index, df_view[df_view['signal'] == -1.0]['bidclose'])
            # Peaks EMA 10
            self.peaks_min_ema_10_line.set_data(df_view[df_view['peaks_min_ema_10'] == 1].index, df_view[df_view['peaks_min_ema_10'] == 1]['ema_10'])
            self.peaks_max_ema_10_line.set_data(df_view[df_view['peaks_max_ema_10'] == 1].index, df_view[df_view['peaks_max_ema_10'] == 1]['ema_10'])
            # Peaks EMA 30
            self.peaks_min_ema_30_line.set_data(df_view[df_view['peaks_min_ema_30'] == 1].index, df_view[df_view['peaks_min_ema_30'] == 1]['ema_30'])
            self.peaks_max_ema_30_line.set_data(df_view[df_view['peaks_max_ema_30'] == 1].index, df_view[df_view['peaks_max_ema_30'] == 1]['ema_30'])
            # Trade open zones
            if 'trade_open_zone_buy' in df_view.columns:
                self.trade_open_zone_min_line.set_data(df_view[df_view['trade_open_zone_buy'] == 1].index, df_view[df_view['trade_open_zone_buy'] == 1]['bidclose'])
            else:
                self.trade_open_zone_min_line.set_data([], [])
            if 'trade_open_zone_sell' in df_view.columns:
                self.trade_open_zone_max_line.set_data(df_view[df_view['trade_open_zone_sell'] == 1].index, df_view[df_view['trade_open_zone_sell'] == 1]['bidclose'])
            else:
                self.trade_open_zone_max_line.set_data([], [])
            # Visualización de la zona de tolerancia
            # Limpia bandas previas si existen
            if hasattr(self, 'tolerance_spans'):
                for span in self.tolerance_spans:
                    span.remove()
            self.tolerance_spans = []
            tolerance = ConfigurationOperation.tolerance_peaks  # Usar el valor centralizado
            # Zonas de compra (lime)
            if 'trade_open_zone_buy' in df_view.columns:
                for idx in df_view[df_view['trade_open_zone_buy'] == 1].index:
                    span = self.ax.axvspan(idx - tolerance, idx + tolerance, color='lime', alpha=0.15, zorder=0)
                    self.tolerance_spans.append(span)
            # Zonas de venta (red)
            if 'trade_open_zone_sell' in df_view.columns:
                for idx in df_view[df_view['trade_open_zone_sell'] == 1].index:
                    span = self.ax.axvspan(idx - tolerance, idx + tolerance, color='red', alpha=0.15, zorder=0)
                    self.tolerance_spans.append(span)
            # Eliminar líneas y marcadores relacionados con centro_picos_max_suave, centro_picos_min_suave, tendencias y marcadores especiales
            # (No crear ni actualizar self.centro_picos_max_suave_line, self.centro_picos_min_suave_line, self.trend_max_up, self.trend_max_down, self.trend_max_flat, self.trend_min_up, self.trend_min_down, self.trend_min_flat, self.last_min_trend_marker, self.last_max_trend_marker)
            # Ajuste de límites
            self.ax.set_ylim(df_view['bidclose'].min() * 0.999, df_view['bidclose'].max() * 1.001)
            return [
                self.price_line, 
                self.peaks_min_inf, 
                self.peaks_max_inf, 
                self.trigger_buy, 
                self.trigger_sell,
                self.peaks_min_ema_10_line,
                self.peaks_max_ema_10_line,
                self.ema_10_line,
                self.peaks_min_ema_30_line,
                self.peaks_max_ema_30_line,
                self.ema_30_line,
                self.trade_open_zone_min_line,
                self.trade_open_zone_max_line
            ]
        except Exception as e:
            print(f"Error updating plot: {str(e)}")
            return []
        
    def animate(self) -> None:
        """Start the animation"""
        # Load initial data
        self.data = self.load_data()
        if self.data.empty:
            print("Failed to load initial data")
            return
            
        self.last_update = datetime.now().timestamp()
        
        # Set initial view limits
        self.ax.set_xlim(0, len(self.data))
        self.ax.set_ylim(self.data['bidclose'].min() * 0.999, self.data['bidclose'].max() * 1.001)
        
        # Create animation with faster interval for smoother zoom
        ani = animation.FuncAnimation(
            self.fig, 
            self.update_plot, 
            frames=None,
            interval=4000,  # Cambiado a 1000 ms (1 segundo) para un refresco más lento
            blit=True
        )
        plt.show()

def run_plotter_for_instrument(instrument):
    config = PlotConfig(
        instrument=instrument.replace("/", "_"),  # Asegúrate que el nombre del archivo coincida
        timeframe=ConfigurationOperation.timeframe
    )
    plotter = ForexPlotter(config)
    plotter.animate()

if __name__ == "__main__":
    from ConfigurationOperation import ConfigurationOperation
    instruments = [i.replace("/", "_") for i in ConfigurationOperation.instruments]
    processes = []
    for instrument in instruments:
        p = multiprocessing.Process(target=run_plotter_for_instrument, args=(instrument,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()