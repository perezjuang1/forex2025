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
        self.trigger_buy, = self.ax.plot([], [], 'D', color='purple', label='Buy Trigger', zorder=4)
        self.trigger_sell, = self.ax.plot([], [], 'd', color='crimson', label='Sell Trigger', zorder=4)

        # Línea EMA 10
        self.ema_10_line, = self.ax.plot([], [], '-', color='yellow', linewidth=2, label='EMA 10', zorder=2)

        # Nueva línea para peaks_min_ema_10 (mínimos EMA 10)
        self.peaks_min_ema_10_line, = self.ax.plot([], [], 'x', color='cyan', label='Min EMA 10')
        # Nueva línea para peaks_max_ema_10 (máximos EMA 10)
        self.peaks_max_ema_10_line, = self.ax.plot([], [], 'x', color='magenta', label='Max EMA 10')

        # Línea para zonas de alto volumen (eliminada)
        # self.high_volume_zone_line, = self.ax.plot([], [], '|', color='yellow', markersize=15, label='High Volume Zone')
        # Confluencia de picos: zona de apertura de trade
        self.trade_open_zone_min_line, = self.ax.plot([], [], '*', color='lime', markersize=18, label='Trade Open Buy (Confluence)', zorder=1)
        self.trade_open_zone_max_line, = self.ax.plot([], [], '*', color='red', markersize=18, label='Trade Open Sell (Confluence)', zorder=1)
        
        # Línea de regresión por hora
        self.hourly_regression_line, = self.ax.plot([], [], '-', color='white', linewidth=2, label='Regresión Lineal por Hora', zorder=2)

        # Add legend with white text
        legend = self.ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
        for text in legend.get_texts():
            text.set_color('white')
        
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

            # Update price line
            self.price_line.set_data(df_view.index, df_view['bidclose'])

            # EMA 10
            if 'ema_10' in df_view.columns:
                self.ema_10_line.set_data(df_view.index, df_view['ema_10'])
            else:
                self.ema_10_line.set_data([], [])

            # Update peaks
            self.peaks_min_inf.set_data(df_view[df_view['peaks_min'] == 1.0].index, 
                                       df_view[df_view['peaks_min'] == 1.0]['bidclose'])
            self.peaks_max_inf.set_data(df_view[df_view['peaks_max'] == 1.0].index, 
                                       df_view[df_view['peaks_max'] == 1.0]['bidclose'])
            
            # Update trigger markers para la nueva señal
            self.trigger_buy.set_data(df_view[df_view['signal'] == 1.0].index, df_view[df_view['signal'] == 1.0]['bidclose'])
            self.trigger_sell.set_data(df_view[df_view['signal'] == -1.0].index, df_view[df_view['signal'] == -1.0]['bidclose'])

            # Nueva línea: peaks_min_ema_10 (mínimos EMA 10)
            self.peaks_min_ema_10_line.set_data(
                df_view[df_view['peaks_min_ema_10'] == 1].index,
                df_view[df_view['peaks_min_ema_10'] == 1]['ema_10']
            )
            # Nueva línea: peaks_max_ema_10 (máximos EMA 10)
            self.peaks_max_ema_10_line.set_data(
                df_view[df_view['peaks_max_ema_10'] == 1].index,
                df_view[df_view['peaks_max_ema_10'] == 1]['ema_10']
            )

            # Línea para zonas de alto volumen (eliminada)
            # if 'high_volume_zone' in df_view.columns:
            #     high_vol_idx = df_view[df_view['high_volume_zone'] == 1].index
            #     high_vol_price = df_view.loc[high_vol_idx, 'bidclose']
            #     self.high_volume_zone_line.set_data(high_vol_idx, high_vol_price)
            # else:
            #     self.high_volume_zone_line.set_data([], [])
            # Confluencia de picos: zona de apertura de trade
            if 'trade_open_zone_min' in df_view.columns:
                min_idx = df_view[df_view['trade_open_zone_min'] == 1].index
                min_price = df_view.loc[min_idx, 'bidclose']
                self.trade_open_zone_min_line.set_data(min_idx, min_price)
            else:
                self.trade_open_zone_min_line.set_data([], [])
            if 'trade_open_zone_max' in df_view.columns:
                max_idx = df_view[df_view['trade_open_zone_max'] == 1].index
                max_price = df_view.loc[max_idx, 'bidclose']
                self.trade_open_zone_max_line.set_data(max_idx, max_price)
            else:
                self.trade_open_zone_max_line.set_data([], [])

            # Línea de regresión por hora (segmentada)
            if 'hourly_regression' in df_view.columns and 'date' in df_view.columns:
                self.hourly_regression_line.set_data([], [])  # Limpiar antes de graficar
                # Agrupar por hora
                segments_x = []
                segments_y = []
                df_view = df_view.copy()
                df_view['hour_group'] = df_view['date'].dt.floor('H')
                for _, group in df_view.groupby('hour_group'):
                    idx = group[~group['hourly_regression'].isna()].index
                    vals = group.loc[idx, 'hourly_regression']
                    if len(idx) > 1:
                        segments_x.append(idx)
                        segments_y.append(vals)
                # Graficar cada segmento por separado
                self.hourly_regression_line.set_data([], [])
                for x, y in zip(segments_x, segments_y):
                    self.hourly_regression_line.set_data(x, y)
                    self.ax.plot(x, y, '-', color='white', linewidth=2, zorder=2)
            else:
                self.hourly_regression_line.set_data([], [])

            # Adjust y-axis limits dynamically based on visible data
            self.ax.set_ylim(df_view['bidclose'].min() * 0.999, df_view['bidclose'].max() * 1.001)
            
            return [self.price_line, 
                    self.peaks_min_inf, 
                    self.peaks_max_inf, 
                    self.trigger_buy, 
                    self.trigger_sell,
                    self.peaks_min_ema_10_line,
                    self.peaks_max_ema_10_line,
                    self.ema_10_line,
                    # self.high_volume_zone_line,  # Eliminada
                    self.trade_open_zone_min_line,
                    self.trade_open_zone_max_line,
                    self.hourly_regression_line]
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