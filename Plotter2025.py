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
        
        # Minimize the window on startup
        try:
            # Try to minimize the window using the backend-specific method
            if hasattr(self.fig.canvas.manager, 'window'):
                # For TkAgg backend
                self.fig.canvas.manager.window.state('iconic')
            elif hasattr(self.fig.canvas.manager, 'window') and hasattr(self.fig.canvas.manager.window, 'iconify'):
                # For Qt backend
                self.fig.canvas.manager.window.iconify()
            else:
                # Fallback: try to set window state to minimized
                self.fig.canvas.manager.set_window_title(f"{self.config.instrument} - Forex Plotter (Minimized)")
        except Exception as e:
            print(f"Could not minimize window for {self.config.instrument}: {e}")
        
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
        # Línea de mediana móvil segmentada por tendencia
        self.median_segments = []  # Para almacenar los segmentos de la mediana
        # Trigger markers
        self.trigger_buy, = self.ax.plot([], [], 'D', color='white', label='Buy Signal', zorder=30)
        self.trigger_sell, = self.ax.plot([], [], 'd', color='blue', label='Sell Signal', zorder=30)
        # Trade zones (combinadas: trend following + reversal)
        self.trade_zone_buy_line, = self.ax.plot([], [], '*', color='lime', markersize=20, label='Trade Zone Buy (Trend+Reversal)', zorder=2)
        self.trade_zone_sell_line, = self.ax.plot([], [], '*', color='red', markersize=20, label='Trade Zone Sell (Trend+Reversal)', zorder=2)

        
        # Leyenda
        from matplotlib.patches import Patch
        handles, labels = self.ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        # Añadir manualmente la mediana móvil y zonas de tolerancia
        median_patch_up = Patch(facecolor='lime', edgecolor='none', alpha=0.8, label='Median (Up Trend)')
        median_patch_down = Patch(facecolor='red', edgecolor='none', alpha=0.8, label='Median (Down Trend)')
        tolerance_patch_buy = Patch(facecolor='lime', edgecolor='none', alpha=0.15, label='Tolerance Zone Buy')
        tolerance_patch_sell = Patch(facecolor='red', edgecolor='none', alpha=0.15, label='Tolerance Zone Sell')
        
        # Solo agregar si no existen ya
        legend_handles = list(unique.values())
        legend_labels = list(unique.keys())
        
        if 'Median (Up Trend)' not in legend_labels:
            legend_handles.append(median_patch_up)
            legend_labels.append('Median (Up Trend)')
        if 'Median (Down Trend)' not in legend_labels:
            legend_handles.append(median_patch_down)
            legend_labels.append('Median (Down Trend)')
        if 'Tolerance Zone Buy' not in legend_labels:
            legend_handles.append(tolerance_patch_buy)
            legend_labels.append('Tolerance Zone Buy')
        if 'Tolerance Zone Sell' not in legend_labels:
            legend_handles.append(tolerance_patch_sell)
            legend_labels.append('Tolerance Zone Sell')
            
        self.ax.legend(legend_handles, legend_labels, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
        
    def toggle_window_state(self):
        """Toggle between minimized and normal window state"""
        try:
            if hasattr(self.fig.canvas.manager, 'window'):
                current_state = self.fig.canvas.manager.window.state()
                if current_state == 'iconic':
                    # Restore window
                    self.fig.canvas.manager.window.state('normal')
                    print(f"Window restored for {self.config.instrument}")
                else:
                    # Minimize window
                    self.fig.canvas.manager.window.state('iconic')
                    print(f"Window minimized for {self.config.instrument}")
        except Exception as e:
            print(f"Could not toggle window state for {self.config.instrument}: {e}")
        
    def _create_continuous_ranges(self, indices):
        """Create continuous ranges from a list of indices"""
        if len(indices) == 0:
            return []
        
        ranges = []
        start_idx = indices[0]
        prev_idx = indices[0]
        
        for idx in indices[1:]:
            if idx != prev_idx + 1:
                # Gap found, end current range and start new one
                ranges.append((start_idx, prev_idx + 1))
                start_idx = idx
            prev_idx = idx
        
        # Add the last range
        ranges.append((start_idx, prev_idx + 1))
        
        return ranges

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
            
            # Línea de mediana móvil segmentada por tendencia
            # Elimina segmentos anteriores
            for seg in getattr(self, 'median_segments', []):
                seg.remove()
            self.median_segments = []
            if 'median_bidclose' in df_view.columns and 'median_trend' in df_view.columns:
                x = df_view.index.values
                y = df_view['median_bidclose'].values
                trend = df_view['median_trend'].values
                # Segmenta por tendencia
                current_color = None
                seg_x = []
                seg_y = []
                for i in range(1, len(x)):
                    color = 'lime' if trend[i] == 'going up' else ('red' if trend[i] == 'going down' else None)
                    if color != current_color and seg_x:
                        # Dibuja el segmento anterior
                        if current_color:
                            line, = self.ax.plot(seg_x, seg_y, color=current_color, linewidth=2, zorder=2)
                            self.median_segments.append(line)
                        seg_x = [x[i-1]]
                        seg_y = [y[i-1]]
                        current_color = color
                    seg_x.append(x[i])
                    seg_y.append(y[i])
                # Dibuja el último segmento
                if seg_x and current_color:
                    line, = self.ax.plot(seg_x, seg_y, color=current_color, linewidth=2, zorder=2)
                    self.median_segments.append(line)
            
            self.trigger_buy.set_data(df_view[df_view['signal'] == 1.0].index, df_view[df_view['signal'] == 1.0]['bidclose'])
            self.trigger_sell.set_data(df_view[df_view['signal'] == -1.0].index, df_view[df_view['signal'] == -1.0]['bidclose'])
            # Peaks min/max
            self.peaks_min_inf.set_data(df_view[df_view['peaks_min'] == 1].index, df_view[df_view['peaks_min'] == 1]['bidclose'])
            self.peaks_max_inf.set_data(df_view[df_view['peaks_max'] == 1].index, df_view[df_view['peaks_max'] == 1]['bidclose'])
            # Trade zones (zonas finales de trading calculadas)
            if 'trade_zone_buy' in df_view.columns:
                self.trade_zone_buy_line.set_data(df_view[df_view['trade_zone_buy'] == 1].index, df_view[df_view['trade_zone_buy'] == 1]['bidclose'])
            else:
                self.trade_zone_buy_line.set_data([], [])
            if 'trade_zone_sell' in df_view.columns:
                self.trade_zone_sell_line.set_data(df_view[df_view['trade_zone_sell'] == 1].index, df_view[df_view['trade_zone_sell'] == 1]['bidclose'])
            else:
                self.trade_zone_sell_line.set_data([], [])
            
            # Visualización de las zonas de tolerancia (las zonas ya están marcadas en el DataFrame)
            # Limpia bandas previas si existen
            if hasattr(self, 'tolerance_spans'):
                for span in self.tolerance_spans:
                    span.remove()
            self.tolerance_spans = []
            
            # Zonas de tolerancia de compra (lime) - mostrar solo los puntos marcados
            if 'tolerance_zone_buy' in df_view.columns:
                tolerance_buy_indices = df_view[df_view['tolerance_zone_buy'] == 1].index
                if len(tolerance_buy_indices) > 0:
                    # Crear bandas continuas para las zonas de tolerancia
                    tolerance_buy_ranges = self._create_continuous_ranges(tolerance_buy_indices)
                    for start_idx, end_idx in tolerance_buy_ranges:
                        span = self.ax.axvspan(start_idx, end_idx, color='lime', alpha=0.15, zorder=0)
                        self.tolerance_spans.append(span)
            
            # Zonas de tolerancia de venta (red) - mostrar solo los puntos marcados
            if 'tolerance_zone_sell' in df_view.columns:
                tolerance_sell_indices = df_view[df_view['tolerance_zone_sell'] == 1].index
                if len(tolerance_sell_indices) > 0:
                    # Crear bandas continuas para las zonas de tolerancia
                    tolerance_sell_ranges = self._create_continuous_ranges(tolerance_sell_indices)
                    for start_idx, end_idx in tolerance_sell_ranges:
                        span = self.ax.axvspan(start_idx, end_idx, color='red', alpha=0.15, zorder=0)
                        self.tolerance_spans.append(span)

            # Ajuste de límites
            self.ax.set_ylim(df_view['bidclose'].min() * 0.999, df_view['bidclose'].max() * 1.001)
            
            # Devuelve también los segmentos de la mediana
            return [
                self.price_line,
                *self.median_segments,
                self.peaks_min_inf,
                self.peaks_max_inf,
                self.trigger_buy,
                self.trigger_sell,
                self.trade_zone_buy_line,
                self.trade_zone_sell_line
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
        
        # Add keyboard event handler for toggling window state
        def on_key_press(event):
            if event.key == 'm' or event.key == 'M':
                self.toggle_window_state()
            elif event.key == 'escape':
                plt.close(self.fig)
        
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Create animation with faster interval for smoother zoom
        ani = animation.FuncAnimation(
            self.fig, 
            self.update_plot, 
            frames=None,
            interval=4000,  # Cambiado a 1000 ms (1 segundo) para un refresco más lento
            blit=True
        )
        
        # Show instructions in console
        print(f"Plotter for {self.config.instrument} started (minimized)")
        print("Press 'M' to toggle minimize/restore window")
        print("Press 'ESC' to close window")
        
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