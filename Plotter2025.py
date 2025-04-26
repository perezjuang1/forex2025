import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import warnings
import os
from datetime import datetime

# Ignore specific matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Matplotlib.*")

@dataclass
class PlotConfig:
    """Configuration for the plot"""
    instrument: str
    timeframe: str
    timeframe_sup: str
    timeframe_sup2: str
    days: int

class ForexPlotter:
    """Class to handle forex data plotting with animation"""
    
    def __init__(self, config: PlotConfig):
        """Initialize the plotter with configuration"""
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.setup_plot()
        self.setup_lines()
        self.data = None
        self.last_update = 0
        self.update_interval = 20  # seconds
        
    def setup_plot(self) -> None:
        """Setup the basic plot configuration"""
        plt.style.use('dark_background')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel(f'Price Move {self.config.instrument}')
        self.ax.grid(True, alpha=0.3)
        
    def setup_lines(self) -> None:
        """Initialize all plot lines"""
        # Price lines
        self.price_line, = self.ax.plot([], [], linestyle='dotted', color='gray', label='Price')
        self.ema_line, = self.ax.plot([], [], linestyle='dotted', color='pink', label='Moving Average')
        
        # Peak markers
        self.peaks_min_inf, = self.ax.plot([], [], linestyle='dotted', marker='o', color='blue', label='Min Peaks')
        self.peaks_max_inf, = self.ax.plot([], [], linestyle='dotted', marker='o', color='blue', label='Max Peaks')
        
       
        # Trigger markers
        self.trigger_buy, = self.ax.plot([], [], '^', color='green', label='Buy Trigger')
        self.trigger_sell, = self.ax.plot([], [], 'v', color='red', label='Sell Trigger')
        
        # Add legend
        self.ax.legend()
        
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
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M')
            
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
            
            # Filter data based on current view
            mask = (self.data.index >= xlim[0]) & (self.data.index <= xlim[1])
            df_view = self.data[mask]
            
            if len(df_view) == 0:
                return []
            
            # Update price and EMA lines
            self.price_line.set_data(df_view.index, df_view['bidclose'])
            self.ema_line.set_data(df_view.index, df_view['ema'])
            
            # Update peaks
            self.peaks_min_inf.set_data(df_view[df_view['peaks_min'] == 1.0].index, 
                                       df_view[df_view['peaks_min'] == 1.0]['bidclose'])
            self.peaks_max_inf.set_data(df_view[df_view['peaks_max'] == 1.0].index, 
                                       df_view[df_view['peaks_max'] == 1.0]['bidclose'])
            
            
            # Update trigger markers
            self.trigger_buy.set_data(df_view[df_view['buy'] == 1.0].index, 
                                     df_view[df_view['buy'] == 1.0]['bidclose'])
            self.trigger_sell.set_data(df_view[df_view['sell'] == 1.0].index, 
                                      df_view[df_view['sell'] == 1.0]['bidclose'])
            
            # Only autoscale if we're not zoomed in
            if xlim[0] == 0 and xlim[1] == len(self.data):
                self.ax.relim()
                self.ax.autoscale_view()
            
            return [self.price_line, self.ema_line, self.peaks_min_inf, self.peaks_max_inf,
                   self.trigger_buy, self.trigger_sell]
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
            interval=100,  # Faster update interval for smoother interaction
            blit=True
        )
        plt.show()

def main():
    """Main function to run the plotter"""
    config = PlotConfig(
        instrument="EUR_USD",
        timeframe="m5",
        timeframe_sup="m15",
        timeframe_sup2="m30",
        days=1
    )
    
    plotter = ForexPlotter(config)
    plotter.animate()

if __name__ == "__main__":
    main() 