import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional

from dataclasses import dataclass
import warnings
import os
from datetime import datetime
import numpy as np

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
        self.ax.set_xlabel('Date', color='white')
        self.ax.set_ylabel(f'Price Move {self.config.instrument}', color='white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.tick_params(colors='white')
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#1a1a1a')
        
    def setup_lines(self) -> None:
        """Initialize all plot lines"""
        # Price lines
        self.price_line, = self.ax.plot([], [], linestyle='dotted', color='#00ff00', label='Price')
        self.ema_line, = self.ax.plot([], [], linestyle='dotted', color='#ff00ff', label='EMA Moving Average')
        self.ema_slow_line, = self.ax.plot([], [], linestyle='solid', color='#00ffff', label='EMA Slow')

        # Add trend line
        self.trend_line, = self.ax.plot([], [], linestyle='solid', color='red', label='Trend', alpha=0.3)

        # Add regression lines
        self.regression_line_1, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 1', alpha=0.5)
        self.regression_line_2, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 2', alpha=0.5)
        self.regression_line_3, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 3', alpha=0.5)
        self.regression_line_4, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 4', alpha=0.5)
        self.regression_line_5, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 5', alpha=0.5)
        self.regression_line_6, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 6', alpha=0.5)
        self.regression_line_7, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 7', alpha=0.5)
        self.regression_line_8, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 8', alpha=0.5)
        self.regression_line_9, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 9', alpha=0.5)
        self.regression_line_10, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 10', alpha=0.5)
        self.regression_line_11, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 11', alpha=0.5)
        self.regression_line_12, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 12', alpha=0.5)
        self.regression_line_13, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 13', alpha=0.5)
        self.regression_line_14, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 14', alpha=0.5)
        self.regression_line_15, = self.ax.plot([], [], linestyle='solid', color='#ff9900', label='Regression 15', alpha=0.5)

        # Peak markers
        self.peaks_min_inf, = self.ax.plot([], [], linestyle='dotted', marker='o', color='#00ccff', label='Min Peaks')
        self.peaks_max_inf, = self.ax.plot([], [], linestyle='dotted', marker='o', color='orange', label='Max Peaks')
       
        # Trigger markers
        self.trigger_buy, = self.ax.plot([], [], '^', color='#00ff00', label='Buy Trigger')
        self.trigger_sell, = self.ax.plot([], [], 'v', color='#ff0000', label='Sell Trigger')
        
        # Add markers for closed buy and sell operations
        self.trigger_close_buy, = self.ax.plot([], [], '*', color='#00ff00', label='Close Buy Trigger')
        self.trigger_close_sell, = self.ax.plot([], [], '*', color='#ff0000', label='Close Sell Trigger')

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
            
            # Ensure xlim is within valid range
            xlim = (max(0, int(xlim[0])), min(len(self.data) - 1, int(xlim[1])))
            
            # Filter data based on current view
            mask = (self.data.index >= xlim[0]) & (self.data.index <= xlim[1])
            df_view = self.data[mask]
            
            if df_view.empty:
                print("No data in the current view range")
                return []

            # Update trend line with the new trend_line column
            self.trend_line.set_data(df_view.index, df_view['trend_line'])
            
            # Update price and EMA lines
            self.price_line.set_data(df_view.index, df_view['bidclose'])
            self.ema_line.set_data(df_view.index, df_view['ema'])
            
            # Update regression lines
            self.regression_line_1.set_data(df_view.index, df_view['regression_line_1'])
            self.regression_line_2.set_data(df_view.index, df_view['regression_line_2'])
            self.regression_line_3.set_data(df_view.index, df_view['regression_line_3'])
            self.regression_line_4.set_data(df_view.index, df_view['regression_line_4'])
            self.regression_line_5.set_data(df_view.index, df_view['regression_line_5'])
            self.regression_line_6.set_data(df_view.index, df_view['regression_line_6'])
            self.regression_line_7.set_data(df_view.index, df_view['regression_line_7'])
            self.regression_line_8.set_data(df_view.index, df_view['regression_line_8'])
            self.regression_line_9.set_data(df_view.index, df_view['regression_line_9'])
            self.regression_line_10.set_data(df_view.index, df_view['regression_line_10'])
            self.regression_line_11.set_data(df_view.index, df_view['regression_line_11'])
            self.regression_line_12.set_data(df_view.index, df_view['regression_line_12'])
            self.regression_line_13.set_data(df_view.index, df_view['regression_line_13'])
            self.regression_line_14.set_data(df_view.index, df_view['regression_line_14'])
            self.regression_line_15.set_data(df_view.index, df_view['regression_line_15'])
            
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

            # Update slow EMA line
            self.ema_slow_line.set_data(df_view.index, df_view['ema_slow'])

            # Update trigger_close_buy markers
            self.trigger_close_buy.set_data(df_view[df_view['buy'] == -1.0].index, 
                                            df_view[df_view['buy'] == -1.0]['bidclose'])

            # Update trigger_close_sell markers
            self.trigger_close_sell.set_data(df_view[df_view['sell'] == -1.0].index, 
                                             df_view[df_view['sell'] == -1.0]['bidclose'])

            # Adjust y-axis limits dynamically based on visible data
            self.ax.set_ylim(df_view['bidclose'].min() * 0.999, df_view['bidclose'].max() * 1.001)
            
            return [self.price_line, self.ema_line,
                    self.ema_slow_line, 
                    self.peaks_min_inf, 
                    self.peaks_max_inf, 
                    self.trigger_buy, 
                    self.trigger_sell, 
                    self.trigger_close_buy, 
                    self.trigger_close_sell,
                    self.trend_line,
                    self.regression_line_1,
                    self.regression_line_2,
                    self.regression_line_3,
                    self.regression_line_4,
                    self.regression_line_5,
                    self.regression_line_6,
                    self.regression_line_7,
                    self.regression_line_8,
                    self.regression_line_9,
                    self.regression_line_10,
                    self.regression_line_11,
                    self.regression_line_12,
                    self.regression_line_13,
                    self.regression_line_14,
                    self.regression_line_15]
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