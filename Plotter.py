import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import warnings

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
        self.peaks_min_inf, = self.ax.plot([], [], linestyle='dotted', marker='^', color='green')
        self.peaks_max_inf, = self.ax.plot([], [], linestyle='dotted', marker='v', color='blue')
        self.peaks_min_sup, = self.ax.plot([], [], linestyle='-', marker='^', color='white')
        self.peaks_max_sup, = self.ax.plot([], [], linestyle='-', marker='v', color='white')
        self.peaks_min_sup2, = self.ax.plot([], [], linestyle='-', marker='^', color='gray')
        self.peaks_max_sup2, = self.ax.plot([], [], linestyle='-', marker='v', color='gray')
        
        # Trade markers
        self.sell_open_inf, = self.ax.plot([], [], ',', color='green')
        self.sell_close_inf, = self.ax.plot([], [], ',', color='green')
        self.buy_open_inf, = self.ax.plot([], [], ',', color='red')
        self.buy_close_inf, = self.ax.plot([], [], ',', color='red')
        
        self.sell_open_sup, = self.ax.plot([], [], ',', color='green')
        self.sell_close_sup, = self.ax.plot([], [], ',', color='green')
        self.buy_open_sup, = self.ax.plot([], [], ',', color='red')
        self.buy_close_sup, = self.ax.plot([], [], ',', color='red')
        
        # Trigger markers
        self.trigger_buy, = self.ax.plot([], [], '^', color='orange')
        self.trigger_sell, = self.ax.plot([], [], 'v', color='orange')
        
        # Add legend
        self.ax.legend()
        
    def load_data(self) -> pd.DataFrame:
        """Load and process the forex data"""
        try:
            df = pd.read_csv(f"{self.config.instrument}_{self.config.timeframe}.csv")
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M')
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
            
    def update_plot(self, frame: int) -> List[plt.Line2D]:
        """Update the plot for each animation frame"""
        df = self.load_data()
        if df.empty:
            return []
            
        # Update price and EMA lines
        self.price_line.set_data(df.index, df['bidclose'])
        self.ema_line.set_data(df.index, df['ema'])
        
        # Update peaks
        self.peaks_min_inf.set_data(df[df['peaks_min'] == 1.0].index, 
                                   df[df['peaks_min'] == 1.0]['bidclose'])
        self.peaks_max_inf.set_data(df[df['peaks_max'] == 1.0].index, 
                                   df[df['peaks_max'] == 1.0]['bidclose'])
        
        # Update trade markers
        self.sell_open_inf.set_data(df[df['sell'] == 1.0].index, 
                                   df[df['sell'] == 1.0]['bidclose'])
        self.sell_close_inf.set_data(df[df['sell'] == -1.0].index, 
                                    df[df['sell'] == -1.0]['bidclose'])
        self.buy_open_inf.set_data(df[df['buy'] == 1.0].index, 
                                  df[df['buy'] == 1.0]['bidclose'])
        self.buy_close_inf.set_data(df[df['buy'] == -1.0].index, 
                                   df[df['buy'] == -1.0]['bidclose'])
        
        # Update trigger markers
        self.trigger_buy.set_data(df[df['buy'] == 1.0].index, 
                                 df[df['buy'] == 1.0]['bidclose'])
        self.trigger_sell.set_data(df[df['sell'] == 1.0].index, 
                                  df[df['sell'] == 1.0]['bidclose'])
        
        # Auto-scale the view
        self.ax.relim()
        self.ax.autoscale_view()
        
        return [self.price_line, self.ema_line, self.peaks_min_inf, self.peaks_max_inf,
                self.sell_open_inf, self.sell_close_inf, self.buy_open_inf, self.buy_close_inf,
                self.trigger_buy, self.trigger_sell]
        
    def animate(self) -> None:
        """Start the animation"""
        ani = animation.FuncAnimation(
            self.fig, 
            self.update_plot, 
            frames=None,
            interval=20000,
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