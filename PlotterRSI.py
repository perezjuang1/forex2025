import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List
import os
from datetime import datetime

class RSIPlotter:
    """Class to handle RSI plotting with animation"""
    
    def __init__(self, file_name: str):
        """Initialize the plotter with the file name"""
        self.file_name = file_name
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.setup_plot()
        self.data = None
        self.last_update = 0
        self.update_interval = 20  # seconds

    def setup_plot(self) -> None:
        """Setup the basic plot configuration"""
        plt.style.use('dark_background')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('RSI')
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        self.ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        self.ax.legend()

    def load_data(self) -> pd.DataFrame:
        """Load and process the forex data"""
        try:
            # Check if file exists
            if not os.path.exists(self.file_name):
                print(f"Error: File {self.file_name} not found in current directory: {os.getcwd()}")
                return pd.DataFrame()
            
            # Read the CSV file
            df = pd.read_csv(self.file_name)
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
            
            # Update RSI line
            self.rsi_line.set_data(df_view.index, df_view['rsi'])
            
            # Only autoscale if we're not zoomed in
            if xlim[0] == 0 and xlim[1] == len(self.data):
                self.ax.relim()
                self.ax.autoscale_view()
            
            return [self.rsi_line]
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
        self.ax.set_ylim(0, 100)  # RSI is typically between 0 and 100
        
        # Create RSI line
        self.rsi_line, = self.ax.plot([], [], linestyle='-', color='cyan', label='RSI')
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, 
            self.update_plot, 
            frames=None,
            interval=100,  # Faster update interval for smoother interaction
            blit=True
        )
        plt.show()

def main():
    """Main function to run the RSI plotter"""
    file_name = "EUR_USD_m5.csv"
    plotter = RSIPlotter(file_name)
    plotter.animate()

if __name__ == "__main__":
    main()
