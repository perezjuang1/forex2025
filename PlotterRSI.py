import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import time
from matplotlib.animation import FuncAnimation

class PlotterRSI:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.animation = None
        # Set dark style
        plt.style.use('dark_background')
        
    def plot_rsi(self, df, instrument, timeframe):
        """
        Plot RSI indicator with overbought/oversold levels
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data with RSI column
        instrument : str
            Trading instrument name
        timeframe : str
            Timeframe of the data
        """
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Plot RSI
        self.ax.plot(df['date'], df['rsi'], label='RSI', color='cyan', linewidth=1)
        
        # Add overbought and oversold levels
        self.ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        self.ax.axhline(y=30, color='lime', linestyle='--', alpha=0.5, label='Oversold (30)')
        
        # Set title and labels
        self.ax.set_title(f'RSI - {instrument} ({timeframe})')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('RSI')
        
        # Set y-axis limits
        self.ax.set_ylim(0, 100)
        
        # Format x-axis dates
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add legend
        self.ax.legend()
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return self.fig
        
    def plot_rsi_with_price(self, df, instrument, timeframe):
        """
        Plot RSI indicator with price data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data with RSI column
        instrument : str
            Trading instrument name
        timeframe : str
            Timeframe of the data
        """
        # Create figure with two subplots
        self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        # Create a sequential index for x-axis
        x = range(len(df))
        
        # Plot price data on top subplot
        ax1.plot(x, df['bidclose'], label='Price', color='yellow', linewidth=1)
        ax1.set_title(f'{instrument} ({timeframe})')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot RSI on bottom subplot
        ax2.plot(x, df['rsi'], label='RSI', color='cyan', linewidth=1)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.axhline(y=30, color='lime', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Set x-axis ticks to show dates
        date_ticks = df['date'].iloc[::len(df)//10]  # Show 10 evenly spaced dates
        ax2.set_xticks(range(0, len(df), len(df)//10))
        ax2.set_xticklabels([d.strftime('%H:%M:%S') for d in date_ticks], rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        return self.fig

    def auto_refresh_rsi(self):
        """
        Auto-refresh RSI plot every 3 seconds using data from EUR_USD_m5.csv
        """
        def update(frame):
            try:
                # Read the latest data
                df = pd.read_csv('EUR_USD_m5.csv')
                
                # Clear the current plot
                self.ax.clear()
                
                # Plot RSI
                self.ax.plot(df['date'], df['rsi'], label='RSI', color='cyan', linewidth=1)
                
                # Add overbought and oversold levels
                self.ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
                self.ax.axhline(y=30, color='lime', linestyle='--', alpha=0.5, label='Oversold (30)')
                
                # Set title and labels
                self.ax.set_title(f'RSI - EUR/USD (M5)')
                self.ax.set_xlabel('Date')
                self.ax.set_ylabel('RSI')
                
                # Set y-axis limits
                self.ax.set_ylim(0, 100)
                
                # Format x-axis dates
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                
                # Add legend
                self.ax.legend()
                
                # Add grid
                self.ax.grid(True, alpha=0.3)
                
                # Adjust layout
                plt.tight_layout()
                
            except Exception as e:
                print(f"Error updating plot: {str(e)}")
        
        # Create initial figure
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            update,
            interval=3000,  # 3 seconds
            cache_frame_data=False
        )
        
        plt.show()
        
    def show(self):
        """Display the plot"""
        if self.fig is not None:
            plt.show()
            
    def save(self, filename):
        """Save the plot to a file"""
        if self.fig is not None:
            self.fig.savefig(filename)
            plt.close(self.fig)

def main():
    # Create an instance of PlotterRSI
    plotter = PlotterRSI()
    
    # Example: Read data from CSV file
    try:
        df = pd.read_csv('EUR_USD_m5.csv')
        
        # Validate data
        if df.empty:
            raise ValueError("The CSV file is empty")
            
        # Check required columns
        required_columns = ['date', 'bidclose', 'rsi']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Convert date column to datetime and handle timezone if present
        df['date'] = pd.to_datetime(df['date'])
        
        # Remove any rows with NaN values
        df = df.dropna(subset=['date', 'bidclose', 'rsi'])
        
        if df.empty:
            raise ValueError("No valid data after cleaning")
            
        # Print data info for debugging
        print("Data shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        print("\nDate range:", df['date'].min(), "to", df['date'].max())
        
        # Plot RSI with price
        plotter.plot_rsi_with_price(df, 'EUR/USD', 'M5')
        
        # Show the plot
        plotter.show()
        
    except FileNotFoundError:
        print("Error: EUR_USD_m5.csv file not found")
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 