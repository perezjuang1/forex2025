import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Tuple, Optional
from dataclasses import dataclass
import warnings
import os
from datetime import datetime
import numpy as np
import glob
import threading
import time
from ConfigurationOperation import ConfigurationOperation

# Ignore specific matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Matplotlib.*")

@dataclass
class PlotConfig:
    """Configuration for the plot"""
    instrument: str
    timeframe: str

class ForexPlotterGUI:
    """Single window plotter with CSV file selector"""
    
    def __init__(self):
        """Initialize the GUI plotter"""
        self.root = tk.Tk()
        self.root.title("Forex Data Plotter")
        self.root.geometry("1400x800")
        
        # Get available CSV files
        self.csv_files = self.get_available_csv_files()
        
        # Current data
        self.current_data = None
        self.current_file = None
        
        # Auto-update settings
        self.auto_update_enabled = True
        self.update_interval = 120  # 2 minutes in seconds
        self.update_thread = None
        self.stop_update = False
        self.auto_scroll_enabled = True  # Auto-scroll to new data
        
        # Setup GUI
        self.setup_gui()
        
        # Start auto-update thread
        self.start_auto_update()
        
    def get_available_csv_files(self):
        """Get all available CSV files in the current directory"""
        csv_files = glob.glob("*.csv")
        return sorted(csv_files)
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for file selection
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File selection label
        ttk.Label(left_panel, text="Available CSV Files:", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        
        # Listbox for file selection
        self.listbox = tk.Listbox(left_panel, width=40, height=20, font=('Arial', 10))
        self.listbox.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        # Populate listbox
        for file in self.csv_files:
            self.listbox.insert(tk.END, file)
        
        # Bind selection event
        self.listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        # Right panel for plot
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.setup_plot()
        self.setup_lines()
        
        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Enable matplotlib navigation tools
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        self.toolbar.update()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Select a CSV file to start plotting")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Auto-update controls
        update_frame = ttk.Frame(left_panel)
        update_frame.pack(pady=(10, 0), fill=tk.X)
        
        # Auto-update checkbox
        self.auto_update_var = tk.BooleanVar(value=True)
        self.auto_update_checkbox = ttk.Checkbutton(
            update_frame, 
            text="Auto-update every 2 minutes", 
            variable=self.auto_update_var,
            command=self.toggle_auto_update
        )
        self.auto_update_checkbox.pack(side=tk.LEFT)
        
        # Manual update button
        self.update_button = ttk.Button(
            update_frame, 
            text="Update Now", 
            command=self.manual_update
        )
        self.update_button.pack(side=tk.RIGHT)
        
        # Full view button
        self.full_view_button = ttk.Button(
            update_frame, 
            text="Full View", 
            command=self.force_full_view
        )
        self.full_view_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Auto-scroll controls
        scroll_frame = ttk.Frame(left_panel)
        scroll_frame.pack(pady=(5, 0), fill=tk.X)
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        self.auto_scroll_checkbox = ttk.Checkbutton(
            scroll_frame, 
            text="Auto-scroll to new data", 
            variable=self.auto_scroll_var,
            command=self.toggle_auto_scroll
        )
        self.auto_scroll_checkbox.pack(side=tk.LEFT)
        
        # Instructions
        instructions = ttk.Label(left_panel, text="Instructions:\n1. Select a CSV file from the list\n2. The plot will update automatically\n3. Use mouse to zoom and pan\n4. Auto-update every 2 minutes", 
                               font=('Arial', 9), justify=tk.LEFT)
        instructions.pack(pady=(10, 0))
        
    def setup_plot(self):
        """Setup the basic plot configuration"""
        plt.style.use('dark_background')
        self.ax.set_xlabel('Time Index', color='white')
        self.ax.set_ylabel('Price', color='white')
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.tick_params(colors='white')
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#1a1a1a')
        
    def setup_lines(self):
        """Initialize all plot lines"""
        # Price line
        self.price_line, = self.ax.plot([], [], linestyle='-', color='#00ff00', label='Price', linewidth=1)
        
        # Peak markers
        self.peaks_min_inf, = self.ax.plot([], [], linestyle='', marker='o', color='#00ccff', 
                                          label='Min Peaks', markersize=6)
        self.peaks_max_inf, = self.ax.plot([], [], linestyle='', marker='o', color='orange', 
                                          label='Max Peaks', markersize=6)
        
        # Median line
        self.median_line, = self.ax.plot([], [], linestyle='-', color='yellow', 
                                        label='Median', linewidth=2, alpha=0.7)
        
        # Short trend line (new indicator)
        self.short_trend_line, = self.ax.plot([], [], linestyle='-', color='cyan', 
                                             label='Short Trend', linewidth=1, alpha=0.8)
        
        # Momentum line (new indicator)
        self.momentum_line, = self.ax.plot([], [], linestyle='-', color='magenta', 
                                          label='Momentum', linewidth=1, alpha=0.6)
        
        # Trigger markers
        self.trigger_buy, = self.ax.plot([], [], 'D', color='white', label='Buy Signal', 
                                        markersize=8, zorder=30)
        self.trigger_sell, = self.ax.plot([], [], 'd', color='blue', label='Sell Signal', 
                                         markersize=8, zorder=30)
        
        # Trade zones
        self.trade_zone_buy_line, = self.ax.plot([], [], '*', color='lime', 
                                                label='Trade Zone Buy', markersize=15, zorder=2)
        self.trade_zone_sell_line, = self.ax.plot([], [], '*', color='red', 
                                                 label='Trade Zone Sell', markersize=15, zorder=2)
        
        # Legend
        self.ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', 
                      loc='upper left', fontsize=9)
        
    def on_file_select(self, event):
        """Handle file selection from listbox"""
        selection = self.listbox.curselection()
        if selection:
            selected_file = self.csv_files[selection[0]]
            print(f"DEBUG: File selected: {selected_file}")
            self.load_and_plot_data(selected_file)
            # Force full view after file change
            self.root.after(100, self.force_full_view)
    
    def load_and_plot_data(self, filename):
        """Load CSV data and update plot"""
        try:
            print(f"DEBUG: Attempting to load {filename}")
            
            # Check if file exists
            if not os.path.exists(filename):
                self.status_var.set(f"Error: {filename} not found")
                print(f"DEBUG: File {filename} not found")
                return
            
            print(f"DEBUG: File {filename} exists, loading data...")
            
            # Load data
            df = pd.read_csv(filename)
            
            print(f"DEBUG: Loaded {len(df)} rows from {filename}")
            
            if df.empty:
                self.status_var.set(f"Error: {filename} is empty")
                print(f"DEBUG: File {filename} is empty")
                return
            
            # Check if data has changed (for auto-updates)
            data_changed = False
            if self.current_data is not None:
                # Compare row count or last timestamp
                if len(df) != len(self.current_data):
                    data_changed = True
                elif len(df) > 0 and len(self.current_data) > 0:
                    # Compare last row
                    if df.iloc[-1].to_dict() != self.current_data.iloc[-1].to_dict():
                        data_changed = True
            else:
                data_changed = True
            
            print(f"DEBUG: Data changed: {data_changed}")
            
            # Update current data
            self.current_data = df
            self.current_file = filename
            
            print(f"DEBUG: Updating plot data...")
            
            # Update plot
            self.update_plot_data()
            
            print(f"DEBUG: Plot updated successfully")
            
            # Update status
            if data_changed:
                self.status_var.set(f"Updated {filename} - {len(df)} rows at {datetime.now().strftime('%H:%M:%S')}")
            else:
                self.status_var.set(f"No changes in {filename} - {len(df)} rows")
            
        except Exception as e:
            self.status_var.set(f"Error loading {filename}: {str(e)}")
            print(f"DEBUG: Error loading {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_plot_data(self):
        """Update the plot with current data"""
        print(f"DEBUG: update_plot_data called for file: {self.current_file}")
        
        if self.current_data is None or self.current_data.empty:
            print(f"DEBUG: No data to plot")
            return
        
        df = self.current_data
        print(f"DEBUG: Plotting {len(df)} rows from {self.current_file}")
        
        # Save current zoom state (only if we have existing data)
        current_xlim = None
        current_ylim = None
        zoomed = False
        viewing_end = False
        
        try:
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            zoomed = (current_xlim != (0, 1) or current_ylim != (0, 1))  # Check if zoomed
            print(f"DEBUG: Zoom state - xlim: {current_xlim}, ylim: {current_ylim}, zoomed: {zoomed}")
            
            # Check if we're viewing the end of the data (for auto-scroll)
            if zoomed and len(df) > 0 and self.auto_scroll_enabled:
                # If we're zoomed and viewing near the end, we'll auto-scroll to new data
                current_end = current_xlim[1]
                data_length = len(df)
                viewing_end = (current_end >= data_length * 0.9)  # If viewing last 10% of data
        except Exception as e:
            # If there's an error getting current limits, assume no zoom
            print(f"DEBUG: Error getting zoom state: {e}")
            zoomed = False
            viewing_end = False
        
        print(f"DEBUG: Starting to update plot elements...")
        
        # Clear previous tolerance spans
        if hasattr(self, 'tolerance_spans'):
            for span in self.tolerance_spans:
                span.remove()
        self.tolerance_spans = []
        
        # Update price line
        self.price_line.set_data(range(len(df)), df['bidclose'])
        print(f"DEBUG: Price line updated with {len(df)} points")
        
        # Update median line if exists
        if 'median_bidclose' in df.columns:
            self.median_line.set_data(range(len(df)), df['median_bidclose'])
        else:
            self.median_line.set_data([], [])
        print(f"DEBUG: Median line updated")
        
        # Update peaks
        if 'peaks_min' in df.columns:
            min_peaks = df[df['peaks_min'] == 1]
            self.peaks_min_inf.set_data(min_peaks.index, min_peaks['bidclose'])
        else:
            self.peaks_min_inf.set_data([], [])
            
        if 'peaks_max' in df.columns:
            max_peaks = df[df['peaks_max'] == 1]
            self.peaks_max_inf.set_data(max_peaks.index, max_peaks['bidclose'])
        else:
            self.peaks_max_inf.set_data([], [])
        print(f"DEBUG: Peaks updated")
        
        # Update short trend line
        if 'short_trend' in df.columns:
            self.short_trend_line.set_data(range(len(df)), df['short_trend'])
        else:
            self.short_trend_line.set_data([], [])
        print(f"DEBUG: Short trend line updated")

        # Update momentum line
        if 'momentum' in df.columns:
            self.momentum_line.set_data(range(len(df)), df['momentum'])
        else:
            self.momentum_line.set_data([], [])
        print(f"DEBUG: Momentum line updated")
        
        # Update trade zones
        if 'trade_zone_buy' in df.columns:
            buy_zones = df[df['trade_zone_buy'] == 1]
            self.trade_zone_buy_line.set_data(buy_zones.index, buy_zones['bidclose'])
        else:
            self.trade_zone_buy_line.set_data([], [])
            
        if 'trade_zone_sell' in df.columns:
            sell_zones = df[df['trade_zone_sell'] == 1]
            self.trade_zone_sell_line.set_data(sell_zones.index, sell_zones['bidclose'])
        else:
            self.trade_zone_sell_line.set_data([], [])
        print(f"DEBUG: Trade zones updated")
        
        # Update tolerance zones
        if 'tolerance_zone_buy' in df.columns:
            tolerance_buy_indices = df[df['tolerance_zone_buy'] == 1].index
            if len(tolerance_buy_indices) > 0:
                tolerance_buy_ranges = self._create_continuous_ranges(tolerance_buy_indices)
                for start_idx, end_idx in tolerance_buy_ranges:
                    span = self.ax.axvspan(start_idx, end_idx, color='lime', alpha=0.15, zorder=0)
                    self.tolerance_spans.append(span)
        
        if 'tolerance_zone_sell' in df.columns:
            tolerance_sell_indices = df[df['tolerance_zone_sell'] == 1].index
            if len(tolerance_sell_indices) > 0:
                tolerance_sell_ranges = self._create_continuous_ranges(tolerance_sell_indices)
                for start_idx, end_idx in tolerance_sell_ranges:
                    span = self.ax.axvspan(start_idx, end_idx, color='red', alpha=0.15, zorder=0)
                    self.tolerance_spans.append(span)
        print(f"DEBUG: Tolerance zones updated")
        
        # Update signals
        if 'signal' in df.columns:
            buy_signals = df[df['signal'] == 1]
            sell_signals = df[df['signal'] == -1]
            self.trigger_buy.set_data(buy_signals.index, buy_signals['bidclose'])
            self.trigger_sell.set_data(sell_signals.index, sell_signals['bidclose'])
        else:
            self.trigger_buy.set_data([], [])
            self.trigger_sell.set_data([], [])
        print(f"DEBUG: Signals updated")
        
        # Update plot limits
        if zoomed and current_xlim is not None and current_ylim is not None:
            # If zoomed, try to maintain the zoom level
            try:
                if len(df) > 0:
                    if viewing_end:
                        # Auto-scroll to show new data at the end
                        new_data_length = len(df)
                        window_size = current_xlim[1] - current_xlim[0]
                        new_xlim = (new_data_length - window_size, new_data_length)
                        self.ax.set_xlim(new_xlim)
                        self.ax.set_ylim(current_ylim)
                    else:
                        # Keep the same zoom level
                        self.ax.set_xlim(current_xlim)
                        self.ax.set_ylim(current_ylim)
                else:
                    # Fallback to full view if no data
                    self.ax.set_xlim(0, len(df))
                    if len(df) > 0:
                        price_min = df['bidclose'].min()
                        price_max = df['bidclose'].max()
                        margin = (price_max - price_min) * 0.01
                        self.ax.set_ylim(price_min - margin, price_max + margin)
            except:
                # If there's an error maintaining zoom, reset to full view
                self.ax.set_xlim(0, len(df))
                if len(df) > 0:
                    price_min = df['bidclose'].min()
                    price_max = df['bidclose'].max()
                    margin = (price_max - price_min) * 0.01
                    self.ax.set_ylim(price_min - margin, price_max + margin)
        else:
            # If not zoomed, show full view
            self.ax.set_xlim(0, len(df))
            if len(df) > 0:
                price_min = df['bidclose'].min()
                price_max = df['bidclose'].max()
                margin = (price_max - price_min) * 0.01
                self.ax.set_ylim(price_min - margin, price_max + margin)
                print(f"DEBUG: Price range - Min: {price_min}, Max: {price_max}, Margin: {margin}")
                print(f"DEBUG: Y-axis limits set to: {price_min - margin} to {price_max + margin}")
        
        # Force refresh of the plot
        self.ax.figure.canvas.draw_idle()
        
        # Update title
        self.ax.set_title(f"{self.current_file} - {len(df)} candles", color='white', fontsize=12)
        
        # Redraw canvas
        self.canvas.draw()
        print(f"DEBUG: Canvas redrawn for {self.current_file}")
    
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
    
    def start_auto_update(self):
        """Start the auto-update thread"""
        self.update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
        self.update_thread.start()
    
    def _auto_update_loop(self):
        """Auto-update loop that runs in a separate thread"""
        while not self.stop_update:
            if self.auto_update_enabled and self.current_file:
                try:
                    # Schedule the update on the main thread
                    self.root.after(0, self._safe_update_data)
                except Exception as e:
                    print(f"Error in auto-update: {e}")
            
            # Wait for the update interval
            time.sleep(self.update_interval)
    
    def _safe_update_data(self):
        """Safely update data from the main thread"""
        if self.current_file and os.path.exists(self.current_file):
            try:
                # Reload and update the current file
                self.load_and_plot_data(self.current_file)
                self.status_var.set(f"Auto-updated {self.current_file} at {datetime.now().strftime('%H:%M:%S')}")
            except Exception as e:
                self.status_var.set(f"Auto-update error: {str(e)}")
    
    def toggle_auto_update(self):
        """Toggle auto-update on/off"""
        self.auto_update_enabled = self.auto_update_var.get()
        if self.auto_update_enabled:
            self.status_var.set("Auto-update enabled")
        else:
            self.status_var.set("Auto-update disabled")
    
    def toggle_auto_scroll(self):
        """Toggle auto-scroll on/off"""
        self.auto_scroll_enabled = self.auto_scroll_var.get()
        if self.auto_scroll_enabled:
            self.status_var.set("Auto-scroll enabled")
        else:
            self.status_var.set("Auto-scroll disabled")
    
    def manual_update(self):
        """Manually update the current file"""
        if self.current_file:
            self.load_and_plot_data(self.current_file)
            self.status_var.set(f"Manually updated {self.current_file} at {datetime.now().strftime('%H:%M:%S')}")
        else:
            self.status_var.set("No file selected for update")
    
    def force_full_view(self):
        """Force a full view of the data to ensure price lines are visible"""
        if self.current_data is not None and not self.current_data.empty:
            df = self.current_data
            print(f"DEBUG: Force full view called for {len(df)} rows")
            
            # Clear any existing zoom
            self.ax.set_xlim(0, len(df))
            if len(df) > 0:
                price_min = df['bidclose'].min()
                price_max = df['bidclose'].max()
                margin = (price_max - price_min) * 0.05  # 5% margin
                self.ax.set_ylim(price_min - margin, price_max + margin)
                print(f"DEBUG: Forced full view - Price range: {price_min} to {price_max}")
                print(f"DEBUG: Y-axis limits set to: {price_min - margin} to {price_max + margin}")
            
            # Force canvas redraw
            self.canvas.draw()
            self.status_var.set(f"Forced full view - {len(df)} candles")
            print(f"DEBUG: Full view applied successfully")
        else:
            print(f"DEBUG: No data available for full view")
            self.status_var.set("No data available for full view")
    
    def run(self):
        """Start the GUI application"""
        # Schedule initial data load after GUI is ready
        if self.csv_files:
            self.root.after(100, self._load_initial_data)
        
        # Start the main loop
        self.root.mainloop()
        
        # Clean up when closing
        self.stop_update = True
    
    def _load_initial_data(self):
        """Load initial data after GUI is ready"""
        if self.csv_files:
            self.listbox.selection_set(0)
            self.load_and_plot_data(self.csv_files[0])
            # Force full view after initial load
            self.root.after(200, self.force_full_view)

def run_single_plotter():
    """Run the single window plotter"""
    plotter = ForexPlotterGUI()
    plotter.run()

if __name__ == "__main__":
    run_single_plotter()