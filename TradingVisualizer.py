import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import os
import glob
from datetime import datetime
import threading
import time

class TradingVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trading Visualizer")
        self.csv_files = self.get_available_csv_files()
        
        # Initialize visibility flags
        self.visibility_flags = {
            'price': tk.BooleanVar(value=True),
            'peaks_min': tk.BooleanVar(value=True),
            'peaks_max': tk.BooleanVar(value=True),
            'signals': tk.BooleanVar(value=True),
            'medians': tk.BooleanVar(value=True),
        }
        
        # Initialize data attributes
        self.current_data = None
        self.current_file = None
        
        # Initialize auto-update variables
        self.update_interval = 120  # 2 minutes in seconds
        self.auto_update_enabled = True
        # Mirror for checkbox-controlled auto-scroll
        self.auto_scroll_enabled = True
        self.stop_update = False
        self.update_thread = None

        self.setup_gui()
        
    def get_available_csv_files(self):
        """Get all available CSV files in the 'data' directory"""
        csv_files = glob.glob(os.path.join('data', '*.csv'))
        return sorted(csv_files)
    
    def setup_gui(self):
        """Setup the GUI layout"""
        self.root.geometry("1400x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for file selection and controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # File selection
        ttk.Label(left_panel, text="CSV Files:").pack(anchor=tk.W, pady=(0, 5))
        self.listbox = tk.Listbox(left_panel, height=8)
        self.listbox.pack(fill=tk.X, pady=(0, 10))
        
        # Populate listbox
        for file in self.csv_files:
            self.listbox.insert(tk.END, file)
        
        # Bind selection event
        self.listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        # Visibility controls
        self.setup_visibility_controls(left_panel)
        
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
        
        # Lock zoom button
        self.lock_zoom_var = tk.BooleanVar(value=False)
        self.lock_zoom_checkbox = ttk.Checkbutton(
            scroll_frame, 
            text="Lock zoom (disable auto-scroll)", 
            variable=self.lock_zoom_var,
            command=self.toggle_zoom_lock
        )
        self.lock_zoom_checkbox.pack(side=tk.RIGHT)
        
        # Instructions
        instructions = ttk.Label(left_panel, text="Instructions:\n1. Select a CSV file from the list\n2. Use checkboxes to show/hide lines\n3. The plot will update automatically\n4. Use mouse to zoom and pan\n5. Check 'Lock zoom' to prevent auto-scroll\n6. Use toolbar buttons for zoom controls", 
                               font=('Arial', 9), justify=tk.LEFT)
        instructions.pack(pady=(10, 0))
    
    def setup_visibility_controls(self, parent):
        """Setup visibility checkboxes for plot lines"""
        # Visibility controls frame
        visibility_frame = ttk.LabelFrame(parent, text="Show/Hide Lines", padding=5)
        visibility_frame.pack(pady=(10, 0), fill=tk.X)
        
        # Create checkboxes for each line type
        checkbox_configs = [
            ('price', 'Price Line'),
            ('peaks_min', 'Min Peaks'),
            ('peaks_max', 'Max Peaks'),
            ('signals', 'Signals'),
            ('medians', 'Medians'),
        ]
        
        # Create checkboxes in a grid layout
        for i, (flag_name, label) in enumerate(checkbox_configs):
            row = i // 2
            col = i % 2
            
            checkbox = ttk.Checkbutton(
                visibility_frame,
                text=label,
                variable=self.visibility_flags[flag_name],
                command=self.update_plot_visibility
            )
            checkbox.grid(row=row, column=col, sticky='w', padx=5, pady=2)
        
        # Add "Show All" and "Hide All" buttons
        button_frame = ttk.Frame(visibility_frame)
        button_frame.grid(row=len(checkbox_configs)//2 + 1, column=0, columnspan=2, pady=(10, 0))
        
        show_all_btn = ttk.Button(button_frame, text="Show All", command=self.show_all_lines)
        show_all_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        hide_all_btn = ttk.Button(button_frame, text="Hide All", command=self.hide_all_lines)
        hide_all_btn.pack(side=tk.LEFT)
        
        # Add statistics panel
        self.setup_statistics_panel(parent)
    
    def setup_statistics_panel(self, parent):
        """Setup statistics panel to show zone analysis information"""
        # Statistics frame
        stats_frame = ttk.LabelFrame(parent, text="Zone Analysis Statistics", padding=5)
        stats_frame.pack(pady=(10, 0), fill=tk.X)
        
        # Create labels for statistics
        self.stats_labels = {}
        
        stats_configs = [
            ('total_peaks', 'Total Peaks:'),
            ('min_peaks', 'Min Peaks:'),
            ('max_peaks', 'Max Peaks:'),

        ]
        
        for i, (key, label) in enumerate(stats_configs):
            row = i // 2
            col = i % 2
            
            # Label
            ttk.Label(stats_frame, text=label, font=('Arial', 9, 'bold')).grid(
                row=row, column=col*2, sticky='w', padx=(5, 2), pady=2)
            
            # Value label
            value_label = ttk.Label(stats_frame, text="0", font=('Arial', 9))
            value_label.grid(row=row, column=col*2+1, sticky='w', padx=(0, 10), pady=2)
            self.stats_labels[key] = value_label
    
    def update_statistics(self, df):
        """Update statistics panel with current data analysis"""
        try:
            if df is None or df.empty:
                return
            
            # Calculate statistics
            total_min_peaks = df['peaks_min'].sum() if 'peaks_min' in df.columns else 0
            total_max_peaks = df['peaks_max'].sum() if 'peaks_max' in df.columns else 0
            total_peaks = total_min_peaks + total_max_peaks
            
            # Calculate flat zones statistics
            
            
            # Calculate percentage of flat zones
            total_rows = len(df)
    
            
            # Update labels
            self.stats_labels['total_peaks'].config(text=str(total_peaks))
            self.stats_labels['min_peaks'].config(text=str(total_min_peaks))
            self.stats_labels['max_peaks'].config(text=str(total_max_peaks))
            
            
        except Exception as e:
            print(f'Error updating statistics: {str(e)}')
    
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
        """Initialize plot lines - price, peaks, signals and medians"""
        self.price_line, = self.ax.plot([], [], linestyle='-', color='#00ff00', label='Price', linewidth=1)
        self.peaks_min_inf, = self.ax.plot([], [], linestyle='', marker='o', color='#ff69b4', label='Min Peaks', markersize=6)
        self.peaks_max_inf, = self.ax.plot([], [], linestyle='', marker='o', color='#32cd32', label='Max Peaks', markersize=6)
        self.buy_signals, = self.ax.plot([], [], linestyle='', marker='^', color='#00ff00', label='Buy Signal', markersize=8)
        self.sell_signals, = self.ax.plot([], [], linestyle='', marker='v', color='#ff0000', label='Sell Signal', markersize=8)
        
        # Median lines
        self.median_close_line, = self.ax.plot([], [], linestyle='--', color='#00ffff', label='Median Close High Upper', linewidth=1, alpha=0.6)
        self.median_open_line, = self.ax.plot([], [], linestyle='--', color='#ff00ff', label='Median Close Low Lower', linewidth=1, alpha=0.6)
        
        # Additional median lines with adjusted percentages
        self.median_high_upper_line, = self.ax.plot([], [], linestyle='-', color='#ffd700', label='Median High Upper', linewidth=2, alpha=0.9)
        self.median_low_lower_line, = self.ax.plot([], [], linestyle='-', color='#ff6600', label='Median Low Lower', linewidth=2, alpha=0.9)
        

        
        # Distanced median zones (vertical bars)

        
        # Legend
        self.ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', 
                      loc='upper left', fontsize=9)
        
    def on_file_select(self, event):
        """Handle file selection from listbox"""
        selection = self.listbox.curselection()
        if selection:
            selected_file = self.csv_files[selection[0]]
            self.load_and_plot_data(selected_file)
            # Force full view after file change
            self.root.after(100, self.force_full_view)
    
    def load_and_plot_data(self, filename):
        """Load CSV data and update plot"""
        try:
            # Check if file exists
            if not os.path.exists(filename):
                self.status_var.set(f"Error: {filename} not found")
                return
            
            # Load data
            df = pd.read_csv(filename)
            
            if df.empty:
                self.status_var.set(f"Error: {filename} is empty")
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
            
            # Update current data
            self.current_data = df
            self.current_file = filename
            
            # Update plot
            self.update_plot_data()
            
            # Update status
            if data_changed:
                self.status_var.set(f"Updated {filename} - {len(df)} rows at {datetime.now().strftime('%H:%M:%S')}")
            else:
                self.status_var.set(f"No changes in {filename} - {len(df)} rows")
            
        except Exception as e:
            self.status_var.set(f"Error loading {filename}: {str(e)}")
            print(f'Error loading {filename}: {str(e)}')
    
    def update_plot_data(self):
        """Update the plot with current data"""
        
        if self.current_data is None or self.current_data.empty:
            return
        
        df = self.current_data
        
        # Save current zoom state (only if we have existing data)
        current_xlim = None
        current_ylim = None
        zoomed = False
        viewing_end = False
        
        try:
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            
            # Better zoom detection - check if we're not at default limits
            
            # Check if we're actually zoomed (not at default view)
            if len(df) > 0:
                prices = df['bidclose'].dropna()
                if len(prices) == 0:
                    price_min, price_max = 0.0, 1.0
                else:
                    price_min = float(prices.min())
                    price_max = float(prices.max())
                if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
                    price_max = price_min + 1e-6
                margin = max((price_max - price_min) * 0.01, 1e-6)
                
                # Consider zoomed if xlim is not full range or ylim is significantly different
                x_range_ratio = (current_xlim[1] - current_xlim[0]) / len(df)
                zoomed = (x_range_ratio < 0.95)  # If viewing less than 95% of data
                
                # Check if we're viewing the end of the data (for auto-scroll)
                if zoomed and self.auto_scroll_enabled:
                    current_end = current_xlim[1]
                    data_length = len(df)
                    viewing_end = (current_end >= data_length * 0.9)  # If viewing last 10% of data
            else:
                zoomed = False
                viewing_end = False
                    
        except Exception as e:
            # If there's an error getting current limits, assume no zoom
            zoomed = False
            viewing_end = False
        
        # Update price line (always visible)
        if self.get_line_visibility('price'):
            self.price_line.set_data(range(len(df)), df['bidclose'])
            self.price_line.set_visible(True)
        else:
            self.price_line.set_visible(False)
        
        # Update peaks
        if 'peaks_min' in df.columns and self.get_line_visibility('peaks_min'):
            peaks_min_x = []
            peaks_min_y = []
            for i, val in enumerate(df['peaks_min']):
                if not pd.isna(val) and val == 1:
                    peaks_min_x.append(i)
                    peaks_min_y.append(df['bidclose'].iloc[i])
            self.peaks_min_inf.set_data(peaks_min_x, peaks_min_y)
            self.peaks_min_inf.set_visible(True)
        else:
            self.peaks_min_inf.set_visible(False)
            
        if 'peaks_max' in df.columns and self.get_line_visibility('peaks_max'):
            peaks_max_x = []
            peaks_max_y = []
            for i, val in enumerate(df['peaks_max']):
                if not pd.isna(val) and val == 1:
                    peaks_max_x.append(i)
                    peaks_max_y.append(df['bidclose'].iloc[i])
            self.peaks_max_inf.set_data(peaks_max_x, peaks_max_y)
            self.peaks_max_inf.set_visible(True)
        else:
            self.peaks_max_inf.set_visible(False)
            
        # Update medians
        if self.get_line_visibility('medians'):
            x_data = range(len(df))
                
            if 'median_close_hight_upper' in df.columns:
                self.median_close_line.set_data(x_data, df['median_close_hight_upper'])
                self.median_close_line.set_visible(True)
            else:
                self.median_close_line.set_visible(False)
                
            if 'median_open_low_lower' in df.columns:
                self.median_open_line.set_data(x_data, df['median_open_low_lower'])
                self.median_open_line.set_visible(True)
            else:
                self.median_open_line.set_visible(False)
            
            # Update additional median lines with adjusted percentages
            if 'median_high_upper' in df.columns:
                self.median_high_upper_line.set_data(x_data, df['median_high_upper'])
                self.median_high_upper_line.set_visible(True)
            else:
                self.median_high_upper_line.set_visible(False)
            
            if 'median_low_lower' in df.columns:
                self.median_low_lower_line.set_data(x_data, df['median_low_lower'])
                self.median_low_lower_line.set_visible(True)
            else:
                self.median_low_lower_line.set_visible(False)
        else:
            self.median_close_line.set_visible(False)
            self.median_open_line.set_visible(False)
            self.median_high_upper_line.set_visible(False)
            self.median_low_lower_line.set_visible(False)
            
        # Update signals
        if 'signal' in df.columns and self.get_line_visibility('signals'):
            buy_x = []
            buy_y = []
            sell_x = []
            sell_y = []
            for i, val in enumerate(df['signal']):
                if not pd.isna(val) and val == 1:  # Buy signal
                    buy_x.append(i)
                    buy_y.append(df['bidclose'].iloc[i])
                elif not pd.isna(val) and val == -1:  # Sell signal
                    sell_x.append(i)
                    sell_y.append(df['bidclose'].iloc[i])
            self.buy_signals.set_data(buy_x, buy_y)
            self.sell_signals.set_data(sell_x, sell_y)
            self.buy_signals.set_visible(True)
            self.sell_signals.set_visible(True)
        else:
            self.buy_signals.set_visible(False)
            self.sell_signals.set_visible(False)
        
        # Update plot limits with better zoom handling
        if len(df) > 0:
            prices = df['bidclose'].dropna()
            if len(prices) == 0:
                price_min, price_max = 0.0, 1.0
            else:
                price_min = float(prices.min())
                price_max = float(prices.max())
            if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
                price_max = price_min + 1e-6
            margin = max((price_max - price_min) * 0.01, 1e-6)
            
            if zoomed and current_xlim is not None and current_ylim is not None:
                # If zoomed, try to maintain the zoom level
                try:
                    if viewing_end and self.auto_scroll_enabled:
                        # Auto-scroll to show new data at the end
                        new_data_length = len(df)
                        window_size = current_xlim[1] - current_xlim[0]
                        new_xlim = (new_data_length - window_size, new_data_length)
                        self.ax.set_xlim(new_xlim)
                        self.ax.set_ylim(current_ylim)
                    else:
                        # Keep the same zoom level (respect manual zoom)
                        self.ax.set_xlim(current_xlim)
                        self.ax.set_ylim(current_ylim)
                except Exception as e:
                    # If there's an error maintaining zoom, reset to full view
                    self.ax.set_xlim(0, len(df))
                    self.ax.set_ylim(price_min - margin, price_max + margin)
            else:
                # If not zoomed, show full view
                self.ax.set_xlim(0, len(df))
                self.ax.set_ylim(price_min - margin, price_max + margin)
        else:
            # No data available
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
        
        # Force refresh of the plot
        self.ax.figure.canvas.draw_idle()
        
        # Update title
        self.ax.set_title(f"{self.current_file} - {len(df)} candles", color='white', fontsize=12)
        
        
        # Redraw canvas
        self.canvas.draw()
    

    
    def create_continuous_zones(self, indices):
        """Create continuous zones from individual indices"""
        if not indices:
            return []
        
        zones = []
        start = indices[0]
        end = indices[0]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                # Continuous
                end = indices[i]
            else:
                # Gap found, save current zone
                zones.append((start, end))
                start = indices[i]
                end = indices[i]
        
        # Add last zone
        zones.append((start, end))
        return zones
    
    def plot_median_zones(self, zones, median_type):
        """Plot zones as vertical bars with enhanced visualization"""
        try:
            if median_type == 'min':
                # Remove existing bars
                for artist in self.ax.get_children():
                    if hasattr(artist, 'get_label') and artist.get_label() == 'Near Min Low Median':
                        artist.remove()
                
                # Add new bars with enhanced styling
                for start, end in zones:
                    self.ax.axvspan(start, end, alpha=0.4, color='#ff4444', 
                                   label='Near Min Low Median', edgecolor='#cc0000', linewidth=1)
                    
                    # Add zone label in the middle
                    mid_point = (start + end) / 2
                    if end - start > 5:  # Only add label for zones wider than 5 points
                        self.ax.text(mid_point, self.ax.get_ylim()[1] * 0.95, 
                                    'SUPPORT', ha='center', va='top', 
                                    fontsize=8, fontweight='bold', color='#cc0000',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            else:  # max
                # Remove existing bars
                for artist in self.ax.get_children():
                    if hasattr(artist, 'get_label') and artist.get_label() == 'Near Max High Median':
                        artist.remove()
                
                # Add new bars with enhanced styling
                for start, end in zones:
                    self.ax.axvspan(start, end, alpha=0.4, color='#00ff00', 
                                   label='Near Max High Median', linewidth=1)
                    
                    # Add zone label in the middle
                    mid_point = (start + end) / 2
                    if end - start > 5:  # Only add label for zones wider than 5 points
                        self.ax.text(mid_point, self.ax.get_ylim()[0] * 1.05, 
                                    'RESISTANCE', ha='center', va='bottom', 
                                    fontsize=8, fontweight='bold', color='#00cc00',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    
        except Exception as e:
            print(f'Error plotting median zones: {str(e)}')
    

    
    def update_plot_visibility(self):
        """Update plot visibility based on checkbox states"""
        try:
            if self.current_data is not None:
                self.update_plot_data()
        except Exception as e:
            print(f'Error updating plot visibility: {str(e)}')
    
    def show_all_lines(self):
        """Show all plot lines"""
        for flag in self.visibility_flags.values():
            flag.set(True)
        self.update_plot_visibility()
    
    def hide_all_lines(self):
        """Hide all plot lines"""
        for flag in self.visibility_flags.values():
            flag.set(False)
        self.update_plot_visibility()
    
    def get_line_visibility(self, line_name):
        """Get visibility state for a specific line"""
        return self.visibility_flags.get(line_name, tk.BooleanVar(value=True)).get()
    
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
                    print(f'Error in auto-update: {str(e)}')
            
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
    
    def toggle_zoom_lock(self):
        """Toggle zoom lock on/off"""
        zoom_locked = self.lock_zoom_var.get()
        if zoom_locked:
            # When zoom is locked, disable auto-scroll
            self.auto_scroll_var.set(False)
            self.auto_scroll_enabled = False
            self.status_var.set("Zoom locked - auto-scroll disabled")
        else:
            # When zoom is unlocked, re-enable auto-scroll if it was enabled
            if self.auto_scroll_var.get():
                self.auto_scroll_enabled = True
                self.status_var.set("Zoom unlocked - auto-scroll enabled")
            else:
                self.status_var.set("Zoom unlocked - auto-scroll disabled")
    
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
            
            # Clear any existing zoom
            self.ax.set_xlim(0, len(df))
            if len(df) > 0:
                prices = df['bidclose'].dropna() if 'bidclose' in df.columns else pd.Series([], dtype=float)
                if len(prices) == 0:
                    price_min, price_max = 0.0, 1.0
                else:
                    price_min = float(prices.min())
                    price_max = float(prices.max())
                if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
                    price_max = price_min + 1e-6
                margin = max((price_max - price_min) * 0.05, 1e-6)  # 5% margin
                self.ax.set_ylim(price_min - margin, price_max + margin)
            
            # Force canvas redraw
            self.canvas.draw()
            self.status_var.set(f"Forced full view - {len(df)} candles")
        else:
            self.status_var.set("No data available for full view")
    
    
    def run(self):
        """Start the GUI application"""
        # Schedule initial data load after GUI is ready
        if self.csv_files:
            self.root.after(100, self._load_initial_data)
        
        # Start auto-update thread
        self.start_auto_update()
        
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

def run_single_visualizer():
    """Run the single window visualizer"""
    visualizer = TradingVisualizer()
    visualizer.run()

if __name__ == "__main__":
    run_single_visualizer() 