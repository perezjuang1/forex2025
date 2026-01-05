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

class ChartViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chart Viewer")
        self.csv_files = self.get_available_csv_files()
        
        # Initialize visibility flags - only indicators actually used in strategy
        self.visibility_flags = {
            # Price and basic indicators
            'price': tk.BooleanVar(value=True),
            'peaks_min': tk.BooleanVar(value=False),  # Used for signal generation but not critical to visualize
            'peaks_max': tk.BooleanVar(value=False),  # Used for signal generation but not critical to visualize
            'signals': tk.BooleanVar(value=True),
            # Trend indicators (all used in strategy)
            'ema_200': tk.BooleanVar(value=True),
            'ema_fast': tk.BooleanVar(value=True),
            'ema_slow': tk.BooleanVar(value=True),
            # Advanced indicators (all used in strategy)
            'rsi': tk.BooleanVar(value=True),
            'macd': tk.BooleanVar(value=True),
            # Market conditions
            'market_conditions': tk.BooleanVar(value=False),
        }
        
        # Initialize data attributes
        self.current_data = None
        self.current_file = None
        
        # Initialize auto-update variables
        self.update_interval = 120  # 2 minutes in seconds
        self.auto_update_enabled = True
        self.auto_scroll_enabled = True
        self.stop_update = False
        self.update_thread = None
        self.loading_thread = None
        self.is_loading = False
        
        # RSI independent window components
        self.rsi_window = None
        self.fig_rsi = None
        self.ax_rsi = None
        self.canvas_rsi = None
        
        # MACD independent window components
        self.macd_window = None
        self.fig_macd = None
        self.ax_macd = None
        self.canvas_macd = None
        
        # Market conditions independent window components (one for each indicator)
        self.liquidity_window = None
        self.fig_liquidity = None
        self.ax_liquidity = None
        self.canvas_liquidity = None
        
        self.volatility_window = None
        self.fig_volatility = None
        self.ax_volatility = None
        self.canvas_volatility = None
        
        self.movement_window = None
        self.fig_movement = None
        self.ax_movement = None
        self.canvas_movement = None
        
        self.spread_window = None
        self.fig_spread = None
        self.ax_spread = None
        self.canvas_spread = None
        
        self.quality_window = None
        self.fig_quality = None
        self.ax_quality = None
        self.canvas_quality = None
        
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
        
        # Create matplotlib figure with organized subplots
        # Layout: Price chart (main) - All indicators moved to independent windows
        self.fig = plt.figure(figsize=(14, 8))
        
        # Only price chart now, all indicators in separate windows
        self.ax = self.fig.add_subplot(1, 1, 1)  # Main price chart only
        
        self.setup_plot()
        self.setup_lines()
        
        # Create independent windows after main GUI is set up
        self.root.after(100, self.setup_rsi_window)
        self.root.after(150, self.setup_macd_window)
        self.root.after(200, self.setup_liquidity_window)
        self.root.after(250, self.setup_volatility_window)
        self.root.after(300, self.setup_movement_window)
        self.root.after(350, self.setup_spread_window)
        self.root.after(400, self.setup_quality_window)
        
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
        instructions = ttk.Label(left_panel, text="Instructions:\n1. Select a CSV file from the list\n2. Use checkboxes to show/hide lines\n3. The plot will update automatically\n4. Use mouse to zoom and pan\n5. Check 'Lock zoom' to prevent auto-scroll\n6. Use toolbar buttons for zoom controls\n7. RSI, MACD and Market Conditions indicators are displayed in separate windows", 
                               font=('Arial', 9), justify=tk.LEFT)
        instructions.pack(pady=(10, 0))
    
    def setup_rsi_window(self):
        """Setup independent window for RSI plot"""
        self.rsi_window = tk.Toplevel(self.root)
        self.rsi_window.title("RSI (Relative Strength Index) - Independent Plot")
        self.rsi_window.geometry("1000x400")
        
        # Create matplotlib figure for RSI
        self.fig_rsi = plt.figure(figsize=(10, 4))
        self.fig_rsi.patch.set_facecolor('#1a1a1a')
        
        # Create subplot for RSI
        self.ax_rsi = self.fig_rsi.add_subplot(1, 1, 1)
        self.ax_rsi.set_ylabel('RSI', color='white', fontsize=10)
        self.ax_rsi.grid(True, alpha=0.3, color='gray')
        self.ax_rsi.tick_params(colors='white')
        self.ax_rsi.set_facecolor('#1a1a1a')
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.set_xlabel('Time Index', color='white', fontsize=9)
        self.ax_rsi.set_title("RSI (Relative Strength Index)", color='white', fontsize=12, pad=10)
        self.ax_rsi.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overbought (70)')
        self.ax_rsi.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Oversold (30)')
        self.ax_rsi.axhline(y=50, color='gray', linestyle=':', linewidth=0.5, alpha=0.3, label='Neutral (50)')
        self.ax_rsi.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='upper right', fontsize=8)
        
        # Embed plot in tkinter
        self.canvas_rsi = FigureCanvasTkAgg(self.fig_rsi, self.rsi_window)
        self.canvas_rsi.draw()
        self.canvas_rsi.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Enable matplotlib navigation tools for RSI window
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar_rsi = NavigationToolbar2Tk(self.canvas_rsi, self.rsi_window)
        self.toolbar_rsi.update()
        
        # Create RSI line in the independent window
        self.rsi_line, = self.ax_rsi.plot([], [], linestyle='-', color='#ffa500', label='RSI', linewidth=1.5, alpha=0.9)
        
        # Handle window close event
        self.rsi_window.protocol("WM_DELETE_WINDOW", self.on_rsi_window_close)
    
    def setup_macd_window(self):
        """Setup independent window for MACD plot"""
        self.macd_window = tk.Toplevel(self.root)
        self.macd_window.title("MACD (Moving Average Convergence Divergence) - Independent Plot")
        self.macd_window.geometry("1000x500")
        
        # Create matplotlib figure for MACD
        self.fig_macd = plt.figure(figsize=(10, 5))
        self.fig_macd.patch.set_facecolor('#1a1a1a')
        
        # Create subplot for MACD
        self.ax_macd = self.fig_macd.add_subplot(1, 1, 1)
        self.ax_macd.set_ylabel('MACD', color='white', fontsize=10)
        self.ax_macd.grid(True, alpha=0.3, color='gray')
        self.ax_macd.tick_params(colors='white')
        self.ax_macd.set_facecolor('#1a1a1a')
        self.ax_macd.set_xlabel('Time Index', color='white', fontsize=9)
        self.ax_macd.set_title("MACD (Moving Average Convergence Divergence)", color='white', fontsize=12, pad=10)
        self.ax_macd.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
        
        # Create MACD lines in the independent window
        self.macd_line, = self.ax_macd.plot([], [], linestyle='-', color='#00ffff', label='MACD', linewidth=1.5, alpha=0.9)
        self.macd_signal_line, = self.ax_macd.plot([], [], linestyle='-', color='#ff00ff', label='Signal', linewidth=1.5, alpha=0.9)
        # MACD histogram will be created dynamically when data is available
        self.macd_hist_line = None
        
        # Add legend
        self.ax_macd.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='upper right', fontsize=8)
        
        # Embed plot in tkinter
        self.canvas_macd = FigureCanvasTkAgg(self.fig_macd, self.macd_window)
        self.canvas_macd.draw()
        self.canvas_macd.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Enable matplotlib navigation tools for MACD window
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar_macd = NavigationToolbar2Tk(self.canvas_macd, self.macd_window)
        self.toolbar_macd.update()
        
        # Handle window close event
        self.macd_window.protocol("WM_DELETE_WINDOW", self.on_macd_window_close)
    
    def setup_liquidity_window(self):
        """Setup independent window for Liquidity plot"""
        self.liquidity_window = tk.Toplevel(self.root)
        self.liquidity_window.title("Liquidity Score - Independent Plot")
        self.liquidity_window.geometry("1000x400")
        
        self.fig_liquidity = plt.figure(figsize=(10, 4))
        self.fig_liquidity.patch.set_facecolor('#1a1a1a')
        
        self.ax_liquidity = self.fig_liquidity.add_subplot(1, 1, 1)
        self.ax_liquidity.set_xlabel('Time Index', color='white', fontsize=10)
        self.ax_liquidity.set_ylabel('Liquidity Score', color='white', fontsize=10)
        self.ax_liquidity.grid(True, alpha=0.3, color='gray')
        self.ax_liquidity.tick_params(colors='white')
        self.ax_liquidity.set_facecolor('#1a1a1a')
        self.ax_liquidity.set_ylim(0, 1)
        self.ax_liquidity.set_title("Liquidity Score", color='white', fontsize=12, pad=10)
        self.ax_liquidity.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (30%)')
        
        self.liquidity_line, = self.ax_liquidity.plot([], [], linestyle='-', color='#00ffff', label='Liquidity', linewidth=2, alpha=0.9)
        self.ax_liquidity.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='upper right', fontsize=9)
        
        self.canvas_liquidity = FigureCanvasTkAgg(self.fig_liquidity, self.liquidity_window)
        self.canvas_liquidity.draw()
        self.canvas_liquidity.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar_liquidity = NavigationToolbar2Tk(self.canvas_liquidity, self.liquidity_window)
        self.toolbar_liquidity.update()
        
        self.liquidity_window.protocol("WM_DELETE_WINDOW", self.on_liquidity_window_close)
    
    def setup_volatility_window(self):
        """Setup independent window for Volatility plot"""
        self.volatility_window = tk.Toplevel(self.root)
        self.volatility_window.title("Volatility Score - Independent Plot")
        self.volatility_window.geometry("1000x400")
        
        self.fig_volatility = plt.figure(figsize=(10, 4))
        self.fig_volatility.patch.set_facecolor('#1a1a1a')
        
        self.ax_volatility = self.fig_volatility.add_subplot(1, 1, 1)
        self.ax_volatility.set_xlabel('Time Index', color='white', fontsize=10)
        self.ax_volatility.set_ylabel('Volatility Score', color='white', fontsize=10)
        self.ax_volatility.grid(True, alpha=0.3, color='gray')
        self.ax_volatility.tick_params(colors='white')
        self.ax_volatility.set_facecolor('#1a1a1a')
        self.ax_volatility.set_ylim(0, 1)
        self.ax_volatility.set_title("Volatility Score", color='white', fontsize=12, pad=10)
        self.ax_volatility.axhline(y=0.25, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (25%)')
        
        self.volatility_line, = self.ax_volatility.plot([], [], linestyle='-', color='#ff00ff', label='Volatility', linewidth=2, alpha=0.9)
        self.ax_volatility.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='upper right', fontsize=9)
        
        self.canvas_volatility = FigureCanvasTkAgg(self.fig_volatility, self.volatility_window)
        self.canvas_volatility.draw()
        self.canvas_volatility.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar_volatility = NavigationToolbar2Tk(self.canvas_volatility, self.volatility_window)
        self.toolbar_volatility.update()
        
        self.volatility_window.protocol("WM_DELETE_WINDOW", self.on_volatility_window_close)
    
    def setup_movement_window(self):
        """Setup independent window for Movement Frequency plot"""
        self.movement_window = tk.Toplevel(self.root)
        self.movement_window.title("Movement Frequency - Independent Plot")
        self.movement_window.geometry("1000x400")
        
        self.fig_movement = plt.figure(figsize=(10, 4))
        self.fig_movement.patch.set_facecolor('#1a1a1a')
        
        self.ax_movement = self.fig_movement.add_subplot(1, 1, 1)
        self.ax_movement.set_xlabel('Time Index', color='white', fontsize=10)
        self.ax_movement.set_ylabel('Movement Frequency', color='white', fontsize=10)
        self.ax_movement.grid(True, alpha=0.3, color='gray')
        self.ax_movement.tick_params(colors='white')
        self.ax_movement.set_facecolor('#1a1a1a')
        self.ax_movement.set_ylim(0, 1)
        self.ax_movement.set_title("Movement Frequency", color='white', fontsize=12, pad=10)
        self.ax_movement.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (30%)')
        
        self.movement_line, = self.ax_movement.plot([], [], linestyle='-', color='#ffff00', label='Movement Freq', linewidth=2, alpha=0.9)
        self.ax_movement.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='upper right', fontsize=9)
        
        self.canvas_movement = FigureCanvasTkAgg(self.fig_movement, self.movement_window)
        self.canvas_movement.draw()
        self.canvas_movement.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar_movement = NavigationToolbar2Tk(self.canvas_movement, self.movement_window)
        self.toolbar_movement.update()
        
        self.movement_window.protocol("WM_DELETE_WINDOW", self.on_movement_window_close)
    
    def setup_spread_window(self):
        """Setup independent window for Spread Score plot"""
        self.spread_window = tk.Toplevel(self.root)
        self.spread_window.title("Spread Score - Independent Plot")
        self.spread_window.geometry("1000x400")
        
        self.fig_spread = plt.figure(figsize=(10, 4))
        self.fig_spread.patch.set_facecolor('#1a1a1a')
        
        self.ax_spread = self.fig_spread.add_subplot(1, 1, 1)
        self.ax_spread.set_xlabel('Time Index', color='white', fontsize=10)
        self.ax_spread.set_ylabel('Spread Score', color='white', fontsize=10)
        self.ax_spread.grid(True, alpha=0.3, color='gray')
        self.ax_spread.tick_params(colors='white')
        self.ax_spread.set_facecolor('#1a1a1a')
        self.ax_spread.set_ylim(0, 1)
        self.ax_spread.set_title("Spread Score", color='white', fontsize=12, pad=10)
        
        self.spread_line, = self.ax_spread.plot([], [], linestyle='-', color='#00ff00', label='Spread', linewidth=2, alpha=0.9)
        self.ax_spread.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='upper right', fontsize=9)
        
        self.canvas_spread = FigureCanvasTkAgg(self.fig_spread, self.spread_window)
        self.canvas_spread.draw()
        self.canvas_spread.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar_spread = NavigationToolbar2Tk(self.canvas_spread, self.spread_window)
        self.toolbar_spread.update()
        
        self.spread_window.protocol("WM_DELETE_WINDOW", self.on_spread_window_close)
    
    def setup_quality_window(self):
        """Setup independent window for Quality Score plot"""
        self.quality_window = tk.Toplevel(self.root)
        self.quality_window.title("Market Quality Score - Independent Plot")
        self.quality_window.geometry("1000x400")
        
        self.fig_quality = plt.figure(figsize=(10, 4))
        self.fig_quality.patch.set_facecolor('#1a1a1a')
        
        self.ax_quality = self.fig_quality.add_subplot(1, 1, 1)
        self.ax_quality.set_xlabel('Time Index', color='white', fontsize=10)
        self.ax_quality.set_ylabel('Quality Score', color='white', fontsize=10)
        self.ax_quality.grid(True, alpha=0.3, color='gray')
        self.ax_quality.tick_params(colors='white')
        self.ax_quality.set_facecolor('#1a1a1a')
        self.ax_quality.set_ylim(0, 1)
        self.ax_quality.set_title("Market Quality Score", color='white', fontsize=12, pad=10)
        self.ax_quality.axhline(y=0.4, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (40%)')
        
        self.quality_line, = self.ax_quality.plot([], [], linestyle='-', color='#ffffff', label='Quality Score', linewidth=2, alpha=0.9)
        self.ax_quality.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='upper right', fontsize=9)
        
        self.canvas_quality = FigureCanvasTkAgg(self.fig_quality, self.quality_window)
        self.canvas_quality.draw()
        self.canvas_quality.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar_quality = NavigationToolbar2Tk(self.canvas_quality, self.quality_window)
        self.toolbar_quality.update()
        
        self.quality_window.protocol("WM_DELETE_WINDOW", self.on_quality_window_close)
    
    def setup_visibility_controls(self, parent):
        """Setup visibility checkboxes organized by category"""
        
        # ========== PRICE & BASIC INDICATORS ==========
        price_frame = ttk.LabelFrame(parent, text="Price & Basic", padding=5)
        price_frame.pack(pady=(10, 0), fill=tk.X)
        
        price_configs = [
            ('price', 'Price'),
            ('peaks_min', 'Min Peaks'),
            ('peaks_max', 'Max Peaks'),
            ('signals', 'Trading Signals'),
        ]
        
        for i, (flag_name, label) in enumerate(price_configs):
            row = i // 2
            col = i % 2
            checkbox = ttk.Checkbutton(
                price_frame,
                text=label,
                variable=self.visibility_flags[flag_name],
                command=self.update_plot_visibility
            )
            checkbox.grid(row=row, column=col, sticky='w', padx=5, pady=2)
        
        # ========== TREND INDICATORS ==========
        trend_frame = ttk.LabelFrame(parent, text="Trend Indicators", padding=5)
        trend_frame.pack(pady=(10, 0), fill=tk.X)
        
        trend_configs = [
            ('ema_200', 'EMA 200'),
            ('ema_fast', 'EMA Fast (12)'),
            ('ema_slow', 'EMA Slow (26)'),
        ]
        
        for i, (flag_name, label) in enumerate(trend_configs):
            row = i // 2
            col = i % 2
            checkbox = ttk.Checkbutton(
                trend_frame,
                text=label,
                variable=self.visibility_flags[flag_name],
                command=self.update_plot_visibility
            )
            checkbox.grid(row=row, column=col, sticky='w', padx=5, pady=2)
        
        # ========== ADVANCED INDICATORS ==========
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Indicators", padding=5)
        advanced_frame.pack(pady=(10, 0), fill=tk.X)
        
        advanced_configs = [
            ('rsi', 'RSI'),
            ('macd', 'MACD'),
        ]
        
        for i, (flag_name, label) in enumerate(advanced_configs):
            row = i // 2
            col = i % 2
            checkbox = ttk.Checkbutton(
                advanced_frame,
                text=label,
                variable=self.visibility_flags[flag_name],
                command=self.update_plot_visibility
            )
            checkbox.grid(row=row, column=col, sticky='w', padx=5, pady=2)
        
        # ========== MARKET CONDITIONS ==========
        market_frame = ttk.LabelFrame(parent, text="Market Conditions", padding=5)
        market_frame.pack(pady=(10, 0), fill=tk.X)
        
        market_checkbox = ttk.Checkbutton(
            market_frame,
            text='Show Market Quality',
            variable=self.visibility_flags['market_conditions'],
            command=self.update_plot_visibility
        )
        market_checkbox.pack(anchor=tk.W, padx=5, pady=2)
        
        # ========== CONTROL BUTTONS ==========
        button_frame = ttk.Frame(parent)
        button_frame.pack(pady=(10, 0), fill=tk.X)
        
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
            ('good_conditions', 'Good Conditions:'),
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
            
            # Calculate market condition statistics
            if 'is_good_condition' in df.columns:
                good_conditions = df['is_good_condition'].sum() if 'is_good_condition' in df.columns else 0
                total_rows = len(df)
                good_percentage = (good_conditions / total_rows * 100) if total_rows > 0 else 0
            else:
                good_conditions = 0
                good_percentage = 0
            
            # Update labels
            self.stats_labels['total_peaks'].config(text=str(total_peaks))
            self.stats_labels['min_peaks'].config(text=str(total_min_peaks))
            self.stats_labels['max_peaks'].config(text=str(total_max_peaks))
            
            # Update market condition stats if available
            if hasattr(self, 'stats_labels') and 'good_conditions' in self.stats_labels:
                self.stats_labels['good_conditions'].config(text=f"{good_conditions} ({good_percentage:.1f}%)")
            
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def setup_plot(self):
        """Setup the basic plot configuration for all subplots"""
        plt.style.use('dark_background')
        
        # Main price chart
        self.ax.set_ylabel('Price', color='white', fontsize=10)
        self.ax.set_xlabel('Time Index', color='white', fontsize=10)
        self.ax.grid(True, alpha=0.3, color='gray')
        self.ax.tick_params(colors='white')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.set_title('Price Chart with Indicators', color='white', fontsize=12, pad=10)
        
        # Figure background
        self.fig.patch.set_facecolor('#1a1a1a')
        
    def setup_lines(self):
        """Initialize plot lines for all indicators"""
        
        # ========== MAIN PRICE CHART ==========
        # Price line
        self.price_line, = self.ax.plot([], [], linestyle='-', color='#00ff00', label='Price', linewidth=1.5)
        
        # Peaks (used in strategy but optional visualization)
        self.peaks_min_inf, = self.ax.plot([], [], linestyle='', marker='o', color='#ff69b4', label='Min Peaks', markersize=5, alpha=0.6)
        self.peaks_max_inf, = self.ax.plot([], [], linestyle='', marker='o', color='#32cd32', label='Max Peaks', markersize=5, alpha=0.6)
        
        # Trading signals (critical - shows actual trade entries)
        self.buy_signals, = self.ax.plot([], [], linestyle='', marker='^', color='#00ff00', label='Buy Signal', markersize=12, markeredgewidth=2, markeredgecolor='white')
        self.sell_signals, = self.ax.plot([], [], linestyle='', marker='v', color='#ff0000', label='Sell Signal', markersize=12, markeredgewidth=2, markeredgecolor='white')
        
        # Trend indicators - EMA (all used in strategy)
        self.ema_200_line, = self.ax.plot([], [], linestyle='-', color='#ffa500', label='EMA 200', linewidth=2.5, alpha=0.9)
        self.ema_fast_line, = self.ax.plot([], [], linestyle='-', color='#00ffff', label='EMA Fast (12)', linewidth=1.5, alpha=0.8)
        self.ema_slow_line, = self.ax.plot([], [], linestyle='-', color='#ff00ff', label='EMA Slow (26)', linewidth=1.5, alpha=0.8)
        
        # ========== LEGENDS ==========
        # Main chart legend
        self.ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', 
                      loc='upper left', fontsize=8, ncol=2)
        
    def on_file_select(self, event):
        """Handle file selection from listbox"""
        selection = self.listbox.curselection()
        if selection:
            selected_file = self.csv_files[selection[0]]
            self.load_and_plot_data(selected_file)
    
    def load_and_plot_data(self, filename):
        """Load CSV data and update plot (non-blocking)"""
        if self.is_loading:
            return  # Already loading, skip
        
        # Start loading in background thread
        self.is_loading = True
        self.status_var.set(f"Loading {os.path.basename(filename)}...")
        self.loading_thread = threading.Thread(target=self._load_data_thread, args=(filename,), daemon=True)
        self.loading_thread.start()
    
    def _load_data_thread(self, filename):
        """Load data in background thread"""
        try:
            # Check if file exists
            if not os.path.exists(filename):
                self.root.after(0, lambda: self.status_var.set(f"Error: {filename} not found"))
                self.is_loading = False
                return
            
            # Load data (this is the blocking operation)
            df = pd.read_csv(filename)
            
            if df.empty:
                self.root.after(0, lambda: self.status_var.set(f"Error: {filename} is empty"))
                self.is_loading = False
                return
            
            # Convert numeric columns to proper types (only indicators used in strategy)
            numeric_columns = ['liquidity_score', 'volatility_score', 'movement_frequency', 
                              'spread_score', 'market_quality_score', 'bidclose', 'bidhigh', 
                              'bidlow', 'bidopen', 'tickqty', 'rsi', 'macd', 'macd_signal', 
                              'macd_hist', 'atr', 'ema_200', 'ema_fast', 'ema_slow',
                              'peaks_min', 'peaks_max', 'signal']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
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
            
            # Update current data and plot from main thread
            # Use a helper function to properly capture the DataFrame
            def update_plot():
                self._update_data_and_plot(df.copy(), filename, data_changed)
            self.root.after(0, update_plot)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error loading {filename}: {str(e)}"))
            import traceback
            traceback.print_exc()
            self.is_loading = False
    
    def _update_data_and_plot(self, df, filename, data_changed):
        """Update data and plot from main thread"""
        try:
            # Update current data
            self.current_data = df
            self.current_file = filename
            
            # Update plot
            self.update_plot_data()
            
            # Force full view after update
            self.force_full_view()
            
            # Update status
            if data_changed:
                self.status_var.set(f"Updated {os.path.basename(filename)} - {len(df)} rows at {datetime.now().strftime('%H:%M:%S')}")
            else:
                self.status_var.set(f"No changes in {os.path.basename(filename)} - {len(df)} rows")
            
        except Exception as e:
            self.status_var.set(f"Error updating plot: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_loading = False
    
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
            
        # Update EMA indicators
        x_data = list(range(len(df)))
        if 'ema_200' in df.columns and self.get_line_visibility('ema_200'):
            ema_200_data = df['ema_200'].ffill().fillna(df['bidclose']).tolist()
            self.ema_200_line.set_data(x_data, ema_200_data)
            self.ema_200_line.set_visible(True)
        else:
            self.ema_200_line.set_visible(False)
        
        if 'ema_fast' in df.columns and self.get_line_visibility('ema_fast'):
            ema_fast_data = df['ema_fast'].ffill().fillna(df['bidclose']).tolist()
            self.ema_fast_line.set_data(x_data, ema_fast_data)
            self.ema_fast_line.set_visible(True)
        else:
            self.ema_fast_line.set_visible(False)
        
        if 'ema_slow' in df.columns and self.get_line_visibility('ema_slow'):
            ema_slow_data = df['ema_slow'].ffill().fillna(df['bidclose']).tolist()
            self.ema_slow_line.set_data(x_data, ema_slow_data)
            self.ema_slow_line.set_visible(True)
        else:
            self.ema_slow_line.set_visible(False)
            
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
            
            # Sync all subplot x-axes with main chart (will be done later after all updates)
        else:
            # No data available
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
        
        # Update RSI in independent window
        if 'rsi' in df.columns and self.get_line_visibility('rsi') and self.ax_rsi is not None and hasattr(self, 'rsi_line'):
            rsi_data = df['rsi'].fillna(50).tolist()
            self.rsi_line.set_data(x_data, rsi_data)
            self.rsi_line.set_visible(True)
            # Update RSI plot limits
            self.ax_rsi.set_xlim(0, len(df))
            self.ax_rsi.set_ylim(0, 100)
            # Update RSI title with filename
            filename = os.path.basename(self.current_file) if self.current_file else "No file"
            self.ax_rsi.set_title(f"RSI (Relative Strength Index) - {filename}", color='white', fontsize=12, pad=10)
            # Force refresh of RSI plot
            if self.canvas_rsi is not None:
                self.canvas_rsi.draw_idle()
        elif self.ax_rsi is not None and hasattr(self, 'rsi_line'):
            self.rsi_line.set_visible(False)
            if self.canvas_rsi is not None:
                self.canvas_rsi.draw_idle()
        
        # Update MACD in independent window
        if 'macd' in df.columns and self.get_line_visibility('macd') and self.ax_macd is not None and hasattr(self, 'macd_line'):
            macd_data = df['macd'].fillna(0).tolist()
            macd_signal_data = df['macd_signal'].fillna(0).tolist() if 'macd_signal' in df.columns else [0] * len(df)
            macd_hist_data = df['macd_hist'].fillna(0).tolist() if 'macd_hist' in df.columns else [0] * len(df)
            
            self.macd_line.set_data(x_data, macd_data)
            self.macd_line.set_visible(True)
            
            if 'macd_signal' in df.columns:
                self.macd_signal_line.set_data(x_data, macd_signal_data)
                self.macd_signal_line.set_visible(True)
            else:
                self.macd_signal_line.set_visible(False)
            
            # Update histogram (bar chart)
            if self.macd_hist_line is not None:
                try:
                    self.macd_hist_line.remove()
                except:
                    pass
            
            # Create new histogram bars
            colors = ['green' if h >= 0 else 'red' for h in macd_hist_data]
            self.macd_hist_line = self.ax_macd.bar(x_data, macd_hist_data, width=0.8, color=colors, alpha=0.6)
            
            # Update MACD plot limits
            if len(macd_data) > 0:
                macd_min = min(macd_data + macd_signal_data + macd_hist_data)
                macd_max = max(macd_data + macd_signal_data + macd_hist_data)
                margin = (macd_max - macd_min) * 0.1 if macd_max != macd_min else 0.1
                self.ax_macd.set_xlim(0, len(df))
                self.ax_macd.set_ylim(macd_min - margin, macd_max + margin)
            
            # Update MACD title with filename
            filename = os.path.basename(self.current_file) if self.current_file else "No file"
            self.ax_macd.set_title(f"MACD (Moving Average Convergence Divergence) - {filename}", color='white', fontsize=12, pad=10)
            
            # Force refresh of MACD plot
            if self.canvas_macd is not None:
                self.canvas_macd.draw_idle()
        elif self.ax_macd is not None and hasattr(self, 'macd_line'):
            self.macd_line.set_visible(False)
            self.macd_signal_line.set_visible(False)
            if self.macd_hist_line is not None:
                try:
                    self.macd_hist_line.remove()
                    self.macd_hist_line = None
                except:
                    pass
            if self.canvas_macd is not None:
                self.canvas_macd.draw_idle()
        
        # Sync window x-axes with main chart (RSI, MACD and Market Conditions windows sync independently above)
        if len(df) > 0:
            xlim = self.ax.get_xlim()
            # Sync RSI window if it exists
            if self.ax_rsi is not None:
                self.ax_rsi.set_xlim(xlim)
                if self.canvas_rsi is not None:
                    self.canvas_rsi.draw()
            # Sync MACD window if it exists
            if self.ax_macd is not None:
                self.ax_macd.set_xlim(xlim)
                if self.canvas_macd is not None:
                    self.canvas_macd.draw()
            # Sync Market Conditions windows if they exist
            if self.ax_liquidity is not None:
                self.ax_liquidity.set_xlim(xlim)
                if self.canvas_liquidity is not None:
                    self.canvas_liquidity.draw()
            if self.ax_volatility is not None:
                self.ax_volatility.set_xlim(xlim)
                if self.canvas_volatility is not None:
                    self.canvas_volatility.draw()
            if self.ax_movement is not None:
                self.ax_movement.set_xlim(xlim)
                if self.canvas_movement is not None:
                    self.canvas_movement.draw()
            if self.ax_spread is not None:
                self.ax_spread.set_xlim(xlim)
                if self.canvas_spread is not None:
                    self.canvas_spread.draw()
            if self.ax_quality is not None:
                self.ax_quality.set_xlim(xlim)
                if self.canvas_quality is not None:
                    self.canvas_quality.draw()
        
        # Force refresh of the plot
        self.ax.figure.canvas.draw_idle()
        
        # Update statistics
        self.update_statistics(df)
        
        # Update titles
        filename = os.path.basename(self.current_file) if self.current_file else "No file"
        self.ax.set_title(f"{filename} - {len(df)} candles", color='white', fontsize=12)
        # RSI, MACD and Market Conditions titles updated in their respective update sections above
        
        # Update market condition indicators in independent windows
        if self.get_line_visibility('market_conditions'):
            self.update_market_condition_plots(df)
        
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
            print(f"Error plotting median zones: {e}")
    

    
    def update_plot_visibility(self):
        """Update plot visibility based on checkbox states"""
        try:
            if self.current_data is not None and hasattr(self, 'root') and self.root is not None:
                self.update_plot_data()
        except (AttributeError, tk.TclError, Exception) as e:
            # Window was destroyed or other error, ignore
            pass
    
    def show_all_lines(self):
        """Show all plot lines"""
        try:
            for flag in self.visibility_flags.values():
                flag.set(True)
            self.update_plot_visibility()
        except (AttributeError, tk.TclError):
            # Window was destroyed, ignore
            pass
    
    def hide_all_lines(self):
        """Hide all plot lines"""
        try:
            for flag in self.visibility_flags.values():
                flag.set(False)
            self.update_plot_visibility()
        except (AttributeError, tk.TclError):
            # Window was destroyed, ignore
            pass
    
    def get_line_visibility(self, line_name):
        """Get visibility state for a specific line"""
        try:
            if line_name in self.visibility_flags:
                return self.visibility_flags[line_name].get()
            else:
                # Default to True if flag doesn't exist
                return True
        except (AttributeError, tk.TclError):
            # Window was destroyed or tkinter is not available, default to True
            return True
    
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
        if self.current_file and os.path.exists(self.current_file) and not self.is_loading:
            try:
                # Reload and update the current file (non-blocking)
                self.load_and_plot_data(self.current_file)
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
    
    def update_market_condition_plots(self, df):
        """Update all market condition plots in independent windows"""
        if df is None or df.empty:
            return
        
        x_data = np.arange(len(df))
        filename = os.path.basename(self.current_file) if self.current_file else "No file"
        
        # Update Liquidity window
        if self.ax_liquidity is not None and hasattr(self, 'liquidity_line'):
            try:
                if 'liquidity_score' in df.columns:
                    liquidity_data = df['liquidity_score'].fillna(0.5).replace([np.inf, -np.inf], 0.5).values
                    self.liquidity_line.set_data(x_data, liquidity_data)
                    self.liquidity_line.set_visible(True)
                    
                    valid_values = [v for v in liquidity_data if not np.isnan(v) and np.isfinite(v)]
                    if valid_values:
                        y_min = min(valid_values)
                        y_max = max(valid_values)
                        if y_max == y_min:
                            center = y_min
                            y_min = max(0, center - 0.1)
                            y_max = min(1, center + 0.1)
                        else:
                            margin = (y_max - y_min) * 0.1
                            y_min = max(0, y_min - margin)
                            y_max = min(1, y_max + margin)
                    else:
                        y_min, y_max = 0, 1
                    
                    self.ax_liquidity.set_xlim(0, max(len(df) - 1, 1))
                    self.ax_liquidity.set_ylim(y_min, y_max)
                    self.ax_liquidity.set_title(f"Liquidity Score - {filename}", color='white', fontsize=12, pad=10)
                    if self.canvas_liquidity is not None:
                        self.canvas_liquidity.draw()
            except Exception as e:
                print(f"Error updating liquidity plot: {e}")
        
        # Update Volatility window
        if self.ax_volatility is not None and hasattr(self, 'volatility_line'):
            try:
                if 'volatility_score' in df.columns:
                    volatility_data = df['volatility_score'].fillna(0.5).replace([np.inf, -np.inf], 0.5).values
                    self.volatility_line.set_data(x_data, volatility_data)
                    self.volatility_line.set_visible(True)
                    
                    valid_values = [v for v in volatility_data if not np.isnan(v) and np.isfinite(v)]
                    if valid_values:
                        y_min = min(valid_values)
                        y_max = max(valid_values)
                        if y_max == y_min:
                            center = y_min
                            y_min = max(0, center - 0.1)
                            y_max = min(1, center + 0.1)
                        else:
                            margin = (y_max - y_min) * 0.1
                            y_min = max(0, y_min - margin)
                            y_max = min(1, y_max + margin)
                    else:
                        y_min, y_max = 0, 1
                    
                    self.ax_volatility.set_xlim(0, max(len(df) - 1, 1))
                    self.ax_volatility.set_ylim(y_min, y_max)
                    self.ax_volatility.set_title(f"Volatility Score - {filename}", color='white', fontsize=12, pad=10)
                    if self.canvas_volatility is not None:
                        self.canvas_volatility.draw()
            except Exception as e:
                print(f"Error updating volatility plot: {e}")
        
        # Update Movement Frequency window
        if self.ax_movement is not None and hasattr(self, 'movement_line'):
            try:
                if 'movement_frequency' in df.columns:
                    movement_data = df['movement_frequency'].fillna(0.5).replace([np.inf, -np.inf], 0.5).values
                    self.movement_line.set_data(x_data, movement_data)
                    self.movement_line.set_visible(True)
                    
                    valid_values = [v for v in movement_data if not np.isnan(v) and np.isfinite(v)]
                    if valid_values:
                        y_min = min(valid_values)
                        y_max = max(valid_values)
                        if y_max == y_min:
                            center = y_min
                            y_min = max(0, center - 0.1)
                            y_max = min(1, center + 0.1)
                        else:
                            margin = (y_max - y_min) * 0.1
                            y_min = max(0, y_min - margin)
                            y_max = min(1, y_max + margin)
                    else:
                        y_min, y_max = 0, 1
                    
                    self.ax_movement.set_xlim(0, max(len(df) - 1, 1))
                    self.ax_movement.set_ylim(y_min, y_max)
                    self.ax_movement.set_title(f"Movement Frequency - {filename}", color='white', fontsize=12, pad=10)
                    if self.canvas_movement is not None:
                        self.canvas_movement.draw()
            except Exception as e:
                print(f"Error updating movement plot: {e}")
        
        # Update Spread window
        if self.ax_spread is not None and hasattr(self, 'spread_line'):
            try:
                if 'spread_score' in df.columns:
                    spread_data = df['spread_score'].fillna(0.5).replace([np.inf, -np.inf], 0.5).values
                    self.spread_line.set_data(x_data, spread_data)
                    self.spread_line.set_visible(True)
                    
                    valid_values = [v for v in spread_data if not np.isnan(v) and np.isfinite(v)]
                    if valid_values:
                        y_min = min(valid_values)
                        y_max = max(valid_values)
                        if y_max == y_min:
                            center = y_min
                            y_min = max(0, center - 0.1)
                            y_max = min(1, center + 0.1)
                        else:
                            margin = (y_max - y_min) * 0.1
                            y_min = max(0, y_min - margin)
                            y_max = min(1, y_max + margin)
                    else:
                        y_min, y_max = 0, 1
                    
                    self.ax_spread.set_xlim(0, max(len(df) - 1, 1))
                    self.ax_spread.set_ylim(y_min, y_max)
                    self.ax_spread.set_title(f"Spread Score - {filename}", color='white', fontsize=12, pad=10)
                    if self.canvas_spread is not None:
                        self.canvas_spread.draw()
            except Exception as e:
                print(f"Error updating spread plot: {e}")
        
        # Update Quality Score window
        if self.ax_quality is not None and hasattr(self, 'quality_line'):
            try:
                if 'market_quality_score' in df.columns:
                    quality_data = df['market_quality_score'].fillna(0.5).replace([np.inf, -np.inf], 0.5).values
                    self.quality_line.set_data(x_data, quality_data)
                    self.quality_line.set_visible(True)
                    
                    valid_values = [v for v in quality_data if not np.isnan(v) and np.isfinite(v)]
                    if valid_values:
                        y_min = min(valid_values)
                        y_max = max(valid_values)
                        if y_max == y_min:
                            center = y_min
                            y_min = max(0, center - 0.1)
                            y_max = min(1, center + 0.1)
                        else:
                            margin = (y_max - y_min) * 0.1
                            y_min = max(0, y_min - margin)
                            y_max = min(1, y_max + margin)
                    else:
                        y_min, y_max = 0, 1
                    
                    self.ax_quality.set_xlim(0, max(len(df) - 1, 1))
                    self.ax_quality.set_ylim(y_min, y_max)
                    self.ax_quality.set_title(f"Market Quality Score - {filename}", color='white', fontsize=12, pad=10)
                    if self.canvas_quality is not None:
                        self.canvas_quality.draw()
            except Exception as e:
                print(f"Error updating quality plot: {e}")
    
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
            
            # Sync RSI window if it exists
            if self.ax_rsi is not None:
                self.ax_rsi.set_xlim(0, len(df))
                if self.canvas_rsi is not None:
                    self.canvas_rsi.draw()
            
            # Sync MACD window if it exists
            if self.ax_macd is not None:
                self.ax_macd.set_xlim(0, len(df))
                if self.canvas_macd is not None:
                    self.canvas_macd.draw()
            
            # Sync Market Conditions windows if they exist
            if self.ax_liquidity is not None:
                self.ax_liquidity.set_xlim(0, len(df))
                if self.canvas_liquidity is not None:
                    self.canvas_liquidity.draw()
            if self.ax_volatility is not None:
                self.ax_volatility.set_xlim(0, len(df))
                if self.canvas_volatility is not None:
                    self.canvas_volatility.draw()
            if self.ax_movement is not None:
                self.ax_movement.set_xlim(0, len(df))
                if self.canvas_movement is not None:
                    self.canvas_movement.draw()
            if self.ax_spread is not None:
                self.ax_spread.set_xlim(0, len(df))
                if self.canvas_spread is not None:
                    self.canvas_spread.draw()
            if self.ax_quality is not None:
                self.ax_quality.set_xlim(0, len(df))
                if self.canvas_quality is not None:
                    self.canvas_quality.draw()
            
            # Force canvas redraw
            self.canvas.draw()
            self.status_var.set(f"Forced full view - {len(df)} candles")
        else:
            self.status_var.set("No data available for full view")
    
    def on_rsi_window_close(self):
        """Handle RSI window close event"""
        if self.rsi_window is not None:
            self.rsi_window.destroy()
        self.rsi_window = None
        self.fig_rsi = None
        self.ax_rsi = None
        self.canvas_rsi = None
        if hasattr(self, 'rsi_line'):
            del self.rsi_line
    
    def on_macd_window_close(self):
        """Handle MACD window close event"""
        if self.macd_window is not None:
            self.macd_window.destroy()
        self.macd_window = None
        self.fig_macd = None
        self.ax_macd = None
        self.canvas_macd = None
        if hasattr(self, 'macd_line'):
            del self.macd_line
        if hasattr(self, 'macd_signal_line'):
            del self.macd_signal_line
        if hasattr(self, 'macd_hist_line'):
            self.macd_hist_line = None
    
    def on_liquidity_window_close(self):
        """Handle Liquidity window close event"""
        if self.liquidity_window is not None:
            self.liquidity_window.destroy()
        self.liquidity_window = None
        self.fig_liquidity = None
        self.ax_liquidity = None
        self.canvas_liquidity = None
        if hasattr(self, 'liquidity_line'):
            del self.liquidity_line
    
    def on_volatility_window_close(self):
        """Handle Volatility window close event"""
        if self.volatility_window is not None:
            self.volatility_window.destroy()
        self.volatility_window = None
        self.fig_volatility = None
        self.ax_volatility = None
        self.canvas_volatility = None
        if hasattr(self, 'volatility_line'):
            del self.volatility_line
    
    def on_movement_window_close(self):
        """Handle Movement Frequency window close event"""
        if self.movement_window is not None:
            self.movement_window.destroy()
        self.movement_window = None
        self.fig_movement = None
        self.ax_movement = None
        self.canvas_movement = None
        if hasattr(self, 'movement_line'):
            del self.movement_line
    
    def on_spread_window_close(self):
        """Handle Spread window close event"""
        if self.spread_window is not None:
            self.spread_window.destroy()
        self.spread_window = None
        self.fig_spread = None
        self.ax_spread = None
        self.canvas_spread = None
        if hasattr(self, 'spread_line'):
            del self.spread_line
    
    def on_quality_window_close(self):
        """Handle Quality Score window close event"""
        if self.quality_window is not None:
            self.quality_window.destroy()
        self.quality_window = None
        self.fig_quality = None
        self.ax_quality = None
        self.canvas_quality = None
        if hasattr(self, 'quality_line'):
            del self.quality_line
    
    
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

def run_chart_viewer():
    """Run the chart viewer application"""
    viewer = ChartViewer()
    viewer.run()

if __name__ == "__main__":
    run_chart_viewer() 