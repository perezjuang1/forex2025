import pandas as pd
import datetime as dt
import time
import threading
import pytz
import logging
import os
import traceback
from TradingConfiguration import TradingConfig

try:
    from FxcmConnection import RobotConnection
    ROBOT_CONNECTION_AVAILABLE = True
except ImportError:
    ROBOT_CONNECTION_AVAILABLE = False
    print("[TRADE MONITOR] Warning: RobotConnection not available - forex tr    def _should_auto_close_profit_trade(self, tracker):
        """Determine if profit trade should be auto-closed"""
        # Close if we have 2 consecutive declines or if we've lost more than 50% of max profits
        current_pips = tracker['current_pips']
        max_pips = tracker['max_pips']
        lost_percentage = (max_pips - current_pips) / max_pips if max_pips > 0 else 0
        
        return (tracker['declining_count'] >= 2 and max_pips > 0) or (lost_percentage >= 0.5 and max_pips > 15)
g features will be disabled")


class TradeMonitor:
    """
    Independent trade monitor that validates open operations every minute.
    Activates profit protection when position reaches at least 15 pips in profit.
    Activates loss protection when position reaches -5 pips or more in loss.
    Tracks pip gains/losses and automatically closes trades when pips start declining from peak (profit) or declining further into loss.
    """
    
    def __init__(self):
        self.monitor_log_file = os.path.join('logs', 'trade_monitor.csv')
        self.pip_tracker = {}  # {trade_id: {'max_pips': float, 'current_pips': float, 'open_price': float, 'side': str, 'instrument': str, 'protection_type': str, 'min_pips': float}}
        self.is_running = False
        self._setup_logging()
        self._ensure_monitor_log_file()
        
        if ROBOT_CONNECTION_AVAILABLE:
            self.robotconnection = RobotConnection(instrument="TRADE_MONITOR")
            self.connection = self.robotconnection.getConnection()
        else:
            self.robotconnection = None
            self.connection = None
            self._log_message("[TRADE MONITOR] No connection - simulation mode", level='warning')

    def _setup_logging(self):
        """Setup logging for the trade monitor"""
        if not os.path.exists('logs'):
            os.makedirs('logs', exist_ok=True)
        
        self.logger = logging.getLogger('TradeMonitor')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            log_file = f'logs/trade_monitor_{dt.datetime.now().strftime("%Y%m%d")}.log'
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Only log to file, not console to reduce spam
            formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)

    def _log_message(self, message: str, level: str = 'info'):
        """Log messages with single line format"""
        try:
            if level == 'info':
                self.logger.info(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'warning':
                self.logger.warning(message)
        except Exception as e:
            print(f"[TRADE MONITOR] Logging error: {e}")

    def _ensure_monitor_log_file(self):
        """Create the CSV log file for trade monitoring if it doesn't exist"""
        try:
            if not os.path.exists(self.monitor_log_file):
                import csv
                with open(self.monitor_log_file, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'trade_id', 'instrument', 'side', 'open_price', 
                        'current_price', 'current_pips', 'max_pips', 'action', 'details'
                    ])
        except Exception as e:
            self._log_message(f"[TRADE MONITOR] Error creating monitor log file: {e}", level='error')

    def _append_monitor_log(self, trade_id: str, instrument: str, side: str, open_price: float,
                           current_price: float, current_pips: float, max_pips: float, action: str, details: str = ''):
        """Append monitoring data to CSV log"""
        try:
            import csv
            europe_london = pytz.timezone('Europe/London')
            ts = dt.datetime.now(europe_london).isoformat()
            
            with open(self.monitor_log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts, trade_id, instrument, side, open_price,
                    current_price, current_pips, max_pips, action, details
                ])
        except Exception as e:
            self._log_message(f"[TRADE MONITOR] Error appending to monitor log: {e}", level='error')

    def get_pip_value(self, instrument: str) -> float:
        """Calculate pip value based on instrument type"""
        # For JPY pairs, pip is 0.01, for others it's 0.0001
        if 'JPY' in instrument:
            return 0.01
        else:
            return 0.0001

    def calculate_pips(self, instrument: str, side: str, open_price: float, current_price: float) -> float:
        """Calculate pips gained/lost for a trade"""
        pip_value = self.get_pip_value(instrument)
        
        if side == 'B':  # BUY trade
            # For BUY: profit when current > open
            pip_diff = (current_price - open_price) / pip_value
        else:  # SELL trade
            # For SELL: profit when current < open
            pip_diff = (open_price - current_price) / pip_value
        
        return round(pip_diff, 2)

    def get_open_trades(self) -> list:
        """Get all open trades from FXCM"""
        trades = []
        try:
            if not ROBOT_CONNECTION_AVAILABLE or self.connection is None:
                self._log_message("[TRADE MONITOR] Connection not available for getting trades", level='warning')
                return trades
            
            trades_table = self.connection.get_table(self.connection.TRADES)
            if trades_table:
                for trade in trades_table:
                    trade_info = {
                        'trade_id': getattr(trade, 'trade_id', None),
                        'instrument': getattr(trade, 'instrument', None),
                        'side': getattr(trade, 'buy_sell', None),
                        'amount': getattr(trade, 'amount', 0),
                        'open_rate': getattr(trade, 'open_rate', 0),
                        'pl': getattr(trade, 'pl', 0),
                        'gross_pl': getattr(trade, 'gross_pl', 0)
                    }
                    trades.append(trade_info)
            
            self._log_message(f"[TRADE MONITOR] Found {len(trades)} open trades")
            return trades
            
        except Exception as e:
            self._log_message(f"[TRADE MONITOR] Error getting open trades: {e}", level='error')
            return trades

    def get_current_price(self, instrument: str) -> dict:
        """Get current bid/ask prices for an instrument"""
        try:
            if not ROBOT_CONNECTION_AVAILABLE or self.connection is None:
                self._log_message(f"[TRADE MONITOR] Connection not available for price of {instrument}", level='warning')
                return {'bid': 0, 'ask': 0}
            
            offers_table = self.connection.get_table(self.connection.OFFERS)
            for offer in offers_table:
                if getattr(offer, 'instrument', None) == instrument:
                    return {
                        'bid': getattr(offer, 'bid', 0),
                        'ask': getattr(offer, 'ask', 0)
                    }
            
            self._log_message(f"[TRADE MONITOR] No price found for {instrument}", level='warning')
            return {'bid': 0, 'ask': 0}
            
        except Exception as e:
            self._log_message(f"[TRADE MONITOR] Error getting current price for {instrument}: {e}", level='error')
            return {'bid': 0, 'ask': 0}

    def close_trade(self, trade_id: str, instrument: str, side: str):
        """Close a specific trade"""
        try:
            if not ROBOT_CONNECTION_AVAILABLE or self.connection is None:
                self._log_message(f"[TRADE MONITOR] Cannot close trade {trade_id} - no connection", level='error')
                return False
            
            # Get account ID
            accounts_response_reader = self.connection.get_table_reader(self.connection.ACCOUNTS)
            account_id = None
            for account in accounts_response_reader:
                account_id = account.account_id
                break
            
            if not account_id:
                self._log_message("[TRADE MONITOR] No account ID found", level='error')
                return False
            
            # Get trade details
            trades_table = self.connection.get_table(self.connection.TRADES)
            fxcorepy = self.robotconnection.fxcorepy
            
            for trade in trades_table:
                if getattr(trade, 'trade_id', None) == trade_id:
                    # Determine opposite side for closing
                    buy_sell = fxcorepy.Constants.SELL if trade.buy_sell == fxcorepy.Constants.BUY else fxcorepy.Constants.BUY
                    
                    # Create close order request
                    request = self.connection.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                        OFFER_ID=trade.offer_id,
                        ACCOUNT_ID=account_id,
                        BUY_SELL=buy_sell,
                        AMOUNT=trade.amount,
                        TRADE_ID=trade.trade_id
                    )
                    
                    self.connection.send_request_async(request)
                    self._log_message(f"[TRADE MONITOR] Closed trade {trade_id} - Instrument: {instrument}, Side: {side}, Amount: {trade.amount}")
                    
                    # Remove from tracker
                    if trade_id in self.pip_tracker:
                        del self.pip_tracker[trade_id]
                    
                    return True
            
            self._log_message(f"[TRADE MONITOR] Trade {trade_id} not found for closing", level='error')
            return False
            
        except Exception as e:
            self._log_message(f"[TRADE MONITOR] Error closing trade {trade_id}: {e}", level='error')
            return False

    def monitor_trades(self):
        """Main monitoring function - checks trades every minute"""
        try:
            trades = self.get_open_trades()
            
            if not trades:
                self._log_message("[TRADE MONITOR] No open trades to monitor")
                return
            
            for trade in trades:
                self._process_single_trade(trade)
                
        except Exception as e:
            self._log_message(f"[TRADE MONITOR] Error in monitor_trades: {e}", level='error')
            self._log_message(f"[TRADE MONITOR] Traceback: {traceback.format_exc()}", level='error')

    def _process_single_trade(self, trade):
        """Process monitoring for a single trade"""
        trade_id = trade['trade_id']
        instrument = trade['instrument']
        side = trade['side']
        open_price = trade['open_rate']
        
        if not all([trade_id, instrument, side, open_price]):
            return
        
        # Get current price
        prices = self.get_current_price(instrument)
        current_price = prices['bid'] if side == 'B' else prices['ask']
        
        if current_price == 0:
            return
        
        # Calculate current pips
        current_pips = self.calculate_pips(instrument, side, open_price, current_price)
        
        # Check if trade should be tracked (activate on 15+ pips profit OR -5 pips loss)
        if trade_id not in self.pip_tracker:
            if current_pips >= 15.0:
                # Activate profit protection
                action, details = self._initialize_trade_tracker(trade_id, current_pips, open_price, side, instrument, 'PROFIT')
            elif current_pips <= -5.0:
                # Activate loss protection
                action, details = self._initialize_trade_tracker(trade_id, current_pips, open_price, side, instrument, 'LOSS')
            else:
                # Trade doesn't meet activation threshold yet - skip monitoring
                self._log_message(f"[TRADE MONITOR] {trade_id} | {instrument} | {side} | Pips: {current_pips} | Status: WAITING (needs 15+ pips profit OR -5 pips loss to activate)")
                return
        else:
            action, details = self._update_trade_tracker(trade_id, current_pips, instrument, side)
        
        # Log the monitoring data
        tracker = self.pip_tracker.get(trade_id, {})
        max_pips = tracker.get('max_pips', current_pips)
        
        self._append_monitor_log(
            trade_id, instrument, side, open_price, current_price,
            current_pips, max_pips, action, details
        )
        
        self._log_message(f"[TRADE MONITOR] {trade_id} | {instrument} | {side} | Pips: {current_pips} | Max: {max_pips} | Action: {action}")

    def _initialize_trade_tracker(self, trade_id, current_pips, open_price, side, instrument, protection_type):
        """Initialize tracking for a new trade (profit protection at 15+ pips OR loss protection at -5 pips)"""
        self.pip_tracker[trade_id] = {
            'max_pips': current_pips if protection_type == 'PROFIT' else current_pips,
            'min_pips': current_pips if protection_type == 'LOSS' else current_pips,
            'current_pips': current_pips,
            'open_price': open_price,
            'side': side,
            'instrument': instrument,
            'declining_count': 0,
            'protection_type': protection_type
        }
        action = 'TRACK_START'
        if protection_type == 'PROFIT':
            details = f'Started PROFIT protection at 15+ pips threshold - Initial pips: {current_pips}'
        else:
            details = f'Started LOSS protection at -5 pips threshold - Initial pips: {current_pips}'
        return action, details

    def _update_trade_tracker(self, trade_id, current_pips, instrument, side):
        """Update tracking for an existing trade"""
        tracker = self.pip_tracker[trade_id]
        previous_pips = tracker['current_pips']
        
        # Update current pips
        tracker['current_pips'] = current_pips
        protection_type = tracker.get('protection_type', 'PROFIT')
        
        # Determine action based on protection type and pip movement
        if protection_type == 'PROFIT':
            # Profit protection logic
            if current_pips > tracker['max_pips']:
                return self._handle_new_pip_high(tracker, current_pips)
            elif current_pips < previous_pips:
                return self._handle_pip_decline(tracker, current_pips, trade_id, instrument, side)
            else:
                return self._handle_stable_pips(tracker, current_pips)
        else:
            # Loss protection logic
            if current_pips < tracker['min_pips']:
                return self._handle_new_pip_low(tracker, current_pips, trade_id, instrument, side)
            elif current_pips > previous_pips:
                return self._handle_loss_recovery(tracker, current_pips)
            else:
                return self._handle_stable_loss_pips(tracker, current_pips)

    def _handle_new_pip_high(self, tracker, current_pips):
        """Handle when trade reaches new pip high"""
        previous_max = tracker['max_pips']
        tracker['max_pips'] = current_pips
        tracker['declining_count'] = 0
        action = 'NEW_HIGH'
        details = f'New pip high reached: {current_pips} (previous max: {previous_max})'
        return action, details

    def _handle_pip_decline(self, tracker, current_pips, trade_id, instrument, side):
        """Handle when pips are declining"""
        tracker['declining_count'] += 1
        action = 'DECLINING'
        details = f'Pips declining: {current_pips} (max was: {tracker["max_pips"]}, declining count: {tracker["declining_count"]})'
        
        # Close trade if conditions are met
        if self._should_auto_close_trade(tracker):
            self._log_message(f"[TRADE MONITOR] AUTO-CLOSING trade {trade_id} - Pips declined from {tracker['max_pips']} to {current_pips}")
            success = self.close_trade(trade_id, instrument, side)
            action = 'AUTO_CLOSE'
            details = f'Auto-closed due to pip decline from {tracker["max_pips"]} to {current_pips} (Success: {success})'
        
        return action, details

    def _handle_stable_pips(self, tracker, current_pips):
        """Handle when pips are stable (not declining)"""
        tracker['declining_count'] = 0
        action = 'MONITOR'
        details = f'Monitoring - Current: {current_pips}, Max: {tracker["max_pips"]}'
        return action, details

    def _handle_new_pip_low(self, tracker, current_pips, trade_id, instrument, side):
        """Handle when trade reaches new pip low (deeper loss)"""
        previous_min = tracker['min_pips']
        tracker['min_pips'] = current_pips
        tracker['declining_count'] += 1
        action = 'NEW_LOW'
        details = f'New pip low reached: {current_pips} (previous min: {previous_min}, declining count: {tracker["declining_count"]})'
        
        # Close trade if conditions are met (3 consecutive deeper losses)
        if self._should_auto_close_loss_trade(tracker):
            self._log_message(f"[TRADE MONITOR] AUTO-CLOSING loss trade {trade_id} - Pips declined to {current_pips} from {previous_min}")
            success = self.close_trade(trade_id, instrument, side)
            action = 'AUTO_CLOSE_LOSS'
            details = f'Auto-closed due to loss decline to {current_pips} (Success: {success})'
        
        return action, details

    def _handle_loss_recovery(self, tracker, current_pips):
        """Handle when pips are recovering from loss"""
        tracker['declining_count'] = 0
        action = 'RECOVERY'
        details = f'Loss recovery - Current: {current_pips}, Min: {tracker["min_pips"]}'
        return action, details

    def _handle_stable_loss_pips(self, tracker, current_pips):
        """Handle when loss pips are stable (not getting worse)"""
        tracker['declining_count'] = 0
        action = 'MONITOR_LOSS'
        details = f'Monitoring loss - Current: {current_pips}, Min: {tracker["min_pips"]}'
        return action, details

    def _should_auto_close_trade(self, tracker):
        """Determine if profit trade should be auto-closed"""
        # Close if we have 3 consecutive declines or if we've lost more than 50% of max profits
        current_pips = tracker['current_pips']
        max_pips = tracker['max_pips']
        lost_percentage = (max_pips - current_pips) / max_pips if max_pips > 0 else 0
        
        return (tracker['declining_count'] >= 3 and max_pips > 0) or (lost_percentage >= 0.5 and max_pips > 15)

    def _should_auto_close_loss_trade(self, tracker):
        """Determine if loss trade should be auto-closed"""
        return tracker['declining_count'] >= 3 and tracker['min_pips'] < 0

    def start_monitoring(self):
        """Start the monitoring loop"""
        self.is_running = True
        self._log_message("[TRADE MONITOR] Starting trade monitoring - checking every minute")
        
        while self.is_running:
            try:
                current_time = dt.datetime.now()
                # Run monitoring at the start of each minute (when seconds = 0)
                if current_time.second == 0:
                    self._log_message(f"[TRADE MONITOR] Running monitoring cycle at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    self.monitor_trades()
                    time.sleep(1)  # Sleep 1 second to avoid running multiple times at second 0
                
                time.sleep(1)  # Check every second for timing
                
            except KeyboardInterrupt:
                self._log_message("[TRADE MONITOR] Monitoring stopped by user")
                break
            except Exception as e:
                self._log_message(f"[TRADE MONITOR] Error in monitoring loop: {e}", level='error')
                time.sleep(60)  # Wait a minute before retrying on error

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.is_running = False
        self._log_message("[TRADE MONITOR] Monitoring stopped")

    def get_monitoring_status(self) -> dict:
        """Get current monitoring status and statistics"""
        return {
            'is_running': self.is_running,
            'tracked_trades': len(self.pip_tracker),
            'tracker_details': self.pip_tracker.copy()
        }


def run_trade_monitor():
    """Function to run the trade monitor independently"""
    try:
        monitor = TradeMonitor()
        monitor.start_monitoring()
    except Exception as e:
        print(f"[TRADE MONITOR] Fatal error: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    print("[TRADE MONITOR] Starting Independent Trade Monitor...")
    run_trade_monitor()
