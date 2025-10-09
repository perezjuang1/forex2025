import pandas as pd
import numpy as np
from PriceAnalyzer import PriceAnalyzer
from TradingConfiguration import TradingConfig
import os

class SimpleBacktest:
    def __init__(self, instrument, timeframe="m1"):
        self.instrument = instrument
        self.timeframe = timeframe
        self.trades = []
        self.equity_curve = []
        
    def load_data(self):
        fileName = os.path.join('data', self.instrument.replace("/", "_") + "_" + self.timeframe + ".csv")
        if not os.path.exists(fileName):
            print(f"No data file found: {fileName}")
            return None
        df = pd.read_csv(fileName, index_col=0)
        return df
    
    def run_backtest(self):
        df = self.load_data()
        if df is None or df.empty:
            return None
        
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {self.instrument}")
        print(f"{'='*60}")
        print(f"Total candles: {len(df)}")
        
        initial_balance = 10000
        balance = initial_balance
        position = None  # {'type': 'B'/'S', 'entry_price': float, 'entry_idx': int}
        
        stop_pips = TradingConfig.stop
        limit_pips = TradingConfig.limit
        
        pip_value = 0.0001 if 'JPY' not in self.instrument else 0.01
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['bidclose']
            
            # Check if position needs to close on stop/limit
            if position:
                entry_price = position['entry_price']
                if position['type'] == 'B':
                    # Check stop and limit for BUY
                    stop_price = entry_price - (stop_pips * pip_value)
                    limit_price = entry_price + (limit_pips * pip_value)
                    
                    if current_price <= stop_price:
                        pnl = -stop_pips * 10  # $10 per pip per lot
                        balance += pnl
                        self.trades.append({
                            'entry_idx': position['entry_idx'],
                            'exit_idx': i,
                            'type': 'B',
                            'entry_price': entry_price,
                            'exit_price': stop_price,
                            'pnl': pnl,
                            'exit_reason': 'stop'
                        })
                        position = None
                        continue
                    elif current_price >= limit_price:
                        pnl = limit_pips * 10
                        balance += pnl
                        self.trades.append({
                            'entry_idx': position['entry_idx'],
                            'exit_idx': i,
                            'type': 'B',
                            'entry_price': entry_price,
                            'exit_price': limit_price,
                            'pnl': pnl,
                            'exit_reason': 'limit'
                        })
                        position = None
                        continue
                        
                else:  # SELL
                    stop_price = entry_price + (stop_pips * pip_value)
                    limit_price = entry_price - (limit_pips * pip_value)
                    
                    if current_price >= stop_price:
                        pnl = -stop_pips * 10
                        balance += pnl
                        self.trades.append({
                            'entry_idx': position['entry_idx'],
                            'exit_idx': i,
                            'type': 'S',
                            'entry_price': entry_price,
                            'exit_price': stop_price,
                            'pnl': pnl,
                            'exit_reason': 'stop'
                        })
                        position = None
                        continue
                    elif current_price <= limit_price:
                        pnl = limit_pips * 10
                        balance += pnl
                        self.trades.append({
                            'entry_idx': position['entry_idx'],
                            'exit_idx': i,
                            'type': 'S',
                            'entry_price': entry_price,
                            'exit_price': limit_price,
                            'pnl': pnl,
                            'exit_reason': 'limit'
                        })
                        position = None
                        continue
            
            # Check for signal-based exit (opposite peak)
            if position:
                if position['type'] == 'B' and row['peaks_max'] == 1:
                    pnl = ((current_price - position['entry_price']) / pip_value) * 10
                    balance += pnl
                    self.trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'type': 'B',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'exit_reason': 'peak_opposite'
                    })
                    position = None
                    
                elif position['type'] == 'S' and row['peaks_min'] == 1:
                    pnl = ((position['entry_price'] - current_price) / pip_value) * 10
                    balance += pnl
                    self.trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'type': 'S',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'exit_reason': 'peak_opposite'
                    })
                    position = None
            
            # Check for new entry signals
            if position is None:
                if row['signal'] == 1:  # BUY
                    position = {'type': 'B', 'entry_price': current_price, 'entry_idx': i}
                elif row['signal'] == -1:  # SELL
                    position = {'type': 'S', 'entry_price': current_price, 'entry_idx': i}
            
            self.equity_curve.append(balance)
        
        # Close any open position at end
        if position:
            pnl = 0
            if position['type'] == 'B':
                pnl = ((current_price - position['entry_price']) / pip_value) * 10
            else:
                pnl = ((position['entry_price'] - current_price) / pip_value) * 10
            balance += pnl
            self.trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': len(df)-1,
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'exit_reason': 'end_of_data'
            })
        
        return self.calculate_metrics(initial_balance, balance)
    
    def calculate_metrics(self, initial_balance, final_balance):
        if not self.trades:
            return {
                'total_trades': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'return_pct': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] < 0]
        
        metrics = {
            'total_trades': len(self.trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': final_balance - initial_balance,
            'return_pct': (final_balance - initial_balance) / initial_balance * 100,
            'avg_win': winners['pnl'].mean() if len(winners) > 0 else 0,
            'avg_loss': losers['pnl'].mean() if len(losers) > 0 else 0,
            'max_win': winners['pnl'].max() if len(winners) > 0 else 0,
            'max_loss': losers['pnl'].min() if len(losers) > 0 else 0,
            'stops': len(trades_df[trades_df['exit_reason'] == 'stop']),
            'limits': len(trades_df[trades_df['exit_reason'] == 'limit']),
            'peak_exits': len(trades_df[trades_df['exit_reason'] == 'peak_opposite'])
        }
        
        return metrics

def print_metrics(instrument, metrics):
    print(f"\nRESULTADOS {instrument}:")
    print(f"  Total trades: {metrics['total_trades']}")
    print(f"  Ganadores: {metrics['winners']} ({metrics['win_rate']:.1f}%)")
    print(f"  Perdedores: {metrics['losers']}")
    print(f"  P&L Total: ${metrics['total_pnl']:.2f}")
    print(f"  Retorno: {metrics['return_pct']:.2f}%")
    print(f"  Avg Win: ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
    print(f"  Max Win: ${metrics['max_win']:.2f}")
    print(f"  Max Loss: ${metrics['max_loss']:.2f}")
    print(f"  Salidas por Stop: {metrics['stops']}")
    print(f"  Salidas por Limit: {metrics['limits']}")
    print(f"  Salidas por Pico: {metrics['peak_exits']}")

if __name__ == "__main__":
    instruments = TradingConfig.get_instruments()
    
    print("\nBACKTESTING SIMPLE - DATOS ACTUALES")
    print("="*60)
    
    all_metrics = {}
    
    for instrument in instruments:
        bt = SimpleBacktest(instrument)
        metrics = bt.run_backtest()
        if metrics and metrics['total_trades'] > 0:
            all_metrics[instrument] = metrics
            print_metrics(instrument, metrics)
    
    if all_metrics:
        print(f"\n{'='*60}")
        print("RESUMEN GENERAL")
        print(f"{'='*60}")
        total_trades = sum(m['total_trades'] for m in all_metrics.values())
        total_pnl = sum(m['total_pnl'] for m in all_metrics.values())
        avg_win_rate = sum(m['win_rate'] for m in all_metrics.values()) / len(all_metrics)
        
        print(f"Total trades (todos): {total_trades}")
        print(f"P&L combinado: ${total_pnl:.2f}")
        print(f"Win rate promedio: {avg_win_rate:.1f}%")

