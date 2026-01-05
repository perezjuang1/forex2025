import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

def analyze_trade_losses(csv_file='logs/triggers_trades_open.csv'):
    """Analyze trading operations to identify loss patterns"""
    
    # Read the trade log
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("=" * 80)
    print("ANÁLISIS DE OPERACIONES - IDENTIFICACIÓN DE PÉRDIDAS")
    print("=" * 80)
    print(f"\nTotal de registros: {len(df)}")
    print(f"Rango de fechas: {df['timestamp'].min()} a {df['timestamp'].max()}\n")
    
    # Separate OPEN and CLOSE operations
    opens = df[df['action'] == 'OPEN'].copy()
    closes = df[df['action'] == 'CLOSE'].copy()
    
    print(f"Operaciones OPEN: {len(opens)}")
    print(f"Operaciones CLOSE: {len(closes)}\n")
    
    # Analyze by instrument
    print("\n" + "=" * 80)
    print("ANÁLISIS POR INSTRUMENTO")
    print("=" * 80)
    
    for instrument in df['instrument'].unique():
        inst_opens = opens[opens['instrument'] == instrument]
        inst_closes = closes[closes['instrument'] == instrument]
        
        print(f"\n{instrument}:")
        print(f"  OPEN: {len(inst_opens)} operaciones")
        print(f"  CLOSE: {len(inst_closes)} operaciones")
        
        # Count by side
        for side in ['B', 'S']:
            side_opens = inst_opens[inst_opens['side'] == side]
            side_closes = inst_closes[inst_closes['side'] == side]
            print(f"  {side} (BUY/SELL): OPEN={len(side_opens)}, CLOSE={len(side_closes)}")
    
    # Analyze close-open patterns (whipsaw detection)
    print("\n" + "=" * 80)
    print("DETECCIÓN DE WHIPSAWS (CIERRES SEGUIDOS DE APERTURAS OPUESTAS)")
    print("=" * 80)
    
    whipsaws = []
    for instrument in df['instrument'].unique():
        inst_df = df[df['instrument'] == instrument].sort_values('timestamp')
        
        for i in range(len(inst_df) - 1):
            current = inst_df.iloc[i]
            next_op = inst_df.iloc[i + 1]
            
            # If current is CLOSE and next is OPEN, check if they're opposite
            if current['action'] == 'CLOSE' and next_op['action'] == 'OPEN':
                time_diff = (next_op['timestamp'] - current['timestamp']).total_seconds()
                
                # Check if opposite sides (this indicates a whipsaw/loss scenario)
                if (current['side'] == 'B' and next_op['side'] == 'S') or \
                   (current['side'] == 'S' and next_op['side'] == 'B'):
                    whipsaws.append({
                        'instrument': instrument,
                        'close_time': current['timestamp'],
                        'open_time': next_op['timestamp'],
                        'time_diff_seconds': time_diff,
                        'close_side': current['side'],
                        'open_side': next_op['side'],
                        'close_price': current.get('price', None),
                        'open_price': next_op.get('price', None)
                    })
    
    whipsaw_df = pd.DataFrame(whipsaws)
    
    if len(whipsaw_df) > 0:
        print(f"\nTotal de whipsaws detectados: {len(whipsaw_df)}")
        print(f"\nTiempo promedio entre cierre y apertura opuesta: {whipsaw_df['time_diff_seconds'].mean():.1f} segundos")
        print(f"Tiempo mínimo: {whipsaw_df['time_diff_seconds'].min():.1f} segundos")
        print(f"Tiempo máximo: {whipsaw_df['time_diff_seconds'].max():.1f} segundos")
        
        print("\nDistribución por instrumento:")
        print(whipsaw_df.groupby('instrument').size())
        
        print("\nPrimeros 10 whipsaws:")
        print(whipsaw_df.head(10)[['instrument', 'close_time', 'open_time', 'time_diff_seconds', 'close_side', 'open_side']])
    else:
        print("\nNo se detectaron whipsaws en el patrón estándar")
    
    # Analyze multiple opens of same type (overtrading)
    print("\n" + "=" * 80)
    print("DETECCIÓN DE OVERTRADING (MÚLTIPLES APERTURAS DEL MISMO TIPO)")
    print("=" * 80)
    
    overtrading = []
    for instrument in df['instrument'].unique():
        inst_opens = opens[opens['instrument'] == instrument].sort_values('timestamp')
        
        # Group consecutive opens of same type
        current_group = []
        for idx, row in inst_opens.iterrows():
            if len(current_group) == 0:
                current_group = [row]
            else:
                last_row = current_group[-1]
                time_diff = (row['timestamp'] - last_row['timestamp']).total_seconds()
                
                # Same side and within 5 minutes (300 seconds)
                if row['side'] == last_row['side'] and time_diff < 300:
                    current_group.append(row)
                else:
                    # End of group
                    if len(current_group) > 1:
                        overtrading.append({
                            'instrument': instrument,
                            'side': current_group[0]['side'],
                            'count': len(current_group),
                            'first_time': current_group[0]['timestamp'],
                            'last_time': current_group[-1]['timestamp'],
                            'duration_seconds': (current_group[-1]['timestamp'] - current_group[0]['timestamp']).total_seconds()
                        })
                    current_group = [row]
        
        # Check last group
        if len(current_group) > 1:
            overtrading.append({
                'instrument': instrument,
                'side': current_group[0]['side'],
                'count': len(current_group),
                'first_time': current_group[0]['timestamp'],
                'last_time': current_group[-1]['timestamp'],
                'duration_seconds': (current_group[-1]['timestamp'] - current_group[0]['timestamp']).total_seconds()
            })
    
    overtrading_df = pd.DataFrame(overtrading)
    
    if len(overtrading_df) > 0:
        print(f"\nTotal de casos de overtrading detectados: {len(overtrading_df)}")
        print(f"Promedio de operaciones duplicadas: {overtrading_df['count'].mean():.1f}")
        print(f"Máximo de operaciones duplicadas: {overtrading_df['count'].max()}")
        
        print("\nDistribución por instrumento:")
        print(overtrading_df.groupby('instrument').size())
        
        print("\nCasos más severos (5+ operaciones):")
        severe = overtrading_df[overtrading_df['count'] >= 5]
        if len(severe) > 0:
            print(severe[['instrument', 'side', 'count', 'first_time', 'last_time']])
    else:
        print("\nNo se detectó overtrading significativo")
    
    # Analyze signal_date patterns (same signal_date = same signal)
    print("\n" + "=" * 80)
    print("ANÁLISIS DE SEÑALES REPETIDAS (MISMO signal_date)")
    print("=" * 80)
    
    signal_groups = df[df['action'] == 'OPEN'].groupby(['instrument', 'signal_date', 'side']).size().reset_index(name='count')
    repeated_signals = signal_groups[signal_groups['count'] > 1].sort_values('count', ascending=False)
    
    if len(repeated_signals) > 0:
        print(f"\nTotal de señales repetidas: {len(repeated_signals)}")
        print(f"Promedio de repeticiones: {repeated_signals['count'].mean():.1f}")
        print(f"Máximo de repeticiones: {repeated_signals['count'].max()}")
        
        print("\nTop 10 señales más repetidas:")
        print(repeated_signals.head(10))
    else:
        print("\nNo se encontraron señales repetidas")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMENDACIONES PARA REDUCIR PÉRDIDAS")
    print("=" * 80)
    
    recommendations = []
    
    if len(whipsaw_df) > 0:
        avg_whipsaw_time = whipsaw_df['time_diff_seconds'].mean()
        if avg_whipsaw_time < 60:
            recommendations.append({
                'problema': 'Whipsaws muy rápidos',
                'descripcion': f'Cambios de señal muy rápidos ({avg_whipsaw_time:.1f}s promedio) causan cierres en pérdida',
                'solucion': 'Aumentar período de confirmación antes de cambiar señal, usar trailing stop en vez de cerrar por señal opuesta'
            })
    
    if len(overtrading_df) > 0:
        max_overtrade = overtrading_df['count'].max()
        if max_overtrade > 3:
            recommendations.append({
                'problema': 'Overtrading',
                'descripcion': f'Hasta {max_overtrade} operaciones del mismo tipo abiertas simultáneamente',
                'solucion': 'Prevenir múltiples aperturas si ya existe una operación activa del mismo tipo'
            })
    
    if len(repeated_signals) > 0:
        max_repeats = repeated_signals['count'].max()
        if max_repeats > 5:
            recommendations.append({
                'problema': 'Señales repetidas excesivas',
                'descripcion': f'Hasta {max_repeats} operaciones con la misma señal',
                'solucion': 'Limitar a 1 operación por señal única, ignorar señales duplicadas hasta que se cierre la operación actual'
            })
    
    if len(recommendations) == 0:
        print("\nNo se identificaron problemas críticos en los patrones básicos")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['problema']}")
            print(f"   Problema: {rec['descripcion']}")
            print(f"   Solución: {rec['solucion']}")
    
    print("\n" + "=" * 80)
    print("RECOMENDACIONES GENERALES")
    print("=" * 80)
    print("""
1. IMPLEMENTAR TRAILING STOP: En vez de cerrar solo por señal opuesta, usar trailing stop
   para proteger ganancias y limitar pérdidas.

2. MEJORAR FILTRO DE CONDICIONES DE MERCADO: 
   - Actualmente solo bloquea si market_quality < 0.2 (muy laxo)
   - Considerar aumentar el umbral a 0.3-0.4 para evitar operaciones en condiciones subóptimas
   - Verificar condiciones ANTES de abrir, no solo en generación de señales

3. PREVENIR OVERTRADING:
   - No abrir múltiples operaciones del mismo tipo
   - Esperar cierre de operación actual antes de abrir nueva del mismo tipo

4. CONFIRMACIÓN DE SEÑALES:
   - Requerir confirmación en múltiples velas antes de cambiar dirección
   - Evitar cambios de señal muy rápidos (menos de 5 minutos)

5. GESTIÓN DE RIESGO MEJORADA:
   - Ajustar stop loss dinámico basado en ATR (ya implementado, verificar que funcione correctamente)
   - Considerar reducir tamaño de posición en condiciones de mercado pobres
   - Implementar máximo drawdown diario

6. ANÁLISIS DE SEÑALES:
   - Revisar por qué se generan múltiples señales del mismo tipo
   - Considerar cooldown period después de cerrar operación antes de abrir nueva
    """)
    
    return whipsaw_df, overtrading_df, repeated_signals

if __name__ == '__main__':
    try:
        whipsaws, overtrading, repeated = analyze_trade_losses()
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

