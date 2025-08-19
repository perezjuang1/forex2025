#!/usr/bin/env python3
"""
Script de diagnóstico para el sistema de trading
Verifica la conexión, señales y apertura de posiciones
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ConnectionFxcm import RobotConnection
from PriceAnalyzer import PriceAnalyzer
from ConfigurationOperation import TradingConfig
import pandas as pd
import datetime as dt
import traceback

def test_connection():
    """Prueba la conexión a FXCM"""
    print("=== PRUEBA DE CONEXIÓN ===")
    try:
        robot_conn = RobotConnection()
        connection = robot_conn.getConnection()
        
        if connection:
            print("✅ Conexión FXCM establecida correctamente")
            print(f"Usuario: {TradingConfig.userid}")
            print(f"URL: {TradingConfig.url}")
            print(f"Tipo: {TradingConfig.connectiontype}")
            return connection
        else:
            print("❌ No se pudo establecer conexión")
            return None
    except Exception as e:
        print(f"❌ Error en conexión: {e}")
        print(traceback.format_exc())
        return None

def test_price_data(connection):
    """Prueba la obtención de datos de precios"""
    print("\n=== PRUEBA DE DATOS DE PRECIOS ===")
    try:
        analyzer = PriceAnalyzer(days=7, instrument="EUR/USD", timeframe="m1")
        
        # Obtener datos
        df = analyzer.get_price_data("EUR/USD", "m1", 7, connection)
        
        if df is None or df.empty:
            print("❌ No se obtuvieron datos de precios")
            return None
        
        print(f"✅ Datos obtenidos: {len(df)} filas")
        print(f"Rango de fechas: {df['date'].min()} - {df['date'].max()}")
        print(f"Últimos precios: {df['bidclose'].tail(5).tolist()}")
        
        return df
    except Exception as e:
        print(f"❌ Error obteniendo datos: {e}")
        print(traceback.format_exc())
        return None

def test_indicators(df):
    """Prueba la generación de indicadores"""
    print("\n=== PRUEBA DE INDICADORES ===")
    try:
        analyzer = PriceAnalyzer(days=7, instrument="EUR/USD", timeframe="m1")
        
        # Generar indicadores
        df_with_indicators = analyzer.set_indicators(df)
        
        if df_with_indicators is None or df_with_indicators.empty:
            print("❌ No se generaron indicadores")
            return None
        
        print(f"✅ Indicadores generados: {len(df_with_indicators)} filas")
        
        # Verificar columnas importantes
        required_columns = ['peaks_min', 'peaks_max', 'trend', 'min_low_flat_zone', 'max_high_flat_zone']
        missing_columns = [col for col in required_columns if col not in df_with_indicators.columns]
        
        if missing_columns:
            print(f"❌ Columnas faltantes: {missing_columns}")
        else:
            print("✅ Todas las columnas requeridas están presentes")
        
        # Mostrar estadísticas de indicadores
        if 'peaks_min' in df_with_indicators.columns:
            peaks_min_count = df_with_indicators['peaks_min'].sum()
            print(f"Picos mínimos: {peaks_min_count}")
        
        if 'peaks_max' in df_with_indicators.columns:
            peaks_max_count = df_with_indicators['peaks_max'].sum()
            print(f"Picos máximos: {peaks_max_count}")
        
        if 'trend' in df_with_indicators.columns:
            trend_counts = df_with_indicators['trend'].value_counts()
            print(f"Distribución de tendencias: {trend_counts.to_dict()}")
        
        return df_with_indicators
    except Exception as e:
        print(f"❌ Error generando indicadores: {e}")
        print(traceback.format_exc())
        return None

def test_signals(df):
    """Prueba la generación de señales"""
    print("\n=== PRUEBA DE SEÑALES ===")
    try:
        analyzer = PriceAnalyzer(days=7, instrument="EUR/USD", timeframe="m1")
        
        # Generar señales
        df_with_signals = analyzer.set_signals_to_trades(df)
        
        if df_with_signals is None or df_with_signals.empty:
            print("❌ No se generaron señales")
            return None
        
        print(f"✅ Señales generadas: {len(df_with_signals)} filas")
        
        # Verificar señales
        if 'signal' in df_with_signals.columns:
            signal_counts = df_with_signals['signal'].value_counts()
            print(f"Distribución de señales: {signal_counts.to_dict()}")
            
            # Buscar señales recientes
            recent_signals = df_with_signals.tail(10)
            buy_signals = recent_signals[recent_signals['signal'] == 1]
            sell_signals = recent_signals[recent_signals['signal'] == -1]
            
            print(f"Señales de compra en últimas 10 velas: {len(buy_signals)}")
            print(f"Señales de venta en últimas 10 velas: {len(sell_signals)}")
            
            if not buy_signals.empty:
                print(f"Última señal de compra: {buy_signals.iloc[-1]['date']}")
            if not sell_signals.empty:
                print(f"Última señal de venta: {sell_signals.iloc[-1]['date']}")
        
        return df_with_signals
    except Exception as e:
        print(f"❌ Error generando señales: {e}")
        print(traceback.format_exc())
        return None

def test_trade_opening(df):
    """Prueba la apertura de trades"""
    print("\n=== PRUEBA DE APERTURA DE TRADES ===")
    try:
        analyzer = PriceAnalyzer(days=7, instrument="EUR/USD", timeframe="m1")
        
        # Verificar si hay operaciones existentes
        has_buy = analyzer.existingOperation("EUR/USD", "B")
        has_sell = analyzer.existingOperation("EUR/USD", "S")
        
        print(f"Operación de compra existente: {has_buy}")
        print(f"Operación de venta existente: {has_sell}")
        
        # Verificar cooldown
        has_recent = analyzer._has_recent_open(cooldown_minutes=10)
        print(f"Cooldown activo (últimos 10 min): {has_recent}")
        
        # Intentar procesar señales
        analyzer.triggers_trades_open(df)
        
        print("✅ Procesamiento de señales completado")
        
    except Exception as e:
        print(f"❌ Error en apertura de trades: {e}")
        print(traceback.format_exc())

def main():
    """Función principal de diagnóstico"""
    print("🔍 DIAGNÓSTICO DEL SISTEMA DE TRADING")
    print("=" * 50)
    
    # 1. Probar conexión
    connection = test_connection()
    if not connection:
        print("\n❌ No se puede continuar sin conexión")
        return
    
    # 2. Probar datos de precios
    df = test_price_data(connection)
    if df is None:
        print("\n❌ No se puede continuar sin datos")
        return
    
    # 3. Probar indicadores
    df_with_indicators = test_indicators(df)
    if df_with_indicators is None:
        print("\n❌ No se puede continuar sin indicadores")
        return
    
    # 4. Probar señales
    df_with_signals = test_signals(df_with_indicators)
    if df_with_signals is None:
        print("\n❌ No se puede continuar sin señales")
        return
    
    # 5. Probar apertura de trades
    test_trade_opening(df_with_signals)
    
    print("\n" + "=" * 50)
    print("🏁 DIAGNÓSTICO COMPLETADO")
    
    # Guardar datos para inspección
    df_with_signals.to_csv("diagnostic_data.csv", index=False)
    print("📁 Datos guardados en 'diagnostic_data.csv' para inspección manual")

if __name__ == "__main__":
    main()
