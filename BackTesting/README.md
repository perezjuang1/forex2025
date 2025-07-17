# BackTesting

Este módulo permite realizar backtesting y simulación de la estrategia de trading definida en el proyecto.

## Archivos principales
- `backtest_runner.py`: Script principal para ejecutar el backtesting sobre datos históricos.

## Uso rápido
1. Coloca los archivos CSV de precios históricos en la raíz del proyecto o especifica la ruta en el parámetro `price_file`.
2. Ejecuta el script:
   ```bash
   python BackTesting/backtest_runner.py
   ```
3. El script simulará las señales y operaciones, mostrando un resumen de las operaciones generadas.

## Personalización
- Puedes modificar el instrumento, timeframe y días de análisis en el bloque `if __name__ == "__main__"`.
- El script reutiliza la lógica y configuración actual del sistema, sin modificar los archivos originales.

## Extensión
Puedes crear nuevos scripts en esta carpeta para analizar resultados, visualizar gráficos o probar variantes de la estrategia.
