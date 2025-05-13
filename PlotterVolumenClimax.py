import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Leer archivo
df = pd.read_csv('EUR_USD_m5.csv')

# Convertir fecha correctamente
df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d%H%M')

# Asegurar tipos
df['bidclose'] = pd.to_numeric(df['bidclose'], errors='coerce')
df['climax_type'] = pd.to_numeric(df['climax_type'], errors='coerce')
df['buy'] = pd.to_numeric(df['buy'], errors='coerce')
df['sell'] = pd.to_numeric(df['sell'], errors='coerce')

# Iniciar gráfico
fig, ax1 = plt.subplots(figsize=(14, 7))

# Línea base de precio
ax1.plot(df['date'], df['bidclose'], label='Precio de Cierre', color='black', linewidth=1.5)

# Clímax de Venta
sell_climax = df[df['climax_type'] == 1].dropna(subset=['date', 'bidclose'])
ax1.scatter(sell_climax['date'].values, sell_climax['bidclose'].values,
            color='red', marker='v', s=100, label='Clímax de Venta')

# Clímax de Compra
buy_climax = df[df['climax_type'] == -1].dropna(subset=['date', 'bidclose'])
ax1.scatter(buy_climax['date'].values, buy_climax['bidclose'].values,
            color='green', marker='^', s=100, label='Clímax de Compra')

# Señales de Compra
buy_signals = df[df['buy'] == 1].dropna(subset=['date', 'bidclose'])
ax1.scatter(buy_signals['date'].values, buy_signals['bidclose'].values,
            color='blue', marker='^', s=80, label='Señal de Compra')

# Señales de Venta
sell_signals = df[df['sell'] == 1].dropna(subset=['date', 'bidclose'])
ax1.scatter(sell_signals['date'].values, sell_signals['bidclose'].values,
            color='orange', marker='v', s=80, label='Señal de Venta')

# Formato de fechas en eje X
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)

# Estética
ax1.set_title('Precio EUR/USD y Clímax de Volumen')
ax1.set_xlabel('Fecha y Hora')
ax1.set_ylabel('Precio')
ax1.grid(True)
ax1.legend()

plt.tight_layout()
plt.show() 