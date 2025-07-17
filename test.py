import pandas as pd

df = pd.read_csv("EUR_USD_m5.csv")

# Ver cuántas señales hay
print(df['signal'].value_counts())

# Ver las filas donde hay señal de compra o venta
print(df[df['signal'] != 0][['date', 'bidclose', 'signal']].tail(20))

print(df['trade_open_zone'].value_counts())
print(df['centro_picos_min_trend'].value_counts())
print(df['centro_picos_max_trend'].value_counts())
