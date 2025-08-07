#Bibliotecas y extensiones
import pandas as pd

ruta_archivo= "C:\\Users\\rafad\\OneDrive\\Documents\\codigos\\Codigos Python\\Deep learning\\TPREDIO.csv"
df = pd.read_csv(ruta_archivo)

print("\n"*3)
print(df.head())
print("\n"*3)
print(df.describe()) 
print("\n"*3)
print(df.info())
print("\n"*3)
