#Bibliotecas y extensiones
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

ruta_archivo= "C:\\Users\\rafad\\OneDrive\\Documents\\codigos\\Codigos Python\\Deep learning\\TPREDIO.csv"
#Cambiar la ruta de archivo segun el entorno(local o colab) y el dispositivo
df = pd.read_csv(ruta_archivo)

#Obtengo informacion general de la base de datos
print("\n"*3)
print(df.head())
print("\n"*3)
print(df.describe()) 
print("\n"*3)
#Obtengo informacion de las columnas y los tipos de datos
print(df.info())
print("\n"*3)

#Obtengo informacion de los valores nulos
print("Analisis de valores nulos")
print("\n")
print(f"Total de valores nulos en el dataset: {df.isnull().sum().sum()}")
print("\n"*2)
print("Valores nulos por columna: ")
print(df.isnull().sum())
print("\n"*2)
print("Porcentaje de valores nulos por columna: ")
print((df.isnull().sum() / len(df)) * 100)
print("\n"*2)

#Use IA para que me ayude a determinar que informacion contiene cada columna y tras un analisis decidi eliminar las siguientes columnas:
#PRECRESTO/Código técnico complementario, raramente usado
#PRECEDCATA/Redundante si tienes PRECHIP
#PRENUPRE/Redundante con PRECHIP (ambos identifican el predio)
#PRENBARRIO/Redundante con PRECBARRIO (mantener el codigo, eliminar el nombre)
#PREMDIRECC/Metodología técnica de dirección
#PRETDIRECC/Tipo técnico de dirección
#PREDSI/Dirección sin identificar (campo técnico)
#PREFCALIF/Fecha de calificación (dato administrativo)
#BARMANPRE/Combinación de otros códigos existentes
#PREEARMAZON/Detalle constructivo muy específico
#PREEMRUROS/Detalle constructivo muy específico
#PREECUBIER/Detalle constructivo muy específico
#PREBENCHAPE/Enchapes muy específicos
#PRECENCHAPE/Enchapes muy específicos
#PREBMOBILI/Mobiliario muy específico
#PRECMOBILI/Mobiliario muy específico
#PREFINCORP/Fecha de incorporación como texto
#PREDIRECC/Dirección como texto libre
#PREUSOPH/Uso predominante como texto
#PREUSONPH/Uso no predominante como texto
#PREUVIVIEN/Unidades de vivienda como texto
#PREUCALIF/Calificación como texto
#PRECLCONS/Clase de construcción como texto
#PRECLASE/Eliminar si hay código equivalente más específico
#PRECZHF/Eliminar si es descripción de texto (mantener códigos numéricos)
#Decidi eliminar aquellas con datos muy especificos, redundantes o tecnicos, ademas elimine algunas de las que tenian mas valores nulos ya que no aportaban mucho

#Elimino las columnas identificadas como innecesarias
columnas_a_eliminar = ['PRECRESTO', 'PRECEDCATA', 'PRENUPRE', 'PREMDIRECC', 'PRETDIRECC', 'PREDSI', 'PREFCALIF', 'BARMANPRE', 'PREEARMAZON', 'PREEMRUROS', 'PREECUBIER', 'PREBENCHAPE', 'PRECENCHAPE', 'PREBMOBILI', 'PRECMOBILI','PRENBARRIO', 'PREDIRECC', 'PREFINCORP', 'PRECLASE', 'PRECZHF','PREUSOPH', 'PREUSONPH', 'PREUVIVIEN', 'PREUCALIF', 'PRECLCONS']
df_limpio = df.drop(columns=columnas_a_eliminar, errors='ignore')

#Verifico las dimensiones del dataframe antes y después de la limpieza para verificar que elimine correctamente las columnas deseadas
print("Dimensiones originales:", df.shape)
print("Dimensiones después de eliminar columnas:", df_limpio.shape)
print("Columnas eliminadas exitosamente")
print("\n"*2)

df=df_limpio

#Muestro información del dataframe limpio y nuevamente evaluo los valores nulos
print("\n")
print("Información del dataframe limpio:")
print(df.info())
print("Análisis de valores nulos en el dataframe limpio:")
print("\n")
print(f"Total de valores nulos en el dataset: {df.isnull().sum().sum()}")
print("\n"*2)
print("Valores nulos por columna: ")
print(df.isnull().sum())
print("\n"*2)
print("Porcentaje de valores nulos por columna: ")
print((df.isnull().sum() / len(df)) * 100)
print("\n"*2)

# Análisis de tipos de datos y identificación de columnas categóricas
print("Análisis de tipos de datos:")
print("\n")
print("Tipos de datos por columna:")
print(df.dtypes)
print("\n"*2)
columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
print("Columnas numéricas:", columnas_numericas)
print("Columnas categóricas:", columnas_categoricas)
print("\n"*2)

# PROCEDIMIENTO PARA VALORES NULOS
print("=== TRATAMIENTO DE VALORES NULOS ===")
print("\n")
# Para columnas numéricas: llenar con la mediana
for col in columnas_numericas:
    if df[col].isnull().sum() > 0:
        mediana = df[col].median()
        df[col].fillna(mediana, inplace=True)
        print(f"Valores nulos en '{col}' llenados con mediana: {mediana}")
# Para columnas categóricas: llenar con la moda (valor más frecuente)
for col in columnas_categoricas:
    if df[col].isnull().sum() > 0:
        moda = df[col].mode().iloc[0] if not df[col].mode().empty else 'DESCONOCIDO'
        df[col].fillna(moda, inplace=True)
        print(f"Valores nulos en '{col}' llenados con moda: {moda}")
print("\n")
print("Verificación después del tratamiento de nulos:")
print(f"Total de valores nulos restantes: {df.isnull().sum().sum()}")
print("\n"*2)

# CONVERSIÓN DE VALORES DE TEXTO A NUMÉRICOS
print("=== CONVERSIÓN DE TEXTO A VALORES NUMÉRICOS ===")
print("\n")
# Crear diccionario para almacenar los encoders
encoders = {}
# Convertir columnas categóricas usando Label Encoder
df_encoded = df.copy()
for col in columnas_categoricas:
    print(f"Convirtiendo columna '{col}':")
    print(f"  Valores únicos antes: {df[col].nunique()}")
    # Crear y aplicar Label Encoder
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  Valores únicos después: {df_encoded[col].nunique()}")
    print(f"  Rango de valores: {df_encoded[col].min()} - {df_encoded[col].max()}")
    print("\n")

df = df_encoded

print("Conversión completada. Verificación final:")
print("\n")
print("Tipos de datos después de la conversión:")
print(df.dtypes)
print("\n"*2)
print("Estadísticas descriptivas del dataset final:")
print(df.describe())
print("\n"*2)
print("Información final del dataset:")
print(df.info())
print("\n"*2)
# Guardar el mapeo de encoders para referencia futura
print("=== MAPEO DE ENCODERS (para referencia) ===")
for col, encoder in encoders.items():
    print(f"\nColumna '{col}':")
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"  Mapeo: {mapping}")
print("\n"*2)
print("Dataset listo")
