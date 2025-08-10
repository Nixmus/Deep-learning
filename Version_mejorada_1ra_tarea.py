#Bibliotecas y extensiones
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import logging
from typing import Dict, List, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.encoders = {}
        self.removed_columns = []
        
        # Columnas a eliminar organizadas por categoría
        self.columns_to_remove = {
            'redundant': ['PRECEDCATA', 'PRENUPRE', 'PRENBARRIO'],
            'technical_specific': ['PRECRESTO', 'PREMDIRECC', 'PRETDIRECC', 'PREDSI', 
                                 'PREEARMAZON', 'PREEMRUROS', 'PREECUBIER'],
            'construction_details': ['PREBENCHAPE', 'PRECENCHAPE', 'PREBMOBILI', 'PRECMOBILI'],
            'administrative': ['PREFCALIF', 'BARMANPRE', 'PREFINCORP'],
            'free_text': ['PREDIRECC', 'PREUSOPH', 'PREUSONPH', 'PREUVIVIEN', 
                         'PREUCALIF', 'PRECLCONS'],
            'classification_redundant': ['PRECLASE', 'PRECZHF']
        }
    
    def load_data(self) -> pd.DataFrame:
        """Cargar datos desde archivo CSV con validaciones"""
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"El archivo {self.file_path} no existe")
            
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Datos cargados exitosamente. Forma: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
    
    def analyze_data_quality(self) -> Dict:
        """Análizar calidad de datos"""
        if self.df is None:
            raise ValueError("Primero debe cargar los datos")
        
        analysis = {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'null_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'dtypes': self.df.dtypes.value_counts().to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
        
        logger.info(f"Análisis de calidad completado: {analysis['shape'][0]} filas, {analysis['shape'][1]} columnas")
        logger.info(f"Uso de memoria: {analysis['memory_usage']:.2f} MB")
        
        return analysis
    
    def remove_unnecessary_columns(self) -> pd.DataFrame:
        """Eliminar columnas innecesarias de forma organizada"""
        all_columns_to_remove = []
        for category, columns in self.columns_to_remove.items():
            all_columns_to_remove.extend(columns)
        
        # Verificar qué columnas existen realmente
        existing_columns_to_remove = [col for col in all_columns_to_remove if col in self.df.columns]
        self.removed_columns = existing_columns_to_remove
        
        # Eliminar columnas
        self.df = self.df.drop(columns=existing_columns_to_remove, errors='ignore')
        
        logger.info(f"Eliminadas {len(existing_columns_to_remove)} columnas")
        logger.info(f"Nueva forma del dataset: {self.df.shape}")
        
        return self.df
    
    def handle_missing_values(self, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """Manejar valores faltantes con estrategias configurables"""
        if strategy is None:
            strategy = {'numeric': 'median', 'categorical': 'mode'}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Tratar columnas numéricas
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                if strategy['numeric'] == 'median':
                    fill_value = self.df[col].median()
                elif strategy['numeric'] == 'mean':
                    fill_value = self.df[col].mean()
                else:
                    fill_value = 0
                
                self.df[col].fillna(fill_value, inplace=True)
                logger.info(f"Valores nulos en '{col}' llenados con {strategy['numeric']}: {fill_value}")
        
        # Tratar columnas categóricas
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                if strategy['categorical'] == 'mode':
                    fill_value = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'DESCONOCIDO'
                else:
                    fill_value = 'DESCONOCIDO'
                
                self.df[col].fillna(fill_value, inplace=True)
                logger.info(f"Valores nulos en '{col}' llenados con: {fill_value}")
        
        logger.info(f"Valores nulos restantes: {self.df.isnull().sum().sum()}")
        return self.df
    
    def encode_categorical_variables(self, save_encoders: bool = True) -> pd.DataFrame:
        """Codificar variables categóricas"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            
            if save_encoders:
                self.encoders[col] = le
            
            logger.info(f"Columna '{col}' codificada: {le.classes_[:5]}... -> 0-{len(le.classes_)-1}")
        
        return self.df
    
    def validate_final_dataset(self) -> bool:
        """Validar dataset final"""
        validations = {
            'no_nulls': self.df.isnull().sum().sum() == 0,
            'all_numeric': all(self.df.dtypes.apply(lambda x: np.issubdtype(x, np.number))),
            'no_infinite': not np.isinf(self.df.select_dtypes(include=[np.number])).any().any(),
            'reasonable_size': self.df.shape[0] > 0 and self.df.shape[1] > 0
        }
        
        all_valid = all(validations.values())
        logger.info(f"Validación del dataset: {'EXITOSA' if all_valid else 'FALLIDA'}")
        
        for validation, result in validations.items():
            logger.info(f"  {validation}: {'✓' if result else '✗'}")
        
        return all_valid
    
    def save_processed_data(self, output_path: str = None) -> str:
        """Guardar datos procesados"""
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_path = os.path.join(os.path.dirname(self.file_path), f"{base_name}_processed.csv")
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Datos procesados guardados en: {output_path}")
        return output_path
    
    def get_preprocessing_report(self) -> Dict:
        """Generar reporte de preprocesamiento"""
        return {
            'original_shape': getattr(self, 'original_shape', 'No disponible'),
            'final_shape': self.df.shape if self.df is not None else 'No disponible',
            'removed_columns': self.removed_columns,
            'encoders_created': list(self.encoders.keys()),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2 if self.df is not None else 0
        }

def main():
    # Configuración
    file_path = "C:\\Users\\rafad\\OneDrive\\Documents\\codigos\\Codigos Python\\Deep learning\\TPREDIO.csv"
    
    try:
        # Inicializar preprocessor
        preprocessor = DataPreprocessor(file_path)
        
        # Cargar y analizar datos
        df = preprocessor.load_data()
        preprocessor.original_shape = df.shape
        
        print("=== ANÁLISIS INICIAL ===")
        analysis = preprocessor.analyze_data_quality()
        print(f"Forma original: {analysis['shape']}")
        print(f"Uso de memoria: {analysis['memory_usage']:.2f} MB")
        print(f"Duplicados: {analysis['duplicates']}")
        
        # Mostrar primeras filas y estadísticas
        print("\nPrimeras 5 filas:")
        print(df.head())
        print("\nEstadísticas descriptivas:")
        print(df.describe())
        
        # Procesar datos
        print("\n=== PROCESAMIENTO ===")
        preprocessor.remove_unnecessary_columns()
        preprocessor.handle_missing_values()
        preprocessor.encode_categorical_variables()
        
        # Validar y guardar
        print("\n=== VALIDACIÓN FINAL ===")
        if preprocessor.validate_final_dataset():
            output_file = preprocessor.save_processed_data()
            
            # Mostrar resultados finales
            print("\n=== RESULTADOS FINALES ===")
            print("Estadísticas del dataset procesado:")
            print(preprocessor.df.describe())
            
            print("\n=== REPORTE DE PREPROCESAMIENTO ===")
            report = preprocessor.get_preprocessing_report()
            for key, value in report.items():
                print(f"{key}: {value}")
            
            print(f"\nDataset procesado guardado en: {output_file}")
            print("Procesamiento completado exitosamente!")
        
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {e}")
        raise

if __name__ == "__main__":
    main()
