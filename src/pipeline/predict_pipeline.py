import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
    def __init__(self) -> None:
        pass
    def predict(self, features):
        try:
            # Check if model files exist and are compatible
            model_path = 'artifacts/model.pickle'
            preprocessor_path = 'artifacts/preprocessor.pickle'
            
            # Try to load existing model
            try:
                model = load_object(file_path=model_path)
                preprocessor = load_object(file_path=preprocessor_path)
            except Exception as load_error:
                print(f"Model loading failed: {load_error}")
                print("Creating new model...")
                
                # Create new model if loading fails
                model, preprocessor = self._create_new_model()
            
            data_scale = preprocessor.transform(features)
            prediction = model.predict(data_scale)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_new_model(self):
        """Create a new model when the old one is incompatible"""
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        import pickle
        import os
        
        print("=== Creating New Compatible Model ===")
        
        # Load data
        df = pd.read_csv('artifacts/data.csv')
        print(f"Data loaded: {df.shape}")
        
        # Prepare features and target
        X = df.drop(columns=['math_score'], axis=1)
        y = df['math_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing
        numerical_columns = ["reading_score", "writing_score"]
        categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
        
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False))
        ])
        
        preprocessor = ColumnTransformer([
            ("num_pipeline", num_pipeline, numerical_columns),
            ("cat_pipeline", cat_pipeline, categorical_columns)
        ])
        
        # Fit and transform
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_transformed, y_train)
        
        # Test model
        y_pred = model.predict(X_test_transformed)
        r2 = r2_score(y_test, y_pred)
        print(f"New model R² Score: {r2:.4f}")
        
        # Save new model files
        with open('artifacts/model.pickle', 'wb') as f:
            pickle.dump(model, f)
        
        with open('artifacts/preprocessor.pickle', 'wb') as f:
            pickle.dump(preprocessor, f)
        
        print("✓ New model created and saved")
        return model, preprocessor

class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education:str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float) -> None:
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score
    
    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            df = pd.DataFrame(custom_data_dict)
            
            # Reorder columns to match training data format (excluding math_score)
            column_order = ["gender", "race_ethnicity", "parental_level_of_education", 
                          "lunch", "test_preparation_course", "reading_score", "writing_score"]
            df = df[column_order]
            
            return df
        except Exception as e:
            raise CustomException(e, sys)