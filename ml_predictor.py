"""
Machine Learning prediction engine for crypto price prediction
Uses XGBoost and ensemble methods for multi-horizon predictions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import config

logger = logging.getLogger(__name__)

class CryptoPredictionModel:
    def __init__(self, crypto_name: str, prediction_horizon: str):
        self.crypto_name = crypto_name
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_columns = []
        self.training_history = []
        
    def prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training/prediction
        """
        # Remove rows with NaN targets
        clean_df = df.dropna(subset=[target_col]).copy()
        
        if clean_df.empty:
            raise ValueError(f"No valid data available for target {target_col}")
        
        # Separate features and target
        exclude_cols = [
            'datetime', 'crypto', 'target_1h', 'target_1d', 'target_1w', 
            'target_return_1h', 'target_return_1d', 'target_return_1w',
            'target_direction_1h', 'target_direction_1d', 'target_direction_1w',
            'target_datetime_1h', 'target_datetime_1d', 'target_datetime_1w'
        ]
        
        feature_cols = [col for col in clean_df.columns if col not in exclude_cols]
        
        X = clean_df[feature_cols].fillna(0)  # Fill remaining NaN with 0
        y = clean_df[target_col]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> pd.DataFrame:
        """
        Select top k features using statistical tests
        """
        selector_key = f"{self.prediction_horizon}_selector"
        
        if selector_key not in self.feature_selectors:
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            self.feature_selectors[selector_key] = selector
            
            selected_features = X.columns[selector.get_support()].tolist()
            logger.info(f"Selected {len(selected_features)} features for {self.prediction_horizon}")
            
        else:
            selector = self.feature_selectors[selector_key]
            X_selected = selector.transform(X)
            selected_features = X.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def scale_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Scale features using StandardScaler
        """
        scaler_key = f"{self.prediction_horizon}_scaler"
        
        if fit or scaler_key not in self.scalers:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[scaler_key] = scaler
        else:
            scaler = self.scalers[scaler_key]
            X_scaled = scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def train_xgboost_regressor(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train XGBoost regression model
        """
        # Adjust cross-validation folds based on data size
        n_samples = len(X)
        max_folds = min(config.MODEL_SETTINGS['cross_validation_folds'], max(2, n_samples // 10))
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=max_folds)
        
        # XGBoost parameters
        xgb_params = config.MODEL_SETTINGS['xgboost_params'].copy()
        xgb_params['objective'] = 'reg:squarederror'
        
        model = xgb.XGBRegressor(**xgb_params)
        
        # Cross-validation with error handling
        try:
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        except Exception as cv_error:
            logger.warning(f"Cross-validation failed: {cv_error}. Skipping CV evaluation.")
            cv_scores = np.array([0])  # Placeholder scores
        
        # Train final model on all data
        model.fit(X, y)
        
        # Model evaluation
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        model_key = f"{self.prediction_horizon}_xgb_regressor"
        self.models[model_key] = model
        
        results = {
            'model_type': 'xgb_regressor',
            'cv_mse_mean': -cv_scores.mean(),
            'cv_mse_std': cv_scores.std(),
            'train_mse': mse,
            'train_mae': mae,
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
        logger.info(f"XGBoost Regressor trained - MSE: {mse:.4f}, MAE: {mae:.4f}")
        return results
    
    def train_xgboost_classifier(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train XGBoost classification model for direction prediction
        """
        # Convert to binary classification (up/down)
        y_binary = (y > 0).astype(int)
        
        # Check for class balance
        unique_classes = y_binary.nunique()
        if unique_classes < 2:
            logger.warning(f"Insufficient class variation: only {unique_classes} unique classes")
            # Return a dummy result to avoid crash
            return {
                'model_type': 'xgb_classifier',
                'cv_accuracy_mean': 0.5,
                'cv_accuracy_std': 0.0,
                'train_accuracy': 0.5,
                'feature_importance': {}
            }
        
        # Adjust cross-validation folds based on data size
        n_samples = len(X)
        max_folds = min(config.MODEL_SETTINGS['cross_validation_folds'], max(2, n_samples // 10))
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=max_folds)
        
        # XGBoost parameters
        xgb_params = config.MODEL_SETTINGS['xgboost_params'].copy()
        xgb_params['objective'] = 'binary:logistic'
        
        model = xgb.XGBClassifier(**xgb_params)
        
        # Cross-validation with error handling
        try:
            cv_scores = cross_val_score(model, X, y_binary, cv=tscv, scoring='accuracy')
        except Exception as cv_error:
            logger.warning(f"Cross-validation failed: {cv_error}. Skipping CV evaluation.")
            cv_scores = np.array([0.5])  # Placeholder scores
        
        # Train final model on all data
        model.fit(X, y_binary)
        
        # Model evaluation
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        accuracy = accuracy_score(y_binary, y_pred)
        
        model_key = f"{self.prediction_horizon}_xgb_classifier"
        self.models[model_key] = model
        
        results = {
            'model_type': 'xgb_classifier',
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'train_accuracy': accuracy,
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
        logger.info(f"XGBoost Classifier trained - Accuracy: {accuracy:.4f}")
        return results
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Main training pipeline
        """
        logger.info(f"Training model for {self.crypto_name} - {self.prediction_horizon}")
        
        # Define target columns based on prediction horizon
        target_mapping = {
            '1h': ('target_return_1h', 'target_direction_1h'),
            '1d': ('target_return_1d', 'target_direction_1d'),
            '1w': ('target_return_1w', 'target_direction_1w')
        }
        
        if self.prediction_horizon not in target_mapping:
            raise ValueError(f"Invalid prediction horizon: {self.prediction_horizon}")
        
        regression_target, classification_target = target_mapping[self.prediction_horizon]
        
        # Prepare data - ensure same samples for both targets
        # Remove rows with NaN in either target
        clean_df = df.dropna(subset=[regression_target, classification_target]).copy()
        
        if clean_df.empty:
            raise ValueError(f"No valid data available for targets {regression_target}, {classification_target}")
        
        X, y_reg = self.prepare_data(clean_df, regression_target)
        _, y_clf = self.prepare_data(clean_df, classification_target)
        
        # Feature selection
        X_selected = self.feature_selection(X, y_reg, k=config.MODEL_SETTINGS['feature_selection_k'])
        
        # Scale features
        X_scaled = self.scale_features(X_selected, fit=True)
        
        # Train models
        regression_results = self.train_xgboost_regressor(X_scaled, y_reg)
        classification_results = self.train_xgboost_classifier(X_scaled, y_clf)
        
        # Store training metadata
        training_info = {
            'timestamp': datetime.now(),
            'crypto': self.crypto_name,
            'horizon': self.prediction_horizon,
            'training_samples': len(X),
            'features_used': len(X_scaled.columns),
            'regression_results': regression_results,
            'classification_results': classification_results
        }
        
        self.training_history.append(training_info)
        
        logger.info(f"Training complete for {self.crypto_name} - {self.prediction_horizon}")
        return training_info
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Make predictions using trained models
        """
        if not self.models:
            raise ValueError("No trained models available. Please train first.")
        
        try:
            # Get latest data point
            latest_data = df.iloc[[-1]].copy()
            
            # Prepare features
            exclude_cols = [
                'datetime', 'crypto', 'target_1h', 'target_1d', 'target_1w', 
                'target_return_1h', 'target_return_1d', 'target_return_1w',
                'target_direction_1h', 'target_direction_1d', 'target_direction_1w',
                'target_datetime_1h', 'target_datetime_1d', 'target_datetime_1w'
            ]
            
            X = latest_data[[col for col in latest_data.columns if col not in exclude_cols]].fillna(0)
            
            # Apply feature selection and scaling
            selector = self.feature_selectors[f"{self.prediction_horizon}_selector"]
            X_selected = pd.DataFrame(
                selector.transform(X), 
                columns=X.columns[selector.get_support()], 
                index=X.index
            )
            
            scaler = self.scalers[f"{self.prediction_horizon}_scaler"]
            X_scaled = pd.DataFrame(
                scaler.transform(X_selected),
                columns=X_selected.columns,
                index=X_selected.index
            )
            
            # Make predictions
            regressor = self.models[f"{self.prediction_horizon}_xgb_regressor"]
            classifier = self.models[f"{self.prediction_horizon}_xgb_classifier"]
            
            # Price return prediction
            predicted_return = regressor.predict(X_scaled)[0]
            
            # Direction prediction
            direction_prob = classifier.predict_proba(X_scaled)[0, 1]  # Probability of price going up
            predicted_direction = int(direction_prob > 0.5)
            
            # Current price for absolute price prediction
            current_price = latest_data['price'].iloc[0]
            predicted_price = current_price * (1 + predicted_return / 100)
            
            # CRITICAL FIX: Use target datetime instead of feature datetime
            # This represents WHEN the prediction is valid, not when it was made
            feature_datetime = latest_data['datetime'].iloc[0]
            if isinstance(feature_datetime, str):
                feature_datetime = pd.to_datetime(feature_datetime)
            
            # Calculate target datetime based on prediction horizon
            if self.prediction_horizon == '1h':
                target_datetime = feature_datetime + pd.Timedelta(hours=1)
            elif self.prediction_horizon == '1d':
                target_datetime = feature_datetime + pd.Timedelta(days=1)
            elif self.prediction_horizon == '1w':
                target_datetime = feature_datetime + pd.Timedelta(weeks=1)
            else:
                target_datetime = feature_datetime  # Fallback
            
            return {
                'timestamp': target_datetime,  # FIXED: Now points to prediction target time
                'feature_timestamp': feature_datetime,  # When the features were from
                'crypto': self.crypto_name,
                'horizon': self.prediction_horizon,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_return': predicted_return,
                'predicted_direction': predicted_direction,
                'direction_confidence': direction_prob,
                'model_confidence': max(direction_prob, 1 - direction_prob)  # Distance from 0.5
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {self.crypto_name} - {self.prediction_horizon}: {e}")
            return None
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'crypto_name': self.crypto_name,
            'prediction_horizon': self.prediction_horizon,
            'models': self.models,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'feature_columns': self.feature_columns,
            'training_history': self.training_history
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.crypto_name = model_data['crypto_name']
        self.prediction_horizon = model_data['prediction_horizon']
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_selectors = model_data['feature_selectors']
        self.feature_columns = model_data['feature_columns']
        self.training_history = model_data['training_history']
        logger.info(f"Model loaded from {filepath}")

class EnsemblePredictionEngine:
    """
    Ensemble engine that manages multiple prediction models
    """
    def __init__(self):
        self.models = {}
        
    def add_model(self, crypto_name: str, prediction_horizon: str) -> CryptoPredictionModel:
        """Add a new prediction model"""
        model_key = f"{crypto_name}_{prediction_horizon}"
        model = CryptoPredictionModel(crypto_name, prediction_horizon)
        self.models[model_key] = model
        return model
    
    def train_all_models(self, data_dict: Dict[str, pd.DataFrame]):
        """Train all models with their respective data"""
        results = {}
        
        for crypto_name in config.CRYPTOCURRENCIES:
            if crypto_name not in data_dict:
                logger.warning(f"No data available for {crypto_name}")
                continue
                
            for horizon in config.PREDICTION_INTERVALS:
                model_key = f"{crypto_name}_{horizon}"
                
                if model_key not in self.models:
                    self.add_model(crypto_name, horizon)
                
                try:
                    training_result = self.models[model_key].train(data_dict[crypto_name])
                    results[model_key] = training_result
                    logger.info(f"Successfully trained {model_key}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_key}: {e}")
                    results[model_key] = {'error': str(e)}
        
        return results
    
    def predict_all(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Generate predictions from all models"""
        predictions = {}
        
        for model_key, model in self.models.items():
            crypto_name = model.crypto_name
            
            if crypto_name not in data_dict:
                logger.warning(f"No data available for prediction: {crypto_name}")
                continue
            
            try:
                prediction = model.predict(data_dict[crypto_name])
                if prediction:
                    predictions[model_key] = prediction
                    
            except Exception as e:
                logger.error(f"Prediction failed for {model_key}: {e}")
        
        return predictions
    
    def save_ensemble(self, directory: str):
        """Save all models in the ensemble"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_key, model in self.models.items():
            filepath = os.path.join(directory, f"{model_key}.joblib")
            model.save_model(filepath)
    
    def load_ensemble(self, directory: str):
        """Load all models in the ensemble"""
        import os
        
        for filename in os.listdir(directory):
            if filename.endswith('.joblib'):
                model_key = filename.replace('.joblib', '')
                crypto_name, horizon = model_key.split('_', 1)
                
                model = CryptoPredictionModel(crypto_name, horizon)
                filepath = os.path.join(directory, filename)
                model.load_model(filepath)
                
                self.models[model_key] = model

if __name__ == "__main__":
    # Test the prediction engine
    from data_collector import DataCollector
    from feature_engineering import FeatureEngineer
    
    # Collect and prepare data
    collector = DataCollector()
    fe = FeatureEngineer()
    
    btc_data = collector.get_crypto_data('bitcoin', days=30)
    market_data = collector.get_traditional_markets_data(days=30)
    
    # Feature engineering
    btc_features = fe.prepare_features(btc_data, market_data)
    
    # Create and train model
    model = CryptoPredictionModel('bitcoin', '1h')
    training_results = model.train(btc_features)
    
    print("Training Results:")
    print(f"Regression MSE: {training_results['regression_results']['train_mse']:.6f}")
    print(f"Classification Accuracy: {training_results['classification_results']['train_accuracy']:.4f}")
    
    # Make prediction
    prediction = model.predict(btc_features)
    if prediction:
        print(f"\nPrediction for {prediction['crypto']} ({prediction['horizon']}):")
        print(f"Current Price: ${prediction['current_price']:.2f}")
        print(f"Predicted Price: ${prediction['predicted_price']:.2f}")
        print(f"Predicted Return: {prediction['predicted_return']:.2f}%")
        print(f"Direction: {'UP' if prediction['predicted_direction'] else 'DOWN'}")
        print(f"Confidence: {prediction['model_confidence']:.2f}") 