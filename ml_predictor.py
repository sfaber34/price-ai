"""
Machine Learning prediction engine for crypto price prediction
Uses XGBoost and ensemble methods for multi-horizon predictions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
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
            'datetime', 'crypto',
            'target_direction_15m', 'target_direction_1h', 'target_direction_4h',
            'target_datetime_15m', 'target_datetime_1h', 'target_datetime_4h',
        ]
        
        feature_cols = [col for col in clean_df.columns if col not in exclude_cols]
        
        X = clean_df[feature_cols].fillna(0)  # Fill remaining NaN with 0
        y = clean_df[target_col]
        
        # CRITICAL: Additional data validation to prevent training failures
        logger.info("Performing final data validation before model training...")
        
        # Check for infinity values
        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.sum() > 0:
            logger.warning(f"Found {inf_mask.sum()} rows with infinity values, removing them")
            X = X[~inf_mask]
            y = y[~inf_mask]
        
        # Check for extremely large values that could cause numerical instability
        max_safe_value = 1e15  # Conservative upper bound for model training
        large_mask = (X.abs() > max_safe_value).any(axis=1)
        if large_mask.sum() > 0:
            logger.warning(f"Found {large_mask.sum()} rows with extremely large values, removing them")
            X = X[~large_mask]
            y = y[~large_mask]
        
        # Final safety check: ensure no NaN values remain in features
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            logger.warning(f"Found {nan_mask.sum()} rows with NaN values, removing them")
            X = X[~nan_mask]
            y = y[~nan_mask]
        
        # Check if we still have sufficient data after cleaning
        if len(X) < 10:
            raise ValueError(f"Insufficient data after cleaning: only {len(X)} samples remaining")
        
        # Final validation: ensure all values are finite and reasonable
        if not np.all(np.isfinite(X.values)):
            logger.error("Non-finite values still present after cleaning")
            # Replace any remaining non-finite values with 0 as last resort
            X = X.replace([np.inf, -np.inf, np.nan], 0)
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Data validation complete - all values are finite and within safe ranges")
        
        return X, y
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> pd.DataFrame:
        """
        Select top k features using f_classif — selects features most informative
        for the direction target (UP/DOWN), not for return magnitude.
        """
        selector_key = f"{self.prediction_horizon}_selector"

        if selector_key not in self.feature_selectors:
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            self.feature_selectors[selector_key] = selector

            selected_features = X.columns[selector.get_support()].tolist()
            logger.info(f"Selected {len(selected_features)} features for {self.prediction_horizon}: {selected_features[:10]}...")
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
    
    def train_xgboost_classifier(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train a calibrated XGBoost classifier for direction prediction.

        Three-step process:
          1. Early stopping on a time-ordered holdout (last 15 % of data) to find
             the optimal number of trees — prevents overfitting to training noise.
          2. TimeSeriesSplit cross-validation on the base model for an unbiased
             accuracy estimate that is reported in logs/metadata.
          3. CalibratedClassifierCV (isotonic, TimeSeriesSplit) wraps the final
             base model so that predict_proba() outputs genuine probabilities that
             correlate with empirical accuracy rather than raw XGBoost scores.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.dummy import DummyClassifier

        # ── Convert target ───────────────────────────────────────────────────
        y_binary = (y > 0).astype(int)
        unique_classes = y_binary.nunique()
        class_counts = y_binary.value_counts()

        # ── Dummy classifier for degenerate datasets ──────────────────────────
        if unique_classes < 2:
            logger.warning(f"Only {unique_classes} class(es) in target — using dummy classifier")
            dummy = DummyClassifier(strategy='most_frequent')
            dummy.fit(X, y_binary)
            self.models[f"{self.prediction_horizon}_xgb_classifier"] = dummy
            baseline = float(max(class_counts)) / len(y_binary)
            return {
                'model_type': 'dummy_classifier',
                'cv_accuracy_mean': baseline,
                'cv_accuracy_std': 0.0,
                'train_accuracy': baseline,
                'best_n_estimators': 0,
                'feature_importance': {col: 0.0 for col in X.columns},
            }

        min_class_pct = float(min(class_counts)) / len(y_binary)
        if min_class_pct < 0.05:
            logger.warning(f"Severe class imbalance: minority = {min_class_pct:.2%}")

        n_samples = len(X)
        min_class_size = int(min(class_counts))

        # How many CV folds can the data support?
        max_folds = min(
            config.MODEL_SETTINGS['cross_validation_folds'],
            max(2, n_samples // 10),
            max(2, min_class_size // 2),
        )

        # ── Base XGBoost params ───────────────────────────────────────────────
        xgb_params = config.MODEL_SETTINGS['xgboost_params'].copy()
        xgb_params['objective'] = 'binary:logistic'
        if min_class_pct < 0.2:
            xgb_params['scale_pos_weight'] = float(max(class_counts)) / float(min(class_counts))
            logger.info(f"Class balancing: scale_pos_weight={xgb_params['scale_pos_weight']:.2f}")

        # ── Step 1: Early stopping to find optimal n_estimators ───────────────
        early_stopping_rounds = config.MODEL_SETTINGS.get('early_stopping_rounds', 30)
        best_n_estimators = xgb_params.get('n_estimators', 100)  # safe fallback

        n_val = max(int(n_samples * 0.15), 50)
        if n_samples - n_val >= 100:
            try:
                X_tr = X.iloc[:n_samples - n_val]
                X_val = X.iloc[n_samples - n_val:]
                y_tr = y_binary.iloc[:n_samples - n_val]
                y_val = y_binary.iloc[n_samples - n_val:]

                es_params = xgb_params.copy()
                es_params['n_estimators'] = 1000  # high ceiling; early stopping prunes it
                es_params['early_stopping_rounds'] = early_stopping_rounds  # XGBoost 2.x: constructor
                es_model = xgb.XGBClassifier(**es_params)
                es_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                best_n_estimators = max(10, es_model.best_iteration + 1)
                logger.info(f"Early stopping: optimal n_estimators={best_n_estimators} "
                            f"({self.crypto_name} {self.prediction_horizon})")
            except Exception as es_err:
                logger.warning(f"Early stopping failed ({es_err}), "
                               f"using fallback n_estimators={best_n_estimators}")
        else:
            logger.info(f"Dataset too small for early stopping ({n_samples} rows), "
                        f"using n_estimators={best_n_estimators}")

        # Final params with early-stopping-selected tree count
        final_params = xgb_params.copy()
        final_params['n_estimators'] = best_n_estimators

        # ── Step 2: TimeSeriesSplit CV on base model (accuracy reporting) ─────
        cv_scores = np.array([0.5])
        try:
            if max_folds >= 2:
                tscv = TimeSeriesSplit(n_splits=max_folds)
                cv_model = xgb.XGBClassifier(**final_params)
                cv_scores = cross_val_score(cv_model, X, y_binary, cv=tscv, scoring='accuracy')
                logger.info(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        except Exception as cv_err:
            logger.warning(f"CV failed ({cv_err}), reporting 0.5 fallback")

        # ── Step 3: Calibrated final model ───────────────────────────────────
        cal_folds = max(2, min(config.MODEL_SETTINGS.get('calibration_folds', 3), max_folds))
        model_type = 'dummy_classifier'
        feature_importance: Dict = {}
        accuracy = float(max(class_counts)) / len(y_binary)

        try:
            base_model = xgb.XGBClassifier(**final_params)
            tscv_cal = TimeSeriesSplit(n_splits=cal_folds)
            model = CalibratedClassifierCV(base_model, cv=tscv_cal, method='isotonic')
            model.fit(X, y_binary)

            # Average feature importances across the calibrated sub-estimators
            importances = [
                cc.estimator.feature_importances_
                for cc in model.calibrated_classifiers_
                if hasattr(cc.estimator, 'feature_importances_')
            ]
            if importances:
                feature_importance = dict(zip(X.columns, np.mean(importances, axis=0)))

            y_pred = model.predict(X)
            accuracy = float(accuracy_score(y_binary, y_pred))
            model_type = 'calibrated_xgb'

        except Exception as cal_err:
            logger.error(f"Calibrated model failed ({cal_err}), falling back to plain XGBoost")
            try:
                model = xgb.XGBClassifier(**final_params)
                model.fit(X, y_binary)
                y_pred = model.predict(X)
                accuracy = float(accuracy_score(y_binary, y_pred))
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                model_type = 'xgb_classifier'
            except Exception as plain_err:
                logger.error(f"Plain XGBoost also failed ({plain_err}), using dummy classifier")
                model = DummyClassifier(strategy='most_frequent')
                model.fit(X, y_binary)
                y_pred = model.predict(X)
                accuracy = float(accuracy_score(y_binary, y_pred))
                feature_importance = {col: 0.0 for col in X.columns}

        self.models[f"{self.prediction_horizon}_xgb_classifier"] = model

        results = {
            'model_type': model_type,
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'train_accuracy': accuracy,
            'best_n_estimators': best_n_estimators,
            'feature_importance': feature_importance,
            'class_distribution': class_counts.to_dict(),
            'min_class_percentage': min_class_pct * 100,
        }

        logger.info(f"{model_type} trained ({self.crypto_name} {self.prediction_horizon}) — "
                    f"CV: {cv_scores.mean():.3f}, best_n_est: {best_n_estimators}, "
                    f"train_acc: {accuracy:.3f}")
        return results
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train a direction classifier for the given horizon.
        Feature selection uses f_classif against the binary direction target so
        the 50 chosen features are the ones that actually predict UP/DOWN, not
        return magnitude.
        """
        logger.info(f"Training direction classifier for {self.crypto_name} - {self.prediction_horizon}")

        clf_target_map = {
            '15m': 'target_direction_15m',
            '1h':  'target_direction_1h',
            '4h':  'target_direction_4h',
        }

        if self.prediction_horizon not in clf_target_map:
            raise ValueError(f"Invalid prediction horizon: {self.prediction_horizon}")

        classification_target = clf_target_map[self.prediction_horizon]

        clean_df = df.dropna(subset=[classification_target]).copy()
        if clean_df.empty:
            raise ValueError(f"No valid data for target {classification_target}")

        # Prepare features and direction target together
        X, y_clf = self.prepare_data(clean_df, classification_target)

        # Feature selection driven by direction target
        X_selected = self.feature_selection(X, y_clf, k=config.MODEL_SETTINGS['feature_selection_k'])

        # Scale
        X_scaled = self.scale_features(X_selected, fit=True)

        # Train classifier only
        classification_results = self.train_xgboost_classifier(X_scaled, y_clf)

        training_info = {
            'timestamp': datetime.now(),
            'crypto': self.crypto_name,
            'horizon': self.prediction_horizon,
            'training_samples': len(X),
            'features_used': len(X_scaled.columns),
            'classification_results': classification_results,
        }

        self.training_history.append(training_info)
        logger.info(
            f"Classifier trained — {self.crypto_name} {self.prediction_horizon} | "
            f"CV acc: {classification_results['cv_accuracy_mean']:.3f} | "
            f"train acc: {classification_results['train_accuracy']:.3f}"
        )
        return training_info
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict direction (UP=1 / DOWN=0) for the latest bar.
        Returns direction, raw up-probability, and confidence (distance from 0.5).
        """
        if not self.models:
            raise ValueError("No trained models available. Please train first.")

        try:
            latest_data = df.iloc[[-1]].copy()

            exclude_cols = [
                'datetime', 'crypto',
                'target_direction_15m', 'target_direction_1h', 'target_direction_4h',
                'target_datetime_15m', 'target_datetime_1h', 'target_datetime_4h',
            ]

            X = latest_data[[c for c in latest_data.columns if c not in exclude_cols]].fillna(0)

            selector = self.feature_selectors[f"{self.prediction_horizon}_selector"]
            X_selected = pd.DataFrame(
                selector.transform(X),
                columns=X.columns[selector.get_support()],
                index=X.index,
            )

            scaler = self.scalers[f"{self.prediction_horizon}_scaler"]
            X_scaled = pd.DataFrame(
                scaler.transform(X_selected),
                columns=X_selected.columns,
                index=X_selected.index,
            )

            classifier = self.models[f"{self.prediction_horizon}_xgb_classifier"]
            direction_prob = classifier.predict_proba(X_scaled)[0, 1]  # P(UP)
            predicted_direction = int(direction_prob > 0.5)

            current_price = latest_data['price'].iloc[0]

            feature_datetime = latest_data['datetime'].iloc[0]
            if isinstance(feature_datetime, str):
                feature_datetime = pd.to_datetime(feature_datetime)

            offsets = {'15m': pd.Timedelta(minutes=15), '1h': pd.Timedelta(hours=1), '4h': pd.Timedelta(hours=4)}
            target_datetime = feature_datetime + offsets.get(self.prediction_horizon, pd.Timedelta(0))

            return {
                'timestamp': target_datetime,
                'feature_timestamp': feature_datetime,
                'crypto': self.crypto_name,
                'horizon': self.prediction_horizon,
                'current_price': current_price,
                'predicted_direction': predicted_direction,   # 1 = UP, 0 = DOWN
                'direction_prob': direction_prob,             # raw P(UP), 0-1
                'model_confidence': max(direction_prob, 1 - direction_prob),  # distance from 0.5
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
    from data_collector import DataCollector
    from feature_engineering import FeatureEngineer

    collector = DataCollector()
    fe = FeatureEngineer()

    btc_data = collector.get_crypto_data('bitcoin', days=30)
    btc_features = fe.prepare_features(btc_data)

    model = CryptoPredictionModel('bitcoin', '15m')
    training_results = model.train(btc_features)

    print("Training Results:")
    print(f"CV Accuracy: {training_results['classification_results']['cv_accuracy_mean']:.4f}")
    print(f"Train Accuracy: {training_results['classification_results']['train_accuracy']:.4f}")

    prediction = model.predict(btc_features)
    if prediction:
        direction = "UP" if prediction['predicted_direction'] == 1 else "DOWN"
        print(f"\nPrediction for {prediction['crypto']} ({prediction['horizon']}):")
        print(f"Current Price:  ${prediction['current_price']:.2f}")
        print(f"Direction:      {direction}")
        print(f"P(UP):          {prediction['direction_prob']:.3f}")
        print(f"Confidence:     {prediction['model_confidence']:.3f}")