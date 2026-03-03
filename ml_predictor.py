"""
Machine Learning prediction engine for crypto price prediction
Uses XGBoost and ensemble methods for multi-horizon predictions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator as _FrozenEstimator
except ImportError:
    _FrozenEstimator = None  # sklearn < 1.6 — fall back to cv='prefit'
import xgboost as xgb
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
import config

logger = logging.getLogger(__name__)


class TreeBasedSelector:
    """Drop-in replacement for SelectKBest that uses a shallow XGBoost to rank
    features by tree-based importance.  Exposes fit/transform/get_support so the
    rest of the pipeline (predict, evaluate_calibration, training_pipeline) works
    unchanged."""

    def __init__(self, k: int = 50):
        self.k = k
        self._support_mask: Optional[np.ndarray] = None
        self._selected_indices: Optional[np.ndarray] = None

    def fit(self, X, y):
        n_features = X.shape[1]
        k = min(self.k, n_features)

        preliminary = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        preliminary.fit(X, y)

        importances = preliminary.feature_importances_
        # Pick top-k indices (descending importance)
        self._selected_indices = np.argsort(importances)[::-1][:k]
        self._support_mask = np.zeros(n_features, dtype=bool)
        self._support_mask[self._selected_indices] = True

        top_names = []
        if hasattr(X, 'columns'):
            top_names = [X.columns[i] for i in self._selected_indices[:5]]
        logger.info(f"TreeBasedSelector: kept {k}/{n_features} features, "
                    f"top-5: {top_names}")
        return self

    def transform(self, X):
        if self._support_mask is None:
            raise RuntimeError("TreeBasedSelector has not been fitted yet")
        if hasattr(X, 'iloc'):
            return X.iloc[:, self._support_mask].values
        return X[:, self._support_mask]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        if self._support_mask is None:
            raise RuntimeError("TreeBasedSelector has not been fitted yet")
        return self._support_mask

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
            'target_direction_15m',
            'target_datetime_15m',
        ]
        
        feature_cols = sorted(col for col in clean_df.columns if col not in exclude_cols)

        # Forward-fill first so early-window rolling NaNs inherit the first valid
        # value rather than being treated as the literal number 0 (which is a real
        # value for many normalised features and would teach the model a spurious signal).
        X = clean_df[feature_cols].ffill().fillna(0)
        y = clean_df[target_col]
        # Ensure integer class labels (0/1) — boolean/object targets break
        # mutual_info_classif and some classifiers.  The target column can be
        # numpy bool, object-dtype with Python True/False, or already int.
        if y.dtype == bool or y.dtype == object:
            y = y.astype(int)
        
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
        Select top k features using a preliminary shallow XGBoost to rank by
        tree-based feature importance.  Captures nonlinear relationships that
        the previous f_classif (linear ANOVA) missed.
        """
        selector_key = f"{self.prediction_horizon}_selector"

        if selector_key not in self.feature_selectors:
            selector = TreeBasedSelector(k=min(k, X.shape[1]))
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
    
    @staticmethod
    def _compute_sample_weights(n_samples: int, decay: float) -> Optional[np.ndarray]:
        """Compute exponential-decay sample weights.

        weight[i] = exp(-decay * (n-1-i) / n)  — i=0 oldest, i=n-1 newest.
        decay=0 returns None (uniform).  decay=1 gives newest=1.0, oldest≈0.37.
        """
        if decay <= 0:
            return None
        idx = np.arange(n_samples, dtype=np.float64)
        weights = np.exp(-decay * (n_samples - 1 - idx) / n_samples)
        logger.info(f"Sample weights: decay={decay}, oldest={weights[0]:.3f}, "
                    f"newest={weights[-1]:.3f}, ratio={weights[-1]/weights[0]:.2f}x")
        return weights

    def train_xgboost_classifier(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train an XGBoost classifier for direction prediction.

        Four-step process:
          1. 80/20 time-ordered split: XGBoost trains on the first 80 %,
             the last 20 % is held out purely for isotonic calibration so
             the calibrator never sees its own training data.
          2. Early stopping on a held-out tail of the training portion to find
             the optimal number of trees.
          3. Bayesian hyperparameter tuning with Optuna (TimeSeriesSplit CV,
             AUC objective — better aligned with probability quality than accuracy).
             Search space includes gamma (tree-growth regularisation) and a wider
             reg_lambda range; max_depth is capped at 4 to prevent overfitting on
             noisy 15 m financial data.
          4. Isotonic calibration on the 20 % holdout so reported confidence
             scores reflect true probabilities rather than raw XGBoost scores.

        Sample weights with exponential decay are applied to all XGBoost fit()
        calls so recent bars have more influence than old ones.
        """
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

        n_samples = len(X)

        # ── 80/20 time-ordered split ──────────────────────────────────────────
        # Training portion: first 80 % — used for early stopping, Optuna, final fit.
        # Calibration portion: last 20 % — held out exclusively for isotonic calibration.
        n_train = int(n_samples * 0.80)
        n_cal   = n_samples - n_train
        X_train = X.iloc[:n_train]
        y_train = y_binary.iloc[:n_train]
        X_cal   = X.iloc[n_train:]
        y_cal   = y_binary.iloc[n_train:]

        min_class_pct = float(min(class_counts)) / len(y_binary)
        if min_class_pct < 0.05:
            logger.warning(f"Severe class imbalance: minority = {min_class_pct:.2%}")

        train_class_counts = y_train.value_counts()
        min_class_size = int(min(train_class_counts)) if len(train_class_counts) > 1 else 1

        # How many CV folds can the training portion support?
        max_folds = min(
            config.MODEL_SETTINGS['cross_validation_folds'],
            max(2, n_train // 10),
            max(2, min_class_size // 2),
        )

        # ── Sample weights (exponential recency decay, training portion only) ─
        decay = config.MODEL_SETTINGS.get('sample_weight_decay', 0.0)
        sample_weights = self._compute_sample_weights(n_train, decay)

        # ── Base XGBoost params ───────────────────────────────────────────────
        xgb_params = config.MODEL_SETTINGS['xgboost_params'].copy()
        xgb_params['objective'] = 'binary:logistic'
        if min_class_pct < 0.2:
            xgb_params['scale_pos_weight'] = float(max(class_counts)) / float(min(class_counts))
            logger.info(f"Class balancing: scale_pos_weight={xgb_params['scale_pos_weight']:.2f}")

        # ── Step 1: Early stopping within training portion ────────────────────
        early_stopping_rounds = config.MODEL_SETTINGS.get('early_stopping_rounds', 30)
        best_n_estimators = xgb_params.get('n_estimators', 100)  # safe fallback

        n_val = max(int(n_train * 0.15), 50)
        if n_train - n_val >= 100:
            try:
                X_tr  = X_train.iloc[:n_train - n_val]
                X_val = X_train.iloc[n_train - n_val:]
                y_tr  = y_train.iloc[:n_train - n_val]
                y_val = y_train.iloc[n_train - n_val:]

                es_weights = sample_weights[:n_train - n_val] if sample_weights is not None else None

                es_params = xgb_params.copy()
                es_params['n_estimators'] = 1000  # high ceiling; early stopping prunes it
                es_params['early_stopping_rounds'] = early_stopping_rounds
                es_model = xgb.XGBClassifier(**es_params)
                es_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    sample_weight=es_weights,
                    verbose=False,
                )
                best_n_estimators = max(10, es_model.best_iteration + 1)
                logger.info(f"Early stopping: optimal n_estimators={best_n_estimators} "
                            f"({self.crypto_name} {self.prediction_horizon})")
            except Exception as es_err:
                logger.warning(f"Early stopping failed ({es_err}), "
                               f"using fallback n_estimators={best_n_estimators}")
        else:
            logger.info(f"Training portion too small for early stopping ({n_train} rows), "
                        f"using n_estimators={best_n_estimators}")

        # ── Step 2: Optuna hyperparameter tuning (AUC objective) ──────────────
        optuna_trials = config.MODEL_SETTINGS.get('optuna_trials', 0)
        best_tuned_params: Dict = {}

        if optuna_trials > 0:
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def objective(trial):
                    params = xgb_params.copy()
                    params['n_estimators']     = best_n_estimators
                    # max_depth capped at 4: depth-6 trees overfit on noisy 15 m data
                    params['max_depth']        = trial.suggest_int('max_depth', 2, 4)
                    params['learning_rate']    = trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
                    params['subsample']        = trial.suggest_float('subsample', 0.6, 1.0)
                    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
                    params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
                    # gamma: min loss-reduction to make a split — key anti-overfit knob
                    params['gamma']            = trial.suggest_float('gamma', 0.0, 2.0)
                    params['reg_alpha']        = trial.suggest_float('reg_alpha', 0.01, 1.0, log=True)
                    # reg_lambda upper bound widened to 10: stronger L2 often helps on noisy data
                    params['reg_lambda']       = trial.suggest_float('reg_lambda', 0.5, 10.0)

                    tscv = TimeSeriesSplit(n_splits=max(2, max_folds))
                    scores = []
                    for train_idx, val_idx in tscv.split(X_train):
                        m = xgb.XGBClassifier(**params)
                        sw = sample_weights[train_idx] if sample_weights is not None else None
                        m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                              sample_weight=sw, verbose=False)
                        val_probs = m.predict_proba(X_train.iloc[val_idx])[:, 1]
                        # AUC — optimises probability ranking, not just 50 % threshold accuracy
                        if len(np.unique(y_train.iloc[val_idx])) == 2:
                            scores.append(roc_auc_score(y_train.iloc[val_idx], val_probs))
                    return float(np.mean(scores)) if scores else 0.5

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)

                best_tuned_params = study.best_params
                logger.info(f"Optuna best CV AUC: {study.best_value:.4f} "
                            f"(n_trials={optuna_trials})")
                logger.info(f"Optuna best params: {best_tuned_params}")

            except ImportError:
                logger.warning("optuna not installed — skipping hyperparameter tuning, "
                               "using hardcoded params")
            except Exception as opt_err:
                logger.warning(f"Optuna tuning failed ({opt_err}), using hardcoded params")

        # Build final params: start from config defaults, overlay Optuna best
        final_params = xgb_params.copy()
        final_params['n_estimators'] = best_n_estimators
        final_params.update(best_tuned_params)

        # ── CV AUC reporting (uses final params on training portion) ──────────
        cv_scores = np.array([0.5])
        try:
            if max_folds >= 2:
                tscv = TimeSeriesSplit(n_splits=max_folds)
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    cv_model = xgb.XGBClassifier(**final_params)
                    sw = sample_weights[train_idx] if sample_weights is not None else None
                    cv_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                                 sample_weight=sw, verbose=False)
                    val_probs = cv_model.predict_proba(X_train.iloc[val_idx])[:, 1]
                    if len(np.unique(y_train.iloc[val_idx])) == 2:
                        scores.append(roc_auc_score(y_train.iloc[val_idx], val_probs))
                if scores:
                    cv_scores = np.array(scores)
                logger.info(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        except Exception as cv_err:
            logger.warning(f"CV failed ({cv_err}), reporting 0.5 fallback")

        # ── Step 3: Train final XGBoost on training portion ───────────────────
        model_type = 'dummy_classifier'
        feature_importance: Dict = {}
        accuracy = float(max(class_counts)) / len(y_binary)

        try:
            model = xgb.XGBClassifier(**final_params)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            y_pred = model.predict(X_train)
            accuracy = float(accuracy_score(y_train, y_pred))
            model_type = 'xgb_classifier'

        except Exception as train_err:
            logger.error(f"XGBoost training failed ({train_err}), falling back to plain XGBoost")
            try:
                fallback_params = xgb_params.copy()
                fallback_params['n_estimators'] = best_n_estimators
                model = xgb.XGBClassifier(**fallback_params)
                model.fit(X_train, y_train, sample_weight=sample_weights)
                y_pred = model.predict(X_train)
                accuracy = float(accuracy_score(y_train, y_pred))
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                model_type = 'xgb_classifier'
            except Exception as plain_err:
                logger.error(f"Plain XGBoost also failed ({plain_err}), using dummy classifier")
                model = DummyClassifier(strategy='most_frequent')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_train)
                accuracy = float(accuracy_score(y_train, y_pred))
                feature_importance = {col: 0.0 for col in X_train.columns}

        # ── Step 4: Isotonic calibration on held-out 20 % ─────────────────────
        # The XGBoost is already trained; we only fit the isotonic monotone
        # mapping on the unseen calibration holdout.
        # sklearn ≥ 1.6: use FrozenEstimator to avoid the cv='prefit' deprecation.
        # sklearn < 1.6: fall back to cv='prefit' (still works, just deprecated).
        final_model = model
        if n_cal >= 30 and model_type == 'xgb_classifier':
            try:
                # sigmoid (Platt scaling) over isotonic: fits only 2 parameters
                # (A·f + B) so it generalises across market regimes and is
                # mathematically guaranteed to be monotone out-of-sample.
                # Isotonic is a staircase with N steps — it memorises the
                # calibration set and can break monotonicity on unseen data.
                if _FrozenEstimator is not None:
                    calibrated = CalibratedClassifierCV(
                        estimator=_FrozenEstimator(model), method='sigmoid'
                    )
                else:
                    calibrated = CalibratedClassifierCV(
                        estimator=model, cv='prefit', method='sigmoid'
                    )
                calibrated.fit(X_cal, y_cal)
                final_model = calibrated
                logger.info(f"Isotonic calibration fitted on {n_cal} held-out samples "
                            f"({self.crypto_name} {self.prediction_horizon})")
            except Exception as cal_err:
                logger.warning(f"Isotonic calibration failed ({cal_err}), "
                               f"storing uncalibrated model")

        self.models[f"{self.prediction_horizon}_xgb_classifier"] = final_model

        results = {
            'model_type': model_type,
            'cv_accuracy_mean': float(cv_scores.mean()),  # AUC score
            'cv_accuracy_std': float(cv_scores.std()),
            'train_accuracy': accuracy,
            'best_n_estimators': best_n_estimators,
            'feature_importance': feature_importance,
            'class_distribution': class_counts.to_dict(),
            'min_class_percentage': min_class_pct * 100,
        }
        if best_tuned_params:
            results['optuna_best_params'] = best_tuned_params

        logger.info(f"{model_type} trained ({self.crypto_name} {self.prediction_horizon}) — "
                    f"CV AUC: {cv_scores.mean():.3f}, best_n_est: {best_n_estimators}, "
                    f"train_acc: {accuracy:.3f}")
        return results
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train a direction classifier for the given horizon.
        Feature selection uses a preliminary shallow XGBoost to rank features by
        tree-based importance, keeping the top-k that predict UP/DOWN.
        """
        logger.info(f"Training direction classifier for {self.crypto_name} - {self.prediction_horizon}")

        clf_target_map = {
            '15m': 'target_direction_15m',
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

        # Train classifier only (no StandardScaler — XGBoost is scale-invariant)
        classification_results = self.train_xgboost_classifier(X_selected, y_clf)

        training_info = {
            'timestamp': datetime.now(timezone.utc).replace(tzinfo=None),
            'crypto': self.crypto_name,
            'horizon': self.prediction_horizon,
            'training_samples': len(X),
            'features_used': len(X_selected.columns),
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

            if not self.feature_columns:
                raise ValueError("Model has no saved feature_columns — was it loaded correctly?")

            missing = [c for c in self.feature_columns if c not in latest_data.columns]
            if missing:
                logger.warning(f"{len(missing)} features missing in live data, filling with 0: {missing[:5]}...")
                for c in missing:
                    latest_data[c] = 0.0

            X = latest_data[self.feature_columns].fillna(0)

            selector = self.feature_selectors[f"{self.prediction_horizon}_selector"]
            X_selected = pd.DataFrame(
                selector.transform(X),
                columns=X.columns[selector.get_support()],
                index=X.index,
            )

            # No StandardScaler — XGBoost is scale-invariant; calibrated probabilities
            # are produced directly by the CalibratedClassifierCV wrapper.
            classifier = self.models[f"{self.prediction_horizon}_xgb_classifier"]
            direction_prob = classifier.predict_proba(X_selected)[0, 1]  # P(UP)
            predicted_direction = int(direction_prob >= 0.5)  # >= matches Polymarket rules

            current_price = latest_data['price'].iloc[0]

            feature_datetime = latest_data['datetime'].iloc[0]
            if isinstance(feature_datetime, str):
                feature_datetime = pd.to_datetime(feature_datetime)

            # datetime = bar open_time; prediction targets the NEXT bar's close vs open
            # (Polymarket-style). Target time = open_time + 15m (this bar) + 15m (next bar).
            offsets = {'15m': pd.Timedelta(minutes=30)}
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