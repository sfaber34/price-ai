"""
Single shared training pipeline used by ALL entry points:
  - crypto_prediction_bot.py  (live prediction)
  - train_optimal_models.py   (production model training)
  - evaluate_calibration.py   (calibration diagnostic)

If you change how data is fetched, features are built, or models are trained,
change it HERE and it takes effect everywhere.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from ml_predictor import CryptoPredictionModel

logger = logging.getLogger(__name__)


# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_data_for_crypto(
    collector: DataCollector,
    crypto: str,
    days: int,
    fetch_1m: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch 15m bars and (optionally) 1m bars for a single crypto.

    Returns {'15m': df_15m, '1m': df_1m}.  Either value may be an empty
    DataFrame if the fetch fails.
    """
    result: Dict[str, pd.DataFrame] = {'15m': pd.DataFrame(), '1m': pd.DataFrame()}

    try:
        df = collector.get_crypto_data(crypto, days=days)
        if not df.empty:
            result['15m'] = df
            logger.info(f"Fetched {len(df)} 15m bars for {crypto}")
    except Exception as e:
        logger.error(f"15m fetch failed for {crypto}: {e}")

    if fetch_1m:
        try:
            df_1m = collector.get_crypto_data_1m(crypto, days=days)
            if not df_1m.empty:
                result['1m'] = df_1m
                logger.info(f"Fetched {len(df_1m)} 1m bars for {crypto}")
        except Exception as e:
            logger.warning(f"1m fetch failed for {crypto}: {e}")

    return result


def fetch_all_cryptos(
    collector: DataCollector,
    days: int,
    fetch_1m: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch data for every crypto in config.CRYPTOCURRENCIES.

    Returns {crypto: {'15m': df, '1m': df}} for each crypto.
    """
    return {
        crypto: fetch_data_for_crypto(collector, crypto, days, fetch_1m)
        for crypto in config.CRYPTOCURRENCIES
    }


# ── Feature preparation ─────────────────────────────────────────────────────

def prepare_features_for_crypto(
    fe: FeatureEngineer,
    df_15m: pd.DataFrame,
    df_1m: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Run the feature pipeline for a single crypto.  This is the ONLY place
    features are built — every caller uses this function.

    Matches the proven evaluate_calibration pipeline:
      15m bars + 1m intrabar only, no cross-asset, no funding, no OI.
    """
    external_data = {'intrabar_1m': df_1m if df_1m is not None and not df_1m.empty else None}
    return fe.prepare_features(df_15m, external_data=external_data)


def prepare_all_cryptos(
    fe: FeatureEngineer,
    raw_data: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """
    Prepare features for every crypto.  raw_data is the output of
    fetch_all_cryptos().
    """
    prepared = {}
    for crypto, data in raw_data.items():
        df_15m = data.get('15m', pd.DataFrame())
        if df_15m.empty:
            continue
        try:
            df = prepare_features_for_crypto(fe, df_15m, data.get('1m'))
            if not df.empty:
                prepared[crypto] = df
                logger.info(f"{crypto}: {df.shape[1]} features, {len(df)} bars")
        except Exception as e:
            logger.error(f"Feature preparation failed for {crypto}: {e}")
    return prepared


# ── Model training ───────────────────────────────────────────────────────────

def train_model(
    crypto: str,
    horizon: str,
    train_df: pd.DataFrame,
) -> CryptoPredictionModel:
    """
    Train a single model.  Returns the fitted CryptoPredictionModel.
    """
    model = CryptoPredictionModel(crypto, horizon)
    model.train(train_df)
    return model


# ── Vectorised batch prediction ──────────────────────────────────────────────

def predict_proba_batch(
    model: CryptoPredictionModel,
    df: pd.DataFrame,
    horizon: str,
) -> np.ndarray:
    """
    Get P(UP) for every row in df using a trained model.

    This is the ONLY prediction path for batch evaluation — evaluate_calibration
    and backtesting both call this instead of reaching into model internals.
    """
    target_col = f'target_direction_{horizon}'
    X, y = model.prepare_data(df, target_col)

    selector = model.feature_selectors[f"{horizon}_selector"]
    scaler = model.scalers[f"{horizon}_scaler"]
    classifier = model.models[f"{horizon}_xgb_classifier"]

    X_sel = pd.DataFrame(
        selector.transform(X),
        columns=X.columns[selector.get_support()],
        index=X.index,
    )
    X_scaled = pd.DataFrame(
        scaler.transform(X_sel),
        columns=X_sel.columns,
        index=X_sel.index,
    )

    probs = classifier.predict_proba(X_scaled)[:, 1]
    return probs, (y > 0).astype(int).values
