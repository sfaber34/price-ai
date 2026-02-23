#!/usr/bin/env python3
"""
Lightweight Flask API that serves the most recent BTC 15m prediction.

Usage:
    python3 prediction_api.py

Endpoint:
    GET /predictions?key=<PREDICTION_API_KEY>

External access:
    http://409tedmon.tplinkdns.com:<PORT>/predictions?key=<your-key>

Set PREDICTION_API_KEY in .env before exposing externally.
Port-forward the port below on your TP-Link router to this machine.
"""
import os
import sqlite3
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY     = os.getenv("PREDICTION_API_KEY", "")
DB_PATH     = os.getenv("DATABASE_PATH", "crypto_predictions.db")
PORT        = int(os.getenv("API_PORT", 8080))


def _get_latest_prediction(crypto: str, horizon: str) -> dict | None:
    """Return the most recent prediction row as a dict, or None."""
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        cur = con.execute(
            """
            SELECT
                id,
                crypto,
                prediction_horizon,
                predicted_price,   -- this is actually P(UP), 0–1
                current_price,
                confidence,
                datetime       AS target_time,
                created_at     AS predicted_at
            FROM predictions
            WHERE crypto = ? AND prediction_horizon = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (crypto, horizon),
        )
        row = cur.fetchone()
        con.close()
        return dict(row) if row else None
    except Exception as e:
        app.logger.error(f"DB error: {e}")
        return None


def _format_market(row: dict) -> dict:
    """Extract per-market fields from a prediction row."""
    prob_up = float(row["predicted_price"])
    return {
        "direction":     "UP" if prob_up >= 0.5 else "DOWN",
        "confidence":    round(float(row["confidence"]), 4),
        "current_price": round(float(row["current_price"]), 2),
    }


@app.route("/predictions")
def predictions():
    # --- auth ---
    if not API_KEY:
        return jsonify({"error": "API key not configured on server"}), 500

    provided = request.args.get("key", "")
    if provided != API_KEY:
        return jsonify({"error": "unauthorized"}), 401

    # --- fetch both markets ---
    btc_row = _get_latest_prediction("bitcoin", "15m")
    eth_row = _get_latest_prediction("ethereum", "15m")

    if btc_row is None and eth_row is None:
        return jsonify({"error": "no predictions available yet — is the bot running?"}), 404

    # Use whichever row exists for the shared timestamp fields
    ref = btc_row or eth_row
    predicted_at = datetime.fromisoformat(ref["predicted_at"])
    prediction_age_ms = int((datetime.now() - predicted_at).total_seconds() * 1000)

    result = {
        "predicted_at":      ref["predicted_at"],
        "target_time":       ref["target_time"],
        "prediction_age_ms": prediction_age_ms,
        "as_of":             datetime.now(timezone.utc).isoformat(),
        "markets": {},
    }

    if btc_row:
        result["markets"]["bitcoin_15m"] = _format_market(btc_row)
    if eth_row:
        result["markets"]["ethereum_15m"] = _format_market(eth_row)

    return jsonify(result)


@app.route("/health")
def health():
    """Unauthenticated health check."""
    return jsonify({"status": "ok", "as_of": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    if not API_KEY or API_KEY == "change-me-to-something-secret":
        print("⚠️  WARNING: Set PREDICTION_API_KEY in .env before exposing this externally!")

    print(f"🚀 Prediction API running on port {PORT}")
    print(f"   Local:    http://localhost:{PORT}/predictions?key=<your-key>")
    print(f"   External: http://409tedmon.tplinkdns.com:{PORT}/predictions?key=<your-key>")
    app.run(host="0.0.0.0", port=PORT, debug=False)
