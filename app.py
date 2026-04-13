"""
Clymb Data Provenance API — EU AI Act Compliance Service
=========================================================
Powered by Patent 24 (GB2607042.5) — McCaffer-Bovill Quality Decay

Endpoints:
  POST /classify    — Classify content as HUMAN/MACHINE/INDETERMINATE
  POST /gate        — Gate training data (admit/reject with provenance)
  POST /decay       — Predict quality decay at generation depth N
  POST /watermark   — Embed provenance watermark in content
  POST /verify      — Verify content provenance and extract watermark
  POST /certify     — Issue a signed provenance certificate
  GET  /health      — Service health check

No ARIA internals are exposed. This is a clean API wrapper.

Author: Clymb Ltd — Patent 24 (GB2607042.5)
"""
from flask import Flask, request, jsonify, redirect, send_from_directory
import hashlib
import hmac
import math
import gzip
import json
import time
import os
import re
import stripe
from datetime import datetime
from collections import Counter
from functools import wraps

app = Flask(__name__, static_folder=".")
app.config["SECRET_KEY"] = os.environ.get("API_SECRET", "clymb-provenance-2026")

# Stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
PRICE_PROFESSIONAL = os.environ.get("STRIPE_PRICE_PRO", "")
PRICE_ENTERPRISE = os.environ.get("STRIPE_PRICE_ENT", "")

# Simple API key auth
API_KEYS = set(os.environ.get("API_KEYS", "demo-key-2026").split(","))


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if key not in API_KEYS:
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated


# ============================================================
# McCaffer-Bovill Quality Decay (Patent 24, Section 4.2)
# ============================================================

def mccaffer_bovill_quality(n, theta_0=0.1):
    """F(n) = cos^2(theta_n / 2) where theta_n = theta_0 * sqrt(n)"""
    theta_n = theta_0 * math.sqrt(max(0, n))
    return math.cos(theta_n / 2) ** 2


def collapse_generation(theta_0=0.1, threshold=0.5):
    """Compute n_collapse: generation where quality drops below threshold."""
    acos_val = math.acos(math.sqrt(threshold))
    return int((2 * acos_val / theta_0) ** 2)


# ============================================================
# Data Origin Classification (Patent 24, Section 5.2)
# ============================================================

HUMAN_COMPRESSION_RATIO = 0.80
HUMAN_ENTROPY = 4.1
HUMAN_UNIQUE_RATIO = 0.85
HUMAN_SENTENCE_LEN = 12
T_HUMAN = 1.0
T_MACHINE = 2.0


def extract_features(text):
    """Extract statistical features for origin classification."""
    features = {}
    text_bytes = text.encode("utf-8")
    compressed = gzip.compress(text_bytes)
    features["compression_ratio"] = len(compressed) / max(1, len(text_bytes))

    char_freq = Counter(text.lower())
    total = max(1, len(text))
    entropy = -sum((c/total) * math.log2(c/total) for c in char_freq.values() if c > 0)
    features["char_entropy"] = entropy

    words = text.lower().split()
    features["unique_word_ratio"] = len(set(words)) / max(1, len(words))

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features["avg_sentence_length"] = sum(len(s.split()) for s in sentences) / max(1, len(sentences))

    sent_lengths = [len(s.split()) for s in sentences]
    if len(sent_lengths) > 1:
        mean = sum(sent_lengths) / len(sent_lengths)
        features["sentence_variance"] = sum((l-mean)**2 for l in sent_lengths) / len(sent_lengths)
    else:
        features["sentence_variance"] = 0

    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    features["trigram_uniqueness"] = len(set(trigrams)) / max(1, len(trigrams)) if trigrams else 1.0

    return features


def classify_origin(text):
    """Classify text as HUMAN/MACHINE/INDETERMINATE."""
    if not text or len(text.strip()) < 50:
        return {"classification": "INDETERMINATE", "confidence": 0.0, "features": {}}

    features = extract_features(text)

    reference = {
        "compression_ratio": (HUMAN_COMPRESSION_RATIO, 0.10),
        "char_entropy": (HUMAN_ENTROPY, 0.8),
        "unique_word_ratio": (HUMAN_UNIQUE_RATIO, 0.15),
        "avg_sentence_length": (HUMAN_SENTENCE_LEN, 8.0),
        "sentence_variance": (5.0, 4.0),
        "trigram_uniqueness": (0.85, 0.10),
    }

    distance_sq = sum(((features.get(k, m) - m) / max(s, 0.001)) ** 2
                      for k, (m, s) in reference.items())
    distance = math.sqrt(distance_sq / max(1, len(reference)))

    if distance < T_HUMAN:
        classification = "HUMAN"
        confidence = max(0, min(1.0, 1.0 - distance / T_HUMAN))
    elif distance > T_MACHINE:
        classification = "MACHINE"
        confidence = max(0, min(1.0, 1.0 - T_MACHINE / distance))
    else:
        classification = "INDETERMINATE"
        confidence = 0.0

    return {
        "classification": classification,
        "confidence": round(confidence, 4),
        "distance": round(distance, 4),
        "features": {k: round(v, 4) for k, v in features.items()},
    }


# ============================================================
# Watermarking (Patent 24, Section 4.4)
# ============================================================

WATERMARK_KEY = os.environ.get("WATERMARK_KEY", "clymb-watermark-key-2026").encode()


def embed_watermark(text, model_name="unknown", metadata=None):
    """Embed provenance watermark using Unicode tag characters."""
    import base64
    payload = {
        "m": model_name[:30],
        "t": int(time.time()),
        "v": "1.0",
    }
    if metadata:
        payload.update(metadata)

    payload_json = json.dumps(payload, separators=(",", ":"))
    sig = hmac.new(WATERMARK_KEY, payload_json.encode(), hashlib.sha256).hexdigest()[:16]
    payload["h"] = sig
    payload_json = json.dumps(payload, separators=(",", ":"))

    b64 = base64.b64encode(payload_json.encode()).decode("ascii")
    tag_str = chr(0xE0001) + "".join(chr(0xE0000 + ord(c)) for c in b64) + chr(0xE007F)

    return text + tag_str


def extract_watermark(text):
    """Extract and verify watermark."""
    import base64
    tag_chars = [c for c in text if 0xE0000 <= ord(c) <= 0xE007F]
    if len(tag_chars) < 3:
        return None

    payload_tags = [c for c in tag_chars if ord(c) not in (0xE0001, 0xE007F)]
    b64_str = "".join(chr(ord(c) - 0xE0000) for c in payload_tags)

    try:
        payload_json = base64.b64decode(b64_str).decode("utf-8")
        payload = json.loads(payload_json)
    except Exception:
        return None

    claimed_sig = payload.pop("h", None)
    verify_json = json.dumps(payload, separators=(",", ":")).encode()
    expected_sig = hmac.new(WATERMARK_KEY, verify_json, hashlib.sha256).hexdigest()[:16]

    payload["verified"] = (claimed_sig == expected_sig)
    return payload


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Clymb Data Provenance API",
        "patent": "GB2607042.5",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/classify", methods=["POST"])
@require_api_key
def api_classify():
    """Classify content as HUMAN/MACHINE/INDETERMINATE."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = classify_origin(data["text"])
    return jsonify(result)


@app.route("/gate", methods=["POST"])
@require_api_key
def api_gate():
    """Gate training data — admit or reject with provenance."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    source = data.get("source", "unknown")
    generation_depth = data.get("generation_depth", 1)

    # Classify origin
    classification = classify_origin(text)

    # Compute fidelity
    fidelity = mccaffer_bovill_quality(generation_depth)

    # Gate decision
    min_fidelity = data.get("min_fidelity", 0.5)
    admitted = fidelity >= min_fidelity and len(text.strip()) >= 50

    # Content quality check
    if classification["classification"] == "MACHINE" and classification["confidence"] > 0.8:
        if data.get("block_machine", False):
            admitted = False

    result = {
        "admitted": admitted,
        "reason": "passed_all_gates" if admitted else "below_fidelity_threshold",
        "fidelity": round(fidelity, 4),
        "generation_depth": generation_depth,
        "n_collapse": collapse_generation(),
        "origin": classification,
        "data_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        "timestamp": datetime.now().isoformat(),
    }

    return jsonify(result)


@app.route("/decay", methods=["POST"])
@require_api_key
def api_decay():
    """Predict quality decay at generation depth N."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    n = data.get("generation_depth", 1)
    theta_0 = data.get("theta_0", 0.1)

    curve = []
    for depth in range(0, max(n + 10, 20)):
        q = mccaffer_bovill_quality(depth, theta_0)
        curve.append({"n": depth, "quality": round(q, 6)})

    return jsonify({
        "query_depth": n,
        "quality_at_depth": round(mccaffer_bovill_quality(n, theta_0), 6),
        "n_collapse_50pct": collapse_generation(theta_0, 0.5),
        "n_collapse_10pct": collapse_generation(theta_0, 0.1),
        "theta_0": theta_0,
        "model": "McCaffer-Bovill F(n) = cos²(θ₀√n / 2)",
        "patent": "GB2607042.5 Section 4.2",
        "decay_curve": curve,
    })


@app.route("/watermark", methods=["POST"])
@require_api_key
def api_watermark():
    """Embed provenance watermark in content."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    watermarked = embed_watermark(
        data["text"],
        model_name=data.get("model", "unknown"),
        metadata=data.get("metadata"),
    )

    return jsonify({
        "watermarked_text": watermarked,
        "original_length": len(data["text"]),
        "watermarked_length": len(watermarked),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/verify", methods=["POST"])
@require_api_key
def api_verify():
    """Verify content provenance and extract watermark."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    # Try watermark extraction
    watermark = extract_watermark(data["text"])

    # Statistical classification
    clean_text = "".join(c for c in data["text"] if not (0xE0000 <= ord(c) <= 0xE007F))
    classification = classify_origin(clean_text)

    # Estimate provenance depth
    depth = 0
    if classification["classification"] == "MACHINE":
        depth = 1
        cr = classification["features"].get("compression_ratio", 0.5)
        if cr < 0.35:
            depth = 2
        if cr < 0.25:
            depth = 3

    return jsonify({
        "watermark_detected": watermark is not None,
        "watermark": watermark,
        "origin_classification": classification,
        "estimated_depth": depth,
        "predicted_fidelity": round(mccaffer_bovill_quality(depth), 4),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/certify", methods=["POST"])
@require_api_key
def api_certify():
    """Issue a signed provenance certificate."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    # Full analysis
    clean_text = "".join(c for c in data["text"] if not (0xE0000 <= ord(c) <= 0xE007F))
    classification = classify_origin(clean_text)
    watermark = extract_watermark(data["text"])

    depth = 1 if classification["classification"] == "MACHINE" else 0

    cert = {
        "content_hash": hashlib.sha256(clean_text.encode()).hexdigest()[:32],
        "origin": classification["classification"],
        "confidence": classification["confidence"],
        "provenance_depth": depth,
        "predicted_fidelity": round(mccaffer_bovill_quality(depth), 4),
        "watermark_verified": watermark.get("verified", False) if watermark else False,
        "issued_by": "Clymb Ltd",
        "patent": "GB2607042.5",
        "timestamp": datetime.now().isoformat(),
    }

    # Sign certificate
    cert_str = json.dumps(cert, sort_keys=True)
    cert["signature"] = hmac.new(
        WATERMARK_KEY, cert_str.encode(), hashlib.sha256
    ).hexdigest()[:32]

    return jsonify(cert)


# ============================================================
# STRIPE CHECKOUT ENDPOINTS
# ============================================================

@app.route("/")
def landing():
    """Serve the landing page."""
    return send_from_directory(".", "index.html")


@app.route("/docs")
def docs():
    """Serve API documentation."""
    return send_from_directory(".", "docs.html")


@app.route("/checkout/professional", methods=["GET"])
def checkout_professional():
    """Create Stripe Checkout session for Professional tier."""
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": PRICE_PROFESSIONAL, "quantity": 1}],
            mode="subscription",
            success_url=request.host_url + "success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=request.host_url + "#pricing",
            metadata={"tier": "professional"},
        )
        return redirect(session.url, code=303)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/checkout/enterprise", methods=["GET"])
def checkout_enterprise():
    """Create Stripe Checkout session for Enterprise tier."""
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": PRICE_ENTERPRISE, "quantity": 1}],
            mode="subscription",
            success_url=request.host_url + "success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=request.host_url + "#pricing",
            metadata={"tier": "enterprise"},
        )
        return redirect(session.url, code=303)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/success")
def checkout_success():
    """Post-checkout success page."""
    session_id = request.args.get("session_id")
    if session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            customer_email = session.get("customer_details", {}).get("email", "")
            # Generate API key for customer
            api_key = "clymb_" + hashlib.sha256(
                f"{session_id}{time.time()}".encode()
            ).hexdigest()[:24]
            # Add to authorized keys
            API_KEYS.add(api_key)
            return jsonify({
                "status": "success",
                "message": "Welcome to Clymb AI Provenance Platform!",
                "api_key": api_key,
                "email": customer_email,
                "docs": request.host_url + "docs",
                "note": "Save your API key — you will need it for all API calls.",
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"status": "success", "message": "Thank you for subscribing!"})


@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    """Handle Stripe webhook events (subscription lifecycle)."""
    payload = request.get_data(as_text=True)
    sig = request.headers.get("Stripe-Signature")
    endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    if endpoint_secret:
        try:
            event = stripe.Webhook.construct_event(payload, sig, endpoint_secret)
        except Exception:
            return jsonify({"error": "Invalid signature"}), 400
    else:
        event = json.loads(payload)

    event_type = event.get("type", "")

    if event_type == "checkout.session.completed":
        session = event["data"]["object"]
        # Log new subscription
        print(f"  [stripe] New subscription: {session.get('customer_email', 'unknown')} — {session.get('metadata', {}).get('tier', '?')}")

    elif event_type == "customer.subscription.deleted":
        # Subscription cancelled — revoke API key
        print(f"  [stripe] Subscription cancelled: {event['data']['object'].get('id', '?')}")

    return jsonify({"received": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
