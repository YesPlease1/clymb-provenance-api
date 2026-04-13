"""
Clymb Data Provenance API — EU AI Act Compliance Service
=========================================================
Fully automated: customer pays -> gets API key -> uses API.
Keys persist across restarts. Email notifications on signup.

Powered by Patent 24 (GB2607042.5) — McCaffer-Bovill Quality Decay
Author: Clymb Ltd
"""
from flask import Flask, request, jsonify, redirect, send_from_directory, render_template_string
import hashlib
import hmac
import math
import gzip
import json
import time
import os
import re
import stripe
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from collections import Counter
from functools import wraps
from pathlib import Path

app = Flask(__name__, static_folder=".")
app.config["SECRET_KEY"] = os.environ.get("API_SECRET", "clymb-provenance-2026")

# Stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
PRICE_PROFESSIONAL = os.environ.get("STRIPE_PRICE_PRO", "")
PRICE_ENTERPRISE = os.environ.get("STRIPE_PRICE_ENT", "")

# Persistent API key storage
KEYS_FILE = Path(__file__).parent / "api_keys.json"
OWNER_EMAIL = "alastair.mccaffer@hotmail.co.uk"


def load_api_keys():
    """Load API keys from persistent storage."""
    keys = {"demo-key-2026": {"tier": "starter", "email": "demo", "created": "2026-01-01"}}
    if KEYS_FILE.exists():
        try:
            stored = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
            keys.update(stored)
        except Exception:
            pass
    return keys


def save_api_keys(keys):
    """Save API keys to persistent storage."""
    KEYS_FILE.write_text(json.dumps(keys, indent=2, ensure_ascii=False), encoding="utf-8")


API_KEYS_DB = load_api_keys()


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if key not in API_KEYS_DB:
            return jsonify({"error": "Invalid API key. Get one at https://clymb.online"}), 401
        # Track usage
        if key in API_KEYS_DB and isinstance(API_KEYS_DB[key], dict):
            API_KEYS_DB[key]["last_used"] = datetime.now().isoformat()
            API_KEYS_DB[key]["calls"] = API_KEYS_DB[key].get("calls", 0) + 1
            if API_KEYS_DB[key]["calls"] % 100 == 0:
                save_api_keys(API_KEYS_DB)
        return f(*args, **kwargs)
    return decorated


def notify_owner(subject, body):
    """Notify Alastair via WhatsApp, Telegram, and notifications file."""
    import requests as _req

    full_message = f"[CLYMB] {subject}\n\n{body}"

    # 1. Save to notifications file (always works)
    try:
        notif_file = Path(__file__).parent / "notifications.json"
        notifs = []
        if notif_file.exists():
            try:
                notifs = json.loads(notif_file.read_text())
            except Exception:
                pass
        notifs.append({
            "subject": subject,
            "body": body,
            "timestamp": datetime.now().isoformat(),
            "read": False,
        })
        notifs = notifs[-200:]
        notif_file.write_text(json.dumps(notifs, indent=2))
    except Exception:
        pass

    # 2. WhatsApp via CallMeBot (free, no API key needed for self-notifications)
    try:
        whatsapp_number = "447720300425"
        whatsapp_key = os.environ.get("CALLMEBOT_KEY", "")
        if whatsapp_key:
            _req.get(
                f"https://api.callmebot.com/whatsapp.php?phone={whatsapp_number}&text={_req.utils.quote(full_message[:1000])}&apikey={whatsapp_key}",
                timeout=15,
            )
    except Exception:
        pass

    # 3. Telegram (via ARIA brain's bot)
    try:
        BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8428473061")
        CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "8668756895")
        if BOT_TOKEN and CHAT_ID:
            # Need the full bot token including the secret part
            bot_token_full = os.environ.get("TELEGRAM_BOT_TOKEN_FULL", "")
            if bot_token_full:
                _req.post(
                    f"https://api.telegram.org/bot{bot_token_full}/sendMessage",
                    json={"chat_id": CHAT_ID, "text": full_message[:4000]},
                    timeout=10,
                )
    except Exception:
        pass

    # 4. ntfy push notification (brain's existing push system)
    try:
        _req.post(
            "http://127.0.0.1:8089/clymb-sales",
            data=full_message[:500].encode(),
            headers={"Title": subject[:100], "Priority": "high"},
            timeout=5,
        )
    except Exception:
        pass


# ============================================================
# McCaffer-Bovill Quality Decay (Patent 24, Section 4.2)
# ============================================================

def mccaffer_bovill_quality(n, theta_0=0.1):
    theta_n = theta_0 * math.sqrt(max(0, n))
    return math.cos(theta_n / 2) ** 2


def collapse_generation(theta_0=0.1, threshold=0.5):
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
    return {"classification": classification, "confidence": round(confidence, 4),
            "distance": round(distance, 4),
            "features": {k: round(v, 4) for k, v in features.items()}}


# ============================================================
# Watermarking (Patent 24, Section 4.4)
# ============================================================

WATERMARK_KEY = os.environ.get("WATERMARK_KEY", "clymb-watermark-key-2026").encode()


def embed_watermark(text, model_name="unknown", metadata=None):
    import base64
    payload = {"m": model_name[:30], "t": int(time.time()), "v": "1.0"}
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
    return jsonify({"status": "healthy", "service": "Clymb Data Provenance API",
                    "patent": "GB2607042.5", "version": "1.0",
                    "customers": len(API_KEYS_DB),
                    "timestamp": datetime.now().isoformat()})


@app.route("/classify", methods=["POST"])
@require_api_key
def api_classify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    return jsonify(classify_origin(data["text"]))


@app.route("/gate", methods=["POST"])
@require_api_key
def api_gate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    text = data["text"]
    generation_depth = data.get("generation_depth", 1)
    classification = classify_origin(text)
    fidelity = mccaffer_bovill_quality(generation_depth)
    min_fidelity = data.get("min_fidelity", 0.5)
    admitted = fidelity >= min_fidelity and len(text.strip()) >= 50
    if classification["classification"] == "MACHINE" and classification["confidence"] > 0.8:
        if data.get("block_machine", False):
            admitted = False
    return jsonify({"admitted": admitted,
                    "reason": "passed_all_gates" if admitted else "below_fidelity_threshold",
                    "fidelity": round(fidelity, 4), "generation_depth": generation_depth,
                    "n_collapse": collapse_generation(), "origin": classification,
                    "data_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                    "timestamp": datetime.now().isoformat()})


@app.route("/decay", methods=["POST"])
@require_api_key
def api_decay():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    n = data.get("generation_depth", 1)
    theta_0 = data.get("theta_0", 0.1)
    curve = [{"n": d, "quality": round(mccaffer_bovill_quality(d, theta_0), 6)}
             for d in range(0, max(n + 10, 20))]
    return jsonify({"query_depth": n,
                    "quality_at_depth": round(mccaffer_bovill_quality(n, theta_0), 6),
                    "n_collapse_50pct": collapse_generation(theta_0, 0.5),
                    "n_collapse_10pct": collapse_generation(theta_0, 0.1),
                    "theta_0": theta_0,
                    "model": "McCaffer-Bovill F(n) = cos\u00b2(\u03b8\u2080\u221an / 2)",
                    "patent": "GB2607042.5 Section 4.2", "decay_curve": curve})


@app.route("/watermark", methods=["POST"])
@require_api_key
def api_watermark():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    watermarked = embed_watermark(data["text"], model_name=data.get("model", "unknown"),
                                  metadata=data.get("metadata"))
    return jsonify({"watermarked_text": watermarked, "original_length": len(data["text"]),
                    "watermarked_length": len(watermarked),
                    "timestamp": datetime.now().isoformat()})


@app.route("/verify", methods=["POST"])
@require_api_key
def api_verify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    watermark = extract_watermark(data["text"])
    clean_text = "".join(c for c in data["text"] if not (0xE0000 <= ord(c) <= 0xE007F))
    classification = classify_origin(clean_text)
    depth = 0
    if classification["classification"] == "MACHINE":
        depth = 1
        cr = classification["features"].get("compression_ratio", 0.5)
        if cr < 0.35: depth = 2
        if cr < 0.25: depth = 3
    return jsonify({"watermark_detected": watermark is not None, "watermark": watermark,
                    "origin_classification": classification, "estimated_depth": depth,
                    "predicted_fidelity": round(mccaffer_bovill_quality(depth), 4),
                    "timestamp": datetime.now().isoformat()})


@app.route("/certify", methods=["POST"])
@require_api_key
def api_certify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    clean_text = "".join(c for c in data["text"] if not (0xE0000 <= ord(c) <= 0xE007F))
    classification = classify_origin(clean_text)
    watermark = extract_watermark(data["text"])
    depth = 1 if classification["classification"] == "MACHINE" else 0
    cert = {"content_hash": hashlib.sha256(clean_text.encode()).hexdigest()[:32],
            "origin": classification["classification"], "confidence": classification["confidence"],
            "provenance_depth": depth,
            "predicted_fidelity": round(mccaffer_bovill_quality(depth), 4),
            "watermark_verified": watermark.get("verified", False) if watermark else False,
            "issued_by": "Clymb Ltd", "patent": "GB2607042.5",
            "timestamp": datetime.now().isoformat()}
    cert_str = json.dumps(cert, sort_keys=True)
    cert["signature"] = hmac.new(WATERMARK_KEY, cert_str.encode(), hashlib.sha256).hexdigest()[:32]
    return jsonify(cert)


# ============================================================
# PAGES
# ============================================================

@app.route("/")
def landing():
    return send_from_directory(".", "index.html")


@app.route("/docs")
def docs_page():
    return send_from_directory(".", "docs.html")


@app.route("/contact")
def contact_page():
    return send_from_directory(".", "contact.html")


@app.route("/contact/send", methods=["POST"])
def contact_send():
    data = request.get_json() or request.form
    name = data.get("name", "Unknown")
    email = data.get("email", "no-email")
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Message is required"}), 400
    notify_owner(
        f"New contact from {name}",
        f"From: {name} <{email}>\n\n{message}"
    )
    return jsonify({"status": "sent", "message": "Thank you! We'll get back to you within 24 hours."})


# ============================================================
# STRIPE CHECKOUT — FULLY AUTOMATED
# ============================================================

@app.route("/checkout/professional", methods=["GET"])
def checkout_professional():
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
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"status": "success", "message": "Thank you!"})

    try:
        session = stripe.checkout.Session.retrieve(session_id)
        customer_details = session.get("customer_details") or {}
        customer_email = customer_details.get("email", "unknown")
        tier = (session.get("metadata") or {}).get("tier", "professional")

        # Generate persistent API key
        api_key = "clymb_" + hashlib.sha256(
            f"{session_id}{customer_email}".encode()
        ).hexdigest()[:24]

        # Save to persistent storage
        API_KEYS_DB[api_key] = {
            "tier": tier,
            "email": customer_email,
            "stripe_session": session_id,
            "created": datetime.now().isoformat(),
            "calls": 0,
        }
        save_api_keys(API_KEYS_DB)

        # Notify owner
        notify_owner(
            f"NEW CUSTOMER: {customer_email} ({tier})",
            f"Email: {customer_email}\nTier: {tier}\nAPI Key: {api_key}\nTime: {datetime.now().isoformat()}"
        )

        # Show success page with API key
        return render_template_string(SUCCESS_PAGE, api_key=api_key, email=customer_email,
                                       tier=tier, docs_url=request.host_url + "docs")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
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

    if event_type == "customer.subscription.deleted":
        # Find and revoke the API key for this customer
        sub = event["data"]["object"]
        customer_id = sub.get("customer", "")
        for key, info in list(API_KEYS_DB.items()):
            if isinstance(info, dict) and info.get("stripe_customer") == customer_id:
                info["revoked"] = True
                info["revoked_at"] = datetime.now().isoformat()
                notify_owner(f"SUBSCRIPTION CANCELLED: {info.get('email', '?')}", f"Key {key} revoked")
        save_api_keys(API_KEYS_DB)

    return jsonify({"received": True})


# ============================================================
# HTML TEMPLATES
# ============================================================

SUCCESS_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Welcome to Clymb</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 600px; margin: 80px auto; padding: 24px; color: #1e293b; }
h1 { color: #10b981; } .key-box { background: #0f172a; color: #86efac; padding: 20px; border-radius: 8px;
font-family: monospace; font-size: 1.1rem; margin: 20px 0; word-break: break-all; }
.warning { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 16px; border-radius: 4px; margin: 16px 0; }
a { color: #3b82f6; } code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
pre { background: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 8px; overflow-x: auto; }
</style></head><body>
<h1>Welcome to Clymb AI Provenance Platform</h1>
<p>Your <strong>{{ tier }}</strong> subscription is active.</p>
<h2>Your API Key</h2>
<div class="key-box">{{ api_key }}</div>
<div class="warning"><strong>Save this key now.</strong> You will need it for all API calls. It won't be shown again.</div>
<h2>Quick Start</h2>
<pre>curl -X POST https://api.clymb.online/classify \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: {{ api_key }}" \\
  -d '{"text": "Your content here..."}'</pre>
<p><a href="{{ docs_url }}">Full API Documentation</a></p>
<p>Questions? Email <a href="mailto:alastair.mccaffer@hotmail.co.uk">alastair.mccaffer@hotmail.co.uk</a></p>
</body></html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
