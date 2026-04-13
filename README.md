# Clymb Data Provenance API

**EU AI Act Compliance — Training Data Quality Assurance**

Hardware-Attested Data Provenance with McCaffer-Bovill Quality Decay Modelling.
Patent GB2607042.5 — Clymb Ltd.

## What This Does

Tells you whether your AI training data is contaminated with AI-generated content, and predicts when model collapse will occur.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Classify content as HUMAN / MACHINE / INDETERMINATE |
| `/gate` | POST | Gate training data — admit or reject with provenance |
| `/decay` | POST | Predict quality decay at generation depth N |
| `/watermark` | POST | Embed provenance watermark in content |
| `/verify` | POST | Verify content provenance and extract watermark |
| `/certify` | POST | Issue a signed provenance certificate |
| `/health` | GET | Service health check |

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Example

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key-2026" \
  -d '{"text": "The implementation of machine learning algorithms requires careful consideration of hyperparameter tuning."}'
```

Response:
```json
{
  "classification": "MACHINE",
  "confidence": 0.72,
  "features": {...}
}
```

## Patent

This service implements aspects of UK Patent Application GB2607042.5:
- Data Origin Classification (Aspect 1)
- Generational Fidelity Decay Modelling (Aspect 2)
- Training Data Gate (Aspect 3)
- Output Watermarking (Aspect 4)
- Output Provenance Verification (Aspect 7)

McCaffer-Bovill Quality Decay: F(n) = cos²(θ₀√n / 2)

## License

Proprietary — Clymb Ltd. Contact: alastair.mccaffer@hotmail.co.uk
