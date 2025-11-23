# ParkIQ: Smart Parking Availability Prediction

---
## 1. Title Slide
ParkIQ – Smart Parking Intelligence
Predicting likely free parking slots in real time
Version: November 2025
Presenter: <Your Name>

Speaker Notes:
Introduce ParkIQ as a data-driven parking decision assistant reducing time-to-park and emissions.

---
## 2. Problem Statement
- Urban drivers waste time searching for parking
- Increased congestion & emissions during peak hours
- Existing solutions often static or rule-based
- Need: actionable, time-specific probability of free slots

Speaker Notes:
Quantify problem (cite typical % of city traffic due to parking search if available). Emphasize real-time decision aid.

---
## 3. Objectives
- Provide ranked list of slots likely free NOW / near-future
- Minimize user friction (fast predictions < 1s)
- Offer explainable probability outputs
- Scalable API + simple web UI

Speaker Notes:
Tie objectives to measurable KPIs: prediction latency, accuracy, user adoption.

---
## 4. Data Source & Structure
CSV: `SPSIRDATA.csv`
Columns: created_at, entry_id, field1 (slot ID e.g. IR20), field2 (binary availability), field3 (unused)
~2700 historic observations spanning multiple days

Speaker Notes:
Data is event-based per slot with timestamp; current pipeline aggregates to temporal probabilities.

---
## 5. Feature Engineering (Current)
- Extract hour (0–23) and dayofweek (0–6)
- Aggregate mean availability per (hour, dayofweek, slot)
- Pivot to wide matrix: rows = time bins, columns = slot IDs

Limitations:
- Loses intra-hour variability
- Potential future leakage (uses full history before split)

Speaker Notes:
Aggregation simplifies modeling but restricts temporal granularity.

---
## 6. Modeling Approach (Current)
Pipeline:
1. StandardScaler (redundant for trees)
2. MultiOutputRegressor(RandomForestRegressor n=300)
Target: probability per slot (regression of mean availability)
Output: Top N (default 10) slots ranked by predicted free probability

Speaker Notes:
Effectively learning a lookup table; propose improvements next.

---
## 7. Proposed Modeling Improvements
- Train on raw events (no pre-aggregation)
- MultiOutputClassifier or per-slot classifier
- Temporal splits (TimeSeriesSplit) to prevent leakage
- Additional features: weekend flag, cyclical hour (sin/cos), recent rolling availability, holiday indicator
- Calibration: reliability curves, isotonic or Platt scaling

Speaker Notes:
Describe expected uplift: better generalization, richer temporal patterns.

---
## 8. System Architecture
Components:
- Data ingestion (CSV / future stream)
- Model training script (`train_model.py`)
- Model artifact `parking_prob_model.pkl`
- Flask API: `/api/predictions?hour=&day=`
- Web UI: `index.html` (form) → `predictions.html` (lazy load results)
- Deployment (Gunicorn / Waitress) + optional containerization

Speaker Notes:
Highlight separation: presentation layer, inference layer, model artifact.

---
## 9. Web Application UX
Index Page:
- Hero section with static parking image
- Parameter form (hour fractional, day selection)
Predictions Page:
- Skeleton shimmer loaders
- Spinner overlay (slow network)
- Lazy fetch & animated row reveals
- Probability color coding & status badges

Speaker Notes:
User journey: input → redirect → progressive enhancement of results.

---
## 10. API Contract
Endpoint: `/api/predictions`
Method: GET
Parameters: `hour` (float), `day` (int 0–6)
Response JSON:
```
{
  "hour": <float>,
  "day": <int>,
  "expected_free": <float>,
  "predictions": [ { "slot": "IR20", "prob": 73.2 }, ... ]
}
```
Error Codes: 400 invalid params, 500 internal error

Speaker Notes:
Stress simplicity & integratability into mobile apps.

---
## 11. Performance & Latency Goals
- Current latency: ~ <1s (inference is small tree ensemble)
- Target latency: <200ms p50, <500ms p95
- Future scaling: model served via WSGI behind reverse proxy (NGINX) or ASGI for async streaming

Speaker Notes:
Discuss caching probabilities for identical time bins.

---
## 12. Metrics & Evaluation (Future)
To implement:
- Per-slot Brier score
- Calibration plots
- AUC / PR curves per occupancy class
- Drift monitoring (slot probability shift) monthly

Speaker Notes:
Frame evaluation roadmap; current version baseline only.

---
## 13. Risks & Mitigations
Risk: Data sparsity → Mitigation: Smoothing / Bayesian priors
Risk: Temporal drift → Scheduled re-training, drift detection
Risk: Overfitting aggregated means → Switch to raw event modeling
Risk: User trust in probabilities → Provide historical context & confidence intervals

Speaker Notes:
Show proactive mitigation plan fosters stakeholder confidence.

---
## 14. Scalability Roadmap
Phase 1: Baseline (current)
Phase 2: Improved feature set & per-slot classifiers
Phase 3: Real-time streaming ingestion (Kafka) & incremental updates
Phase 4: GIS integration (distance weighting) + mobile SDK
Phase 5: Multi-city expansion & A/B testing panel

Speaker Notes:
Tie to business growth milestones.

---
## 15. Demo Flow
1. Enter hour/day on landing page
2. Redirect to predictions page with query params
3. Skeleton shimmer → spinner (if slow) → populated table
4. Observe ranked probabilities & status categories

Speaker Notes:
Keep demo concise (<3 minutes).

---
## 16. Security & Privacy Considerations
- No PII collected
- Future: anonymize geolocation events
- Rate limiting for public API
- Potential JWT for partner integration

Speaker Notes:
Assure compliance baseline.

---
## 17. Deployment & Ops
- Gunicorn + Flask (Linux recommended)
- Logging: structured JSON for predictions
- Monitoring: latency & error rate dashboards
- CI/CD: Test training script + lint templates before deploy

Speaker Notes:
Mention containerization option (Dockerfile) for portability.

---
## 18. Future Enhancements
- Confidence intervals via quantile regression or ensembles
- Explainability: feature attribution (SHAP on per-slot classifier)
- Adaptive learning: online updates for peak anomalies
- User feedback loop (“Was slot free?”) to refine model

Speaker Notes:
Demonstrate innovation pipeline.

---
## 19. Summary
ParkIQ delivers rapid, interpretable parking availability probabilities.
Structured path to improved accuracy & scalability.
Clear roadmap for model and platform evolution.

Speaker Notes:
Reinforce value proposition & readiness for pilot.

---
## 20. Q&A
Thank you.
Contact: <email/handle>

Speaker Notes:
Invite feedback; propose next steps meeting.

---
## Appendix: Current Training Script Highlights
Filename: `train_model.py`
- Aggregation & pivot
- RandomForest multi-output regression
- Potential improvement: switch to classification & raw events

---
## Appendix: Requirements
`requirements.txt` includes Flask, pandas, scikit-learn, numpy, scipy, joblib, gunicorn.

---
## Export Instructions
Convert to PowerPoint:
1. Install pandoc + reference template.
2. Command:
```
pandoc ParkIQ_Presentation.md -t pptx -o ParkIQ_Presentation.pptx
```
Or use Marp:
```
marp ParkIQ_Presentation.md --pptx
```

---
## Design Tips (Optional)
- Use consistent color palette (blue/cyan gradients)
- Replace code blocks with diagrams for exec audience
- Add iconography for each phase (data, model, API, UI)
