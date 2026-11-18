# PathSnap: two-laptop map matcher

Laptop 1 = model + FastAPI. Laptop 2 = Streamlit evaluator UI.

## Layout
`data_prep/` Member 1 · `models/` Member 2 · `decoder/` Member 3 · `api/` + `ui/` + `scripts/` Member 4 ·
`configs/` · `results/` · `maps/` · `report_figures/` · `logs/`

## Setup (both laptops)
```
git clone <repo-url> && cd <repo>
python -m venv .venv
.venv\Scripts\activate          # Windows      (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Raw data is NOT in git. Put the trip folders anywhere and set the path in `configs/config.yaml`
(`data_roots.dev`) or via env var `PATHSNAP_DEV_ROOT`. Expected layout: `<root>/<trip_id>/<trip_id>.{track,nodes,arcs,route}`.

## Laptop 1 (server)
1. Find the LAN IP (`ipconfig` / `ip addr`). Allow Python through the firewall, or open port 8000.
2. Dev mode (student_test):
   `uvicorn api.main:app --host 0.0.0.0 --port 8000`
3. Final mode at the event:
   `set PATHSNAP_MODE=final` and `set PATHSNAP_FINAL_ROOT=<evaluator folder>` (PowerShell: `$env:PATHSNAP_MODE="final"`), then the same uvicorn command.

## Laptop 2 (UI)
```
curl http://SERVER_IP:8000/health         # must print status ok, and Laptop 1's terminal must log the request
set PATHSNAP_API=http://SERVER_IP:8000
streamlit run ui/app.py
```

## Tests
- `python -m scripts.offline_check` : no server; all trips; proves prediction works with `.route` files removed.
- `python -m scripts.smoke_test http://SERVER_IP:8000` : live API checks from Laptop 2.
- `python -m scripts.make_dev_map 00000056` : report map (dev only; reads reference).

## API
`GET /health` · `GET /trips` · `POST /predict {"trip_id": "00000056"}` · `POST /replay` (needs `decoder.replay`)
Prediction code never reads `.route`; only `api/evaluator.py` does, and it returns aggregates only.

## Status of the engine
`engine: auto` uses `data_prep.candidates.generate`, `models.scorer.score`, `decoder.viterbi.decode` if importable
(adapt `RealEngine` in `api/pipeline.py` to their signatures); otherwise it falls back to a nearest-road
SCAFFOLD, flagged in every response and in the UI. Never report scaffold numbers.

## AI assistance
Record according to the event instructions: <fill in>.
