# PathSnap: two-laptop map matcher

Laptop 1 = model + FastAPI. Laptop 2 = Streamlit evaluator UI.

## Layout
`data_prep/` Member 1 · `models/` Member 2 · `decoder/` Member 3 · `api/` + `ui/` + `scripts/` Member 4 ·
`configs/` · `results/` · `maps/` · `report_figures/` · `logs/`

## Deploy on two laptops

Connect both laptops to a network that allows them to reach each other.

| Computer | Files to copy from this workspace | Process |
| --- | --- | --- |
| Laptop 1 | The integrated `member4/` folder, including `models/extra_trees.joblib` and `vendor/member3/run.py`, plus the trip data | FastAPI on port 8000 |
| Laptop 2 | `member4/ui/` and optionally `member4/scripts/smoke_test.py`, preserving their folder names | Streamlit on port 8501 |

### Prepare Python on each laptop

Open a terminal inside each laptop's `member4/` folder. Create a fresh environment using Python 3.12. You can skip copying any existing virtual environments.

**Linux or macOS**

```bash
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate
```

**Windows Command Prompt**

```bat
py -3.12 -m venv .venv-demo
.venv-demo\Scripts\activate.bat
```

### Laptop 1 — start FastAPI

Place `student_test/` on Laptop 1. The default location is `../Student folder (2)/student_test` relative to `member4/`. For a different location, set `PATHSNAP_DEV_ROOT` to the absolute path of the directory containing the numbered trip folders. Each trip needs its `.track`, `.nodes` and `.arcs` files. A `.route` file supplies the optional evaluation metric.

```sh
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's Wi-Fi or Ethernet IPv4 address in its network settings. For example, if it is `192.168.1.20`, the API base URL is `http://192.168.1.20:8000`. Allow incoming TCP connections to port 8000 from the local network on Laptop 1.

### Laptop 2 — connect and start Streamlit

Open `http://192.168.1.20:8000/health` and `http://192.168.1.20:8000/trips` in **Laptop 2's browser**, substituting Laptop 1's actual IP. Health should report `status: ok` and `engine: real`. The trips endpoint should list the available trip IDs. Laptop 1's terminal should log both requests from Laptop 2.

```sh
python -m pip install streamlit requests folium streamlit-folium
python -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501
```

On Laptop 2, open `http://127.0.0.1:8501`, enter Laptop 1's API base URL in the **Backend URL** sidebar field, and press Enter. Select a trip and click **Run**. To prefill the sidebar on future launches, set `PATHSNAP_API` in the launch terminal before starting Streamlit.

### Evaluation day and shutdown

On Laptop 1, stop Uvicorn with **Ctrl+C**. Set both variables below in that terminal, replacing the example folder with the evaluator's trip root, then repeat the Uvicorn command. On Laptop 2, refresh the UI and confirm the sidebar shows `mode=final`.

| Terminal | Evaluation mode | Evaluator trip root |
| --- | --- | --- |
| Linux or macOS | `export PATHSNAP_MODE=final` | `export PATHSNAP_FINAL_ROOT="/absolute/path/to/evaluator-trips"` |
| Windows Command Prompt | `set "PATHSNAP_MODE=final"` | `set "PATHSNAP_FINAL_ROOT=C:\data\evaluator-trips"` |

At the end of the demo, press **Ctrl+C** in the Streamlit terminal on Laptop 2 and the Uvicorn terminal on Laptop 1.

## Tests
- `python -m scripts.offline_check` : no server; all trips; proves prediction works with `.route` files removed.
- `python -m scripts.smoke_test http://SERVER_IP:8000` : live API checks from Laptop 2.
- `python -m scripts.make_dev_map 00000056` : report map (dev only; reads reference).

## API
`GET /health` · `GET /trips` · `POST /predict {"trip_id": "00000056"}` · `POST /replay {"trip_id": "00000056"}`
Prediction code never reads `.route`; only `api/evaluator.py` does, and it returns aggregates only.

## Integrated prediction engine
The default `real` engine generates Member 1's UTM candidates and features, applies Member 2's saved
Extra Trees pipeline, and calls Member 3's directed-road decoder. `vendor/member3/UPSTREAM.json`
records the downloaded source revision. The vendor files retain the upstream implementation.
The API adapters translate live candidates, probabilities, route lines, breaks, and replay to the UI.
`PATHSNAP_MODEL` can select another compatible saved model bundle.

Replay exposes the automatically selected ambiguous anchor and lets the UI inspect how its chosen
arc evolves across later GPS prefixes. Some trips have no changed road choice in that replay window.
The API's route-arc hit rate measures membership in the supplied route, with its convention included
in each response. Offline point accuracy and sequence evaluation require the documented label alignment.

Run `python -m scripts.integration_check` from this directory to exercise all ten development trips,
replay, deterministic repeated predictions, invalid IDs, and prediction using input files without a route file.
Results are saved in `results/integration_check.json`.
