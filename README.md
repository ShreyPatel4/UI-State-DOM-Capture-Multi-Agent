# DOM-UI State Capture Multi-Agent
End-to-end agents that runs browser tasks, tracks flows in Postgres, and stores screenshots/DOM snapshots in MinIO. Code lives under `ui_state_capture_agent/`.

## Quick start
- Requirements: Python 3.10+, Postgres, MinIO, Chromium via Playwright, optional GPU for HF policy models.
- Setup: `cd ui_state_capture_agent && python -m venv .venv && source .venv/bin/activate`
- Install: `pip install -r requirements.txt && playwright install chromium`
- Configure: `cp .env.example .env` then set `DATABASE_URL`, `MINIO_*`, and either `OPENAI_API_KEY` or `HF_MODEL_NAME`; set `HEADLESS=false` to watch the browser.
- Prepare storage: start Postgres + MinIO, then create tables once with `python -c "from src.models import init_db; init_db()"`.

## Run the FastAPI app
- From `ui_state_capture_agent`: `uvicorn src.server.api:app --host 0.0.0.0 --port 8000`
- UI: open `http://localhost:8000`, submit a natural-language query, and watch steps/logs as they stream in; artifacts land in MinIO under the configured bucket.
- API trigger:
```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"query":"open the demo app and add a todo"}'
```
- Status endpoints: `GET /api/flows` for recent runs and `GET /api/flows/{id}/status` or `/flows/{id}` for details.

## Run from the terminal
- Kick off the same agent loop without HTTP: `python scripts/run_task.py --query "open demo.softlight.app and complete the onboarding checklist"`
- Export captured runs and assets: `python scripts/export_dataset.py --out dataset/export_$(date +%Y%m%d) --limit-flows 5 [--tar dataset/export.tar.gz]`
