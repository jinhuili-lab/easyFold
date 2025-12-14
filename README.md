# AF3 Mini GUI v3 (FastAPI)

## What’s new
- **User Home** now also shows live server performance charts (CPU/GPU/Mem), updating every 2s.
- **Admin Profile** adds host path settings:
  - input / output / models / afdb
- **Submit pipeline**
  - UI -> AF3 JSON payload
  - payload written as `<job_uid>.json`
  - docker command generated using Admin Profile volumes
- Each job has a unique `job_uid` to avoid cross-user mix-ups.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export AF3_GUI_JWT_SECRET='change-me'
export AF3_ADMIN_USER='admin'
export AF3_ADMIN_PASS='Admin_123456!'

python -m uvicorn app:app --reload
```

Open: http://127.0.0.1:8000

## Notes
- GPU utilization uses `nvidia-smi` if available; otherwise shows 0.
- This version **generates** docker commands; it does not execute docker yet.
