import io
import json
import os
import asyncio
import zipfile
import uuid
from uuid import uuid4
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text as sql_text

from db import init_db, get_db, SessionLocal, User, Job, AdminProfile
from auth import (
    hash_password,
    verify_password,
    create_token,
    COOKIE_NAME,
    get_current_user_optional,
    require_admin,
)

try:
    import psutil
except Exception:
    psutil = None

app = FastAPI(title="AF3 Mini GUI v3")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

HIST_MAX = 60  # 5 minutes @ 5s refresh
_hist_labels = deque(maxlen=HIST_MAX)
_hist_cpu = deque(maxlen=HIST_MAX)
_hist_gpu = deque(maxlen=HIST_MAX)
_hist_mem = deque(maxlen=HIST_MAX)


def _ensure_columns_sqlite(db: Session):
    # Best-effort lightweight migration for sqlite
    try:
        url = str(db.get_bind().url)
        if not url.startswith("sqlite"):
            return
        cols = {r[1] for r in db.execute(sql_text("PRAGMA table_info(jobs)")).fetchall()}
        if "finished_at" not in cols:
            db.execute(sql_text("ALTER TABLE jobs ADD COLUMN finished_at DATETIME"))
        if "job_uid" not in cols:
            db.execute(sql_text("ALTER TABLE jobs ADD COLUMN job_uid VARCHAR(64) DEFAULT ''"))
        if "docker_cmd" not in cols:
            db.execute(sql_text("ALTER TABLE jobs ADD COLUMN docker_cmd TEXT DEFAULT ''"))
        db.commit()
    except Exception:
        db.rollback()

    try:
        url = str(db.get_bind().url)
        if not url.startswith("sqlite"):
            return
        cols = {r[1] for r in db.execute(sql_text("PRAGMA table_info(admin_profile)")).fetchall()}
        for c, typ in [("input_dir", "VARCHAR(512)"), ("output_dir", "VARCHAR(512)"), ("models_dir", "VARCHAR(512)"), ("afdb_dir", "VARCHAR(512)")]:
            if c not in cols:
                db.execute(sql_text(f"ALTER TABLE admin_profile ADD COLUMN {c} {typ} DEFAULT ''"))
        db.commit()
    except Exception:
        db.rollback()


def ensure_admin(db: Session):
    admin_user = os.getenv("AF3_ADMIN_USER", "admin")
    admin_pass = os.getenv("AF3_ADMIN_PASS", "Admin_123456!")
    existing = db.query(User).filter(User.username == admin_user).first()
    if existing:
        existing.is_admin = True
        db.commit()
        return
    u = User(username=admin_user, password_hash=hash_password(admin_pass), is_admin=True)
    db.add(u)
    db.commit()


def ensure_admin_profile(db: Session):
    prof = db.query(AdminProfile).filter(AdminProfile.id == 1).first()
    if not prof:
        prof = AdminProfile(id=1, input_dir="", output_dir="", models_dir="", afdb_dir="", updated_at=datetime.utcnow())
        db.add(prof)
        db.commit()
    return prof


def redirect_after_login(user: User) -> RedirectResponse:
    return RedirectResponse(url="/admin/home" if user.is_admin else "/home", status_code=303)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 401:
        return RedirectResponse("/login", status_code=303)
    if exc.status_code == 403:
        return RedirectResponse("/home", status_code=303)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})




def _status_is_completed(s: str | None) -> bool:
    return (s or "").lower() in {"done", "completed", "finished", "success", "succeeded", "failed", "error"}


def _status_is_running(s: str | None) -> bool:
    return (s or "").lower() in {"running"}


def _status_is_pending(s: str | None) -> bool:
    return (s or "").lower() in {"pending", "queued"}


def docker_inspect_status(container_name: str) -> tuple[str, int | None]:
    """Return (status, exit_code). status in: running/exited/notfound/unknown"""
    try:
        p = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", container_name],
            capture_output=True, text=True, check=False
        )
        if p.returncode != 0:
            return ("notfound", None)
        out = (p.stdout or "").strip()
        if not out:
            return ("unknown", None)
        parts = out.split()
        st = parts[0]
        code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        return (st, code)
    except Exception:
        return ("unknown", None)


def docker_stop(container_name: str) -> bool:
    try:
        p = subprocess.run(["docker", "stop", container_name], capture_output=True, text=True)
        return p.returncode == 0
    except Exception:
        return False


def launch_local_docker(job: Job, prof: AdminProfile) -> tuple[bool, str]:
    """Launch docker in background; returns (ok, container_name)."""
    container = f"af3_{job.job_uid}"
    cmd = build_docker_cmd(prof, str(Path(job.workdir, f"check-{uuid4().hex[:10]}") / f"{job.job_uid}.json"), job.job_uid)
    job.docker_cmd = cmd
    try:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return True, container
    except Exception:
        return False, container


async def scheduler_loop():
    """Background loop: update running jobs, mark finished_at, and start pending jobs (FIFO) with max_running."""
    while True:
        try:
            db = SessionLocal()
            prof = db.query(AdminProfile).filter(AdminProfile.id == 1).first()
            if not prof:
                await asyncio.sleep(5)
                continue

            # enforce local docker only for now
            max_running = int(getattr(prof, "max_running", 1) or 1)
            if max_running < 1:
                max_running = 1

            # 1) Update running jobs statuses based on docker inspect
            running_jobs = db.query(Job).filter(Job.status == "running").all()
            for j in running_jobs:
                container = f"af3_{j.job_uid}"
                st, code = docker_inspect_status(container)
                if st == "running":
                    continue
                if st in {"exited", "dead"}:
                    # decide success/fail
                    if code == 0:
                        j.status = "done"
                    else:
                        j.status = "failed"
                    j.finished_at = datetime.utcnow()
                elif st == "notfound":
                    # container missing; mark as failed only if it had been running for a while
                    j.status = "failed"
                    j.finished_at = datetime.utcnow()
                else:
                    # keep as running for unknown
                    pass
            db.commit()

            # 2) Start pending jobs if slots available (FIFO)
            running_count = db.query(Job).filter(Job.status == "running").count()
            slots = max_running - running_count
            if slots > 0 and (prof.executor_type or "local_docker") == "local_docker":
                pendings = db.query(Job).filter(Job.status == "pending").order_by(Job.created_at.asc()).limit(slots).all()
                for j in pendings:
                    ok, _ = launch_local_docker(j, prof)
                    j.status = "running" if ok else "failed"
                    if not ok:
                        j.finished_at = datetime.utcnow()
                db.commit()

            db.close()
        except Exception:
            try:
                db.close()
            except Exception:
                pass
        await asyncio.sleep(5)

@app.on_event("startup")
def on_startup():
    init_db()
    db = SessionLocal()
    try:
        _ensure_columns_sqlite(db)
        ensure_admin(db)
        ensure_admin_profile(db)
    finally:
        db.close()
    asyncio.create_task(scheduler_loop())



def parse_pagination(request: Request):
    try:
        page = int(request.query_params.get("page", "1"))
    except ValueError:
        page = 1
    try:
        per_page = int(request.query_params.get("per_page", "10"))
    except ValueError:
        per_page = 10
    if per_page not in (10, 20, 50):
        per_page = 10
    page = max(page, 1)
    return page, per_page


def paginate_query(query, page: int, per_page: int):
    total = query.count()
    total_pages = max(1, (total + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages
    items = query.offset((page - 1) * per_page).limit(per_page).all()
    return items, total, total_pages, page, per_page


def _gpu_util_percent() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.5,
        )
        vals = []
        for line in out.strip().splitlines():
            line = line.strip()
            if line.isdigit():
                vals.append(int(line))
        if not vals:
            return 0
        return int(sum(vals) / len(vals))
    except Exception:
        return 0


def _sample_metrics():
    if psutil is None:
        cpu = 0
        mem = 0
    else:
        cpu = int(psutil.cpu_percent(interval=None))
        mem = int(psutil.virtual_memory().percent)
    gpu = _gpu_util_percent()
    return cpu, gpu, mem


@app.get("/api/metrics")
async def api_metrics():
    cpu, gpu, mem = _sample_metrics()
    ts = datetime.now().strftime("%H:%M")
    _hist_labels.append(ts)
    _hist_cpu.append(cpu)
    _hist_gpu.append(gpu)
    _hist_mem.append(mem)
    return {
        "labels": list(_hist_labels),
        "cpu": list(_hist_cpu),
        "gpu": list(_hist_gpu),
        "mem": list(_hist_mem),
        "cpu_now": cpu,
        "gpu_now": gpu,
        "mem_now": mem,
    }

@app.get("/api/job_counts")
async def api_job_counts(scope: str = "me", db: Session = Depends(get_db), user: User | None = Depends(get_current_user_optional)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    include_all = (scope == "all") and bool(user.is_admin)

    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    labels = [d.strftime("%Y-%m-%d") for d in days]

    completed_set = {"done", "completed", "finished", "success", "succeeded"}

    completed = [0] * 7
    pending = [0] * 7

    q = db.query(Job)
    if not include_all:
        q = q.filter(Job.owner_id == user.id)

    start_dt = datetime.combine(days[0], datetime.min.time())
    end_dt = datetime.combine(today + timedelta(days=1), datetime.min.time())
    jobs = q.filter(Job.created_at >= start_dt, Job.created_at < end_dt).all()

    day_index = {days[i]: i for i in range(7)}
    for j in jobs:
        d = j.created_at.date()
        i = day_index.get(d)
        if i is None:
            continue
        if (j.status or "").lower() in completed_set:
            completed[i] += 1
        else:
            pending[i] += 1

    return {"labels": labels, "completed": completed, "pending": pending, "scope": ("all" if include_all else "me")}


@app.get("/api/jobs/{job_id}/log")
async def api_job_log(job_id: int, db: Session = Depends(get_db), user: User | None = Depends(get_current_user_optional)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not user.is_admin and job.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    prof = db.query(AdminProfile).filter(AdminProfile.id == 1).first()
    name = f"af3_{job.job_uid}"
    candidates = []
    if prof and prof.output_dir:
        candidates.append(Path(os.path.expandvars(prof.output_dir)) / f"{name}.logs")
    if job.workdir:
        candidates.append(Path(job.workdir) / f"{name}.logs")

    content = ""
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                content = p.read_text(encoding="utf-8", errors="replace")
                break
        except Exception:
            continue

    if not content:
        content = "(log file not found yet)\n"

    tail = "\n".join(content.splitlines()[-500:])
    return {"job_id": job_id, "job_uid": job.job_uid, "log": tail}



@app.get("/", response_class=HTMLResponse)
async def root(user: User | None = Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    return redirect_after_login(user)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password."})

    token = create_token(user.id, user.username, user.is_admin)
    resp = redirect_after_login(user)
    resp.set_cookie(COOKIE_NAME, token, httponly=True, secure=False, samesite="lax", max_age=86400)
    return resp


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    username = username.strip()
    if len(username) < 3:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username too short (>= 3)."})
    if len(password) < 8:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Password too short (>= 8)."})

    existing = db.query(User).filter(User.username == username).first()
    if existing:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already exists."})

    user = User(username=username, password_hash=hash_password(password), is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_token(user.id, user.username, user.is_admin)
    resp = RedirectResponse(url="/home", status_code=303)
    resp.set_cookie(COOKIE_NAME, token, httponly=True, secure=False, samesite="lax", max_age=86400)
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp


@app.get("/home", response_class=HTMLResponse)
async def home(request: Request, user: User | None = Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.is_admin:
        return RedirectResponse("/admin/home", status_code=303)
    return templates.TemplateResponse("user_home.html", {"request": request, "user": user, "jobs_scope": "me"})


def build_af3_payload(job_name: str, chain_ids: str, sequence: str, model_seed: int, dialect: str, version: int):
    ids = [x.strip() for x in (chain_ids or "A").split(",") if x.strip()]
    seq = (sequence or "").strip().replace("\n", "").replace("\r", "")
    return {
        "name": job_name or "AF3 Job",
        "sequences": [{"protein": {"id": ids, "sequence": seq}}],
        "modelSeeds": [int(model_seed)],
        "dialect": dialect or "alphafold3",
        "version": int(version),
    }


def build_docker_cmd(profile: AdminProfile, host_json_path: str, job_uid: str = "preview"):
    name = f"af3_{job_uid}"
    json_name = Path(host_json_path).name

    def vol(host, container):
        host = host or "<SET_IN_PROFILE>"
        return f'--volume {host}:{container}'

    return " \\\n    ".join([
        "docker run -it -d",
        f"--name {name}",
        vol(profile.input_dir, "/root/input"),
        vol(profile.output_dir, "/root/output"),
        vol(profile.models_dir, "/root/models"),
        vol(profile.afdb_dir, "/root/public_databases"),
        "--gpus all",
        "cford38/alphafold3",
        "python run_alphafold.py",
        f"--json_path=/root/input/{json_name}",
        "--model_dir=/root/models",
        "--output_dir=/root/output",
        f"1>/root/output/{name}.logs 2>&1",
    ])



@app.get("/submit", response_class=HTMLResponse)
async def submit_page(request: Request, user: User | None = Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.is_admin:
        return RedirectResponse("/admin/home", status_code=303)
    return templates.TemplateResponse("user_submit.html", {"request": request, "user": user, "error": None})


@app.post("/submit/check", response_class=HTMLResponse)
async def submit_check(
    request: Request,
    job_name: str = Form(default="AF3 Job"),
    chain_ids: str = Form(default="A"),
    sequence: str = Form(default=""),
    model_seed: int = Form(default=1),
    dialect: str = Form(default="alphafold3"),
    version: int = Form(default=1),
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.is_admin:
        return RedirectResponse("/admin/home", status_code=303)

    payload = build_af3_payload(job_name, chain_ids, sequence, model_seed, dialect, version)
    prof = db.query(AdminProfile).filter(AdminProfile.id == 1).first() or ensure_admin_profile(db)

    tmp_uid = uuid.uuid4().hex[:12]
    tmp_dir = JOBS_DIR / f"check_{tmp_uid}"
    tmp_dir.mkdir(exist_ok=True)
    host_json = tmp_dir / "test.json"
    host_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_uid = f"check-{uuid4().hex[:10]}"

    cmd = build_docker_cmd(prof, str(host_json), tmp_uid)
    return templates.TemplateResponse(
        "user_submit.html",
        {
            "request": request,
            "user": user,
            "error": None,
            "command": cmd,
            "payload_preview": json.dumps(payload, indent=2),
            "right_help": "Check builds AF3 JSON + docker command draft. Submit will create a unique job UID and persist everything.",
            "form_job_name": job_name,
            "form_chain_ids": chain_ids,
            "form_sequence": sequence,
            "form_seed": model_seed,
            "form_dialect": dialect,
            "form_version": version,
        },
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit_job(
    request: Request,
    job_name: str = Form(default="AF3 Job"),
    chain_ids: str = Form(default="A"),
    sequence: str = Form(default=""),
    model_seed: int = Form(default=1),
    dialect: str = Form(default="alphafold3"),
    version: int = Form(default=1),
    db: Session = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.is_admin:
        return RedirectResponse("/admin/home", status_code=303)

    seq = (sequence or "").strip()
    if len(seq) < 10:
        return templates.TemplateResponse("user_submit.html", {"request": request, "user": user, "error": "Sequence too short."})

    payload = build_af3_payload(job_name, chain_ids, sequence, model_seed, dialect, version)

    job_uid = uuid.uuid4().hex
    job_dir = JOBS_DIR / f"job_{job_uid}"
    job_dir.mkdir(exist_ok=True)

    prof = db.query(AdminProfile).filter(AdminProfile.id == 1).first() or ensure_admin_profile(db)
    input_dir = Path(os.path.expandvars(prof.input_dir)) if prof.input_dir else job_dir
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        input_dir = job_dir

    host_json = input_dir / f"{job_uid}.json"
    host_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_uid = f"check-{uuid4().hex[:10]}"

    cmd = build_docker_cmd(prof, str(host_json), tmp_uid)
    job = Job(
        job_uid=job_uid,
        owner_id=user.id,
        name=(job_name.strip() or "AF3 Job"),
        status="submitted",
        params_json=json.dumps(payload, indent=2),
        workdir=str(job_dir),
        docker_cmd=cmd,
    )
    db.add(job)
    db.commit()

    return RedirectResponse("/jobs", status_code=303)


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_list(request: Request, db: Session = Depends(get_db), user: User | None = Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.is_admin:
        return RedirectResponse("/admin/jobs", status_code=303)

    page, per_page = parse_pagination(request)
    q = db.query(Job).filter(Job.owner_id == user.id).order_by(Job.created_at.desc())
    jobs, total, total_pages, page, per_page = paginate_query(q, page, per_page)

    for j in jobs:
        try:
            j.created_at = j.created_at.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        try:
            j.finished_at = j.finished_at.strftime("%Y-%m-%d %H:%M:%S") if getattr(j, "finished_at", None) else ""
        except Exception:
            j.finished_at = ""

    start_index = (page - 1) * per_page + 1

    return templates.TemplateResponse("user_jobs.html", {"request": request, "user": user, "jobs": jobs, "total": total, "total_pages": total_pages, "page": page, "per_page": per_page, "start_index": start_index})


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: int, db: Session = Depends(get_db), user: User | None = Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.is_admin:
        return RedirectResponse(f"/admin/jobs/{job_id}", status_code=303)

    job = db.query(Job).filter(Job.id == job_id, Job.owner_id == user.id).first()
    if not job:
        return RedirectResponse("/jobs", status_code=303)

    preview = ""
    if job.result_path and os.path.exists(job.result_path):
        with open(job.result_path, "r", encoding="utf-8", errors="replace") as f:
            preview = f.read()[:4000]
    return templates.TemplateResponse("user_job_detail.html", {"request": request, "user": user, "job": job, "preview": preview})


@app.post("/jobs/download")
async def jobs_download(job_ids: str = Form(default=""), db: Session = Depends(get_db), user: User | None = Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.is_admin:
        return RedirectResponse("/admin/jobs", status_code=303)

    ids = [int(x) for x in job_ids.split(",") if x.strip().isdigit()]
    if not ids:
        return RedirectResponse("/jobs", status_code=303)

    jobs = db.query(Job).filter(Job.owner_id == user.id, Job.id.in_(ids)).all()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for j in jobs:
            if j.workdir and os.path.isdir(j.workdir):
                for root, _, files in os.walk(j.workdir):
                    for fn in files:
                        full = Path(root) / fn
                        arc = f"{j.job_uid}/" + str(full.relative_to(j.workdir))
                        z.write(full, arcname=arc)
            z.writestr(f"{j.job_uid}/payload.json", j.params_json or "{}")
            if j.docker_cmd:
                z.writestr(f"{j.job_uid}/docker_command.sh", j.docker_cmd + "\n")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="af3_jobs.zip"' })


@app.get("/guide", response_class=HTMLResponse)
async def guide(request: Request, user: User | None = Depends(get_current_user_optional)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    if user.is_admin:
        return RedirectResponse("/admin/home", status_code=303)
    return templates.TemplateResponse("user_guide.html", {"request": request, "user": user})


@app.get("/admin/home", response_class=HTMLResponse)
async def admin_home(request: Request, admin: User = Depends(require_admin)):
    return templates.TemplateResponse("admin_home.html", {"request": request, "user": admin, "jobs_scope": "all"})


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    page, per_page = parse_pagination(request)
    q = db.query(User).order_by(User.created_at.desc())
    users, total, total_pages, page, per_page = paginate_query(q, page, per_page)
    return templates.TemplateResponse("admin_users.html", {"request": request, "user": admin, "users": users, "total": total, "total_pages": total_pages, "page": page, "per_page": per_page})


@app.post("/admin/users/new")
async def admin_user_new(username: str = Form(...), password: str = Form(...), is_admin: str = Form(default="0"), db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    username = username.strip()
    if len(password) < 8:
        return RedirectResponse("/admin/users", status_code=303)
    if db.query(User).filter(User.username == username).first():
        return RedirectResponse("/admin/users", status_code=303)
    u = User(username=username, password_hash=hash_password(password), is_admin=(is_admin == "1"))
    db.add(u); db.commit()
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/{uid}/delete")
async def admin_user_delete(uid: int, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    if uid == admin.id:
        return RedirectResponse("/admin/users", status_code=303)
    db.query(Job).filter(Job.owner_id == uid).delete()
    db.query(User).filter(User.id == uid).delete()
    db.commit()
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/{uid}/password")
async def admin_user_password(uid: int, new_password: str = Form(...), db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    if len(new_password) < 8:
        return RedirectResponse("/admin/users", status_code=303)
    u = db.query(User).filter(User.id == uid).first()
    if not u:
        return RedirectResponse("/admin/users", status_code=303)
    u.password_hash = hash_password(new_password)
    db.commit()
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/{uid}/edit")
async def admin_user_edit(uid: int, new_username: str = Form(...), is_admin: str = Form(default="0"), db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    u = db.query(User).filter(User.id == uid).first()
    if not u:
        return RedirectResponse("/admin/users", status_code=303)
    new_username = new_username.strip()
    if new_username and new_username != u.username:
        if db.query(User).filter(User.username == new_username).first():
            return RedirectResponse("/admin/users", status_code=303)
        u.username = new_username
    if u.id == admin.id:
        u.is_admin = True
    else:
        u.is_admin = (is_admin == "1")
    db.commit()
    return RedirectResponse("/admin/users", status_code=303)


@app.get("/admin/profile", response_class=HTMLResponse)
async def admin_profile(request: Request, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    prof = db.query(AdminProfile).filter(AdminProfile.id == 1).first() or ensure_admin_profile(db)
    if prof is None:
        # fallback: create a default profile row
        ensure_admin_profile(db)
        prof = db.query(AdminProfile).filter(AdminProfile.id==1).first()
    return templates.TemplateResponse("admin_profile.html", {"request": request, "user": admin, "profile": prof, "prof": prof})


@app.post("/admin/profile")
async def admin_profile_save(
    input_dir: str = Form(default=""),
    output_dir: str = Form(default=""),
    models_dir: str = Form(default=""),
    afdb_dir: str = Form(default=""),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    prof = db.query(AdminProfile).filter(AdminProfile.id == 1).first() or ensure_admin_profile(db)
    prof.input_dir = (input_dir or "").strip()
    prof.output_dir = (output_dir or "").strip()
    prof.models_dir = (models_dir or "").strip()
    prof.afdb_dir = (afdb_dir or "").strip()
    prof.updated_at = datetime.utcnow()
    db.commit()
    return RedirectResponse("/admin/profile", status_code=303)


@app.get("/admin/jobs", response_class=HTMLResponse)
async def admin_jobs(request: Request, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    page, per_page = parse_pagination(request)
    q = db.query(Job).options(joinedload(Job.owner)).order_by(Job.created_at.desc())
    jobs, total, total_pages, page, per_page = paginate_query(q, page, per_page)
    for j in jobs:
        try:
            j.created_at = j.created_at.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        try:
            j.finished_at = j.finished_at.strftime("%Y-%m-%d %H:%M:%S") if getattr(j, "finished_at", None) else ""
        except Exception:
            j.finished_at = ""
    start_index = (page - 1) * per_page + 1
    return templates.TemplateResponse("admin_jobs.html", {"request": request, "user": admin, "jobs": jobs, "total": total, "total_pages": total_pages, "page": page, "per_page": per_page, "start_index": start_index})


@app.get("/admin/jobs/{job_id}", response_class=HTMLResponse)
async def admin_job_detail(request: Request, job_id: int, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    job = db.query(Job).options(joinedload(Job.owner)).filter(Job.id == job_id).first()
    if not job:
        return RedirectResponse("/admin/jobs", status_code=303)
    return templates.TemplateResponse("admin_job_detail.html", {"request": request, "user": admin, "job": job, "preview": ""})


def zip_jobs(jobs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for j in jobs:
            if j.workdir and os.path.isdir(j.workdir):
                for root, _, files in os.walk(j.workdir):
                    for fn in files:
                        full = Path(root) / fn
                        arc = f"{j.job_uid}/" + str(full.relative_to(j.workdir))
                        z.write(full, arcname=arc)
            z.writestr(f"{j.job_uid}/payload.json", j.params_json or "{}")
            if j.docker_cmd:
                z.writestr(f"{j.job_uid}/docker_command.sh", j.docker_cmd + "\n")
    buf.seek(0)
    return buf


@app.post("/admin/jobs/download")
async def admin_jobs_download(job_ids: str = Form(default=""), db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    ids = [int(x) for x in job_ids.split(",") if x.strip().isdigit()]
    if not ids:
        return RedirectResponse("/admin/jobs", status_code=303)
    jobs = db.query(Job).options(joinedload(Job.owner)).filter(Job.id.in_(ids)).all()
    buf = zip_jobs(jobs)
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="af3_admin_jobs.zip"' })




@app.post("/admin/jobs/stop")
async def admin_jobs_stop(request: Request, job_ids: str = Form(...), db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    ids = [int(x) for x in (job_ids or "").split(",") if x.strip().isdigit()]
    for jid in ids:
        j = db.query(Job).filter(Job.id == jid).first()
        if not j:
            continue
        container = f"af3_{j.job_uid}"
        docker_stop(container)
        j.status = "stopped"
        j.finished_at = datetime.utcnow()
    db.commit()
    return RedirectResponse(url="/admin/jobs", status_code=303)

@app.post("/admin/jobs/delete")
async def admin_jobs_delete(job_ids: str = Form(default=""), db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    ids = [int(x) for x in job_ids.split(",") if x.strip().isdigit()]
    if not ids:
        return RedirectResponse("/admin/jobs", status_code=303)
    db.query(Job).filter(Job.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    return RedirectResponse("/admin/jobs", status_code=303)
