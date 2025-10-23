import os
import json
import math
import sqlite3
from functools import wraps
from typing import List, Dict, Optional

from flask import (
    Flask, render_template, request, jsonify, Response,
    redirect, url_for, flash, session
)

import numpy as np
from sympy import symbols, lambdify, diff, N
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------------------------------------------------------------
# Config: default plot window
# -----------------------------------------------------------------------------
DATA_XMIN_DEFAULT = -1999
DATA_XMAX_DEFAULT = 1999
DATA_YMIN_DEFAULT = -1999.0
DATA_YMAX_DEFAULT = 1999

# -----------------------------------------------------------------------------
# Flask setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

# -----------------------------------------------------------------------------
# SQLite auth: helpers + init
# -----------------------------------------------------------------------------
DB_PATH = os.getenv("AUTH_DB_PATH", "auth.db")

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

with app.app_context():
    init_db()

# -----------------------------------------------------------------------------
# Auth helpers
# -----------------------------------------------------------------------------
def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    with get_db() as db:
        return db.execute(
            "SELECT id, username, email FROM users WHERE id = ?",
            (uid,)
        ).fetchone()

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

@app.context_processor
def inject_user():
    return {"user": current_user()}

# -----------------------------------------------------------------------------
# Auth routes
# -----------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email    = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm  = request.form.get("confirm") or ""

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        pw_hash = generate_password_hash(password)
        try:
            with get_db() as db:
                db.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (username, email, pw_hash),
                )
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
            return render_template("register.html")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email_or_username = (request.form.get("email_or_username") or "").strip()
        password = request.form.get("password") or ""

        with get_db() as db:
            row = db.execute(
                "SELECT id, username, email, password_hash "
                "FROM users WHERE email = ? OR username = ?",
                (email_or_username.lower(), email_or_username)
            ).fetchone()

        if row and check_password_hash(row["password_hash"], password):
            session["user_id"] = row["id"]
            session["username"] = row["username"]
            flash("Logged in successfully.", "success")
            nxt = request.args.get("next") or url_for("home")
            return redirect(nxt)

        flash("Invalid credentials.", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

# -----------------------------------------------------------------------------
# Sympy parsing
# -----------------------------------------------------------------------------
x = symbols("x")
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

def parse_expression(expr_text: str):
    if "=" in expr_text or ";" in expr_text:
        raise ValueError("Only pure expressions in x are allowed (no assignments).")
    expr = parse_expr(expr_text, transformations=TRANSFORMS)
    if expr.free_symbols - {x}:
        raise ValueError("Use only the variable x in your expression.")
    return expr

# -----------------------------------------------------------------------------
# Basic pages
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/main", methods=["GET", "POST"])
@login_required
def main():
    expr_text = request.form.get("expr", "sin(x)")
    xmin = request.form.get("xmin", str(int(DATA_XMIN_DEFAULT)))
    xmax = request.form.get("xmax", str(int(DATA_XMAX_DEFAULT)))
    grid = request.form.get("grid", "on")
    title = request.form.get("title", "")
    return render_template(
        "index2.html",
        expr=expr_text, xmin=xmin, xmax=xmax, grid=grid, title=title
    )

# -----------------------------------------------------------------------------
# /data — sampling for on-screen plot (no coordinate/cell mapping)
# -----------------------------------------------------------------------------
@app.route("/data")
def data():
    expr_text = request.args.get("expr", "sin(x)")
    xmin = request.args.get("xmin", str(DATA_XMIN_DEFAULT))
    xmax = request.args.get("xmax", str(DATA_XMAX_DEFAULT))

    try:
        expr = parse_expression(expr_text)
        xmin = float(xmin)
        xmax = float(xmax)
        if xmin >= xmax:
            raise ValueError("xmin must be less than xmax.")

        f = lambdify(x, expr, modules=["numpy"])
        xs = np.linspace(xmin, xmax, 1000)
        ys = f(xs)

        xs_list = xs.tolist()
        ys_list: List[Optional[float]] = []
        for v in np.array(ys, dtype=np.complex128):
            if np.isfinite(v.real) and abs(v.imag) < 1e-12:
                ys_list.append(float(v.real))
            else:
                ys_list.append(None)

        return jsonify({"x": xs_list, "y": ys_list, "expr": expr_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------------------------------------------------------
# OpenAI Recommendations (optional; falls back if API not available)
# -----------------------------------------------------------------------------
RECO_MODEL = os.getenv("RECO_MODEL", "gpt-4o-mini")

def _recommend_prompt(expr, xmin, xmax, context_title=None):
    title_part = f' Title: "{context_title}".' if context_title else ""
    return f"""
You are helping a math grapher app suggest the next 3 expressions to plot. related to what user just plot

User just plotted:
- y = {expr}
Task:
1) Suggest 3 diverse follow-ups that are interesting from this start.
2) Add a short reason (<=14 words) for each.
3) Return STRICT JSON only:
{{
  "suggestions": [
    {{"label":"short title","expr":"valid_sympy_expression_in_x","xmin":-50,"xmax":50,"why":"short reason"}}
  ]
}}
4) so for example if user draw y=x, then may be y=x+1 or y=2x etc
No text outside JSON.{title_part}
""".strip()

def _fallback_suggestions(expr: str, xmin: float, xmax: float) -> List[Dict[str, object]]:
    ex = (expr or "").replace(" ", "").lower()

    def rng(a, b):
        try:
            return int(a), int(b)
        except Exception:
            return int(DATA_XMIN_DEFAULT), int(DATA_XMAX_DEFAULT)

    rxmin, rxmax = rng(xmin, xmax)

    if "sin(" in ex:
        return [
            {"label": "cos(x)",  "expr": "cos(x)",  "xmin": rxmin, "xmax": rxmax, "why": "Phase-shifted companion"},
            {"label": "tan(x)",  "expr": "tan(x)",  "xmin": -5, "xmax": 5, "why": "Compare poles and period"},
            {"label": "sin(2x)", "expr": "sin(2*x)","xmin": rxmin,"xmax": rxmax, "why": "Double the frequency"},
        ]
    if "cos(" in ex:
        return [
            {"label": "sin(x)",  "expr": "sin(x)",  "xmin": rxmin, "xmax": rxmax, "why": "Quadrature pair"},
            {"label": "tan(x)",  "expr": "tan(x)",  "xmin": -5, "xmax": 5, "why": "Asymptote comparison"},
            {"label": "cos(2x)", "expr": "cos(2*x)","xmin": rxmin,"xmax": rxmax, "why": "Higher frequency"},
        ]
    if "tan(" in ex:
        return [
            {"label": "sin(x)",  "expr": "sin(x)",  "xmin": rxmin, "xmax": rxmax, "why": "Smooth, no poles"},
            {"label": "cos(x)",  "expr": "cos(x)",  "xmin": rxmin, "xmax": rxmax, "why": "Phase contrast"},
            {"label": "atan(x)", "expr": "atan(x)", "xmin": rxmin, "xmax": rxmax, "why": "Inverse behavior"},
        ]

    if ex in {"x", "+x"} or ex.startswith("1*x") or ex == "x*1":
        return [
            {"label": "Shift right", "expr": "x+1",   "xmin": rxmin, "xmax": rxmax, "why": "Simple translation"},
            {"label": "Scale up",    "expr": "2*x",   "xmin": rxmin, "xmax": rxmax, "why": "Slope change"},
            {"label": "Square",      "expr": "x**2",  "xmin": -10,   "xmax": 10,    "why": "Compare curvature"},
        ]
    if any(tok in ex for tok in ["x**2","x^2"]):
        return [
            {"label": "x^3",   "expr": "x**3",       "xmin": rxmin, "xmax": rxmax, "why": "Odd vs even"},
            {"label": "abs(x)","expr": "Abs(x)",     "xmin": rxmin, "xmax": rxmax, "why": "V-shape contrast"},
            {"label": "bell",  "expr": "exp(-x**2)", "xmin": -10,   "xmax": 10,    "why": "Gaussian shape"},
        ]
    if "exp(" in ex:
        return [
            {"label": "ln(x)","expr": "log(x)", "xmin": 0, "xmax": max(10, rxmax), "why": "Inverse family"},
            {"label": "shift","expr": "exp(x)-1","xmin": rxmin,"xmax": rxmax,"why": "Baseline shift"},
            {"label": "decay","expr": "exp(-x)","xmin": rxmin,"xmax": rxmax,"why": "Opposite trend"},
        ]
    if "log(" in ex:
        return [
            {"label": "sqrt","expr": "sqrt(x)", "xmin": 0, "xmax": max(10, rxmax), "why": "Concave root"},
            {"label": "shift","expr": "log(x+1)", "xmin": 0, "xmax": max(10, rxmax), "why": "Domain shift"},
            {"label": "exp","expr": "exp(x)", "xmin": rxmin, "xmax": rxmax, "why": "Inverse growth"},
        ]
    return [
        {"label": "cos(x)",  "expr": "cos(x)",  "xmin": rxmin, "xmax": rxmax, "why": "Common companion"},
        {"label": "tan(x)",  "expr": "tan(x)",  "xmin": -5,    "xmax": 5,     "why": "Add asymptotes"},
        {"label": "sin(2x)", "expr": "sin(2*x)","xmin": rxmin, "xmax": rxmax, "why": "Frequency change"},
    ]

@app.post("/recommend")
def recommend():
    try:
        body = request.get_json(force=True, silent=True) or {}
        expr_text = (body.get("expr") or "").strip()
        xmin = float(body.get("xmin", DATA_XMIN_DEFAULT))
        xmax = float(body.get("xmax", DATA_XMAX_DEFAULT))
        title = (body.get("title") or "").strip()

        if not expr_text:
            return jsonify({"suggestions": []})

        # Try OpenAI first, fall back if unavailable
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = _recommend_prompt(expr_text, xmin, xmax, context_title=title)
            raw = None
            try:
                chat = client.chat.completions.create(
                    model=RECO_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                raw = chat.choices[0].message.content.strip()
            except TypeError:
                chat = client.chat.completions.create(
                    model=RECO_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=300,
                )
                raw = chat.choices[0].message.content.strip()

            data = json.loads(raw)
            suggestions = data.get("suggestions", [])[:3]
            clean = []
            for s in suggestions:
                clean.append({
                    "label": str(s.get("label", "Suggestion")),
                    "expr": str(s.get("expr", "sin(x)")),
                    "xmin": int(s.get("xmin", DATA_XMIN_DEFAULT)),
                    "xmax": int(s.get("xmax", DATA_XMAX_DEFAULT)),
                    "why":  str(s.get("why", "Looks interesting"))
                })
            if clean:
                return jsonify({"suggestions": clean})
        except Exception:
            pass

        return jsonify({"suggestions": _fallback_suggestions(expr_text, xmin, xmax)})
    except Exception as e:
        return jsonify({
            "suggestions": _fallback_suggestions("generic", DATA_XMIN_DEFAULT, DATA_XMAX_DEFAULT),
            "error": str(e)
        }), 200
# -----------------------------------------------------------------------------
# Integer path (steps of 100) + Arduino send
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Integer path (steps of 100) + Arduino send (UPDATED)
# -----------------------------------------------------------------------------
try:
    import serial  # optional
except Exception:
    serial = None

STEP_SIZE = 100                 # each grid tick = 100 motor steps
STEP_MIN  = -2900               # clamp in step space
STEP_MAX  =  2900
DOMAIN_FULL_SCALE = STEP_MAX / STEP_SIZE   # 29.0 domain units corresponds to 2900 steps
Y_TARGET_DOMAIN   = DOMAIN_FULL_SCALE * 0.9  # target typical magnitude (~26.1) to avoid clamp
def _send_cells_to_arduino(cells, echo_preview_chars: int = 200, read_seconds: float = 5.0):
    """
    Send cells as a single JSON line and print TX/RX to server stdout (PyCharm).
    Example TX: {"cells":[[0,0],[100,100],...]}
    Also reads back lines for up to read_seconds and prints them with <<< prefix.
    """
    port = os.getenv("ARDUINO_PORT", "/dev/tty.usbserial-1140")
    baud = int(os.getenv("ARDUINO_BAUD", "9600"))
    if not port:
        print(">>> [ARDUINO TX] SKIP (ARDUINO_PORT not set)")
        return "ARDUINO_PORT not set"
    if serial is None:
        print(">>> [ARDUINO TX] SKIP (pyserial not available)")
        return "pyserial not available"

    try:
        import time
        line = json.dumps({"cells": cells}, separators=(",", ":")) + "\n"

        # ---- TX preview ----
        preview = line[:echo_preview_chars] + (" ...[truncated]" if len(line) > echo_preview_chars else "")
        print(f">>> [ARDUINO TX] {preview}")
        print(f">>> [ARDUINO TX] cells_count={len(cells)} port={port} baud={baud}")

        with serial.Serial(port, baudrate=baud, timeout=0.5) as ser:
            time.sleep(2.0)  # let the board auto-reset & open its port
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            # write & flush
            ser.write(line.encode("utf-8"))
            ser.flush()

            # ---- RX loop ----
            t0 = time.time()
            rx_log = []
            while time.time() - t0 < read_seconds:
                try:
                    rx = ser.readline().decode("utf-8", errors="ignore").strip()
                except Exception:
                    rx = ""
                if rx:
                    rx_log.append(rx)
                    print(f"<<< [ARDUINO RX] {rx}")

        # short summary for HTTP response payload
        head = ", ".join(rx_log[:3]) if rx_log else ""
        return f"sent {len(cells)} cells to {port}@{baud}; rx_lines={len(rx_log)}; head='{head[:120]}'"

    except Exception as e:
        print(f"!!! [ARDUINO ERR] {e}")
        return f"send failed: {e}"


def to_step(v: float) -> int:
    """
    Convert a *domain* value v to the nearest 100-step tick and clamp.
    Domain 1.0 -> 100 steps, so outputs are ..., -2900, -2800, ... , 2900.
    """
    s = int(round(v)) * STEP_SIZE
    return max(STEP_MIN, min(STEP_MAX, s))

def _build_centered_step_xs() -> list[int]:
    """
    X sequence in *step space* starting at 0:
    0, +100, +200, ... +2900, then -100, -200, ... -2900
    """
    pos = [i * STEP_SIZE for i in range(0, STEP_MAX // STEP_SIZE + 1)]     # 0..2900
    neg = [-i * STEP_SIZE for i in range(1, STEP_MAX // STEP_SIZE + 1)]     # -100..-2900
    return pos + neg
def _autoscale_gain(
    y_vals: np.ndarray,
    target_domain: float = Y_TARGET_DOMAIN,
    allow_upscale: bool = True,
    max_upscale: float = None,   # None => no explicit cap (still clamped by step limits)
) -> float:
    """
    Choose a gain so the 98th percentile of |y| lands near ±target_domain.
    If allow_upscale is True, we also amplify small signals (e.g., sin).
    """
    finite = y_vals[np.isfinite(y_vals)]
    if finite.size == 0:
        return 1.0

    mag = np.abs(finite)
    q98 = float(np.quantile(mag, 0.98))
    if q98 <= 0:
        return 1.0

    g = target_domain / q98  # how much to scale to hit the target
    if not allow_upscale:
        g = min(1.0, g)  # only shrink
    else:
        if max_upscale is not None:
            g = min(max_upscale, g)  # cap amplification
    # never negative or NaN
    return float(max(0.0, g))

# ---------------- helpers: snap + affine detection ----------------
from sympy import Poly, diff, N  # ensure imported

def round_half_away_from_zero(v: float) -> int:
    """Round halves away from zero: 0.5->1, -0.5->-1, etc."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))

def to_step_from_domain(v_domain: float) -> int:
    """
    Convert a domain value to the nearest 100-step tick (±2900 clamp).
    Domain 1.0 -> 100 steps. Uses half-away-from-zero.
    """
    s = round_half_away_from_zero(v_domain) * STEP_SIZE
    return max(STEP_MIN, min(STEP_MAX, s))

def _affine_params(expr):
    """
    If expr is affine: expr == a*x + b (degree <= 1), return (a, b) as floats.
    Otherwise return None.
    """
    try:
        p = Poly(expr, x)
        if p.total_degree() <= 1:
            a = float(N(p.coeffs()[0])) if p.total_degree() == 1 else 0.0
            # safer extraction for constant term
            b = float(N(p.eval(0)))
            # If degree==1 but Poly put constant first, derive a,b explicitly:
            # a = float(N(p.diff().eval(0)))  # derivative at any x is slope
            # b = float(N(expr.subs(x, 0)))   # intercept
            # But the above two lines are more robust across sympy versions:
            a = float(N(diff(expr, x)))
            b = float(N(expr.subs(x, 0)))
            return a, b
        return None
    except Exception:
        # fallback: derivative check for const slope
        try:
            s = diff(expr, x)
            if not getattr(s, "free_symbols", set()):
                a = float(N(s))
                b = float(N(expr.subs(x, 0)))
                return a, b
        except Exception:
            pass
        return None
def _quantize_domain_sequence(y_domain: np.ndarray,
                              clamp_domain: float = DOMAIN_FULL_SCALE,
                              step_size: int = STEP_SIZE) -> List[int]:
    """
    Quantize a sequence of domain values to integer ticks (1 tick = 1 domain unit),
    with error diffusion to avoid plateaus. Returns step-space ints (multiples of 100).
    """
    out_steps: List[int] = []
    err = 0.0
    lo, hi = -clamp_domain, clamp_domain

    for v in y_domain:
        if not np.isfinite(v):
            out_steps.append(None)  # caller can skip
            continue
        # diffuse accumulated error so we don't get long flats
        vv = v + err
        k = round(vv)  # integer domain tick
        err = vv - k
        # clamp in domain units, then to steps
        k = int(max(lo, min(hi, k)))
        out_steps.append(int(k * step_size))
    return out_steps
@app.route("/coords")
def coords():
    """
    Build a *single-stroke* polyline in step space for Arduino:
      • x ticks fixed by the chosen step: 0, step, 2*step, …, 2900, then -step, …, -2900
      • Nonlinear: autoscale Y and snap to 'step'-grid
      • Linear y=a*x+b: exact straight line in step space
          - y=x: force y_step == x_step exactly
          - general a,b: y_domain = a*(x_step/step) + b, then snap to 'step'-grid

    Query params:
      expr   : sympy expression (required)
      step   : integer step size in motor steps (optional; default 100)
               The effective step will be a divisor of 2900 (we coerce via gcd).
      xmin,xmax,ymin,ymax,dx : accepted for metadata only
      print=1 : print preview on server
      send=1  : send to Arduino (if serial available)
    """
    expr_text = request.args.get("expr", "sin(x)")
    do_print  = request.args.get("print", "0") == "1"
    do_send   = request.args.get("send",  "0") == "1"

    # --- parse metadata args (not used to change fixed x ticks) ---
    try:
        xmin = float(request.args.get("xmin", "-29"))
        xmax = float(request.args.get("xmax", "29"))
        ymin = float(request.args.get("ymin", "-29"))
        ymax = float(request.args.get("ymax", "29"))
        dx   = float(request.args.get("dx",   "1"))
    except Exception:
        return jsonify({"error": "bad numeric parameters"}), 400

    # --- choose effective step size (divisor of STEP_MAX=2900) ---
    try:
        requested_step = int(request.args.get("step", str(STEP_SIZE)))
    except Exception:
        requested_step = STEP_SIZE

    # guardrails: reasonable min/max
    requested_step = max(1, min(500, requested_step))
    # make it divide STEP_MAX exactly by taking gcd
    eff_step = math.gcd(STEP_MAX, requested_step)
    # keep it practical (avoid 1 unless explicitly asked)
    if requested_step >= 2 and eff_step == 1:
        eff_step = requested_step  # if they *really* want non-divisor, allow, but ticks won't hit 2900 exactly

    # helpers bound to the effective step
    def build_centered_step_xs(step_sz: int) -> list[int]:
        pos = [i * step_sz for i in range(0, STEP_MAX // step_sz + 1)]     # 0..+2900
        neg = [-i * step_sz for i in range(1, STEP_MAX // step_sz + 1)]     # -step..-2900
        return pos + neg

    def round_half_away_from_zero(v: float) -> int:
        return int(math.copysign(math.floor(abs(v) + 0.5), v))

    def to_step_from_domain_eff(v_domain: float, step_sz: int) -> int:
        s = round_half_away_from_zero(v_domain) * step_sz
        return max(STEP_MIN, min(STEP_MAX, s))

    # effective domain full-scale and target for autoscale gain
    domain_full_scale_eff = STEP_MAX / eff_step               # e.g., step=50 -> 58.0
    y_target_domain_eff   = domain_full_scale_eff * 0.9       # keep same headroom as before

    try:
        # compile expression
        expr = parse_expression(expr_text)
        f = lambdify(x, expr, modules=["numpy"])

        # fixed x ticks for the chosen step
        x_steps  = build_centered_step_xs(eff_step)                  # [0..+2900] + [-step..-2900]
        x_domain = np.array(x_steps, dtype=float) / float(eff_step)  # domain x = steps/eff_step

        cells: list[tuple[int, int]] = [(0, 0)]  # origin once

        # linear detection (your existing helper)
        aff = _affine_params(expr)
        if aff is not None:
            a, b = aff
            gain = 1.0  # record for meta

            if abs(a - 1.0) < 1e-12 and abs(b) < 1e-12:
                # exact diagonal regardless of step size
                last = cells[-1]
                for xv in x_steps:
                    yv = xv
                    cur = (xv, yv)
                    if cur != last:
                        cells.append(cur)
                        last = cur
            else:
                # general linear: y = a*x + b
                last = cells[-1]
                # y(0)=b
                y0s = to_step_from_domain_eff(b, eff_step)
                if last != (0, y0s):
                    cells.append((0, y0s))
                    last = (0, y0s)

                for xv in x_steps[1:]:
                    x_dom = xv / float(eff_step)
                    y_dom = a * x_dom + b
                    yv = to_step_from_domain_eff(y_dom, eff_step)
                    cur = (xv, yv)
                    if cur != last:
                        cells.append(cur)
                        last = cur
        else:
            # nonlinear: evaluate + autoscale using the effective domain scale
            y_raw_complex = np.array(f(x_domain), dtype=np.complex128)
            y_real = np.where(
                np.isfinite(y_raw_complex.real) & (np.abs(y_raw_complex.imag) < 1e-12),
                y_raw_complex.real,
                np.nan
            )
            gain = _autoscale_gain(
                y_real,
                target_domain=y_target_domain_eff,
                allow_upscale=True,
                max_upscale=domain_full_scale_eff  # e.g., ~58 when step=50
            )

            last = cells[-1]
            # first at x=0
            if np.isfinite(y_real[0]):
                y0s = to_step_from_domain_eff(float(gain * y_real[0]), eff_step)
                if last != (0, y0s):
                    cells.append((0, y0s))
                    last = (0, y0s)

            # remaining ticks
            for xv, yv_dom in zip(x_steps[1:], y_real[1:]):
                if not np.isfinite(yv_dom):
                    continue
                yv = to_step_from_domain_eff(float(gain * yv_dom), eff_step)
                cur = (xv, yv)
                if cur != last:
                    cells.append(cur)
                    last = cur

        payload = {
            "cells": cells,
            "meta": {
                "expr": expr_text,
                "step_size_requested": requested_step,
                "step_size_effective": eff_step,
                "clamp": [STEP_MIN, STEP_MAX],
                "order": f"0 -> +{STEP_MAX} step {eff_step}, then -{eff_step} -> -{STEP_MAX}",
                "auto_gain": gain,
                "domain_full_scale_effective": domain_full_scale_eff,
                "y_target_domain_effective": y_target_domain_eff,
                "compat_params": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "dx": dx},
            }
        }

        if do_print:
            prev = cells if len(cells) <= 200 else (cells[:200] + [("...", "...")])
            print("\n=== SINGLE-STROKE CELLS ===")
            print(f"count={len(cells)}; step_eff={eff_step}; clamp=±{STEP_MAX}; gain={payload['meta']['auto_gain']:.6g}")
            print(prev)
            print("=== END ===\n")

        if do_send:
            payload["send_status"] = _send_cells_to_arduino(cells)

        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # In production, consider: app.run(host="0.0.0.0", port=8000)
    app.run(debug=True)
