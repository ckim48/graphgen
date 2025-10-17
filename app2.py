import os
import json
import math
import sqlite3
from functools import wraps
from typing import List, Tuple, Dict, Optional

from flask import (
    Flask, render_template, request, jsonify, Response,
    redirect, url_for, flash, session
)

import numpy as np
from sympy import symbols, lambdify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from werkzeug.security import generate_password_hash, check_password_hash

# Optional serial sending
try:
    import serial  # pyserial
except Exception:
    serial = None

# -----------------------------------------------------------------------------
# Config: default plot window and integer grid size
# -----------------------------------------------------------------------------
DATA_XMIN_DEFAULT = -1999
DATA_XMAX_DEFAULT = 1999
DATA_YMIN_DEFAULT = -1999.0
DATA_YMAX_DEFAULT = 1999
GRID_N = 91  # centered integer indices: -(N-1)//2 .. +(N-1)//2 (odd number required)

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
# Math / helpers
# -----------------------------------------------------------------------------
def _build_centered_samples(xmin: float, xmax: float, dx: float, ndp: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (xi, split_mask) where xi starts at 0, goes to +xmax, then (split) to -dx, -2dx ... down to xmin.
    split_mask is a boolean array with True at positions that should 'lift the pen' (i.e., break continuity).
    If 0 is outside [xmin, xmax] or dx<=0, fall back to normal xmin→xmax sampling with no split.
    """
    if dx <= 0 or not (xmin <= 0.0 <= xmax):
        steps = int(math.floor((xmax - xmin)/dx)) + 1 if dx > 0 else 0
        xi = xmin + np.arange(steps, dtype=float) * dx if steps > 0 else np.array([], dtype=float)
        if ndp is not None and steps > 0:
            xi = np.round(xi, ndp)
        return xi, np.zeros_like(xi, dtype=bool)

    # right side: 0, dx, 2dx, ... <= xmax
    r_steps = int(math.floor((xmax - 0.0)/dx)) + 1
    xi_r = 0.0 + np.arange(r_steps, dtype=float) * dx

    # left side: -dx, -2dx, ... >= xmin
    l_steps = int(math.floor((0.0 - xmin)/dx))      # number of negative steps
    xi_l = -dx - np.arange(l_steps, dtype=float) * dx  # -dx, -2dx, ...

    xi = np.concatenate([xi_r, xi_l])
    if ndp is not None:
        xi = np.round(xi, ndp)

    # split after finishing right side so we "lift pen" before going to negatives
    split_mask = np.zeros_like(xi, dtype=bool)
    if len(xi_r) > 0 and len(xi_l) > 0:
        split_mask[len(xi_r)] = True  # marker to break continuity at that index
    return xi, split_mask

def _apply_axes_transform(
    seq,
    *,
    swap_xy: bool = False,
    invert_x: bool = False,
    invert_y: bool = False,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
):
    """
    Transform a sequence of points [(x,y), ...] (ints or floats or strings)
    into stepper-friendly integer tuples with optional swap/invert/scale/offset.
    None points are kept as None (pen lifts in our pipelines).
    """
    out = []
    for p in seq:
        if p is None:
            out.append(None)
            continue
        x, y = p
        x = float(x); y = float(y)

        if swap_xy:
            x, y = y, x
        if invert_x:
            x = -x
        if invert_y:
            y = -y

        x = int(round(x * scale_x + offset_x))
        y = int(round(y * scale_y + offset_y))
        out.append((x, y))
    return out

def _uniform_fit(points: List[Tuple[float, float]],
                 txmin: float, txmax: float,
                 tymin: float, tymax: float,
                 margin_frac: float = 0.05) -> List[Tuple[float, float]]:
    """
    Uniformly scale+center (x,y) points to fit inside [txmin,txmax]×[tymin,tymax]
    with a margin percentage (0..0.45). Preserves aspect (shape).
    """
    if not points:
        return points

    m = max(0.0, min(0.45, float(margin_frac)))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    sxmin, sxmax = min(xs), max(xs)
    symin, symax = min(ys), max(ys)

    sw = max(1e-12, sxmax - sxmin)
    sh = max(1e-12, symax - symin)

    # Target center and drawable area after margins
    tcx = 0.5 * (txmin + txmax)
    tcy = 0.5 * (tymin + tymax)
    tw  = max(1e-12, (txmax - txmin) * (1.0 - 2.0 * m))
    th  = max(1e-12, (tymax - tymin) * (1.0 - 2.0 * m))

    s = min(tw / sw, th / sh)  # uniform (preserve shape)

    scx = 0.5 * (sxmin + sxmax)
    scy = 0.5 * (symin + symax)

    out: List[Tuple[float, float]] = []
    for x, y in points:
        nx = (x - scx) * s + tcx
        ny = (y - scy) * s + tcy
        out.append((nx, ny))
    return out

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
# Data sampling for on-screen plot
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
# Grid mapping helpers (centered integer grid, size N)
# -----------------------------------------------------------------------------
def _bresenham_int(x0: int, y0: int, x1: int, y1: int):
    points = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points

def _to_centered_cells_N(
    xs: np.ndarray,
    ys: List[Optional[float]],
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    N: int = GRID_N,
    skip_axes: bool = False
) -> List[Tuple[int,int]]:
    """
    Map (x,y) into integer cells on a centered N×N grid using the *given*
    fixed world ranges only. No re-centering, no square normalization.

    x in [xmin,xmax] -> ix in [-half, +half]
    y in [ymin,ymax] -> iy in [-half, +half]

    We choose symmetric scaling so that x=0 -> ix=0 and y=0 -> iy=0.
    """
    if N % 2 == 0:
        raise ValueError("N must be odd so the grid is symmetric around zero.")
    half = (N - 1) // 2

    # Independent symmetric scales for x and y
    sx = half / max(abs(xmin), abs(xmax), 1e-12)
    sy = half / max(abs(ymin), abs(ymax), 1e-12)

    def inside_soft(ixf: float, iyf: float) -> bool:
        # allow ±(half+0.5) so rounding stays in-bounds
        bound = half + 0.5
        return -bound <= ixf <= bound and -bound <= iyf <= bound

    grid_pts: List[Tuple[int,int]] = []
    last_int: Optional[Tuple[int,int]] = None

    for xx, yy in zip(xs, ys):
        if yy is None or not (math.isfinite(xx) and math.isfinite(yy)):
            last_int = None
            continue

        gx = xx * sx
        gy = yy * sy

        if not inside_soft(gx, gy):
            last_int = None
            continue

        ix = int(round(gx))
        iy = int(round(gy))
        ix = max(-half, min(half, ix))
        iy = max(-half, min(half, iy))

        if skip_axes and (ix == 0 or iy == 0):
            last_int = None
            continue

        cur = (ix, iy)
        if last_int is None:
            if not grid_pts or grid_pts[-1] != cur:
                grid_pts.append(cur)
            last_int = cur
            continue

        if cur == last_int:
            continue

        for px, py in _bresenham_int(last_int[0], last_int[1], cur[0], cur[1]):
            if skip_axes and (px == 0 or py == 0):
                continue
            if not grid_pts or grid_pts[-1] != (px, py):
                grid_pts.append((px, py))

        last_int = cur

    return grid_pts

# -----------------------------------------------------------------------------
# Legacy raster helpers (kept for compatibility / debugging)
# -----------------------------------------------------------------------------
def _map_to_grid(xs: np.ndarray, ys: List[Optional[float]],
                 xmin: float, xmax: float, ymin: float, ymax: float, q: int = 10
                 ) -> Tuple[List[Optional[int]], List[Optional[int]], int, int]:
    cols = rows = 2 * q
    sx = (cols - 1) / (xmax - xmin) if xmax != xmin else 1.0
    sy = (rows - 1) / (ymax - ymin) if ymax != ymin else 1.0

    gx: List[Optional[int]] = []
    gy: List[Optional[int]] = []

    for xx, y in zip(xs, ys):
        if y is None or (isinstance(y, float) and not math.isfinite(y)):
            gx.append(None); gy.append(None); continue
        if not (math.isfinite(xx) and math.isfinite(y)):
            gx.append(None); gy.append(None); continue
        if y < ymin or y > ymax or xx < xmin or xx > xmax:
            gx.append(None); gy.append(None); continue

        col = int(round((xx - xmin) * sx))
        row = int(round((y  - ymin) * sy))
        col = max(0, min(cols - 1, col))
        row = max(0, min(rows - 1, row))
        gx.append(col); gy.append(row)

    return gx, gy, cols, rows

def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points

def _rasterize_path(gx: List[Optional[int]], gy: List[Optional[int]]) -> List[Dict[str, int]]:
    path: List[Dict[str, int]] = []
    prev: Optional[Tuple[int,int]] = None

    i = 0
    n = len(gx)
    while i < n:
        while i < n and (gx[i] is None or gy[i] is None):
            prev = None
            i += 1
        if i >= n:
            break

        start = (gx[i], gy[i])
        if prev != start:
            path.append({"x": start[0], "y": start[1], "pen": 0})
            prev = start
        i += 1

        while i < n and gx[i] is not None and gy[i] is not None:
            p = (gx[i], gy[i])
            if p != prev:
                for (cx, cy) in _bresenham_line(prev[0], prev[1], p[0], p[1]):
                    if path and path[-1]["x"] == cx and path[-1]["y"] == cy:
                        continue
                    path.append({"x": cx, "y": cy, "pen": 1})
                prev = p
            i += 1

    return path

def _to_cells_raw(
    xs: np.ndarray,
    ys: List[Optional[float]],
    xmin: int, xmax: int,
    ymin: int, ymax: int
) -> List[Tuple[int, int]]:
    """
    Map samples directly to integer (x,y) with no normalization.
    - Round x,y to nearest ints.
    - Clip to [xmin,xmax] × [ymin,ymax].
    - Connect successive valid points with Bresenham.
    """
    pts: List[Tuple[int, int]] = []
    last: Optional[Tuple[int, int]] = None

    def inside(ix: int, iy: int) -> bool:
        return xmin <= ix <= xmax and ymin <= iy <= ymax

    for xx, yy in zip(xs, ys):
        if yy is None or not (math.isfinite(xx) and math.isfinite(yy)):
            last = None
            continue

        ix = int(round(xx))
        iy = int(round(yy))

        if not inside(ix, iy):
            last = None
            continue

        cur = (ix, iy)
        if last is None:
            if not pts or pts[-1] != cur:
                pts.append(cur)
            last = cur
            continue

        if cur == last:
            continue

        # Densify with Bresenham, then clip inside window
        for px, py in _bresenham_line(last[0], last[1], cur[0], cur[1]):
            if inside(px, py) and (not pts or pts[-1] != (px, py)):
                pts.append((px, py))

        last = cur

    return pts

def _to_cells_raw_with_splits(
    xs: np.ndarray,
    ys: List[Optional[float]],
    split_mask: Optional[np.ndarray],
    xmin: int, xmax: int, ymin: int, ymax: int
) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    last: Optional[Tuple[int, int]] = None

    def inside(ix: int, iy: int) -> bool:
        return xmin <= ix <= xmax and ymin <= iy <= ymax

    for i, (xx, yy) in enumerate(zip(xs, ys)):
        # lift the pen at split boundaries produced by single_stroke center sampling
        if split_mask is not None and i < len(split_mask) and split_mask[i]:
            last = None

        if yy is None or not (math.isfinite(xx) and math.isfinite(yy)):
            last = None
            continue

        ix = int(round(xx))
        iy = int(round(yy))
        if not inside(ix, iy):
            last = None
            continue

        cur = (ix, iy)
        if last is None:
            if not pts or pts[-1] != cur:
                pts.append(cur)
            last = cur
            continue

        if cur == last:
            continue

        for px, py in _bresenham_line(last[0], last[1], cur[0], cur[1]):
            if inside(px, py) and (not pts or pts[-1] != (px, py)):
                pts.append((px, py))

        last = cur

    return pts

def _to_points_raw_float(
    xs: np.ndarray,
    ys: List[Optional[float]],
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    ndp: Optional[int] = None,
    fixed_str: bool = False,
) -> List[Tuple[float, float]]:
    """
    Keep raw (x,y) as floats, clip to [xmin,xmax]×[ymin,ymax], optional rounding.
    If fixed_str=True, return ('x','y') as strings with ndp fixed decimals.
    """
    pts: List[Tuple[float, float]] = []
    for xx, yy in zip(xs, ys):
        if yy is None or not (math.isfinite(xx) and math.isfinite(yy)):
            continue
        if not (xmin <= xx <= xmax and ymin <= yy <= ymax):
            continue

        if ndp is not None:
            xx_r = round(float(xx), ndp)
            yy_r = round(float(yy), ndp)
        else:
            xx_r = float(xx)
            yy_r = float(yy)

        if fixed_str and ndp is not None:
            fmt = f"{{:.{ndp}f}}"
            pts.append((fmt.format(xx_r), fmt.format(yy_r)))
        else:
            pts.append((xx_r, yy_r))

    return pts

# -----------------------------------------------------------------------------
# /coords — float points and/or integer cells with optional send
# -----------------------------------------------------------------------------
@app.route("/coords")
def coords():
    """
    Return sampled data for y=f(x) and (optionally) send step-scaled coords to Arduino.

    Query parameters:
      expr           : expression in x (e.g. sin(x))
      xmin,xmax      : float bounds (domain units)
      ymin,ymax      : optional y-clip (domain units)
      dx             : sampling step (domain units)
      ndp            : decimal places for rounding when fixed_str=1 (0..8)
      fixed_str      : "1"/"0" -> keep trailing zeros (requires ndp)
      single_stroke  : "1"/"0" -> center-out sampling (default "1")
      fit            : "1"/"0" -> uniform fit to window (in domain units)
      fit_margin     : float   -> 0..0.45 (default 0.05)
      sx             : steps per X unit (default 100)
      sy             : steps per Y unit (default = sx)
      cells          : "1"/"0" -> include integer cells path (domain→steps) (default "0")
      pretty         : "1"/"0" -> pretty-print JSON
      print          : "1"/"0" -> print payload server-side
      send           : "1"/"0" -> send to Arduino (forces cells=1)
    """
    # ----- read params -----
    expr_text = request.args.get("expr", "sin(x)")

    xmin = float(request.args.get("xmin", DATA_XMIN_DEFAULT))
    xmax = float(request.args.get("xmax", DATA_XMAX_DEFAULT))

    ymin_s = request.args.get("ymin")
    ymax_s = request.args.get("ymax")
    yclip = (ymin_s is not None) and (ymax_s is not None)
    ymin = float(ymin_s) if yclip else DATA_YMIN_DEFAULT
    ymax = float(ymax_s) if yclip else DATA_YMAX_DEFAULT

    dx = float(request.args.get("dx", 1))

    ndp_param = request.args.get("ndp")
    try:
        ndp = int(ndp_param) if ndp_param not in (None, "") else None
    except Exception:
        ndp = None
    if ndp is not None:
        ndp = max(0, min(8, ndp))

    fixed_str     = request.args.get("fixed_str", "0") == "1"
    single_stroke = request.args.get("single_stroke", "1") == "1"
    fit           = request.args.get("fit", "0") == "1"
    try:
        margin = float(request.args.get("fit_margin", "0.05") or 0.05)
    except Exception:
        margin = 0.05
    margin = max(0.0, min(0.45, margin))

    # anisotropic step scaling (domain → steps)
    try:
        sx = float(request.args.get("sx", "100"))
    except Exception:
        sx = 100.0
    try:
        sy = float(request.args.get("sy", str(sx)))
    except Exception:
        sy = sx

    want_cells = request.args.get("cells", "0") == "1"
    pretty     = request.args.get("pretty", "0") == "1"
    do_print   = request.args.get("print", "0") == "1"
    do_send    = request.args.get("send", "0") == "1"
    if do_send:
        want_cells = True  # ensure we build a discrete path to send

    # ----- validate -----
    if dx <= 0:
        return jsonify({"error": "dx must be positive"}), 400
    if xmin >= xmax:
        return jsonify({"error": "xmin < xmax required"}), 400

    try:
        # ----- parse and compile f(x) -----
        expr = parse_expression(expr_text)
        f = lambdify(x, expr, modules=["numpy"])

        # ----- sample x’s in DOMAIN UNITS -----
        if single_stroke and (xmin <= 0.0 <= xmax):
            xi, split_mask = _build_centered_samples(xmin, xmax, dx, ndp)
        else:
            steps = int(math.floor((xmax - xmin) / dx)) + 1
            xi = xmin + np.arange(steps, dtype=float) * dx
            if ndp is not None:
                xi = np.round(xi, ndp)
            split_mask = None

        # ----- evaluate y = f(x) (domain units) -----
        yi = f(xi)
        ys: List[Optional[float]] = []
        for v in np.array(yi, dtype=np.complex128):
            if not np.isfinite(v.real) or abs(v.imag) > 1e-12:
                ys.append(None)
            else:
                val = float(v.real)
                if yclip and not (ymin <= val <= ymax):
                    ys.append(None)
                else:
                    ys.append(val)

        # ----- build float points (domain), clip + optional rounding -----
        points_domain = _to_points_raw_float(
            xi, ys, xmin, xmax, ymin, ymax, ndp=ndp, fixed_str=fixed_str
        )

        # optional uniform fit to the domain window
        if fit and points_domain:
            pts = [(float(px), float(py)) for (px, py) in points_domain]
            fitted = _uniform_fit(pts, xmin, xmax, ymin, ymax, margin_frac=margin)
            if fixed_str and ndp is not None:
                fmt = f"{{:.{ndp}f}}"
                points_domain = [(fmt.format(a), fmt.format(b)) for (a, b) in fitted]
            else:
                points_domain = fitted

        # ----- DOMAIN → STEPS (anisotropic) -----
        points_steps = [(float(px) * sx, float(py) * sy) for (px, py) in points_domain]

        payload: Dict[str, object] = {
            "meta": {
                "expr": expr_text,
                "xmin": xmin, "xmax": xmax,
                "ymin": ymin, "ymax": ymax,
                "dx": dx,
                "ndp": ndp,
                "fixed_str": fixed_str,
                "fit": fit,
                "fit_margin": margin,
                "single_stroke": bool(single_stroke and (xmin <= 0.0 <= xmax)),
                "sx": sx,
                "sy": sy,
                "units": "points already scaled to STEPS",
            },
            "points": points_steps,
        }

        # ----- integer cells path (domain ints → steps) -----
        if want_cells:
            xi_i_min, xi_i_max = int(round(xmin)), int(round(xmax))
            yi_i_min, yi_i_max = int(round(ymin)), int(round(ymax))

            # Build a discrete polyline in DOMAIN integer space
            cells_domain = _to_cells_raw_with_splits(
                xi, ys, split_mask, xi_i_min, xi_i_max, yi_i_min, yi_i_max
            )

            # Scale to STEPS with anisotropic factors
            cells_steps = [(int(round(cx * sx)), int(round(cy * sy))) for (cx, cy) in cells_domain]
            payload["cells"] = cells_steps

            # Optional: send to Arduino
            if do_send:
                preview = str(cells_steps)
                if len(preview) > 1000:
                    preview = preview[:1000] + "... (truncated)"
                print("\n=== TX → Arduino (cells, steps) ===")
                print(preview)
                print("=== END TX ===\n")

                payload["send_status"] = _send_cells_to_arduino(cells_steps)

        if do_print:
            print(json.dumps(payload, indent=2 if pretty else None))

        return Response(
            json.dumps(payload, indent=2 if pretty else None),
            mimetype="application/json",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------------------------------------------------------
# /path — legacy int cell path or float mode
# -----------------------------------------------------------------------------
@app.route("/path")
def path():
    """
    Build path data for the current expression.

    Query params:
      expr          : string  (e.g., "x", "sin(x)")
      xmin, xmax    : floats  (world x window)
      ymin, ymax    : floats  (world y window)
      dx            : float   sampling step for float mode (default 0.1)
      float_points  : "1"/"0" emit raw float (x,y) pairs instead of integer cells (default "0")
      ndp           : int     number of decimal places to round x,y to in float mode; omit to keep raw
      fixed_str     : "1"/"0" if "1", return fixed-width strings with trailing zeros (requires ndp)
      print         : "1"/"0" print payload to server console
      pretty        : "1"/"0" pretty-print JSON
      send          : "1"/"0" send payload to Arduino via _send_cells_to_arduino()
    """
    expr_text = request.args.get("expr", "sin(x)")
    xmin = float(request.args.get("xmin", -30))
    xmax = float(request.args.get("xmax",  30))
    ymin = float(request.args.get("ymin", -25))
    ymax = float(request.args.get("ymax",  25))

    # Sampling + float formatting
    dx = float(request.args.get("dx", 1))                  # step for float mode
    float_mode = request.args.get("float_points", "0") == "1"

    ndp_param = request.args.get("ndp", 3)                # decimals for rounding
    ndp = int(ndp_param) if ndp_param is not None and ndp_param != "" else None
    if ndp is not None:
        ndp = max(0, min(ndp, 8))                            # clamp to a sensible range

    fixed_str = request.args.get("fixed_str", "1") == "1"    # keep trailing zeros as strings

    do_print = request.args.get("print", "0") == "1"
    pretty   = request.args.get("pretty", "0") == "1"
    do_send  = request.args.get("send", "0") == "1"

    # Basic validation
    if xmin >= xmax:
        return jsonify({"error": "xmin must be less than xmax."}), 400
    if ymin >= ymax:
        return jsonify({"error": "ymin must be less than ymax."}), 400
    if dx <= 0:
        return jsonify({"error": "dx must be positive."}), 400

    try:
        # Parse + make function
        expr = parse_expression(expr_text)
        f = lambdify(x, expr, modules=["numpy"])

        if float_mode:
            # -------- FLOAT MODE --------
            steps = int(math.floor((xmax - xmin) / dx)) + 1
            xi = xmin + np.arange(steps, dtype=float) * dx
            if ndp is not None:
                xi = np.round(xi, ndp)

            yi_val = f(xi)
            ys: List[Optional[float]] = []
            for v in np.array(yi_val, dtype=np.complex128):
                if np.isfinite(v.real) and abs(v.imag) < 1e-12:
                    ys.append(float(v.real))
                else:
                    ys.append(None)

            points = _to_points_raw_float(
                xi, ys, xmin, xmax, ymin, ymax,
                ndp=ndp, fixed_str=fixed_str
            )

            payload = {
                "meta": {
                    "expr": expr_text,
                    "xmin": xmin, "xmax": xmax,
                    "ymin": ymin, "ymax": ymax,
                    "normalized": False,
                    "dx": dx,
                    "mode": "float_points",
                    "ndp": ndp,
                    "fixed_str": fixed_str
                },
                "points": points
            }

            if do_print:
                print("\n=== POINTS (float) len={} ===".format(len(points)))
                print(json.dumps(points, indent=2))
                print("=== END POINTS ===\n")

            if do_send:
                # If Arduino expects text with fixed decimals, call with fixed_str=1 & ndp set.
                send_status = _send_cells_to_arduino(points)
                payload["send_status"] = send_status

            return Response(
                json.dumps(payload, indent=2) if pretty else json.dumps(payload),
                mimetype="application/json"
            )

        else:
            # -------- INTEGER CELL MODE (legacy) --------
            xi = np.arange(int(math.ceil(xmin)), int(math.floor(xmax)) + 1, dtype=float)

            yi_val = f(xi)
            ys: List[Optional[float]] = []
            for v in np.array(yi_val, dtype=np.complex128):
                if np.isfinite(v.real) and abs(v.imag) < 1e-12:
                    ys.append(float(v.real))
                else:
                    ys.append(None)

            cells_raw = _to_cells_raw(
                xi, ys,
                int(round(xmin)), int(round(xmax)),
                int(round(ymin)), int(round(ymax))
            )

            payload = {
                "meta": {
                    "expr": expr_text,
                    "xmin": int(round(xmin)), "xmax": int(round(xmax)),
                    "ymin": int(round(ymin)), "ymax": int(round(ymax)),
                    "normalized": False,
                    "dx": dx,
                    "mode": "int_cells"
                },
                "cells": cells_raw
            }

            if do_print:
                print("\n=== CELLS (raw int) len={} ===".format(len(cells_raw)))
                print(json.dumps(cells_raw, indent=2))
                print("=== END CELLS ===\n")

            if do_send:
                send_status = _send_cells_to_arduino(cells_raw)
                payload["send_status"] = send_status

            return Response(
                json.dumps(payload, indent=2) if pretty else json.dumps(payload),
                mimetype="application/json"
            )

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
# Helpers used by combo routes
# -----------------------------------------------------------------------------
def _compute_cells_centered(expr_text: str, xmin: float, xmax: float,
                            ymin: float, ymax: float, N: int = GRID_N
                           ) -> List[Tuple[int, int]]:
    """Return centered grid cells for the given expression and window."""
    expr = parse_expression(expr_text)
    f = lambdify(x, expr, modules=["numpy"])
    xs = np.linspace(xmin, xmax, 1200)
    ys_val = f(xs)

    ys: List[Optional[float]] = []
    for v in np.array(ys_val, dtype=np.complex128):
        if np.isfinite(v.real) and abs(v.imag) < 1e-12:
            ys.append(float(v.real))
        else:
            ys.append(None)

    cells_centered = _to_centered_cells_N(xs, ys, xmin, xmax, ymin, ymax, N=N, skip_axes=False)
    return cells_centered

def _send_cells_to_arduino(cells: List[Tuple[int,int]]) -> str:
    """Send the given cells to Arduino. Returns a status string."""
    port = os.getenv("ARDUINO_PORT", "/dev/tty.usbserial-130")
    baud = int(os.getenv("ARDUINO_BAUD", "9600"))
    if not port:
        return "ARDUINO_PORT not set"
    if serial is None:
        return "pyserial not available"
    try:
        import time
        with serial.Serial(port, baudrate=baud, timeout=10) as ser:
            time.sleep(2)                 # allow Arduino auto-reset
            ser.reset_input_buffer()      # clear boot noise
            line = str(cells) + "\n"      # one-line payload
            ser.write(line.encode("utf-8"))
            # Optional: read a few seconds of responses for debugging
            t0 = time.time()
            while time.time() - t0 < 3:
                rx = ser.readline().decode(errors="ignore").strip()
                if rx:
                    print("Arduino:", rx)
        return f"sent {len(cells)} cells to {port}@{baud}"
    except Exception as e:
        return f"send failed: {e}"

def _generate_recommendations(expr_text: str, xmin: float, xmax: float, title: Optional[str] = None
                             ) -> List[Dict[str, object]]:
    """Try OpenAI-based recommendations; fall back to local suggestions."""
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
        clean: List[Dict[str, object]] = []
        for s in suggestions:
            clean.append({
                "label": str(s.get("label", "Suggestion")),
                "expr": str(s.get("expr", "sin(x)")),
                "xmin": int(s.get("xmin", DATA_XMIN_DEFAULT)),
                "xmax": int(s.get("xmax", DATA_XMAX_DEFAULT)),
                "why":  str(s.get("why", "Looks interesting"))
            })
        if clean:
            return clean
    except Exception:
        pass
    # Local fallback
    return _fallback_suggestions(expr_text, xmin, xmax)

# -----------------------------------------------------------------------------
# Combo routes
# -----------------------------------------------------------------------------
@app.post("/recommend-then-send")
def recommend_then_send():
    body = request.get_json(force=True, silent=True) or {}
    expr_text = str(body.get("expr", "sin(x)")).strip()
    xmin = float(body.get("xmin", DATA_XMIN_DEFAULT))
    xmax = float(body.get("xmax", DATA_XMAX_DEFAULT))
    # FIX: correct defaults for y-range
    ymin = float(body.get("ymin", DATA_YMIN_DEFAULT))
    ymax = float(body.get("ymax", DATA_YMAX_DEFAULT))
    title = (body.get("title") or "").strip()
    pick_idx = int(body.get("pick_idx", 0))  # which suggestion to send

    try:
        # Step 1: recommend
        suggestions = _generate_recommendations(expr_text, xmin, xmax, title)

        # Step 2: choose one and send to Arduino
        if not suggestions:
            return jsonify({
                "flow": "recommend_then_send",
                "base_expr": expr_text,
                "suggestions": [],
                "sent": None,
                "send_status": "no suggestions to send"
            }), 200

        sel = suggestions[max(0, min(pick_idx, len(suggestions)-1))]
        sel_expr = sel["expr"]
        sel_xmin = float(sel["xmin"])
        sel_xmax = float(sel["xmax"])

        cells = _compute_cells_centered(sel_expr, sel_xmin, sel_xmax, ymin, ymax, N=GRID_N)
        send_status = _send_cells_to_arduino(cells)

        return jsonify({
            "flow": "recommend_then_send",
            "base_expr": expr_text,
            "suggestions": suggestions,
            "sent": {
                "expr": sel_expr, "xmin": sel_xmin, "xmax": sel_xmax,
                "ymin": ymin, "ymax": ymax, "gridN": GRID_N,
                "cell_count": len(cells), "status": send_status
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.post("/send-then-recommend")
def send_then_recommend():
    body = request.get_json(force=True, silent=True) or {}
    expr_text = str(body.get("expr", "sin(x)")).strip()
    xmin = float(body.get("xmin", DATA_XMIN_DEFAULT))
    xmax = float(body.get("xmax", DATA_XMAX_DEFAULT))
    # FIX: correct default for ymax
    ymin = float(body.get("ymin", DATA_YMIN_DEFAULT))
    ymax = float(body.get("ymax", DATA_YMAX_DEFAULT))
    title = (body.get("title") or "").strip()

    try:
        # Step 1: compute cells and send for the CURRENT expr
        cells = _compute_cells_centered(expr_text, xmin, xmax, ymin, ymax, N=GRID_N)
        send_status = _send_cells_to_arduino(cells)

        # Step 2: produce recommendations for "what to try next"
        suggestions = _generate_recommendations(expr_text, xmin, xmax, title)

        return jsonify({
            "flow": "send_then_recommend",
            "sent": {
                "expr": expr_text, "xmin": xmin, "xmax": xmax,
                "ymin": ymin, "ymax": ymax, "gridN": GRID_N,
                "cell_count": len(cells), "status": send_status
            },
            "suggestions": suggestions
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # In production, consider: app.run(host="0.0.0.0", port=8000)
    app.run(debug=False)
