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
            nxt = request.args.get("next") or url_for("index")
            return redirect(nxt)

        flash("Invalid credentials.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

# -----------------------------------------------------------------------------
# Math / Grapher
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

@app.route("/", methods=["GET", "POST"])
def index():
    expr_text = request.form.get("expr", "sin(x)")
    xmin = request.form.get("xmin", "-10")
    xmax = request.form.get("xmax", "10")
    grid = request.form.get("grid", "on")
    title = request.form.get("title", "")

    return render_template(
        "index.html",
        expr=expr_text,
        xmin=xmin,
        xmax=xmax,
        grid=grid,
        title=title,
    )

@app.route("/data")
def data():
    expr_text = request.args.get("expr", "sin(x)")
    xmin = request.args.get("xmin", "-10")
    xmax = request.args.get("xmax", "10")

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


def _map_to_grid(
    xs: np.ndarray,
    ys: List[Optional[float]],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    q: int = 10
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


@app.route("/path")
def path():
    expr_text = request.args.get("expr", "sin(x)")
    xmin = float(request.args.get("xmin", -10))
    xmax = float(request.args.get("xmax", 10))
    ymin = float(request.args.get("ymin", -10))
    ymax = float(request.args.get("ymax", 10))
    q = int(request.args.get("q", 10))
    do_print = request.args.get("print", "0") == "1"
    pretty = request.args.get("pretty", "0") == "1"

    if xmin >= xmax:
        return jsonify({"error": "xmin must be less than xmax."}), 400
    if ymin >= ymax:
        return jsonify({"error": "ymin must be less than ymax."}), 400
    if q <= 0:
        return jsonify({"error": "q must be positive."}), 400

    try:
        expr = parse_expression(expr_text)
        f = lambdify(x, expr, modules=["numpy"])
        xs = np.linspace(xmin, xmax, 1000)
        ys_val = f(xs)

        ys: List[Optional[float]] = []
        for v in np.array(ys_val, dtype=np.complex128):
            if np.isfinite(v.real) and abs(v.imag) < 1e-12:
                yreal = float(v.real)
                ys.append(yreal if (ymin <= yreal <= ymax) else None)
            else:
                ys.append(None)

        gx, gy, cols, rows = _map_to_grid(xs, ys, xmin, xmax, ymin, ymax, q=q)
        path_arr = _rasterize_path(gx, gy)

        payload = {
            "grid": {"cols": cols, "rows": rows, "origin": "center", "q": q},
            "meta": {"expr": expr_text, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
            "path": path_arr
        }

        if do_print:
            print("\n=== PATH ARRAY (len={}): ===".format(len(path_arr)))
            print(json.dumps(path_arr, indent=2))
            print("=== END PATH ARRAY ===\n")

        if pretty:
            return Response(json.dumps(payload, indent=2), mimetype="application/json")
        else:
            return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------------------------------------------------------
# OpenAI Recommendations (robust with fallbacks)
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
    {{"label":"short title","expr":"valid_sympy_expression_in_x","xmin":-10,"xmax":10,"why":"short reason"}}
  ]
}}
4) so for example if user draw y=x, then may be y=x+1 or y=2x etc
No text outside JSON.
""".strip()

def _fallback_suggestions(expr: str, xmin: float, xmax: float) -> List[Dict[str, object]]:
    """Local suggestions if OpenAI fails or is unavailable."""
    ex = (expr or "").replace(" ", "").lower()

    def rng(a, b):
        try:
            return int(a), int(b)
        except Exception:
            return -10, 10

    rxmin, rxmax = rng(xmin, xmax)

    # Trig
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

    # Linear & polynomial heuristics
    if ex in {"x", "+x"} or ex.startswith("1*x") or ex == "x*1":
        return [
            {"label": "Shift right", "expr": "x+1",   "xmin": rxmin, "xmax": rxmax, "why": "Simple translation"},
            {"label": "Scale up",    "expr": "2*x",   "xmin": rxmin, "xmax": rxmax, "why": "Slope change"},
            {"label": "Square",      "expr": "x**2",  "xmin": -5,    "xmax": 5,     "why": "Compare curvature"},
        ]
    if any(tok in ex for tok in ["x**2","x^2"]):
        return [
            {"label": "x^3",   "expr": "x**3",       "xmin": rxmin, "xmax": rxmax, "why": "Odd vs even"},
            {"label": "abs(x)","expr": "Abs(x)",     "xmin": rxmin, "xmax": rxmax, "why": "V-shape contrast"},
            {"label": "bell",  "expr": "exp(-x**2)", "xmin": -4,    "xmax": 4,     "why": "Gaussian shape"},
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

    # Generic
    return [
        {"label": "cos(x)",  "expr": "cos(x)",  "xmin": rxmin, "xmax": rxmax, "why": "Common companion"},
        {"label": "tan(x)",  "expr": "tan(x)",  "xmin": -5,    "xmax": 5,     "why": "Add asymptotes"},
        {"label": "sin(2x)", "expr": "sin(2*x)","xmin": rxmin, "xmax": rxmax, "why": "Frequency change"},
    ]

@app.post("/recommend")
def recommend():
    """
    Body (JSON): { "expr": "sin(x)", "xmin": -10, "xmax": 10, "title": "optional" }
    Returns:     { "suggestions": [ {label, expr, xmin, xmax, why}, ... ] }
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        expr = (body.get("expr") or "").strip()
        xmin = float(body.get("xmin", -10))
        xmax = float(body.get("xmax", 10))
        title = (body.get("title") or "").strip()

        if not expr:
            return jsonify({"suggestions": []})

        # Try OpenAI first
        try:
            from openai import OpenAI
            client = OpenAI(api_key="")
            prompt = _recommend_prompt(expr, xmin, xmax, context_title=title)

            raw = None
            try:
                chat = client.chat.completions.create(
                    model=RECO_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=300,
                    response_format={"type": "json_object"},  # strict JSON if supported
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

            try:
                data = json.loads(raw)
                suggestions = data.get("suggestions", [])[:3]
                clean = []
                for s in suggestions:
                    clean.append({
                        "label": str(s.get("label", "Suggestion")),
                        "expr": str(s.get("expr", "sin(x)")),
                        "xmin": int(s.get("xmin", -10)),
                        "xmax": int(s.get("xmax", 10)),
                        "why":  str(s.get("why", "Looks interesting"))
                    })
                if clean:
                    return jsonify({"suggestions": clean})
            except Exception:
                pass  # fall through to fallback

        except Exception:
            pass  # SDK/API key/network issues → use fallback

        # Fallbacks always return something meaningful
        return jsonify({"suggestions": _fallback_suggestions(expr, xmin, xmax)})

    except Exception as e:
        # Last resort: still provide suggestions
        return jsonify({
            "suggestions": _fallback_suggestions("generic", -10, 10),
            "error": str(e)
        }), 200

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # In production, consider: app.run(host="0.0.0.0", port=8000)
    app.run(debug=True)
