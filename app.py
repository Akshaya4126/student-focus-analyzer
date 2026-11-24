"""AI-Based Student Focus Analyzer — Flask application."""
from __future__ import annotations

import io
import json
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, url_for

from model import build_recommendations, predict_focus
from train_model import ensure_artifacts

ROOT = Path(__file__).resolve().parent
DB_DEFAULT = ROOT / "instance" / "student_focus.db"
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"


def connect_db(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with connect_db(path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            goal TEXT NOT NULL,
            planned_minutes REAL NOT NULL,
            study_minutes REAL NOT NULL,
            distraction_minutes REAL NOT NULL DEFAULT 0,
            distraction_count INTEGER NOT NULL DEFAULT 0,
            break_minutes REAL NOT NULL DEFAULT 0,
            concentration INTEGER NOT NULL CHECK(concentration BETWEEN 1 AND 10),
            focus_category TEXT NOT NULL,
            model_confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created ON study_sessions(created_at)")


def rows_as_dicts(rows):
    return [dict(row) for row in rows]


def session_recommendations(row):
    return build_recommendations(row["focus_category"], row["study_minutes"],
        row["distraction_minutes"], row["distraction_count"], row["break_minutes"], row["concentration"])


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(SECRET_KEY=os.environ.get("FOCUS_SECRET_KEY", "local-development-key-change-me"), DB_PATH=os.environ.get("FOCUS_DB_PATH", str(DB_DEFAULT)))
    if test_config:
        app.config.update(test_config)
    init_db(app.config["DB_PATH"])
    ensure_artifacts(ROOT / "dataset.csv", MODEL_PATH, METRICS_PATH)
    app.config["FOCUS_MODEL"] = __import__("model").load_model(MODEL_PATH)
    app.config["MODEL_METRICS"] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    def fetch_sessions(limit=None):
        sql = "SELECT * FROM study_sessions ORDER BY created_at DESC, id DESC"
        params = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        with connect_db(app.config["DB_PATH"]) as con:
            return rows_as_dicts(con.execute(sql, params).fetchall())

    def summary():
        today = date.today().isoformat()
        week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        with connect_db(app.config["DB_PATH"]) as con:
            row = con.execute("""SELECT COUNT(*) AS sessions,
                COALESCE(SUM(study_minutes),0) AS study,
                COALESCE(SUM(distraction_minutes),0) AS distractions,
                COALESCE(SUM(break_minutes),0) AS breaks,
                COALESCE(AVG(concentration),0) AS avg_concentration,
                SUM(CASE WHEN substr(created_at,1,10)=? THEN study_minutes ELSE 0 END) AS today_study,
                SUM(CASE WHEN substr(created_at,1,10)>=? THEN study_minutes ELSE 0 END) AS week_study
                FROM study_sessions""", (today, week_start)).fetchone()
        return dict(row)

    @app.context_processor
    def shared_context():
        return {"current_year": date.today().year}

    @app.get("/")
    def home():
        return render_template("index.html", page="home", stats=summary())

    @app.get("/dashboard")
    def dashboard():
        stats = summary()
        recent = fetch_sessions(5)
        return render_template("dashboard.html", page="dashboard", stats=stats, recent=recent)

    @app.get("/study")
    def study():
        return render_template("study.html", page="study")

    @app.get("/analysis")
    def analysis():
        latest = fetch_sessions(1)
        latest_session = latest[0] if latest else None
        explanation = None
        if latest_session:
            explanation = predict_focus(app.config["FOCUS_MODEL"], latest_session["study_minutes"],
                latest_session["distraction_minutes"], latest_session["break_minutes"],
                latest_session["concentration"])["explanation"]
        return render_template("analysis.html", page="analysis", latest=latest_session,
            explanation=explanation, metrics=app.config["MODEL_METRICS"])

    @app.get("/recommendations")
    def recommendations():
        latest = fetch_sessions(1)
        tips = session_recommendations(latest[0]) if latest else []
        return render_template("recommendations.html", page="recommendations", latest=latest[0] if latest else None, tips=tips)

    @app.get("/reports")
    def reports():
        return render_template("reports.html", page="reports", stats=summary())

    @app.get("/history")
    def history():
        sessions = fetch_sessions()
        return render_template("history.html", page="history", sessions=sessions)

    @app.post("/history/<int:session_id>/delete")
    def delete_session(session_id):
        with connect_db(app.config["DB_PATH"]) as con:
            cursor = con.execute("DELETE FROM study_sessions WHERE id=?", (session_id,))
        if cursor.rowcount == 0:
            abort(404)
        flash("Study session deleted.", "success")
        return redirect(url_for("history"))

    @app.get("/api/stats")
    def api_stats():
        return jsonify(summary())

    @app.post("/api/predict")
    def api_predict():
        data = request.get_json(silent=True) or request.form
        try:
            study_m = float(data.get("study_minutes", ""))
            distraction_m = float(data.get("distraction_minutes", 0))
            break_m = float(data.get("break_minutes", 0))
            concentration = int(data.get("concentration", ""))
            distraction_count = int(data.get("distraction_count", 0))
            validate_values(study_m, distraction_m, break_m, concentration, distraction_count)
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        result = predict_focus(app.config["FOCUS_MODEL"], study_m, distraction_m, break_m, concentration)
        result["recommendations"] = build_recommendations(result["category"], study_m, distraction_m, distraction_count, break_m, concentration)
        return jsonify(result)

    @app.post("/api/sessions")
    def api_save_session():
        data = request.get_json(silent=True) or {}
        subject = str(data.get("subject", "")).strip()
        goal = str(data.get("goal", "")).strip()
        try:
            planned = float(data.get("planned_minutes", ""))
            study_m = float(data.get("study_minutes", ""))
            distraction = float(data.get("distraction_minutes", 0))
            breaks = float(data.get("break_minutes", 0))
            concentration = int(data.get("concentration", ""))
            distraction_count = int(data.get("distraction_count", 0))
            validate_values(study_m, distraction, breaks, concentration, distraction_count, planned)
            if not subject or len(subject) > 80:
                raise ValueError("Enter a subject name (up to 80 characters).")
            if not goal or len(goal) > 180:
                raise ValueError("Enter a study goal (up to 180 characters).")
            if study_m <= 0 or study_m > 1440:
                raise ValueError("Study time must be greater than 0 and no more than 1,440 minutes.")
            if distraction > planned + breaks + 0.1:
                raise ValueError("Distraction time cannot exceed the planned study time plus breaks.")
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

        prediction = predict_focus(app.config["FOCUS_MODEL"], study_m, distraction, breaks, concentration)
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        with connect_db(app.config["DB_PATH"]) as con:
            cursor = con.execute("""INSERT INTO study_sessions
                (subject,goal,planned_minutes,study_minutes,distraction_minutes,distraction_count,
                break_minutes,concentration,focus_category,model_confidence,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (subject, goal, planned, study_m, distraction, distraction_count, breaks,
                 concentration, prediction["category"], prediction["confidence"], created_at))
            session_id = cursor.lastrowid
        prediction.update({"id": session_id, "created_at": created_at, "subject": subject, "goal": goal,
            "recommendations": build_recommendations(prediction["category"], study_m, distraction, distraction_count, breaks, concentration)})
        return jsonify(prediction), 201

    @app.get("/charts/<chart_name>.png")
    def chart_image(chart_name):
        allowed = {"daily", "weekly", "distractions", "focus", "subjects"}
        if chart_name not in allowed:
            abort(404)
        data = fetch_sessions()
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        blue, pale, ink, muted = "#3978e8", "#b8d0ff", "#24324b", "#8290a5"
        if not data:
            ax.axis("off")
            ax.text(.5, .55, "Your saved study data will appear here", ha="center", va="center", fontsize=12, color=muted, transform=ax.transAxes)
            ax.text(.5, .42, "Log a session to start building this report", ha="center", va="center", fontsize=9, color=muted, transform=ax.transAxes)
        else:
            df = pd.DataFrame(data)
            df["day"] = pd.to_datetime(df["created_at"]).dt.strftime("%b %d")
            if chart_name == "daily":
                grouped = df.groupby("day", sort=False)["study_minutes"].sum().iloc[::-1].tail(7)
                ax.bar(grouped.index, grouped.values / 60, color=blue, width=.58, zorder=3)
                ax.set_ylabel("Study hours", color=muted)
                ax.set_title("Study time · recent days", loc="left", color=ink, fontsize=12, pad=14)
            elif chart_name == "weekly":
                df["week"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None).dt.to_period("W").astype(str)
                grouped = df.groupby("week")["study_minutes"].sum().tail(6)
                labels = [pd.Period(w).start_time.strftime("%b %d") for w in grouped.index]
                ax.plot(labels, grouped.values / 60, color=blue, marker="o", linewidth=2.5, markersize=6)
                ax.fill_between(range(len(grouped)), grouped.values / 60, color=pale, alpha=.3)
                ax.set_ylabel("Study hours", color=muted)
                ax.set_title("Weekly study time", loc="left", color=ink, fontsize=12, pad=14)
            elif chart_name == "distractions":
                names = ["Distraction", "Break", "Focused study"]
                values = [df.distraction_minutes.sum(), df.break_minutes.sum(), max(0, df.study_minutes.sum()-df.distraction_minutes.sum())]
                ax.barh(names, values, color=["#f2a65a", "#9db8f2", blue], height=.55)
                ax.set_xlabel("Minutes logged", color=muted)
                ax.set_title("Time breakdown", loc="left", color=ink, fontsize=12, pad=14)
            elif chart_name == "focus":
                order = ["Low focus", "Medium focus", "High focus"]
                grouped = df.groupby("focus_category").size().reindex(order, fill_value=0)
                ax.bar(order, grouped.values, color=["#ef9b8f", "#f0c36e", "#58b99a"], width=.58)
                ax.set_ylabel("Sessions", color=muted)
                ax.set_title("Predicted focus categories", loc="left", color=ink, fontsize=12, pad=14)
            else:
                grouped = df.groupby("subject")["study_minutes"].sum().sort_values(ascending=False).head(6).iloc[::-1]
                ax.barh(grouped.index, grouped.values / 60, color=blue, height=.58)
                ax.set_xlabel("Study hours", color=muted)
                ax.set_title("Time by subject", loc="left", color=ink, fontsize=12, pad=14)
            ax.tick_params(colors=muted, labelsize=8)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.grid(axis="x" if chart_name in {"distractions", "subjects"} else "y", color="#edf1f7", linewidth=.8, zorder=0)
            ax.set_axisbelow(True)
            ax.margins(x=.08)
        fig.tight_layout(pad=1.4)
        image = io.BytesIO()
        fig.savefig(image, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        image.seek(0)
        response = send_file(image, mimetype="image/png")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    return app


def validate_values(study_m, distraction_m, break_m, concentration, distraction_count, planned=None):
    values = [study_m, distraction_m, break_m]
    if any(not np.isfinite(value) for value in values):
        raise ValueError("Enter valid numeric session values.")
    if study_m <= 0 or study_m > 1440:
        raise ValueError("Study duration must be greater than 0 and no more than 1,440 minutes.")
    if distraction_m < 0 or distraction_m > 1440:
        raise ValueError("Distraction duration must be between 0 and 1,440 minutes.")
    if break_m < 0 or break_m > 480:
        raise ValueError("Break duration must be between 0 and 480 minutes.")
    if concentration < 1 or concentration > 10:
        raise ValueError("Concentration rating must be from 1 to 10.")
    if distraction_count < 0 or distraction_count > 1000:
        raise ValueError("Distraction count must be between 0 and 1,000.")
    if planned is not None and (not np.isfinite(planned) or planned <= 0 or planned > 1440):
        raise ValueError("Planned duration must be between 1 and 1,440 minutes.")


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
