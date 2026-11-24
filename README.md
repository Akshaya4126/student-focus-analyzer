# AI-Based Student Focus Analyzer

A beginner-friendly Flask mini project for reflecting on study sessions. Students can use a browser timer, record distraction and break time, rate their own concentration, and review an estimate from a real Scikit-learn classifier. Session records are stored in SQLite and used for the dashboard and reports.

> **Important:** The classifier was trained on a reproducible synthetic teaching dataset, not on real student outcomes. Its categories and confidence are educational estimates, not objective measurements, diagnosis, or a validated attention assessment. The app does not use a webcam, paid API, login service, or API key.

## Abstract

Students often have difficulty identifying which study habits help them stay engaged. This project provides a lightweight tool for recording study duration, distractions, breaks, and self-reported concentration. A Random Forest model predicts a Low, Medium, or High focus category from those inputs. A Flask backend stores completed sessions in SQLite, while a responsive HTML/CSS/JavaScript interface presents summaries, study history, personalized suggestions, and charts generated from saved records.

## Problem statement

Study effort and focus patterns are difficult to review when sessions, interruptions, and breaks are not recorded consistently. A simple reflection tool can help learners identify patterns and test practical changes without claiming to monitor or measure their minds.

## Objectives

- Build a working local study timer with start, pause, resume, and save controls.
- Record a subject, study goal, planned and actual duration, distractions, breaks, and a 1–10 concentration rating.
- Train and use a real Scikit-learn classifier with a held-out test split.
- Clearly distinguish the student's self-rating from the model's predicted category and class-probability confidence.
- Persist session history in SQLite and generate charts from saved data.
- Provide recommendations that adapt to the predicted category and session details.

## Existing system

A basic stopwatch or spreadsheet can record time but usually does not connect session details to a trained classification model, recommendations, history, and visual summaries in one local application.

## Proposed system

Focusly combines a browser timer, session reflection form, Random Forest focus estimate, dynamically selected study suggestions, and SQLite-backed analytics. It is designed as a mini-project demonstration and does not claim scientific validation.

## Software requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or newer
- Visual Studio Code (recommended) and the Python extension
- A modern browser
- Packages listed in `requirements.txt`

## Hardware requirements

A basic laptop/desktop with 4 GB RAM and a few hundred megabytes of free disk space is sufficient. No GPU, webcam, or internet connection is required after Python packages have been installed. The optional Google Fonts import falls back to system fonts when offline.

## Technologies used

- Python and Flask — routes, JSON API, HTML rendering
- Scikit-learn — Random Forest classifier, train/test split and evaluation
- Pandas and NumPy — dataset and analytics processing
- Matplotlib — charts rendered from actual SQLite data
- SQLite — durable local study history
- HTML, CSS, JavaScript — responsive UI, client-side timer and interactions
- Joblib — saved trained model artifact

## System architecture

```text
Browser (HTML/CSS/JavaScript)
  ├── Study timer + reflection form ── POST /api/sessions ──┐
  ├── Dashboard / history / reports ── Flask page routes     │
  └── Chart images ── /charts/<name>.png                     │
                                                             ▼
Flask application ── Scikit-learn Random Forest ── SQLite study_sessions
                 └── Matplotlib + Pandas ◀───────── saved session records
```

## Project structure

```text
student_focus_analyzer/
├── app.py                  # Flask application, API, pages, SQLite and charts
├── model.py                # Dataset recipe, training, inference and recommendations
├── train_model.py          # Train/retrain command-line entry point
├── dataset.csv             # 900-row reproducible synthetic training dataset
├── model.joblib            # Trained Random Forest (created by training command)
├── model_metrics.json      # Held-out accuracy and feature importance
├── requirements.txt
├── README.md
├── instance/
│   └── student_focus.db    # Created automatically on first run
├── templates/              # Home, dashboard, study, analysis, recommendations, reports, history
├── static/css/style.css
├── static/js/app.js
├── tests/test_app.py
└── screenshots/            # Running-app screenshots
```

## Modules

1. **Home page:** project introduction and links to app sections.
2. **Student dashboard:** total study/distraction/break time, average self-rating, daily/weekly summaries, chart, and recent sessions.
3. **Study timer:** planned duration with 25/45/60-minute presets, start, pause, resume, finish, and save. Timer state is local to the current browser tab; do not close or reload the tab before saving.
4. **Distraction tracker:** record distraction minutes/count, break minutes, and concentration rating from 1 to 10.
5. **Focus analysis:** show the latest model estimate, class confidence, saved input signals, test-set accuracy, and model limitations.
6. **Recommendations:** create next-step suggestions from the saved session's category and values.
7. **Productivity reports:** daily and weekly time, distraction breakdown, model-category comparison, and subject-wise time.
8. **Study history:** review and delete saved session records.

## Machine-learning methodology

`train_model.py` uses a deterministic, seeded generator in `model.py` to create 900 sample rows. Features are `study_minutes`, `distraction_minutes`, `break_minutes`, and `concentration`. The labels are produced by a documented weighted teaching recipe with a small random-noise term, then mapped to Low, Medium, or High focus. This provides reproducible example data for learning the workflow; it is **not** a dataset of actual student assessments.

The script uses a stratified 75%/25% train/test split (`random_state=42`) and fits a Scikit-learn `RandomForestClassifier` with 220 trees. `model_metrics.json` records test-set accuracy, per-class precision/recall/F1, sample counts, and feature importances. On an individual prediction, the highest class probability is shown as **model class confidence**. It is not the user's self-rating and is not a guarantee the prediction is correct. The actual student concentration rating remains a separate input and display value.

Re-train after changing the dataset or the training recipe:

```bash
python train_model.py
```

## Installation and execution

### In VS Code on Windows

1. Install Python 3.10+ and VS Code. In VS Code, install the Microsoft Python extension.
2. Download and unzip the project. In VS Code, select **File → Open Folder…** and open `student_focus_analyzer`.
3. Open **Terminal → New Terminal**. Create and activate a virtual environment:

   ```powershell
   py -3 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt instead with `.venv\Scripts\activate.bat`, or run `.venv\Scripts\python.exe` directly.

4. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Train the model and create its CSV/artifacts (the app can also create missing artifacts on first start):

   ```powershell
   python train_model.py
   ```

6. Start the web app:

   ```powershell
   python app.py
   ```

7. Open **http://127.0.0.1:5000** in a browser. Use the **Study session** page, enter a subject and goal, start the timer, pause/resume if needed, and select **Finish & save**. Complete reflection values before saving. Open Dashboard, Focus analysis, Recommendations, Reports, and Study history to see saved results.
8. Stop the server with **Ctrl+C** in the terminal. Session data remains in `instance/student_focus.db` after restarting.

### macOS/Linux commands

Replace the environment commands with `python3 -m venv .venv` and `source .venv/bin/activate`; then use `python -m pip install -r requirements.txt`, `python train_model.py`, and `python app.py`.

## API notes

- `GET /health` — local health check
- `GET /api/stats` — SQLite dashboard totals as JSON
- `POST /api/predict` — validate features and get an unsaved prediction/recommendations
- `POST /api/sessions` — validate, predict, and save a completed study session
- `GET /charts/daily.png` (also `weekly`, `distractions`, `focus`, `subjects`) — charts from saved session records

## Expected output

The first run opens to an empty, private workspace. No fabricated sessions are inserted. After a student saves a session, the app shows the prediction with confidence, adds the session to SQLite, and updates the dashboard/history/charts. The sample ML dataset is used only for model training and evaluation.

## Advantages

- Clear separation of self-report and model output
- Real Scikit-learn training, test split, metrics, and inference
- Offline-friendly local data storage, no paid APIs or keys
- Simple architecture suitable for explaining in a college demonstration
- Responsive interface with live session timer and real-data visualizations

## Limitations

- Synthetic training labels are not validated against real student focus outcomes; accuracy applies only to the synthetic test split.
- Model confidence is not a guarantee of correctness or a measurement of attention.
- The timer runs in the open browser tab and is not synchronized across devices.
- SQLite is a single-user local database without cloud backup, account support, or multi-user access control.
- Analytics are descriptive and should not be used to compare students or make academic/health decisions.

## Future enhancements

- Collect ethically consented, anonymized, carefully labeled research data and validate the model before making stronger claims.
- Add export/import of sessions, optional local profiles, configurable timer presets, and accessibility preferences.
- Add tests for browser-specific timing and a downloadable report.
- Explore calibration and fairness checks for a properly validated model.

## Testing

Run the included automated checks:

```bash
python -m unittest discover -s tests -v
```

The tests cover page startup, input validation, prediction, SQLite save/delete, analytics endpoints, and chart image generation.

## Conclusion

The AI-Based Student Focus Analyzer demonstrates an end-to-end machine-learning web project while making the limits of synthetic data explicit. It helps students reflect on their study habits and is intended as a learning tool—not a scientific focus monitor.
