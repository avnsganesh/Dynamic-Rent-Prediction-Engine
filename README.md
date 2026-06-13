# UrbanNest Analytics – Dynamic Rent Prediction Engine

> **🚀 Live Streamlit App:** [https://saikrishna888-urbannest-rent-predictor.hf.space](https://saikrishna888-urbannest-rent-predictor.hf.space)
>
> **HF Space:** [huggingface.co/spaces/SaiKrishna888/UrbanNest-Rent-Predictor](https://huggingface.co/spaces/SaiKrishna888/UrbanNest-Rent-Predictor)

A PropTech ML pipeline that predicts monthly rent across **Mumbai, Pune, Delhi, and Hisar** using a Random Forest Regressor optimised with Grid Search, Random Search, and Bayesian Optimization (Optuna), tracked via trackio, served via Streamlit, and containerised with Docker.

---

## Project Structure

```text
Assignment_4/
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── train.ipynb                      ← Task 1: Full ML pipeline (4 sections)
├── app.py                           ← Task 2: Streamlit prediction dashboard
├── Dockerfile                       ← Task 3: Container definition
├── Dataset/
│   ├── train.csv                    ← 11,128 training records
│   └── test.csv                     ← 2,782 test records
├── models/
│   ├── best_rf_model.pkl            ← Trained Random Forest model
│   └── preprocess_artifacts.pkl     ← LabelEncoders + feature metadata
├── plots/
│   ├── trials_vs_error.png          ← Convergence plot (all 3 methods)
│   └── optuna_hyperparameter_space.png  ← Bayesian space exploration
└── screenshots/
    ├── trackio_dashboard_2.png      ← trackio dashboard (3 named runs)
    ├── docker_build_1.png           ← docker build output (part 1)
    ├── docker_build_2.png           ← docker build output (part 2)
    ├── docker_ps.png                ← docker ps output
    └── streamlit_working_1.png      ← app running in browser
```

---

## Implementation

### Task 1 – `train.ipynb` (ML Pipeline)

The notebook follows **4 clearly numbered sections** matching the starter template:

| Section | What it does |
|---------|-------------|
| **§1 Data Loading & Preprocessing** | Loads CSVs, fills NaNs (median / `'Unknown'`), fits `LabelEncoder` on combined train+test values for each categorical column, saves `models/preprocess_artifacts.pkl` |
| **§2 Hyperparameter Tuning** | Initialises `trackio`, then runs all three HPO strategies with 5-fold CV and a 60-trial budget each |
| **§3 Evaluation & Plots** | Generates `plots/trials_vs_error.png` (3-method overlay) and `plots/optuna_hyperparameter_space.png` (Bayesian space scatter) |
| **§4 Final Testing & Saving** | Prints best params from all 3 methods, retrains the winner on full `train.csv`, reports MAE on `test.csv`, saves `models/best_rf_model.pkl` |

**Search spaces used:**

| Method | Space | Trials |
|--------|-------|--------|
| Grid Search (`GridSearchCV`) | `n_estimators`∈{50,100,150,200} × `max_depth`∈{10,15,20,25,30} × `min_samples_split`∈{2,5,8} → **60 combos** | 60 |
| Random Search (`RandomizedSearchCV`) | Integer ranges: 50–200 / 10–30 / 2–10 | 60 |
| Bayesian Optimization (`optuna`) | Same integer ranges via `suggest_int` | 60 |

### Task 2 – `app.py` (Streamlit Dashboard)

- Loads `models/best_rf_model.pkl` + `models/preprocess_artifacts.pkl` with `@st.cache_resource`
- All 14 dataset features mapped to widgets (`st.selectbox` for categoricals, `st.number_input` for numerics)
- On **Predict** click: applies `LabelEncoder.transform()` then calls `model.predict()`
- Displays result via `st.success`

### Task 3 – `Dockerfile`

Uses `python:3.11-slim`, installs dependencies, exposes **port 8501**, and runs:
```
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
```

---

## How to Run Locally (without Docker)

### 1. Set up the virtual environment

```bash
# Create and activate venv
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the training notebook (generates model + plots)

```bash
# Execute all cells and save outputs
jupyter nbconvert --to notebook --execute --inplace train.ipynb --ExecutePreprocessor.timeout=1200

# Or simply open in Jupyter and run all cells
jupyter notebook train.ipynb
```

### 3. Launch the Streamlit app

> [!IMPORTANT]
> You **must** run Streamlit from the **same venv** used to train the model.
> Using a different Python (e.g. Anaconda) causes a scikit-learn version mismatch and crashes on `pickle.load`.

```bash
# Windows (always use the venv's Streamlit binary)
.\venv\Scripts\streamlit run app.py

# macOS / Linux
./venv/bin/streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## How to Open the trackio Dashboard

trackio logs every HPO run automatically when the notebook is executed.

```bash
# Start the trackio dashboard (in any terminal with venv active)
trackio ui

# or if that doesn't work
python -m trackio ui
```

Then open **http://localhost:4321** (default trackio port).  
You will see three logged runs: **Grid Search**, **Random Search**, and **Bayesian Optimization (Optuna)** with their MAE scores, time, iterations, and best hyperparameters.

> **Screenshot:** Take a screenshot of the dashboard and save it as `screenshots/trackio_dashboard.png`.

---

## Docker – Build & Run

### Build the image

```bash
docker build -t urbannest-rent .
```

> **Screenshot:** Capture the build output and save as `screenshots/docker_build.png`.

### Run the container (port-forwarded to localhost:8501)

```bash
docker run -p 8501:8501 urbannest-rent
```

Open **http://localhost:8501** in your browser.

### Verify the container is running

```bash
docker ps
```

Expected output shows `urbannest-rent` with port mapping `0.0.0.0:8501->8501/tcp`.

> **Screenshots:**
> - `docker ps` output → `screenshots/docker_ps.png`
> - App in browser → `screenshots/streamlit_working.png`

### Stop the container

```bash
# Find the container ID from docker ps, then:
docker stop <container_id>
```

---

## Hugging Face Spaces Deployment (Task 4)

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to **New Space** → choose **Docker** as the SDK
3. Push this repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
4. HF Spaces builds the Docker container automatically using the `Dockerfile`
5. Your public URL will be: `https://<your-username>-<space-name>.hf.space`
6. **Paste this URL at the very top of this README**

---

## Dataset

| File | Records | Purpose |
|------|---------|---------|
| `Dataset/train.csv` | 11,128 | HPO tuning + final model training |
| `Dataset/test.csv`  | 2,782  | Final MAE evaluation (held-out) |

**Target variable:** `price` (monthly rent in INR)  
**Categorical features encoded:** `location`, `city`, `Status`, `property_type`

---

## Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| ML Model | `scikit-learn` RandomForestRegressor |
| Grid Search | `sklearn.model_selection.GridSearchCV` |
| Random Search | `sklearn.model_selection.RandomizedSearchCV` |
| Bayesian Opt | `optuna` |
| Experiment Tracking | `trackio` |
| Web App | `streamlit` |
| Containerisation | Docker (`python:3.11-slim`) |
| Cloud Deployment | Hugging Face Spaces (Docker SDK) |
