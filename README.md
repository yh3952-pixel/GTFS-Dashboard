完美 ✅ 这是最终英文版的 **`README.md`**，你可以直接复制整段内容粘贴到项目根目录保存为 `README.md`：

---

```markdown
# 🗽 GTFS-Dashboard

A lightweight **Streamlit dashboard** for visualizing and analyzing **New York City transportation (MTA GTFS)** data — including subway, bus, LIRR, and MNR.

---

## 🚀 Features
- Real-time GTFS feed visualization (subway, bus, commuter rail)
- Borough-color-coded interactive maps
- Cached data loading for faster updates
- Streamlit-based clean UI with auto-refresh
- Modularized structure: `app_streamlit.py`, `utils_streamlit.py`, `utils.py`

---

## 🧩 Project Structure
```

GTFS-Dashboard/
│
├── app_streamlit.py        # Main Streamlit app
├── utils_streamlit.py      # Streamlit helper functions
├── utils.py                # Data processing utilities
├── requirements.txt        # Python dependencies
├── assets/                 # Icons, map templates, CSS
└── .streamlit/             # Streamlit config

````

*(Local folders like `GTFS/`, `cache/`, `.pem` are excluded via `.gitignore`.)*

---

## ⚙️ Setup & Run

### 1️⃣ Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the dashboard

```bash
streamlit run app_streamlit.py
```

Then open the displayed local URL (usually [http://localhost:8501](http://localhost:8501)).

---

## 📊 Data Source

* **MTA GTFS Realtime Feeds**
  [https://data.ny.gov/Transportation/](https://data.ny.gov/Transportation/)
* Optional: add your API keys in `.streamlit/secrets.toml` (excluded from repo).

---

## 🧠 Notes

* Large raw data (`GTFS/`, `cache/`, `BUS_MAPPING/`, etc.) are **ignored** via `.gitignore`.
  If needed, re-generate locally or provide download scripts.
* SSL certificate files (`.pem`) are **private** and should never be committed.

---

## 👤 Author

**yh3952-pixel**
📧 [yh3952@columbia.edu](mailto:yh3952@columbia.edu)
M.S. Mechanical Engineering – Columbia University (2025)

````

---

✅ 保存后执行：
```bash
git add README.md
git commit -m "add project README"
git push
````

然后刷新你的 GitHub 仓库页面（`https://github.com/yh3952-pixel/GTFS-Dashboard`），
你会看到 README 自动渲染成美观的项目首页。
