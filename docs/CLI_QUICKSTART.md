# 🚀 Quick Start - DNA Pattern Explorer

## Get Started in 3 Minutes

### Step 1: Install Dependencies (1 min)

```bash
cd e:\git\dna
pip install -r requirements.txt
```

### Step 2: Start Web Server (30 seconds)

```bash
# Start the FastAPI web application
python app.py
```

You should see:
```
🚀 Starting DNA Pattern Explorer...
📍 Dashboard: http://localhost:8000
📚 API Docs: http://localhost:8000/api/docs
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ Database initialized
```

### Step 3: Open Dashboard (30 seconds)

Open your browser to: **http://localhost:8000**

You'll see the main dashboard with:
- 🧬 Model Zoo statistics
- 📊 Recent experiments
- 🎨 Theme switcher (Dark/Light)
- 🌐 Language switcher (العربية/English)

---

## 🎯 What You Can Do

### 1. **Upload Models** (Models Page)

Visit: http://localhost:8000/static/models.html

```
1. Click "Upload Model" button
2. Select your .safetensors or .pt file
3. Fill in model metadata (name, type, modality)
4. Click "Upload"
```

**Or download from HuggingFace:**
```
1. Click "Download from HuggingFace"
2. Enter model name (e.g., "huawei-noah/TinyBERT_General_4L_312D")
3. Click "Download"
```

### 2. **Run Pattern Mining** (Experiment Page)

Visit: http://localhost:8000/static/experiment.html

```
1. Select a model from dropdown
2. Configure hyperparameters:
   - Epochs: 100
   - Learning Rate: 1e-4
   - SIREN Layers: 5
   - SIREN Hidden: 256
3. Click "Start Experiment"
4. Watch real-time progress
```

### 3. **Visualize Patterns in 3D** (Visualization Page)

Visit: http://localhost:8000/static/viz.html

```
1. Select a pattern from dropdown
2. Click "Load Visualization"
3. Explore 3D scatter plot:
   - Rotate: Click + Drag
   - Zoom: Scroll wheel
   - Pan: Right-click + Drag
```

**What you'll see:**
- **X/Y/Z axes:** Weight coordinate space
- **Colors:** Values (Viridis colorscale)
- **Interactive:** Plotly.js 3D controls

### 4. **Explore API** (API Documentation)

Visit: http://localhost:8000/api/docs

- **Swagger UI:** Interactive API testing
- **Try it out:** Test endpoints directly
- **Schema:** View request/response models

---

## 📱 Web Interface Features

### Dashboard (/)
- **Stats cards:** Models count, Experiments run, Patterns discovered
- **Recent experiments:** Status, model, timestamps
- **Quick navigation:** Access all pages

### Models Page
- **Model grid:** Card-based display
- **Upload:** Drag-and-drop support
- **Download:** HuggingFace integration
- **Details:** View metadata on click

### Visualization Page
- **3D Plotly:** Interactive scatter plots
- **Theme-aware:** Colors adapt to dark/light
- **Export:** Save as PNG or HTML
- **Responsive:** Works on all screen sizes

---

## 🔧 API Usage (For Developers)

### REST API

All endpoints at: `http://localhost:8000/api/`

**List models:**
```bash
curl http://localhost:8000/api/models
```

**Create experiment:**
```bash
curl -X POST http://localhost:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": 1,
    "name": "TinyBERT Pattern Mining",
    "epochs": 100,
    "learning_rate": 0.0001
  }'
```

### JavaScript API Client

```javascript
<script src="/static/js/api.js"></script>

const models = await API.models.list();
const experiment = await API.experiments.create({...});
const vizData = await API.patterns.getViz(1);
```

---

## 🎨 Customization

### Theme Switcher
Click top-right to toggle:
- 🌙 **Dark Mode** (default)
- ☀️ **Light Mode**

### Language Switcher
- 🇸🇦 **العربية** (RTL layout)
- 🇬🇧 **English** (LTR layout)

---

## 📊 Data Location

```
e:\git\dna\data\
├── dna_explorer.db        ← SQLite database
└── models/               ← Uploaded files
```

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check port
netstat -ano | findstr :8000

# Run on different port (edit app.py)
uvicorn.run("app:app", port=8001)
```

### Reset database
```bash
rm data/dna_explorer.db
python app.py  # Auto-recreates
```

---

## 🎓 Next Steps

1. **Explore Code:** [`app.py`](file:///e:/git/dna/app.py), [`api/`](file:///e:/git/dna/api/)
2. **Add Pages:** Create custom dashboards
3. **Deploy:** Use Docker or production WSGI server

---

**Start mining patterns! 🧬**

```bash
python app.py
# Open: http://localhost:8000
```
