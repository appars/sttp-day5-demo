# Day 5 Demo — Complete Setup Guide
## MacBook Pro · Wednesday Step-by-Step

---

## ✅ PHASE 0 — Check Prerequisites (5 min)

Open **Terminal** (Cmd + Space → type Terminal → Enter) and run each check:

```bash
# Check Python (need 3.9 or higher)
python3 --version

# Check Git
git --version

# Check pip
pip3 --version
```

If Python is missing → download from **python.org/downloads** (get 3.11)
If Git is missing → run `xcode-select --install` in terminal

---

## 📁 PHASE 1 — Create Project Folder (2 min)

```bash
# Go to your home directory
cd ~

# Create the project folder
mkdir sttp-day5-demo

# Enter the folder
cd sttp-day5-demo

# Confirm you are inside it
pwd
# Should show: /Users/YOUR_NAME/sttp-day5-demo
```

---

## 📋 PHASE 2 — Copy the Notebook File (2 min)

The notebook was downloaded from Claude.
It is likely in your **Downloads** folder.

```bash
# Copy notebook from Downloads into the project folder
cp ~/Downloads/Day5_Demo_Pipeline.ipynb .

# Confirm it is there
ls -la
# Should show: Day5_Demo_Pipeline.ipynb
```

If you saved it somewhere else, use Finder to locate it,
then drag it into the sttp-day5-demo folder.

---

## 🗂️ PHASE 3 — Create .gitignore (2 min)

Before Git setup, tell Git what NOT to track:

```bash
# Create .gitignore file
cat > .gitignore << 'EOF'
# Python virtual environment
sttp-demo/
__pycache__/
*.pyc
*.pyo

# Jupyter checkpoints
.ipynb_checkpoints/

# Generated files (created by notebook at runtime)
sample_xray.png
gradcam_result.png
xray_model.onnx
audit.log
main.py
Dockerfile
requirements.txt

# Mac system files
.DS_Store
EOF

# Confirm it was created
cat .gitignore
```

---

## 🔧 PHASE 4 — Initialize Git Repository (5 min)

```bash
# Initialize Git in the project folder
git init

# Set your name and email (first time only)
git config --global user.name "Apparsamy Perumal"
git config --global user.email "your.email@cmrit.ac.in"

# Add the notebook to Git
git add Day5_Demo_Pipeline.ipynb .gitignore

# First commit
git commit -m "feat: add Day 5 STTP demo notebook and gitignore"

# Confirm commit
git log --oneline
# Should show: one commit with your message
```

### Optional — Push to GitHub (skip if no GitHub account)

```bash
# 1. Go to github.com → New Repository
#    Name: sttp-day5-demo
#    Visibility: Private
#    Do NOT add README or .gitignore (we already have one)
#    Click Create Repository

# 2. Copy the remote URL shown on GitHub, then run:
git remote add origin https://github.com/YOUR_USERNAME/sttp-day5-demo.git
git branch -M main
git push -u origin main
```

---

## 🐍 PHASE 5 — Create Python Virtual Environment (5 min)

A virtual environment keeps demo packages separate from everything else.
This is the key step to avoid environment errors.

```bash
# Make sure you are in the project folder
cd ~/sttp-day5-demo

# Create virtual environment named sttp-demo
python3 -m venv sttp-demo

# Activate it
source sttp-demo/bin/activate

# Confirm it is active (you should see (sttp-demo) in your prompt)
which python
# Should show: /Users/YOUR_NAME/sttp-day5-demo/sttp-demo/bin/python
```

---

## 📦 PHASE 6 — Install All Packages (15–20 min)

With the virtual environment active, install everything:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all packages needed for the demo
pip install \
  torch torchvision \
  onnx onnxruntime \
  fastapi "uvicorn[standard]" \
  pillow numpy matplotlib \
  requests python-multipart \
  ipykernel jupyter

# This takes 10-15 minutes — normal, PyTorch is large
```

**Verify key packages installed correctly:**

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import onnxruntime; print('ONNX Runtime: OK')"
python -c "import fastapi; print('FastAPI: OK')"
python -c "import torchvision; print('TorchVision: OK')"
```

All four lines should print without errors.

---

## 🔑 PHASE 7 — Register Jupyter Kernel (2 min)

This tells VS Code which Python to use for the notebook:

```bash
# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user \
  --name=sttp-demo \
  --display-name="STTP Day5 Demo (Python 3)"

# Confirm it was registered
jupyter kernelspec list
# Should show sttp-demo in the list
```

---

## 💻 PHASE 8 — Install and Configure VS Code (10 min)

### Install VS Code
Download from **code.visualstudio.com** → install the .dmg → drag to Applications

### Install Extensions
Open VS Code → press `Cmd + Shift + X` → search and install:

- **Python** (by Microsoft) — install
- **Jupyter** (by Microsoft) — install

### Open the Project

```bash
# Open the project folder in VS Code from Terminal
cd ~/sttp-day5-demo
code .
```

### Select the Right Kernel
1. In VS Code, click on `Day5_Demo_Pipeline.ipynb`
2. Top right corner → click **"Select Kernel"**
3. Choose **"STTP Day5 Demo (Python 3)"**
4. If not visible → click "Jupyter Kernel..." → "Python Environments" → pick sttp-demo

### VS Code Font Size (important for demo screen sharing)
- Press `Cmd + ,` to open Settings
- Search "font size"
- Set **Editor: Font Size** to **16**
- This ensures audience can read your code clearly

---

## 🚀 PHASE 9 — Run the Notebook (30 min)

In VS Code with the notebook open:

### Run All Cells
- Press `Cmd + Shift + P` → type "Run All" → select **"Notebook: Run All Cells"**
- Or click **Run All** button at top of notebook

### What to watch for in each step:

**Step 0 (Install):** Should print ✅ for each package
```
  ✅ torch torchvision
  ✅ onnx onnxruntime
  ...
🎉 All packages installed!
```

**Step 0 (Imports):** Should print device and version info
```
  PyTorch  : 2.x.x
  Device   : cpu   ← (mps on Apple Silicon is fine too)
  Labels   : ['Normal', 'Pneumonia']
✅ Imports ready
```

**Step 1 (Load Model):** ResNet18 downloads ImageNet weights (~45 MB, one-time)
```
✅ ResNet18 loaded
✅ X-ray downloaded → sample_xray.png
```
Image of chest X-ray should appear below the cell.

**Step 2 (Grad-CAM):** Three-panel image appears (Original | Heatmap | Overlay)
```
  Prediction   : Normal   ← or Pneumonia (random weights, that's OK)
  Confidence   : xx.x%
✅ Saved → gradcam_result.png
```

**Step 3 (ONNX):** Model exported and verified
```
✅ Exported & validated → xray_model.onnx  (44.7 MB)
  Prediction : Normal
  Latency    : xx.x ms
💡 Same result — ZERO PyTorch. This runs anywhere!
```

**Step 4 (Write main.py):** File created — check left sidebar
```
✅ main.py written
```

**Step 5 (Write Dockerfile):** Files created
```
✅ Dockerfile written
✅ requirements.txt written
📁 Your complete deployment package:
  ✅  main.py          ...
  ✅  xray_model.onnx  ...
```

**Step 6 (Audit Log):** Log entries displayed
```
📋 Contents of audit.log:
  🔴  12:34:56  Pneumonia  conf:0.847 ...
  🟢  12:35:14  Normal     conf:0.923 ...
```

**Step 7 (Summary):** Just a markdown cell — no code to run.

### Commit generated files to Git (optional)
```bash
cd ~/sttp-day5-demo
git add sample_xray.png gradcam_result.png xray_model.onnx \
        main.py Dockerfile requirements.txt audit.log
git commit -m "feat: add generated demo outputs from notebook run"
```

---

## 🌐 PHASE 10 — Test FastAPI (10 min)

The notebook is still open. Now open a **second terminal**:

```bash
# Open a NEW terminal window: Cmd + T (or Cmd + N in Terminal)

# Navigate to project folder and activate environment
cd ~/sttp-day5-demo
source sttp-demo/bin/activate

# Start the FastAPI server
uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

### Test in Browser
Open Chrome/Safari → go to **http://localhost:8000/docs**

You should see the Swagger UI with two endpoints:
- `GET /health`
- `POST /predict`

### Test Health Check
```bash
# Open a THIRD terminal tab (Cmd + T)
cd ~/sttp-day5-demo
source sttp-demo/bin/activate

curl http://localhost:8000/health
# Should return: {"status":"ok","model":"xray_model.onnx"}
```

### Test Prediction via Swagger UI
1. In browser at `localhost:8000/docs`
2. Click **POST /predict**
3. Click **"Try it out"**
4. Click **"Choose File"** → select `sample_xray.png`
5. Click **"Execute"**
6. See JSON response appear with prediction + confidence

### Test Prediction via curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_xray.png;type=image/png"
```

Expected response:
```json
{
  "prediction": "Normal",
  "confidence": 0.6234,
  "probabilities": {"Normal": 0.6234, "Pneumonia": 0.3766},
  "latency_ms": 38.4,
  "timestamp": "2026-06-03T...",
  "model_version": "resnet18-onnx-v1.0",
  "filename": "sample_xray.png"
}
```

Stop the server: press **Ctrl + C** in the uvicorn terminal.

---

## 🐳 PHASE 11 — Test Docker (20 min)

### Start Rancher Desktop
- Open Rancher Desktop from Applications
- Wait for it to fully start (the green indicator in menu bar)
- Confirm Docker is working:
```bash
docker --version
docker ps
# Should show empty table (no error)
```

### Build the Docker Image
```bash
# Make sure you are in the project folder
cd ~/sttp-day5-demo

# Build the image (takes 5-10 minutes FIRST TIME — normal)
docker build -t xray-api:v1 .
```

You will see layers being downloaded and built:
```
[1/3] FROM docker.io/library/python:3.11-slim
[2/3] RUN pip install --no-cache-dir -r requirements.txt
[3/3] COPY xray_model.onnx .
Successfully built xxxxxxxx
Successfully tagged xray-api:v1
```

### Run the Container
```bash
docker run -p 8000:8000 xray-api:v1
```

Should show:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Test the Container
```bash
# In another terminal
curl http://localhost:8000/health
# Should return: {"status":"ok","model":"xray_model.onnx"}
```

### Stop the Container
Press **Ctrl + C** in the docker run terminal.

### Tag for "production" (optional but good to show)
```bash
docker tag xray-api:v1 xray-api:latest
docker images
# Shows both xray-api:v1 and xray-api:latest
```

---

## 🔢 PHASE 12 — kubectl Demo with Rancher Desktop (Optional)

If you want to show Kubernetes (more impressive than plain Docker):

```bash
# Create deployment YAML
cat > deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xray-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: xray-api
  template:
    metadata:
      labels:
        app: xray-api
    spec:
      containers:
      - name: xray-api
        image: xray-api:v1
        imagePullPolicy: Never
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: xray-api-svc
spec:
  selector:
    app: xray-api
  ports:
  - port: 8000
    targetPort: 8000
  type: NodePort
EOF

# Deploy to local k3s cluster
kubectl apply -f deployment.yaml

# Watch pods start (most visual moment)
kubectl get pods -w
# Shows two pods going Running -> Running

# Scale live during demo
kubectl scale deployment xray-api --replicas=3
kubectl get pods
# Now shows 3 pods

# Port forward to test
kubectl port-forward svc/xray-api-svc 8001:8000 &
curl http://localhost:8001/health

# Clean up after demo
kubectl delete -f deployment.yaml
```

Commit the deployment file:
```bash
git add deployment.yaml
git commit -m "feat: add Kubernetes deployment manifest"
```

---

## ✅ PHASE 13 — Final Verification Checklist

Run through this before closing everything:

```bash
cd ~/sttp-day5-demo

# Verify all files exist
ls -lh
```

You should see ALL of these:
```
Day5_Demo_Pipeline.ipynb   ← the notebook
sample_xray.png            ← test X-ray image
gradcam_result.png         ← Grad-CAM output
xray_model.onnx            ← exported model (~44 MB)
main.py                    ← FastAPI application
Dockerfile                 ← container instructions
requirements.txt           ← locked dependencies
audit.log                  ← HIPAA log
deployment.yaml            ← Kubernetes manifest (if created)
sttp-demo/                 ← virtual environment
.gitignore
```

### Git status check
```bash
git log --oneline
git status
# Working tree should be clean
```

### Full pipeline smoke test (do this in one go)
```bash
# 1. Activate environment
source sttp-demo/bin/activate

# 2. Quick Python check
python -c "
import torch, onnxruntime, fastapi
print('PyTorch:', torch.__version__)
print('ONNX Runtime: OK')
print('FastAPI: OK')
print('All good — ready for Friday!')
"

# 3. ONNX inference check
python -c "
import onnxruntime as rt, numpy as np
sess = rt.InferenceSession('xray_model.onnx')
dummy = np.random.randn(1,3,224,224).astype('float32')
out = sess.run(None, {sess.get_inputs()[0].name: dummy})
print('ONNX inference: OK, output shape:', out[0].shape)
"

# 4. Docker image check
docker images xray-api
# Should show xray-api:v1
```

---

## 🎯 Demo Day Quick-Start (Friday 5:30 PM)

Just before going live, run these 4 commands:

```bash
# 1. Go to project
cd ~/sttp-day5-demo

# 2. Activate environment
source sttp-demo/bin/activate

# 3. Open VS Code (notebook already has outputs from Thursday dry run)
code .

# 4. Start Docker container (pre-built, instant start)
docker run -p 8000:8000 xray-api:v1
```

Then open Rancher Desktop, open Chrome at `localhost:8000/docs`, open your slides — you are ready.

---

## 🆘 Troubleshooting

| Problem | Fix |
|---------|-----|
| `python3: command not found` | Install from python.org/downloads |
| `(sttp-demo)` not showing in prompt | Run `source sttp-demo/bin/activate` again |
| `ModuleNotFoundError: torch` | Wrong kernel selected — pick sttp-demo in VS Code |
| ONNX Runtime error on Apple Silicon | Run `pip install onnxruntime` (not onnxruntime-silicon) |
| Docker: `Cannot connect to daemon` | Open Rancher Desktop, wait for green indicator |
| Port 8000 already in use | Run `lsof -ti:8000 | xargs kill -9` |
| Notebook kernel dead | Click "Restart Kernel" in VS Code top bar |
| `git: command not found` | Run `xcode-select --install` |

---

*Prepared for STTP Day 5 — AI-Driven Medical Computer Vision*
*Prof. Apparsamy Perumal · CMRIT Bengaluru · 5th June 2026*
