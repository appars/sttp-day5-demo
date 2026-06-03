# 🏥 STTP Day 5 — Healthcare AI Deployment Pipeline
### AI-Driven Medical Computer Vision: Generative and Agentic AI for Healthcare Applications
**CoE — Image Processing and Video Analytics · CMRIT Bengaluru**

> **Session:** Day 5 · 5th June 2026 · 6–8 PM · Online via Google Meet
> **Speaker:** Prof. Apparsamy Perumal · Professor of Practice, CSE

---

## 🎯 What You Will Build

A **complete, production-grade healthcare AI deployment pipeline** — from a trained
chest X-ray model all the way to a GitOps-managed Kubernetes deployment:

```
Day 2 Model (ResNet18 CNN)
       ↓
Day 4 Grad-CAM Explainability
       ↓
ONNX Export  →  FastAPI REST API  →  Docker Container
       ↓
Docker Hub Registry
       ↓
ArgoCD (GitOps)  →  Kubernetes (Rancher Desktop k3s)
       ↓
Live Clinical API  ✅
```

---

## 🔗 How Every Day Connects

| Day | Speaker | Topic | Connection to Day 5 |
|-----|---------|-------|---------------------|
| Day 1 | Prof. Krishna Sowjanya K | Medical Imaging Foundations | The NIH chest X-ray **dataset** |
| Day 2 | Prof. Abdul Ashiq O K | CNNs & Transfer Learning | The **ResNet18 model** we deploy |
| Day 3 | Prof. Varun Khare | Annotation & Video Analytics | The **labelled data** that trained it |
| Day 4 | Dr. Sumathi D | XAI + Generative AI + Ethics | **Grad-CAM** + HIPAA audit trail |
| **Day 5** | **Prof. Apparsamy Perumal** | **Deployment + Agentic AI** | **This repo** ← |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your MacBook Pro                      │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  Jupyter    │    │   FastAPI    │    │   Docker   │  │
│  │  Notebook   │───▶│   main.py    │───▶│  Container │  │
│  │  (VS Code)  │    │  /predict    │    │ xray-api   │  │
│  └─────────────┘    └──────────────┘    └────────────┘  │
│         │                                      │         │
│         ▼                                      ▼         │
│  ┌─────────────┐                    ┌──────────────────┐ │
│  │ xray_model  │                    │  Docker Hub      │ │
│  │   .onnx     │                    │ appars/xray-api  │ │
│  │  (43 MB)    │                    └──────────────────┘ │
│  └─────────────┘                             │           │
│                                              ▼           │
│  ┌───────────────────────────────────────────────────┐   │
│  │         Rancher Desktop (k3s cluster)             │   │
│  │                                                   │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────────┐  │   │
│  │  │ ArgoCD   │──▶│  Deploy  │──▶│  Pod 1       │  │   │
│  │  │ (GitOps) │   │  ment    │   │  Pod 2       │  │   │
│  │  │ :8080    │   │          │   │  Pod 3       │  │   │
│  │  └──────────┘   └──────────┘   └──────────────┘  │   │
│  │       ▲              NodePort → localhost:32285   │   │
│  └───────┼───────────────────────────────────────────┘   │
│          │                                               │
└──────────┼───────────────────────────────────────────────┘
           │
    ┌──────┴──────┐
    │   GitHub    │
    │  (Git repo) │
    │  k8s/       │
    └─────────────┘
```

---

## 📋 Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10 or 3.11 | [python.org/downloads](https://python.org/downloads) |
| Git | any | `xcode-select --install` (Mac) |
| VS Code | latest | [code.visualstudio.com](https://code.visualstudio.com) |
| Rancher Desktop | latest | [rancherdesktop.io](https://rancherdesktop.io) |
| Docker Hub account | free | [hub.docker.com](https://hub.docker.com) |
| GitHub account | free | [github.com](https://github.com) |

**VS Code Extensions required:**
- Python (by Microsoft)
- Jupyter (by Microsoft)

---

## ⚡ Student Quick Start (Clone & Run)

If you are a **workshop student** cloning this repo, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/appars/sttp-day5-demo.git
cd sttp-day5-demo

# 2. Create and activate virtual environment
python3 -m venv sttp-demo
source sttp-demo/bin/activate          # Mac/Linux
# sttp-demo\Scripts\activate           # Windows

# 3. Install all packages
pip install --upgrade pip
pip install \
  torch torchvision \
  onnx onnxruntime onnxscript \
  fastapi "uvicorn[standard]" \
  pillow numpy matplotlib \
  requests python-multipart \
  ipykernel jupyter

# 4. Register Jupyter kernel
python -m ipykernel install --user \
  --name=sttp-demo \
  --display-name="STTP Day5 Demo (Python 3)"

# 5. Open in VS Code
code .
```

Then open `Day5_Demo_Pipeline_v2.ipynb` → select kernel **STTP Day5 Demo** → run cells.

---

## 📁 Repository Structure

```
sttp-day5-demo/
│
├── Day5_Demo_Pipeline_v2.ipynb   ← Main demo notebook (run this)
├── README.md                     ← This file
├── .gitignore
│
├── k8s/                          ← Kubernetes manifests (ArgoCD reads these)
│   ├── deployment.yaml           ← 2 replicas, readiness probe
│   └── service.yaml              ← NodePort service
│
└── sttp-demo/                    ← Virtual environment (not in Git)
```

**Files generated at runtime** (by notebook — not in Git):
```
main.py              ← FastAPI application
Dockerfile           ← Container build instructions
requirements.txt     ← Python dependencies
xray_model.onnx      ← Exported ONNX model (43 MB)
sample_xray.png      ← Downloaded chest X-ray
gradcam_result.png   ← Grad-CAM explainability output
audit.log            ← HIPAA-compliant inference log
```

---

## 🚀 Complete Setup Guide (Step by Step)

### PHASE 1 — Project Setup

```bash
# Create project folder
cd ~
mkdir sttp-day5-demo
cd sttp-day5-demo

# Initialise Git
git init
git config --global user.name "Your Name"
git config --global user.email "your.email@institution.ac.in"
```

### PHASE 2 — Python Environment

```bash
# Create virtual environment
python3 -m venv sttp-demo
source sttp-demo/bin/activate

# Confirm it is active
which python
# Should show: .../sttp-day5-demo/sttp-demo/bin/python

# Install all packages (includes onnxscript for PyTorch 2.4+)
pip install --upgrade pip
pip install \
  torch torchvision \
  onnx onnxruntime onnxscript \
  fastapi "uvicorn[standard]" \
  pillow numpy matplotlib \
  requests python-multipart \
  ipykernel jupyter

# Verify
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import onnxruntime; print('ONNX Runtime: OK')"
python -c "import fastapi; print('FastAPI: OK')"
python -c "import onnxscript; print('onnxscript: OK')"
```

### PHASE 3 — Run the Notebook

Open VS Code → select **STTP Day5 Demo** kernel → run cells **one by one**:

| Cell | Step | Expected Output |
|------|------|-----------------|
| 1 | Install packages | `✅ All packages installed!` |
| 2 | Imports | `✅ Imports ready` |
| 3 | Load ResNet18 + X-ray | X-ray image appears |
| 4 | Grad-CAM | 3-panel heatmap appears |
| 5 | ONNX export + merge | `✅ Exported, merged & validated → xray_model.onnx (43.x MB)` |
| 6 | Write main.py | `✅ main.py written` |
| 7 | Test API (after uvicorn) | JSON prediction response |
| 8 | Write Dockerfile | `✅ Dockerfile written` |
| 9 | Audit log | Log entries displayed |

> ⚠️ **Important:** Do NOT use "Run All". Cell 7 (API test) requires
> uvicorn to be running first — see Phase 4 below.

### PHASE 4 — Test FastAPI

```bash
# Open a NEW terminal tab — keep this running
cd ~/sttp-day5-demo
source sttp-demo/bin/activate
uvicorn main:app --reload
```

Wait for:
```
INFO:     Application startup complete.
```

Then open Chrome → **http://localhost:8000/docs**

Test it:
- Click **POST /predict** → Try it out
- Choose File → select `sample_xray.png`
- Click **Execute**
- See JSON prediction with confidence score ✅

```bash
# Or test via curl
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_xray.png;type=image/png"
```

### PHASE 5 — Docker Build & Run

```bash
# Build image (5-10 min first time — normal)
cd ~/sttp-day5-demo
docker build -t xray-api:v1 .

# Run container
docker run -p 8000:8000 xray-api:v1

# Test
curl http://localhost:8000/health
# {"status":"ok","model":"xray_model.onnx"}
```

### PHASE 6 — Push to Docker Hub

```bash
# Login
docker login
# Enter your Docker Hub username and password

# Tag with your username
docker tag xray-api:v1 YOUR_DOCKERHUB_USERNAME/xray-api:v1

# Push
docker push YOUR_DOCKERHUB_USERNAME/xray-api:v1

# Make repository PUBLIC on hub.docker.com
# (Repository → Settings → Make Public)
```

### PHASE 7 — Update k8s Manifests

```bash
# Update the image name in deployment.yaml
# Replace YOUR_DOCKERHUB_USERNAME with your actual username
sed -i '' 's|appars/xray-api:v1|YOUR_DOCKERHUB_USERNAME/xray-api:v1|g' \
  k8s/deployment.yaml

# Commit and push to GitHub
git add k8s/
git commit -m "feat: add k8s manifests for ArgoCD deployment"
git push origin main
```

### PHASE 8 — Install ArgoCD on Rancher Desktop

```bash
# Confirm k3s is running
kubectl get nodes
# Should show: rancher-desktop   Ready

# Create ArgoCD namespace
kubectl create namespace argocd

# Install ArgoCD
kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for all pods to be Running (2-3 minutes)
kubectl get pods -n argocd -w
# Press Ctrl+C when all show Running

# Get admin password (copy this!)
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d && echo

# Port-forward ArgoCD UI (keep this terminal open)
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

Open Chrome → **https://localhost:8080**
- Username: `admin`
- Password: (copied above)

### PHASE 9 — Create ArgoCD Application

```bash
# Install ArgoCD CLI
brew install argocd

# Login
argocd login localhost:8080 \
  --username admin \
  --password YOUR_ARGOCD_PASSWORD \
  --insecure

# Create application pointing to YOUR GitHub repo
argocd app create xray-api \
  --repo https://github.com/YOUR_GITHUB_USERNAME/sttp-day5-demo \
  --path k8s \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace default \
  --sync-policy automated \
  --self-heal

# Check status
argocd app get xray-api
# Should show: Health: Healthy, Sync: Synced
```

### PHASE 10 — Verify Full Pipeline

```bash
# Check pods running
kubectl get pods
# NAME                        READY   STATUS    RESTARTS
# xray-api-xxxx-yyyy          1/1     Running   0
# xray-api-xxxx-zzzz          1/1     Running   0

# Get the NodePort
NODE_PORT=$(kubectl get svc xray-api-svc \
  -o jsonpath='{.spec.ports[0].nodePort}')
echo "API running at: http://localhost:$NODE_PORT"

# Test via NodePort
curl http://localhost:$NODE_PORT/health
# {"status":"ok","model":"xray_model.onnx"}

# Or port-forward directly
kubectl port-forward svc/xray-api-svc 9000:8000 &
curl http://localhost:9000/health
```

Open Chrome → **http://localhost:NODE_PORT/docs** → full Swagger UI via Kubernetes! ✅

---

## 🎯 GitOps Live Demo (Friday Money Shot)

Show the audience live auto-deployment — **this is the most impressive moment:**

```bash
# Step 1 — Scale from 2 to 3 replicas (one line change in Git)
sed -i '' 's/replicas: 2/replicas: 3/' k8s/deployment.yaml

# Step 2 — Commit and push
git add k8s/deployment.yaml
git commit -m "scale: increase to 3 replicas for peak load"
git push origin main

# Step 3 — Switch to ArgoCD UI at https://localhost:8080
# Watch ArgoCD detect the change and spin up a 3rd pod automatically!
```

**What to say to the audience:**

> *"I changed one line in a YAML file and pushed to Git.
> That's the only action I took. Now watch ArgoCD — it detects
> the change, syncs the cluster to match Git, and Kubernetes
> spins up a third pod automatically. No SSH. No manual deploy.
> Git is the single source of truth. This is GitOps."*

---

## 🔑 Key Commands Reference

```bash
# ── Environment ──────────────────────────────────────────────
source sttp-demo/bin/activate        # Activate virtual env
deactivate                            # Deactivate

# ── FastAPI ──────────────────────────────────────────────────
uvicorn main:app --reload             # Start API server
uvicorn main:app --host 0.0.0.0 --port 8000  # Production start

# ── Docker ───────────────────────────────────────────────────
docker build -t xray-api:v1 .        # Build image
docker run -p 8000:8000 xray-api:v1  # Run container
docker images                         # List images
docker ps                             # List running containers
docker push appars/xray-api:v1       # Push to Docker Hub

# ── Kubernetes ───────────────────────────────────────────────
kubectl get pods                      # List pods
kubectl get svc                       # List services
kubectl get nodes                     # Confirm cluster running
kubectl describe pod POD_NAME         # Debug a pod
kubectl logs -l app=xray-api          # View pod logs
kubectl scale deployment xray-api --replicas=3  # Scale up

# ── ArgoCD ───────────────────────────────────────────────────
argocd app get xray-api               # App status
argocd app sync xray-api              # Manual sync
argocd app history xray-api           # Deployment history

# ── Port Forwards (keep in separate terminals) ────────────────
kubectl port-forward svc/argocd-server -n argocd 8080:443
kubectl port-forward svc/xray-api-svc 9000:8000
```

---

## 🎯 Demo Day Checklist (Friday 5:30 PM)

Run through this **before** going live at 6 PM:

```bash
# 1. Start Rancher Desktop — wait for green indicator

# 2. Activate environment
cd ~/sttp-day5-demo
source sttp-demo/bin/activate

# 3. Confirm Docker image exists
docker images appars/xray-api
# Should show: appars/xray-api   v1   ...

# 4. Confirm Kubernetes pods running
kubectl get pods
# Should show 2-3 pods Running

# 5. Confirm ArgoCD running
kubectl get pods -n argocd | grep Running | wc -l
# Should show: 5

# 6. Start port forwards (two separate terminal tabs)
kubectl port-forward svc/argocd-server -n argocd 8080:443 &
kubectl port-forward svc/xray-api-svc 9000:8000 &

# 7. Open in browser — confirm both load
# https://localhost:8080  → ArgoCD (green Healthy + Synced)
# http://localhost:9000/docs → Swagger UI (API ready)

# 8. Open VS Code with notebook outputs already visible
code .

# 9. Open slides in PowerPoint

# 10. Enable Do Not Disturb on Mac
```

---

## 🆘 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: onnxscript` | PyTorch 2.4+ needs it | `pip install onnxscript` |
| ONNX exports two files (.onnx + .onnx.data) | PyTorch 2.4 new exporter | Use v2 notebook — merge cell included |
| Docker can't find xray_model.onnx.data | Split model files | Run merge cell in notebook first |
| `Cannot connect to Docker daemon` | Rancher Desktop not running | Open Rancher Desktop, wait for green |
| Port 8000 already in use | Previous server running | `lsof -ti:8000 \| xargs kill -9` |
| ArgoCD pods stuck Pending | Insufficient resources | `kubectl describe pod -n argocd` |
| Image pull error in pods | Docker Hub image is private | Go to hub.docker.com → make repo Public |
| `kubectl: command not found` | Not in PATH | Rancher Desktop → Preferences → enable kubectl |
| ArgoCD shows OutOfSync | Git change detected | Click Sync in UI or `argocd app sync xray-api` |
| Notebook kernel dead | Kernel crashed | Click Restart Kernel in VS Code top bar |
| Wrong Python in kernel | venv not selected | VS Code top right → Select Kernel → sttp-demo |
| `opset_version 13` warnings | Old opset setting | Use v2 notebook — opset 18 already set |

---

## 📦 Package Versions (tested and working)

```
torch==2.4.x or 2.5.x
torchvision==0.19.x or 0.20.x
onnx==1.16.x
onnxruntime==1.18.x
onnxscript==0.1.x          ← Required for PyTorch 2.4+
fastapi==0.111.x
uvicorn==0.29.x
pillow==10.x
numpy==1.26.x
matplotlib==3.9.x
```

---

## 📚 Resources & Further Reading

| Resource | Link | What for |
|----------|------|----------|
| HuggingFace Hub | huggingface.co | Pre-trained models, datasets, demos |
| MLflow | mlflow.org | Experiment tracking |
| Weights & Biases | wandb.ai | Real-time training dashboards |
| Papers with Code | paperswithcode.com | Benchmarks + reproducible baselines |
| MONAI | monai.io | Medical imaging AI framework |
| LangChain | github.com/langchain-ai/langchain | Agentic AI framework |
| NIH ChestX-ray14 | nihcc.app.box.com/v/ChestXray-NIHCC | Real chest X-ray dataset |
| ArgoCD Docs | argo-cd.readthedocs.io | GitOps tool documentation |
| FastAPI Docs | fastapi.tiangolo.com | API framework documentation |
| ONNX Runtime | onnxruntime.ai | Cross-platform model inference |

---

## 🔮 Next Steps for Students

1. **Fine-tune the model** on the full NIH ChestX-ray14 dataset (50,000 labelled images)
2. **Add `/explain` endpoint** — returns Grad-CAM overlay as an image
3. **Add multi-class support** — 14 disease categories instead of binary
4. **Build an agentic wrapper** — LangChain agent that calls `/predict`, retrieves
   clinical guidelines from a vector DB, and drafts a structured clinical report
5. **Add CI/CD pipeline** — GitHub Actions builds and pushes Docker image on every commit,
   ArgoCD deploys automatically
6. **Federated learning** — train the model across multiple hospitals without
   sharing patient data

---

## 📄 License

This project is created for educational purposes as part of the STTP workshop.
The ResNet18 model uses ImageNet pre-trained weights (see PyTorch model zoo licence).
The sample chest X-ray is from Wikimedia Commons (public domain).

---

*STTP — AI-Driven Medical Computer Vision: Generative and Agentic AI for Healthcare Applications*
*CoE — Image Processing and Video Analytics · CMRIT Bengaluru*
*Prof. Apparsamy Perumal · Day 5 · 5th June 2026*
