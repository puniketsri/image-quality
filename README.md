# Image Quality Gate

An automated ML pipeline that scores contractor-submitted images 
across 4 quality dimensions — blur, exposure, crop errors, and 
metadata completeness — and routes them to pass, human review, 
or auto-reject.

Built on real image data from Indiaum Solutions' global 
collection operations (50K+ images across 7 countries).

## Stack
- PyTorch (EfficientNet-B0) — image classification
- OpenCV — classical feature extraction  
- FastAPI — inference API
- PostgreSQL — result logging
- Docker — containerised deployment
- MLflow — experiment tracking

## Project Status
🔵 Phase 1 — Dataset creation (in progress)  
⚪ Phase 2 — OpenCV feature scorers  
⚪ Phase 3 — Model architecture  
⚪ Phase 4 — Training  
⚪ Phase 5 — FastAPI + Docker deployment  

## Running locally
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Dataset
Training data sourced from proprietary contractor image 
submissions. Not included in this repo.