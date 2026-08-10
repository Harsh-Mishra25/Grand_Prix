# GrandPrix

A track/surface condition classifier — trained to detect **wet**, **dry**, and **damp** conditions from video footage.

## Project Structure

```
GrandPrix/
├── Backend/
│   ├── model/              # Model definition / saved weights
│   ├── classifier.py       # Model architecture / inference logic
│   ├── extract_frames.py   # Extracts frames from raw video clips
│   ├── train.py             # Training loop
│   ├── trend.py             # Trend/analysis logic
│   ├── main.py               # API entry point (FastAPI)
│   └── requirements.txt
├── Data/                    # Extracted frames, organized by class (NOT committed — see .gitignore)
│   ├── damp/
│   ├── dry/
│   └── wet/
├── Frontend/
│   └── index.html
└── RawClips/                 # Raw source videos, organized by class (NOT committed — see .gitignore)
    ├── damp/
    ├── dry/
    └── wet/
```

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd GrandPrix/Backend
```

### 2. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> If you get a `CommandNotFoundException` or execution policy error in PowerShell, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Data

Raw video clips go in `RawClips/<class>/` where `<class>` is one of `wet`, `dry`, `damp`.

Videos are **not tracked in Git** (see `.gitignore`) since GitHub isn't suited for large binary files. Use one of the following instead:
- **Git LFS** — if you want videos versioned alongside the code
- **External storage** (Google Drive, S3, etc.) — with a shared link/download step documented here

## Usage

### 1. Extract frames from raw clips
```bash
python extract_frames.py
```
This reads from `RawClips/` and writes labeled frames to `Data/<class>/`.

### 2. Train the model
```bash
python train.py
```

### 3. Run the API server
```bash
uvicorn main:app --reload
```

## Requirements

See [`requirements.txt`](./requirements.txt). Core dependencies:
- `torch`, `torchvision`, `transformers`, `accelerate` — model & training
- `datasets`, `evaluate`, `scikit-learn` — data handling & metrics
- `opencv-python`, `pillow` — video/image processing
- `fastapi`, `uvicorn`, `python-multipart` — backend API

## Notes
- Update file paths in `extract_frames.py` / `train.py` if you're not running from the `Backend/` directory.
- Add a `config.py` (optional) to centralize class labels, batch size, learning rate, etc. as the project grows.