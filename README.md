## AI Tech Project: Audio Classification

This repository contains code and data for an audio classification project using a custom Naive Bayes classifier and feature extraction from audio files.

### Project Overview
The goal is to classify audio samples into categories (e.g., musical instruments, environmental sounds) using features such as MFCCs, zero-crossing rate, spectral centroid, and bandwidth. The project includes data preprocessing, feature extraction, model training, evaluation, and visualization.

### Environment Setup
1. **Clone the repository** (if not already done):
	```sh
	git clone <repo-url>
	cd <repo-folder>
	```

2. **Install [uv](https://github.com/astral-sh/uv) (if not already installed):**
	```sh
	pip install uv
	```

3. **Sync Python dependencies:**
	```sh
	uv sync
	```

4. **Activate your virtual environment:**
	```sh
	source .venv/bin/activate
	```

### Usage
1. **Open and run the Jupyter notebook:**
	- Launch Jupyter Lab or Notebook:
	  ```sh
	  jupyter lab
	  # or
	  jupyter notebook
	  ```
	- Open `main.ipynb` and run all cells in order.

2. **Data**
    - Download the data from [here](https://www.kaggle.com/datasets/ivanj0/audiodata)
	- Place the `train/` and `val/` directories as structured in the repo.
	- Ensure `train.csv` and `val.csv` are present and correctly formatted.

### Project Structure
- `main.ipynb` — Main notebook for data processing, training, and evaluation
- `naivebayes.py` — Custom Naive Bayes classifier implementation
- `train/`, `val/` — Folders containing audio files for training and validation
- `train.csv`, `val.csv` — Metadata for audio files
- `pyproject.toml` — Project dependencies


