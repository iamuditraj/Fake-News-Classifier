# Fake News Classifier

A machine learning project to classify news articles as Fake or Real.

## Features
- **Exploratory Data Analysis**: Visualizing class balance and token distributions.
- **Preprocessing**: Robust text cleaning and stratified dataset splitting.
- **Modeling**: Baseline and transformer-based models for classification.
- **API**: FastAPI backend for serving predictions.

## Performance
Current baseline metrics:
- **Accuracy**: 95.7%
- **Macro F1**: 0.957
- **ROC-AUC**: 0.991

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iamuditraj/Fake-News-Classifier.git
   cd Fake-News-Classifier
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage
- Explore notebooks in `notebooks/`
- Run the FastAPI app: `uvicorn app.main:app --reload`
