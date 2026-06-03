# ECG Heartbeat Classification with Explainable AI

A machine learning system that automatically classifies ECG heartbeats
as Normal or Abnormal using the MIT-BIH Arrhythmia Database, with
SHAP and LIME explanations for every prediction.

## Project Overview

Cardiovascular disease is one of the leading causes of death worldwide.
Manual ECG interpretation is time consuming and requires specialist
knowledge. This system automates heartbeat classification and provides
transparent explanations for each prediction — making it suitable for
clinical decision support.

## Features

- Loads and processes real ECG recordings from 48 patients
- Segments 102,382 individual heartbeats automatically
- Extracts 13 clinically meaningful features per beat
- Classifies each beat as Normal or Abnormal using Random Forest
- Explains predictions using SHAP and LIME
- Interactive Streamlit web application with file upload support

## Project Structure
ecg_project/
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_segmentation.ipynb
│   ├── 03_eda_features.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_xai_shap_lime.ipynb
├── app/
│   └── streamlit_app.py
├── data/
│   └── mit-bih-arrhythmia-database-1.0.0/
├── models/
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── feature_names.txt
└── README.md

## Results

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Random Forest | 97.26%   | 97.52%    | 92.07% | 94.72%   |
| SVM           | 94.79%   | 87.98%    | 93.24% | 90.54%   |

Random Forest was selected as the final model.

## Dataset

MIT-BIH Arrhythmia Database (PhysioNet)
- 48 patient recordings
- 30 minutes per recording at 360 Hz
- 112,647 annotated heartbeats
- Expert cardiologist annotations

## Installation

1. Clone the repository
```bash
git clone https://github.com/saugatpokharel/ecg-heartbeat-classification-xai.git
cd ecg-heartbeat-classification-xai
```

2. Create conda environment
```bash
conda create -n ecg_project python=3.10
conda activate ecg_project
```

3. Install dependencies
```bash
pip install wfdb numpy pandas matplotlib seaborn scikit-learn
pip install shap lime streamlit jupyter joblib scipy
```

4. Download the dataset
Download the MIT-BIH Arrhythmia Database from:
https://physionet.org/content/mitdb/1.0.0/

Place the files in: data/mit-bih-arrhythmia-database-1.0.0/

## Running the Notebooks

Open Jupyter Notebook and run the notebooks in order:

```bash
conda activate ecg_project
jupyter notebook
```

Run in this order:
1. 01_data_loading.ipynb
2. 02_segmentation.ipynb
3. 03_eda_features.ipynb
4. 04_model_training.ipynb
5. 05_xai_shap_lime.ipynb

## Running the Streamlit App

```bash
conda activate ecg_project
cd ecg_project
streamlit run app/streamlit_app.py
```

Open your browser at http://localhost:8501

## Technologies Used

- Python 3.10
- WFDB — ECG file reading
- NumPy, Pandas — data processing
- Matplotlib, Seaborn — visualisation
- Scikit-learn — machine learning
- SHAP — global and local explanations
- LIME — local explanations
- Streamlit — web application
- Joblib — model persistence
- Jupyter Notebook — development environment

## Author

Saugat Pokharel
BSc Computing(Final Year Project)
Griffith College Cork
Student Number: 3093315
Supervisor: Atif Atif

## Acknowledgements

MIT-BIH Arrhythmia Database:
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
