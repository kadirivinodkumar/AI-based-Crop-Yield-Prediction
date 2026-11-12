# 🌾 AI-Based Crop Yield Prediction

An AI-powered agricultural project designed to analyze agricultural data and develop machine learning and deep learning models for crop yield prediction.

The project combines historical agricultural datasets, environmental information, satellite-image data, and predictive modeling components to support data-driven agricultural analysis.

## 📌 Project Overview

Agriculture is influenced by several factors, including rainfall, temperature, soil conditions, seasonal variations, and environmental changes. Predicting crop yield accurately can help farmers and agricultural planners make better decisions.

**AI-Based Crop Yield Prediction** explores the use of machine learning and deep learning techniques to analyze agricultural datasets and estimate crop yield.

The project includes data preprocessing, feature engineering, CNN-based image modeling, LSTM-based modeling, model fusion, and a dashboard interface.

## 🎯 Objectives

- Analyze historical agricultural and environmental datasets.
- Preprocess and prepare data for model training.
- Extract meaningful features from agricultural data.
- Explore CNN-based techniques for satellite-image analysis.
- Explore LSTM-based techniques for sequential data.
- Combine model components through a fusion approach.
- Present project results through a dashboard interface.

## ✨ Key Components

| Component | Description |
|---|---|
| Data Preprocessing | Cleaning and preparing agricultural datasets |
| Feature Engineering | Extracting useful features for prediction |
| CNN Model | Image-based feature learning |
| LSTM Model | Sequential data modeling |
| Fusion Model | Combining model components |
| Dashboard | Interface for interacting with the application |
| Evaluation | Model evaluation and performance analysis |

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Development Environment:** Jupyter Notebook, VS Code
- **Version Control:** Git and GitHub

*The exact packages and versions are listed in `requirements.txt`.*

## 📂 Project Structure

```text
AI-based-Crop-Yield-Prediction/
│
├── dashboard/
│   ├── app.py
│   ├── static/
│   │   ├── css/
│   │   ├── images/
│   │   └── js/
│   └── templates/
│
├── data/
│   ├── raw/
│   └── satellite_images/
│
├── notebooks/
│   └── agr i yield.ipynb
│
├── src/
│   ├── cnn_model.py
│   ├── cross_validation.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── fusion_model.py
│   ├── lstm_model.py
│   ├── sentinel2_downloader.py
│   ├── train.py
│   └── utils.py
│
├── config.py
├── requirements.txt
├── run_pipeline.py
├── pred_out.txt
├── .gitignore
└── README.md
```

## 📊 Dataset

The project contains agricultural and environmental datasets under `data/raw/`, including:

- Crop recommendation data
- Rainfall data
- Temperature data
- Daily and monthly temperature records
- Sunspot data
- Pesticide data
- Crop yield datasets

Satellite-image data is also used as a project input.

**Note:** Large satellite-image `.npy` files are excluded from GitHub to avoid repository size limitations. Keep the required files in `data/satellite_images/` locally before running the relevant image-processing components.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kadirivinodkumar/AI-based-Crop-Yield-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd AI-based-Crop-Yield-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Running the Project

### Run the main pipeline

```bash
python run_pipeline.py
```

### Run the dashboard

```bash
python dashboard/app.py
```

If the dashboard starts successfully, open the local URL displayed in your terminal.

**Important:** Ensure the required datasets, satellite-image files, and dependencies are available before running the corresponding components.

## 🧠 Model Components

### 1. CNN Model
A Convolutional Neural Network component for learning spatial and visual features from image data.

### 2. LSTM Model
A Long Short-Term Memory component for learning patterns in sequential agricultural or environmental data.

### 3. Fusion Model
A model component designed to combine information from multiple model pathways for prediction.

### 4. Model Evaluation
The project includes evaluation and cross-validation modules to assess model performance.

## 🌱 Applications

- Agricultural data analysis
- Crop yield estimation research
- Environmental factor analysis
- Agricultural decision-support systems
- Data-driven farming applications

## 🚀 Future Enhancements

- Integrate real-time weather information.
- Expand satellite-image analysis.
- Improve model accuracy through hyperparameter optimization.
- Add support for more crop varieties and regions.
- Deploy the dashboard as a web application.
- Provide interactive visualizations and prediction reports.

## 👨‍💻 Author

**Vinod Kumar**

GitHub: [@kadirivinodkumar](https://github.com/kadirivinodkumar)

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome. Feel free to fork this repository and submit a pull request.

## 📄 License

This project is intended for educational and research purposes. Add an open-source license if you wish to permit reuse and redistribution.

---

⭐ If you find this project useful, consider giving the repository a star!
