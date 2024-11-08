# This project was completed under the guidance of the prestigious Institute IIT Kanpur. 
##Additionally, it was developed with the assistance of large language models (LLMs), and their performance was compared to traditional engineering practices to evaluate their effectiveness. 

You can view the project completion certificate [here](https://drive.google.com/file/d/1AjuCPdJ0h783YdzFhsf15YjXaiUEoedc/view).



# Anomaly Detection Using ML and NN

## Introduction

Welcome to the Anomaly Detection project! This repository contains code and resources for detecting anomalies in the CICIDS2017 dataset using machine learning (ML) and neural networks (NN). Anomaly detection is critical in various fields such as fraud detection, network security, health monitoring, and more. This project leverages state-of-the-art techniques to identify unusual patterns that do not conform to expected behavior.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Preprocessing**: Tools for cleaning and preparing CICIDS2017 datasets for anomaly detection.
- **Machine Learning Models**: Implementation of various ML algorithms such as Isolation Forest, One-Class SVM, and others.
- **Neural Network Models**: Advanced NN architectures for deep learning-based anomaly detection.
- **Visualization**: Functions to visualize the results and performance of models.
- **Evaluation Metrics**: Comprehensive evaluation metrics to assess the performance of the models.

## Installation

To get started with this project, clone the repository and install the required dependencies. Ensure you have Python 3.8 or above.

```bash
git clone https://github.com/yourusername/anomaly-detection-ml-nn.git
cd anomaly-detection-ml-nn
pip install -r requirements.txt
```

## Usage

### Data Preparation

The CICIDS2017 dataset must be located in the "CSVs" folder in the same directory as the program. Before running the models, you need to preprocess your data to clean and correct errors.

```bash
python data_preprocessing.py
```

### Training Models

You can train different models using the provided scripts. For instance, to train an Isolation Forest model:

```bash
python train_isolation_forest.py --input data/all_data.csv --output models/isolation_forest.pkl
```

Similarly, to train a neural network model:

```bash
python train_neural_network.py --input data/all_data.csv --output models/neural_network.h5
```

### Evaluation

Evaluate the performance of your trained models using the evaluation scripts. For example:

```bash
python evaluate_model.py --model models/isolation_forest.pkl --input data/all_data.csv
```

## Project Structure

- `data/`: Contains raw and processed datasets.
- `models/`: Directory to save trained models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and prototyping.
- `scripts/`: Python scripts for preprocessing, training, and evaluation.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

## Datasets

The project uses the CICIDS2017 dataset, a comprehensive dataset containing network traffic data for various attack types. The dataset must be preprocessed to handle empty entries and incorrect characters, and to merge all CSV files into a single file for easier processing.

## Models

### Machine Learning Models

- **Isolation Forest**: An ensemble method that isolates observations by randomly selecting a feature and then randomly selecting a split value.
- **One-Class SVM**: A support vector machine algorithm that identifies the boundary of normal data points.

### Neural Network Models

- **Autoencoders**: Neural networks trained to reconstruct input data, which can then be used to detect anomalies based on reconstruction error.
- **LSTM Networks**: Long Short-Term Memory networks suitable for sequential data.

## Results

The performance of the models is evaluated using metrics such as Precision, Recall, F1-Score, and ROC-AUC. Visualization tools are provided to help interpret the results.

## Certification
### Link: https://drive.google.com/file/d/1AjuCPdJ0h783YdzFhsf15YjXaiUEoedc/view
Certificate Granted by Indian Institute of Technology


