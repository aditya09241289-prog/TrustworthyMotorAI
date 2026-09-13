# Trustworthy Motor AI

A trustworthy artificial intelligence framework for **IoT-based motor fault diagnosis**, combining machine learning, sensor-data analysis, trust evaluation, data integrity checks, and explainable AI.

The project explores how AI-based predictive maintenance systems can become more reliable by considering not only **what the model predicts**, but also **how trustworthy the underlying sensor information and decision are**.

---

## Overview

Industrial motors are commonly monitored using sensor data such as vibration, current, and other operating measurements.

Traditional machine-learning fault diagnosis systems generally focus on classification accuracy. However, real-world IoT environments introduce additional challenges:

- Sensor noise and uncertainty
- Tampered or corrupted measurements
- Unreliable sensor sources
- Model uncertainty
- Difficulty explaining automated decisions

This project addresses these concerns through a **trust-aware motor intelligence pipeline**.

```text
IoT Motor Sensor Data
        ↓
Data Generation / Acquisition
        ↓
Data Processing
        ↓
Fault Diagnosis Model
        ↓
Trust Evaluation
        ↓
Explainability
        ↓
Trustworthy AI Decision
````

---

## Key Features

### Motor Fault Diagnosis

Machine-learning models are used to analyse motor-related sensor data and identify potential operating conditions and faults.

The project supports an end-to-end workflow including:

```text
Data → Training → Evaluation → Prediction
```

### Trust-Aware AI

Instead of relying only on the predicted class, the system incorporates a **trust evaluation layer**.

The goal is to distinguish between:

```text
Prediction
   +
Data / Sensor Trust
   ↓
More Reliable Decision
```

This provides a foundation for safer AI-assisted predictive-maintenance systems.

### Data Integrity / Tampering Analysis

The project includes functionality for generating and analysing modified or tampered sensor data.

This allows the system to explore how changes in sensor information can affect:

* Fault predictions
* Trust estimates
* Model behaviour
* Final decisions

### Explainable AI

The project includes an explainability component to help interpret model predictions rather than treating the AI system as a black box.

The objective is to make predictions easier to inspect and understand.

### Interactive Dashboard

A Streamlit-based dashboard provides an interactive interface for exploring the system.

The dashboard is designed to expose model outputs and trust-related information in a user-friendly way.

---

## System Architecture

```text
                    ┌───────────────────────┐
                    │   IoT Motor Sensors   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Data Processing /     │
                    │ Feature Preparation   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Fault Diagnosis AI  │
                    │                       │
                    │  Motor Classification│
                    └───────────┬───────────┘
                                │
                     ┌──────────┴──────────┐
                     │                     │
                     ▼                     ▼
            ┌─────────────────┐   ┌──────────────────┐
            │ Trust Evaluation│   │ Explainability   │
            └────────┬────────┘   └────────┬─────────┘
                     │                     │
                     └──────────┬──────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Trustworthy AI        │
                    │ Motor Decision        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Interactive Dashboard │
                    └───────────────────────┘
```

---

## Technology Stack

### Machine Learning

* Python
* Scikit-learn
* NumPy
* Pandas

### Dashboard

* Streamlit

### Data / Analysis

* CSV-based sensor datasets
* Jupyter Notebooks
* Matplotlib
* Data preprocessing and evaluation pipelines

### Development

* Git
* GitHub
* Virtual environments
* PowerShell

---

## Project Structure

```text
Trustworthy_IoT_Motor_AI/
│
├── data/
│   └── motor_iot_dataset.csv
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── evaluate_paderborn_model.py
│   ├── explain.py
│   ├── fusion.py
│   ├── generate_tampered_data.py
│   ├── paderborn_loader.py
│   ├── train_fault_model.py
│   ├── train_paderborn_motor.py
│   └── train_trust_model.py
│
├── dashboard.py
├── dashboard_backup.py
├── run_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Main Components

### `data_generator.py`

Responsible for generating or preparing motor-related data used by the project pipeline.

### `train_fault_model.py`

Trains the motor fault-diagnosis model.

### `train_trust_model.py`

Builds the trust-evaluation component used to assess the reliability of the system's information or predictions.

### `generate_tampered_data.py`

Generates modified sensor data for studying the effect of data manipulation and integrity issues.

### `fusion.py`

Combines relevant model or trust outputs into the overall decision-making pipeline.

### `explain.py`

Provides model-explanation functionality to make AI decisions easier to interpret.

### `paderborn_loader.py`

Provides functionality for loading and working with the Paderborn motor-dataset workflow used in the project.

### `evaluate_paderborn_model.py`

Evaluates the motor-diagnosis model using the Paderborn data workflow.

### `run_pipeline.py`

Provides an end-to-end pipeline for running the major processing and modelling stages.

### `dashboard.py`

Launches the interactive Streamlit interface.

---

## Research Motivation

The central motivation of this project is:

> **A highly accurate AI model is not automatically a trustworthy AI model.**

In IoT-based predictive maintenance, the quality and reliability of sensor information can directly influence the final AI decision.

This project therefore investigates a broader pipeline:

```text
Sensor Data
     ↓
Fault Diagnosis
     ↓
Trust Assessment
     ↓
Explainable Decision
     ↓
Trustworthy AI
```

The approach is intended to support AI systems that are not only predictive, but also more transparent and aware of data reliability.

---

## Research Questions

The project provides a foundation for investigating questions such as:

1. Can trust evaluation improve the reliability of AI-based motor fault diagnosis?
2. How does sensor-data tampering affect fault-diagnosis predictions?
3. Can explainable AI make predictive-maintenance decisions easier to interpret?
4. How can trust information be incorporated into an IoT motor-diagnosis pipeline?
5. Can trustworthy AI techniques improve the practical adoption of intelligent predictive-maintenance systems?

---

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/aditya09241289-prog/TrustworthyMotorAI.git
cd TrustworthyMotorAI
```

### 2. Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the Dashboard

```powershell
streamlit run dashboard.py
```

The Streamlit application will open in your browser.

---

## Pipeline

The project can be viewed as the following workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Fault Model Training
   ↓
Fault Prediction
   ↓
Trust Model
   ↓
Tampering / Integrity Analysis
   ↓
Explainability
   ↓
Final Trustworthy Decision
   ↓
Dashboard
```

---

## Dataset and Experimental Workflow

The project contains a local motor IoT dataset and also includes tooling for working with the Paderborn motor dataset workflow.

The repository can therefore support experimentation involving:

* Motor operating conditions
* Fault diagnosis
* Sensor-data manipulation
* Trust evaluation
* Model evaluation
* Explainable AI

Specific experimental results should be reported alongside the corresponding experiment, dataset split, and evaluation protocol rather than treated as universal model performance.

---

## Current Status

The current prototype demonstrates:

* IoT motor data processing
* Motor fault-diagnosis modelling
* Trust-model development
* Sensor-data tampering experiments
* Explainable AI functionality
* Paderborn dataset evaluation workflow
* End-to-end processing pipeline
* Interactive Streamlit dashboard

---

## Future Work

Potential extensions include:

* Real-time IoT sensor streaming
* Edge-device deployment
* Online trust estimation
* Advanced anomaly detection
* Probabilistic uncertainty estimation
* Federated learning for distributed motor monitoring
* Real-world industrial motor validation
* Cloud-based monitoring
* Automated maintenance recommendations
* Integration with industrial IoT platforms

---

## Disclaimer

This repository is a research and educational prototype.

The system should not be used as the sole basis for industrial maintenance, safety-critical decisions, or equipment shutdown without appropriate engineering validation and domain-specific testing.

---

## Author

**Aditya Pawar**
