# Trustworthy AI for IoT-Connected Electric Motors

This project combines electric motor fault classification with AI-based IoT sensor trust analysis.

## Research Goal

The objective is to distinguish between:

- Genuine physical motor faults
- Normal motor operation
- Anomalous or potentially manipulated IoT sensor data

The system independently analyzes:

1. **Motor Fault AI** — classifies motor conditions such as normal operation, bearing faults, and overload faults.
2. **IoT Sensor Trust AI** — determines whether sensor readings appear trustworthy or potentially manipulated.
3. **Trustworthy Decision Layer** — combines both AI outputs to determine whether a detected physical fault should be accepted or investigated further.
4. **Transparent Diagnostic Explanation** — highlights the sensor features that contributed most to the diagnostic result.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard.py