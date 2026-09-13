from pathlib import Path
from src.data_generator import generate_dataset
from src.train_fault_model import train_fault_model
from src.train_trust_model import train_trust_model

ROOT=Path(__file__).resolve().parent
for d in ['data','models','results']: (ROOT/d).mkdir(exist_ok=True)
df=generate_dataset()
csv=ROOT/'data'/'motor_iot_dataset.csv'
df.to_csv(csv,index=False)
print('Dataset created')
print('Training motor fault model...')
print(train_fault_model(csv,ROOT/'models',ROOT/'results'))
print('Training IoT trust model...')
print(train_trust_model(csv,ROOT/'models',ROOT/'results'))
print('DONE - run: streamlit run dashboard.py')
