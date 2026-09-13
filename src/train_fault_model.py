import json,joblib,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
F=['vibration','temperature','current','rpm']
def train_fault_model(csv,models,results):
    df=pd.read_csv(csv); df=df[df.trust_label=='trusted']
    Xtr,Xte,ytr,yte=train_test_split(df[F],df.fault_label,test_size=.25,random_state=42,stratify=df.fault_label)
    m=RandomForestClassifier(n_estimators=300,random_state=42).fit(Xtr,ytr)
    acc=float(accuracy_score(yte,m.predict(Xte)))
    joblib.dump(m,models/'fault_model.joblib')
    json.dump({'accuracy':acc},open(results/'fault_metrics.json','w'),indent=2)
    return {'fault_accuracy':acc}
