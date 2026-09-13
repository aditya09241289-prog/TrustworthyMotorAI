def explain_sample(x):
    ref={'vibration':1.0,'temperature':40.0,'current':10.0,'rpm':1500.0}
    rows=[{'feature':k,'value':float(v),'relative_deviation':round(abs(float(v)-ref[k])/abs(ref[k]),3)} for k,v in x.items()]
    return sorted(rows,key=lambda z:z['relative_deviation'],reverse=True)
