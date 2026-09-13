from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Trustworthy IoT Motor AI', page_icon='⚙️', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'results' / 'fusion_results.csv'

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if not CSV.exists():
    st.error('fusion_results.csv was not found.')
    st.code(str(CSV))
    st.stop()

try:
    df = load_data(CSV)
except Exception as e:
    st.error(f'Could not load fusion results: {e}')
    st.stop()

# --------------------------- helpers ---------------------------
def pct(x): return max(0.0, min(1.0, float(x)))
def clean(x): return str(x).replace('_', ' ').title()

def theme(status):
    d = {
        'HIGH_CONFIDENCE': ('#2ee6b8', 'rgba(46,230,184,.12)'),
        'MEDIUM_CONFIDENCE': ('#35d4ff', 'rgba(53,212,255,.12)'),
        'SUSPICIOUS': ('#ffd35a', 'rgba(255,211,90,.12)'),
        'UNTRUSTWORTHY': ('#ff5b78', 'rgba(255,91,120,.12)'),
    }
    return d.get(str(status), ('#a78bfa', 'rgba(167,139,250,.12)'))

def message(decision):
    return {
        'ACCEPT_MOTOR_PREDICTION': 'Motor prediction accepted because the sensor evidence is trustworthy.',
        'REJECT_SENSOR_DATA': 'Sensor evidence is untrustworthy. The motor prediction should not be accepted.',
        'REQUIRE_FURTHER_CHECK': 'The result requires additional verification before a final action.',
        'ACCEPT_WITH_CAUTION': 'Motor prediction is available, but the result should be interpreted with caution.',
    }.get(str(decision), 'The fusion engine generated a final trustworthy AI decision.')

def card(label, value, sub, accent, icon):
    st.markdown(f'''<div class="metric-card"><div class="metric-top"><span>{label}</span><b style="color:{accent}">{icon}</b></div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>''', unsafe_allow_html=True)

def box(label, value, accent='#35d4ff'):
    st.markdown(f'''<div class="info-box"><div class="info-label">{label}</div><div class="info-value" style="color:{accent}">{value}</div></div>''', unsafe_allow_html=True)

def model_item(name, value, sub, score, accent):
    st.markdown(f'''<div class="model-item"><div class="ring" style="border-color:{accent}">{score:.0f}%</div><div><div class="model-name">{name}</div><div class="model-value" style="color:{accent}">{value}</div><div class="model-sub">{sub}</div></div></div>''', unsafe_allow_html=True)

# --------------------------- CSS ---------------------------
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#080b13;--panel:#10141f;--line:#252d3c;--text:#edf2f8;--muted:#78849a}
.stApp{background:radial-gradient(circle at 50% -20%,rgba(53,212,255,.08),transparent 35%),#080b13;color:var(--text);font-family:Manrope,sans-serif}
#MainMenu,footer{visibility:hidden}[data-testid="stHeader"]{background:rgba(8,11,19,.92)}
.block-container{max-width:1550px;padding-top:1.3rem;padding-bottom:2rem}
[data-testid="stSidebar"]{background:#0d111b;border-right:1px solid #202737}
[data-testid="stSidebar"]>div:first-child{padding-top:1rem}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:.25rem 0 1.1rem;border-bottom:1px solid var(--line);margin-bottom:1.1rem}.brand{display:flex;gap:.85rem;align-items:center}.mark{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.35rem;background:linear-gradient(135deg,rgba(53,212,255,.25),rgba(46,230,184,.18));border:1px solid rgba(53,212,255,.35)}.brand-title{font-size:1.3rem;font-weight:800}.brand-sub{font: .58rem 'DM Mono';color:var(--muted);margin-top:.15rem}.live{padding:.45rem .75rem;border:1px solid rgba(46,230,184,.22);background:rgba(46,230,184,.07);border-radius:999px;color:#2ee6b8;font:.62rem 'DM Mono'}.dot{display:inline-block;width:7px;height:7px;background:#2ee6b8;border-radius:50%;margin-right:.4rem;box-shadow:0 0 12px #2ee6b8}
.side-brand{padding:.4rem 0 1.1rem;border-bottom:1px solid var(--line);font-size:1.05rem;font-weight:800}.side-title{color:#5f6b80;font:.58rem 'DM Mono';letter-spacing:.14em;margin:1rem 0 .45rem}.nav{padding:.62rem .7rem;color:#69758a;font-size:.75rem;border-radius:9px;margin:.25rem 0}.nav.active{color:#e2fbff;background:linear-gradient(90deg,rgba(46,230,184,.14),rgba(53,212,255,.06));border:1px solid rgba(46,230,184,.15)}[data-testid="stSidebar"] label{color:#98a4b7!important;font-size:.66rem!important;font-weight:700!important}[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#151a25!important;border:1px solid #293143!important;border-radius:10px!important}
.metric-card,.panel,.model-panel{background:linear-gradient(135deg,rgba(255,255,255,.018),rgba(255,255,255,.004)),var(--panel);border:1px solid var(--line);border-radius:15px;box-shadow:0 12px 32px rgba(0,0,0,.15)}.metric-card{padding:1rem;min-height:120px}.metric-top{display:flex;justify-content:space-between;color:#78849a;font:.58rem 'DM Mono';letter-spacing:.08em;text-transform:uppercase}.metric-value{font-size:1.85rem;font-weight:800;margin-top:.65rem}.metric-sub,.panel-sub,.model-sub{font:.57rem 'DM Mono';color:#667287;margin-top:.25rem}.panel,.model-panel{padding:1rem}.panel-title{font-size:.95rem;font-weight:800;color:#e9eef6}.section{font-size:1.03rem;font-weight:800;margin:1.4rem 0 .7rem}.decision{border-radius:16px;padding:1.1rem 1.2rem;border:1px solid;margin-bottom:1rem}.decision-kicker{font:.58rem 'DM Mono';letter-spacing:.12em}.decision-title{font-size:1.35rem;font-weight:800;margin-top:.3rem}.decision-copy{font-size:.74rem;color:#bac4d3;line-height:1.55;margin-top:.4rem}.info-box{background:#141925;border:1px solid #252d3d;border-radius:11px;padding:.72rem .78rem;margin-bottom:.55rem}.info-label{color:#748096;font:.56rem 'DM Mono';text-transform:uppercase;letter-spacing:.07em}.info-value{font:.73rem 'DM Mono';font-weight:500;margin-top:.35rem;overflow-wrap:anywhere}.confidence-label{font-size:.6rem;color:#7b8799;margin-top:.7rem}.confidence-value{font-size:1.35rem;font-weight:800;margin:.3rem 0 .45rem}[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#2ee6b8,#35d4ff)!important}
.model-item{display:flex;align-items:center;gap:.7rem;padding:.75rem 0;border-bottom:1px solid #202737}.ring{width:46px;height:46px;min-width:46px;border-radius:50%;border:4px solid;display:flex;align-items:center;justify-content:center;background:#0b0f18;font:.58rem 'DM Mono';color:#e3eaf4}.model-name{font-size:.7rem;font-weight:800}.model-value{font:.62rem 'DM Mono';margin-top:.12rem}.architecture{background:radial-gradient(circle at 50% 30%,rgba(53,212,255,.06),transparent 42%),var(--panel);border:1px solid var(--line);border-radius:15px;padding:1.1rem}.arch-row{display:flex;justify-content:center;gap:.8rem;flex-wrap:wrap}.node{min-width:170px;text-align:center;padding:.85rem;border:1px solid #2b3446;background:#131925;border-radius:11px;color:#dce4ee;font-size:.68rem;font-weight:700}.node.a{border-color:rgba(53,212,255,.35)}.arrow{text-align:center;color:#556278;padding:.25rem;font-size:1.1rem}.foot{color:#59657a;font:.57rem 'DM Mono';text-align:center;padding:1.3rem 0 0}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
</style>
''', unsafe_allow_html=True)

# --------------------------- sidebar ---------------------------
conditions = sorted(df['true_condition'].dropna().astype(str).unique())
with st.sidebar:
    st.markdown('<div class="side-brand">⚙ TrustMotor AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-title">OVERVIEW</div><div class="nav active">● Dashboard</div><div class="nav">• Analytics</div><div class="nav">• Model Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-title">SAMPLE EXPLORER</div>', unsafe_allow_html=True)
    cond = st.selectbox('Motor Condition', conditions)
    subset = df[df['true_condition'].astype(str) == cond].reset_index(drop=True)
    idx = st.selectbox('Sample', list(range(len(subset))), format_func=lambda x:f'Sample {x+1:02d}')
    st.markdown('<div class="side-title">SYSTEM</div><div class="nav">• Motor AI</div><div class="nav">• Sensor Trust</div><div class="nav">• Fusion Engine</div>', unsafe_allow_html=True)

r = subset.iloc[idx]
source = str(r['source_file']); true_cond = str(r['true_condition']); motor = str(r['motor_prediction']); mc = pct(r['motor_confidence'])
true_trust = str(r['true_trust_label']); trust = str(r['trust_prediction']); tc = pct(r['trust_confidence'])
status = str(r['final_status']); decision = str(r['final_decision']); accent, soft = theme(status)

# --------------------------- header ---------------------------
st.markdown('''<div class="topbar"><div class="brand"><div class="mark">⚙</div><div><div class="brand-title">AI Performance</div><div class="brand-sub">MOTOR DIAGNOSIS · SENSOR INTEGRITY · FUSION INTELLIGENCE</div></div></div><div class="live"><span class="dot"></span>SYSTEM ONLINE</div></div>''', unsafe_allow_html=True)

trusted_n = int((df['trust_prediction'].astype(str).str.lower()=='trusted').sum())
rejected_n = int((df['final_decision']=='REJECT_SENSOR_DATA').sum())
cols = st.columns(4)
with cols[0]: card('Total Samples', f'{len(df):,}', 'Paderborn dataset processed', '#35d4ff', '◈')
with cols[1]: card('Motor Conditions', df['true_condition'].nunique(), 'Condition classes detected', '#a78bfa', '◉')
with cols[2]: card('Trusted Data', trusted_n, 'Sensor evidence accepted', '#2ee6b8', '◌')
with cols[3]: card('Rejected Data', rejected_n, 'Sensor evidence rejected', '#ff5b78', '◒')

st.markdown('<div class="section">Live AI Analysis</div>', unsafe_allow_html=True)
main, right = st.columns([3.1,1.05], gap='large')
with main:
    st.markdown(f'''<div class="decision" style="background:{soft};border-color:{accent};box-shadow:0 0 34px {soft}"><div class="decision-kicker" style="color:{accent}">● FINAL TRUSTWORTHY AI DECISION</div><div class="decision-title" style="color:{accent}">{clean(status).upper()}</div><div class="decision-copy">{message(decision)}</div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Selected Motor Sample</div><div class="panel-sub">LIVE SAMPLE ANALYSIS · FUSION RESULTS</div>', unsafe_allow_html=True)
    box('Source File', source, '#edf2f8')
    a,b,c = st.columns(3, gap='medium')
    with a:
        st.markdown('<div class="panel"><div class="panel-title">⚙ Motor AI</div><div class="panel-sub">CONDITION CLASSIFICATION</div>', unsafe_allow_html=True)
        box('True Condition', true_cond); box('AI Prediction', motor, '#2ee6b8')
        st.markdown('<div class="confidence-label">PREDICTION CONFIDENCE</div>', unsafe_allow_html=True); st.markdown(f'<div class="confidence-value">{mc*100:.2f}%</div>', unsafe_allow_html=True); st.progress(mc); st.markdown('</div>', unsafe_allow_html=True)
    with b:
        ta = '#2ee6b8' if trust.lower()=='trusted' else '#ff5b78'
        st.markdown('<div class="panel"><div class="panel-title">🛡 Sensor Trust</div><div class="panel-sub">IOT INTEGRITY VERIFICATION</div>', unsafe_allow_html=True)
        box('Actual Sensor Status', true_trust, '#a78bfa'); box('Trust AI Prediction', trust.upper(), ta)
        st.markdown('<div class="confidence-label">TRUST CONFIDENCE</div>', unsafe_allow_html=True); st.markdown(f'<div class="confidence-value">{tc*100:.2f}%</div>', unsafe_allow_html=True); st.progress(tc); st.markdown('</div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="panel"><div class="panel-title">🔗 Fusion Engine</div><div class="panel-sub">MULTI-AI DECISION LAYER</div>', unsafe_allow_html=True)
        box('Final Status', clean(status), accent); box('Final Decision', clean(decision), '#ffd35a'); box('Fusion Insight', message(decision), '#cbd4e1'); st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="model-panel"><div class="panel-title">Model Status</div><div class="panel-sub">AI PIPELINE HEALTH</div>', unsafe_allow_html=True)
    model_item('Motor AI', motor, 'Condition classification', mc*100, '#35d4ff')
    ta = '#a78bfa' if trust.lower()=='trusted' else '#ff5b78'; model_item('Sensor Trust AI', trust.upper(), 'Integrity verification', tc*100, ta)
    fs = ((mc+tc)/2*100) if status!='UNTRUSTWORTHY' else tc*100; model_item('Fusion Engine', clean(status), 'Final trustworthy decision', fs, accent)
    st.markdown(f'<div class="info-box" style="margin-top:.7rem"><div class="info-label">High Confidence Decisions</div><div class="info-value">{int((df["final_status"]=="HIGH_CONFIDENCE").sum())} samples</div></div></div>', unsafe_allow_html=True)

# --------------------------- charts ---------------------------
st.markdown('<div class="section">System Analytics</div>', unsafe_allow_html=True)
cl, cr = st.columns(2, gap='large')
status_colors={'HIGH_CONFIDENCE':'#2ee6b8','MEDIUM_CONFIDENCE':'#35d4ff','SUSPICIOUS':'#ffd35a','UNTRUSTWORTHY':'#ff5b78'}
decision_colors={'ACCEPT_MOTOR_PREDICTION':'#2ee6b8','ACCEPT_WITH_CAUTION':'#ffd35a','REQUIRE_FURTHER_CHECK':'#a78bfa','REJECT_SENSOR_DATA':'#ff5b78'}

def style(fig, h=320):
    fig.update_layout(height=h,paper_bgcolor='#10141f',plot_bgcolor='#10141f',showlegend=False,margin=dict(l=10,r=10,t=10,b=20),font=dict(family='Manrope',color='#aeb8c8'),xaxis=dict(title=None,gridcolor='#10141f'),yaxis=dict(title=None,gridcolor='#202737',zeroline=False))
    return fig
with cl:
    st.markdown('<div class="panel"><div class="panel-title">Final Status Distribution</div><div class="panel-sub">FUSION ENGINE OUTPUT</div>', unsafe_allow_html=True)
    x=df['final_status'].value_counts().reset_index(); x.columns=['status','count']; fig=px.bar(x,x='status',y='count',color='status',color_discrete_map=status_colors,text='count'); fig.update_traces(textposition='outside',marker_line_width=0); st.plotly_chart(style(fig),use_container_width=True,config={'displayModeBar':False}); st.markdown('</div>', unsafe_allow_html=True)
with cr:
    st.markdown('<div class="panel"><div class="panel-title">Final Decision Distribution</div><div class="panel-sub">ACCEPTANCE VS REJECTION INTELLIGENCE</div>', unsafe_allow_html=True)
    x=df['final_decision'].value_counts().reset_index(); x.columns=['decision','count']; fig=px.bar(x,x='decision',y='count',color='decision',color_discrete_map=decision_colors,text='count'); fig.update_traces(textposition='outside',marker_line_width=0); st.plotly_chart(style(fig),use_container_width=True,config={'displayModeBar':False}); st.markdown('</div>', unsafe_allow_html=True)

bl, br = st.columns([1.2,1], gap='large')
with bl:
    st.markdown('<div class="panel"><div class="panel-title">Motor Condition Predictions</div><div class="panel-sub">MODEL CLASS DISTRIBUTION</div>', unsafe_allow_html=True)
    x=df['motor_prediction'].value_counts().reset_index(); x.columns=['condition','count']; fig=go.Figure(go.Bar(x=x['condition'],y=x['count'],marker_color='#35d4ff',text=x['count'],textposition='outside')); st.plotly_chart(style(fig,300),use_container_width=True,config={'displayModeBar':False}); st.markdown('</div>', unsafe_allow_html=True)
with br:
    st.markdown('<div class="panel"><div class="panel-title">Sensor Trust Predictions</div><div class="panel-sub">TRUSTED VS TAMPERED DATA</div>', unsafe_allow_html=True)
    x=df['trust_prediction'].value_counts().reset_index(); x.columns=['trust','count']; fig=px.pie(x,names='trust',values='count',hole=.68,color='trust',color_discrete_map={'trusted':'#2ee6b8','tampered':'#ff5b78'}); fig.update_layout(height=300,paper_bgcolor='#10141f',showlegend=True,margin=dict(l=10,r=10,t=10,b=10),font=dict(family='Manrope',color='#aeb8c8'),annotations=[dict(text=f'<b>{len(df)}</b><br>samples',x=.5,y=.5,font=dict(color='#eef3fa',size=18),showarrow=False)]); fig.update_traces(textinfo='none'); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False}); st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section">Selected Sample Details</div>', unsafe_allow_html=True)
details=pd.DataFrame({'Property':['Source File','True Motor Condition','Motor AI Prediction','Motor Confidence','True Sensor Status','Sensor Trust Prediction','Trust Confidence','Final Status','Final Decision'],'Value':[source,true_cond,motor,f'{mc*100:.2f}%',true_trust,trust,f'{tc*100:.2f}%',status,decision]})
st.dataframe(details,use_container_width=True,hide_index=True)

st.markdown('<div class="section">Trustworthy AI Architecture</div>', unsafe_allow_html=True)
st.markdown('''<div class="architecture"><div class="arch-row"><div class="node a">PADERBORN MOTOR<br>SENSOR DATASET</div></div><div class="arrow">↓</div><div class="arch-row"><div class="node">FEATURE EXTRACTION<br>& SIGNAL ANALYSIS</div></div><div class="arrow">↓</div><div class="arch-row"><div class="node a">⚙ MOTOR CONDITION AI<br>FAULT CLASSIFICATION</div><div class="node a">🛡 SENSOR TRUST AI<br>TAMPER DETECTION</div></div><div class="arrow">↓</div><div class="arch-row"><div class="node a">🔗 FUSION ENGINE<br>TRUST-AWARE DECISION</div></div><div class="arrow">↓</div><div class="arch-row"><div class="node a">TRUSTWORTHY AI DECISION</div></div></div>''', unsafe_allow_html=True)
st.markdown('<div class="foot">TRUSTWORTHY IOT MOTOR AI · MOTOR DIAGNOSIS · SENSOR TRUST VERIFICATION · AI FUSION</div>', unsafe_allow_html=True)
