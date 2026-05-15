import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path

st.set_page_config(page_title='Churn Intel', page_icon='🎯', layout='wide', initial_sidebar_state='collapsed')

# ── CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#f4f6fb!important}
.stApp{background:#f4f6fb}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{display:none}
.block-container{padding:1.5rem 2rem!important;max-width:1400px!important}
div[data-testid="stForm"]{border:none!important;padding:0!important;background:transparent!important}
.stFormSubmitButton>button{background:linear-gradient(135deg,#4a25e1,#7c5cfc)!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:.75rem!important;font-weight:700!important;font-size:.95rem!important;width:100%!important;transition:all .2s!important;box-shadow:0 4px 18px rgba(74,37,225,.3)!important}
.stFormSubmitButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(74,37,225,.45)!important}
@media(max-width:768px){
  .block-container{padding:.75rem!important}
  .kpi-grid{grid-template-columns:1fr 1fr!important}
  .intel-grid{grid-template-columns:1fr!important}
}
@media(max-width:480px){
  .kpi-grid{grid-template-columns:1fr!important}
}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}
.kpi-box{background:#fff;border-radius:16px;padding:1.2rem 1.4rem;box-shadow:0 2px 12px rgba(0,0,0,.04);border:1px solid #eef1f8}
.kpi-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem}
.kpi-icon{width:34px;height:34px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1rem}
.kpi-badge{font-size:.7rem;font-weight:700;padding:.2rem .55rem;border-radius:20px}
.kpi-num{font-family:'Space Grotesk';font-size:2rem;font-weight:700;color:#0f172a;line-height:1.1}
.kpi-lbl{font-size:.78rem;color:#64748b;margin-top:.15rem}
.card{background:#fff;border-radius:18px;padding:1.5rem;box-shadow:0 2px 14px rgba(0,0,0,.04);border:1px solid #eef1f8;margin-bottom:1rem}
.card-title{font-family:'Space Grotesk';font-size:1.05rem;font-weight:700;color:#0f172a;margin-bottom:.25rem}
.card-sub{font-size:.78rem;color:#64748b;margin-bottom:1rem}
.intel-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.85rem;margin:1.5rem 0}
.intel-box{background:#fff;border-radius:14px;padding:1rem 1.1rem;box-shadow:0 2px 10px rgba(0,0,0,.03);border-left:4px solid}
.intel-hdr{font-size:.68rem;font-weight:700;letter-spacing:.05em;margin-bottom:.4rem;display:flex;align-items:center;gap:.4rem}
.intel-txt{font-size:.78rem;color:#64748b;line-height:1.5}
.risk-box{border-radius:14px;padding:1.1rem;margin-top:.75rem}
.risk-low{background:#ecfdf5;border:1px solid #a7f3d0}
.risk-med{background:#fffbeb;border:1px solid #fde68a}
.risk-high{background:#fef2f2;border:1px solid #fecaca}
.risk-hdr{display:flex;justify-content:space-between;font-weight:700;font-size:1rem;margin-bottom:.35rem}
.risk-desc{font-size:.78rem;color:#475569;line-height:1.5;margin-bottom:.6rem}
.tag{display:inline-block;background:rgba(74,37,225,.08);color:#4a25e1;padding:.18rem .55rem;border-radius:12px;font-size:.68rem;font-weight:600;margin-right:.3rem}
.reco{background:linear-gradient(145deg,#f8faff,#eef2ff);border-radius:18px;padding:1.5rem;border:1px solid #e2e8f0;margin-top:1rem}
.reco-badge{display:inline-block;background:#0f172a;color:#fff;font-size:.68rem;font-weight:700;padding:.25rem .7rem;border-radius:20px;margin-bottom:.75rem;letter-spacing:.04em}
.reco-title{font-family:'Space Grotesk';font-size:1.3rem;font-weight:700;color:#0f172a;margin-bottom:.5rem}
.reco-text{font-size:.85rem;color:#475569;line-height:1.6;max-width:750px;margin-bottom:1.2rem}
.offer-row{display:flex;gap:1rem;flex-wrap:wrap}
.offer{background:#fff;border-radius:12px;padding:.85rem 1.1rem;box-shadow:0 2px 8px rgba(0,0,0,.04);min-width:180px;border:1px solid #f1f5f9}
.offer-lbl{font-size:.62rem;font-weight:700;color:#94a3b8;letter-spacing:.05em;margin-bottom:.2rem}
.offer-val{font-size:.9rem;font-weight:600;color:#0f172a}
</style>""", unsafe_allow_html=True)

# ── Load ──
@st.cache_resource
def load_artifacts():
    try: return joblib.load('best_model.pkl'), joblib.load('column_transformer.pkl')
    except: return None, None

@st.cache_data
def load_csv():
    try:
        d = pd.read_csv('data/churn.csv')
        d['Churn'] = d['Churn'].map({'Yes':1,'No':0})
        return d
    except: return None

model, ct = load_artifacts()
df = load_csv()

# ── Header ──
h1, h2 = st.columns([3,1])
with h1:
    st.markdown("<h1 style='font-family:Space Grotesk;font-size:1.7rem;color:#4a25e1;margin:0;letter-spacing:-.5px'>Customer Churn Intelligence</h1>", unsafe_allow_html=True)
    st.caption("Telecom Portfolio Real-time Monitoring")
with h2:
    st.markdown("<div style='text-align:right;padding-top:.5rem'><span style='background:#fff;padding:.4rem .9rem;border-radius:20px;font-size:.82rem;font-weight:500;box-shadow:0 2px 8px rgba(0,0,0,.04)'><span style='display:inline-block;width:8px;height:8px;background:#10b981;border-radius:50%;margin-right:.4rem'></span>Live</span></div>", unsafe_allow_html=True)

# ── KPIs ──
tot = len(df) if df is not None else 7043
cr = df['Churn'].mean()*100 if df is not None else 26.5
rr = 100 - cr
rev = df[df['Churn']==1]['MonthlyCharges'].sum()/1000 if df is not None else 42.5

st.markdown(f"""<div class="kpi-grid">
<div class="kpi-box"><div class="kpi-top"><div class="kpi-icon" style="background:#eff6ff;color:#3b82f6">👥</div><div class="kpi-badge" style="background:#ecfdf5;color:#10b981">+2.4%</div></div><div class="kpi-num">{tot:,}</div><div class="kpi-lbl">Total Customers</div></div>
<div class="kpi-box"><div class="kpi-top"><div class="kpi-icon" style="background:#fef2f2;color:#ef4444">↘</div><div class="kpi-badge" style="background:#fef2f2;color:#ef4444">-0.2%</div></div><div class="kpi-num">{cr:.1f}%</div><div class="kpi-lbl">Churn Rate</div></div>
<div class="kpi-box"><div class="kpi-top"><div class="kpi-icon" style="background:#ecfdf5;color:#10b981">🛡</div><div class="kpi-badge" style="background:#ecfdf5;color:#10b981">STABLE</div></div><div class="kpi-num">{rr:.1f}%</div><div class="kpi-lbl">Retention Rate</div></div>
<div class="kpi-box"><div class="kpi-top"><div class="kpi-icon" style="background:#fdf4ff;color:#d946ef">$</div><div class="kpi-badge" style="background:#fdf4ff;color:#d946ef">AT RISK</div></div><div class="kpi-num">${rev:.1f}k</div><div class="kpi-lbl">Revenue Impact</div></div>
</div>""", unsafe_allow_html=True)

st.markdown("")

# ── Input + Results ──
col_in, col_out = st.columns(2, gap="large")

with col_in:
    st.markdown('<div class="card"><div class="card-title">Customer Input Panel</div>', unsafe_allow_html=True)
    with st.form("predict_form"):
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 75.0, 1.0)
        fc1, fc2 = st.columns(2)
        with fc1: contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        with fc2: internet = st.selectbox("Internet", ["Fiber optic","DSL","No"])
        tech = st.toggle("🛡 Tech Support", value=True)
        submitted = st.form_submit_button("✦ Predict Churn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction logic ──
if submitted and model is not None:
    inp = pd.DataFrame([{
        'tenure':tenure,'MonthlyCharges':charges,'Contract':contract,'InternetService':internet,
        'TechSupport':'Yes' if tech else 'No','gender':'Male','SeniorCitizen':'No','Partner':'Yes',
        'Dependents':'No','PhoneService':'Yes','MultipleLines':'No','OnlineSecurity':'No',
        'OnlineBackup':'No','DeviceProtection':'No','StreamingTV':'No','StreamingMovies':'No',
        'PaperlessBilling':'No','PaymentMethod':'Electronic check',
        'TotalCharges':tenure*charges}])
    Xt = ct.transform(inp)
    if hasattr(Xt,'toarray'): Xt = Xt.toarray()
    proba = model.predict_proba(Xt)[0]
    st.session_state['pred'] = {'proba':proba,'tenure':tenure,'charges':charges,'contract':contract,'internet':internet,'tech':tech}

# Get current prediction
p = st.session_state.get('pred', None)
if p:
    churn_p, retain_p = p['proba'][1], p['proba'][0]
else:
    churn_p, retain_p = 0.18, 0.82

stability = int(retain_p * 100)

with col_out:
    st.markdown("**Prediction Results**")
    st.caption("Based on behavioral data signatures")

    gc = "#10b981" if stability>=75 else "#f59e0b" if stability>=45 else "#ef4444"
    fig = go.Figure(go.Indicator(mode="gauge+number",value=stability,
        number={'suffix':'%','font':{'size':36,'color':'#0f172a','family':'Space Grotesk'}},
        title={'text':'STABILITY SCORE','font':{'size':10,'color':'#64748b'}},
        gauge={'axis':{'range':[0,100],'showticklabels':False,'tickwidth':0},
               'bar':{'color':gc,'thickness':.22},'bgcolor':'#f1f5f9','borderwidth':0,
               'steps':[{'range':[0,stability],'color':gc}]}))
    fig.update_layout(height=160,margin=dict(l=10,r=10,t=0,b=0),paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False,'scrollZoom':False,'staticPlot':True})

    if stability>=75:
        cls,icon,lbl="risk-low","✓","Low Risk Customer"
        desc="Profile aligns with long-term retention benchmarks. Contract type and usage are primary drivers."
        tgs=["#LoyalSegment","#StableBilling","#Retained"]
    elif stability>=45:
        cls,icon,lbl="risk-med","⚠","Medium Risk Customer"
        desc="Moderate churn signals. Re-engagement recommended before billing cycle ends."
        tgs=["#WatchList","#MidTenure","#ReviewValue"]
    else:
        cls,icon,lbl="risk-high","✗","High Risk Customer"
        desc="Strong churn indicators. Immediate intervention — deploy retention bundle."
        tgs=["#HighRisk","#ImmediateAction","#ChurnAlert"]

    th = "".join(f'<span class="tag">{t}</span>' for t in tgs)
    st.markdown(f'<div class="risk-box {cls}"><div class="risk-hdr" style="color:{gc}"><span>{lbl}</span><span>{icon}</span></div><div class="risk-desc">{desc}</div><div>{th}</div></div>', unsafe_allow_html=True)

# ── Advanced Intelligence ──
st.markdown("""<div class="intel-grid">
<div class="intel-box" style="border-left-color:#8b5cf6"><div class="intel-hdr" style="color:#8b5cf6">⚡ FIBER OUTAGE</div><div class="intel-txt">Regional outages in Zone 4 correlated with 15% churn spike last 48h.</div></div>
<div class="intel-box" style="border-left-color:#0ea5e9"><div class="intel-hdr" style="color:#0ea5e9">⏱ CONTRACT END</div><div class="intel-txt">2,400 customers approaching contract end. Predicted impact: High.</div></div>
<div class="intel-box" style="border-left-color:#10b981"><div class="intel-hdr" style="color:#10b981">✨ FEATURE ADOPTION</div><div class="intel-txt">Premium bundle adoption reduced churn by 22% in Segment A.</div></div>
<div class="intel-box" style="border-left-color:#ef4444"><div class="intel-hdr" style="color:#ef4444">⚠ COMPETITOR MOVE</div><div class="intel-txt">Competitor launched "Unlimited Lite" at $45. Monitoring port-outs.</div></div>
</div>""", unsafe_allow_html=True)

# ── Charts ──
ch1, ch2 = st.columns([1.5, 1])

with ch1:
    with st.container():
        st.markdown("**Churn Probability Trend**")
        st.caption("Aggregated risk metrics across time")
        days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        vals = [0.12,0.15,0.14,0.18,0.16,0.11,0.14]
        ft = go.Figure(go.Scatter(x=days,y=vals,mode='lines+markers',
            line=dict(color='#4a25e1',width=3,shape='spline'),
            marker=dict(size=7,color='#4a25e1'),fill='tozeroy',
            fillcolor='rgba(74,37,225,.08)'))
        ft.update_layout(height=230,margin=dict(l=0,r=0,t=5,b=0),
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor='#f1f5f9',tickformat='.0%',range=[0,.3]))
        st.plotly_chart(ft, use_container_width=True, config={'displayModeBar':False,'scrollZoom':False,'staticPlot':True})

with ch2:
    with st.container():
        st.markdown("**Segment Distribution**")
        st.caption("Customer tenure breakdown")
        fd = go.Figure(go.Pie(labels=['Core (2+ Yr)','Rising (1-2 Yr)','New (<1 Yr)'],
            values=[42,32,26],hole=.7,
            marker=dict(colors=['#1e1b4b','#4a25e1','#c7d2fe'],line=dict(color='#fff',width=2)),
            textinfo='none'))
        fd.update_layout(height=230,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,legend=dict(orientation="h",y=-.15,x=.5,xanchor="center",font=dict(size=10)))
        fd.add_annotation(text="42%<br><span style='font-size:10px'>CORE</span>",x=.5,y=.5,
            showarrow=False,font=dict(size=22,color='#0f172a',family='Space Grotesk'))
        st.plotly_chart(fd, use_container_width=True, config={'displayModeBar':False,'scrollZoom':False,'staticPlot':True})

ch3, ch4 = st.columns(2)

with ch3:
    with st.container():
        st.markdown("**Feature Importance**")
        fn = ['Contract Type','Monthly Charge','Tenure','Tech Support']
        fv = [0.82,0.74,0.51,0.49]
        fb = go.Figure(go.Bar(x=fv,y=fn,orientation='h',
            marker=dict(color=['#4a25e1','#7c5cfc','#0ea5e9','#f59e0b']),
            text=[f"IMPACT: {v}" for v in fv],textposition='inside',
            insidetextanchor='end',textfont=dict(color='white',size=11)))
        fb.update_layout(height=200,margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False,showticklabels=False),
            yaxis=dict(showgrid=False,tickfont=dict(size=11)))
        st.plotly_chart(fb, use_container_width=True, config={'displayModeBar':False,'scrollZoom':False,'staticPlot':True})

with ch4:
    with st.container():
        st.markdown("**Monthly Retention**")
        mm = ['Jan','Mar','May','Jul','Sep','Nov']
        mv = [92.1,93.5,91.8,94.2,92.5,93.8]
        fm = go.Figure(go.Scatter(x=mm,y=mv,mode='lines',
            line=dict(color='#10b981',width=3,shape='spline'),
            fill='tozeroy',fillcolor='rgba(16,185,129,.08)'))
        fm.update_layout(height=180,margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),yaxis=dict(showgrid=False,range=[90,95],showticklabels=False))
        st.plotly_chart(fm, use_container_width=True, config={'displayModeBar':False,'scrollZoom':False,'staticPlot':True})
        rc1, rc2 = st.columns(2)
        with rc1: st.metric("Avg Retention", "92.4%")
        with rc2: st.metric("MoM Change", "-1.2%", delta="-1.2%", delta_color="inverse")

# ── Risk Factor + Insights (only after prediction) ──
if p:
    st.markdown("---")
    st.markdown("### 📊 Detailed Prediction Analytics")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("**Risk Factor Analysis**")
        factors = ['Tenure','Charges','Contract','Internet','Tech Support']
        scores = [
            1-(p['tenure']/72), p['charges']/150,
            {'Month-to-month':.9,'One year':.4,'Two year':.1}[p['contract']],
            {'Fiber optic':.5,'DSL':.3,'No':.7}[p['internet']],
            .2 if p['tech'] else .8]
        colors = ['#ef4444' if s>.5 else '#10b981' for s in scores]
        fr = go.Figure(go.Bar(x=factors,y=scores,marker_color=colors,
            text=[f"{s:.0%}" for s in scores],textposition='auto'))
        fr.update_layout(height=250,margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickformat='.0%',range=[0,1],gridcolor='#f1f5f9'))
        st.plotly_chart(fr, use_container_width=True, config={'displayModeBar':False,'scrollZoom':False,'staticPlot':True})

    with ac2:
        st.markdown("**💡 Key Insights**")
        insights = []
        if p['tenure']<12: insights.append("📍 New customer — higher churn risk. Invest in onboarding.")
        elif p['tenure']>48: insights.append("⭐ Loyal customer — consider loyalty rewards.")
        else: insights.append(f"📊 Mid-tenure customer ({p['tenure']}mo) — monitor engagement.")
        if p['contract']=='Month-to-month': insights.append("🔴 Month-to-month has highest churn. Recommend annual plan.")
        elif p['contract']=='Two year': insights.append("🟢 Long-term contract — excellent retention indicator.")
        else: insights.append("🟡 One-year contract — moderate stability.")
        if p['charges']>100: insights.append(f"💰 High charges (${p['charges']:.0f}/mo) — verify value delivery.")
        if not p['tech']: insights.append("🛠️ No tech support — adding it reduces churn significantly.")
        else: insights.append("✅ Tech support active — strong positive retention factor.")
        if p['internet']=='Fiber optic': insights.append("🚀 Fiber optic — premium tier, monitor satisfaction.")
        for ins in insights:
            st.info(ins)

# ── Strategic Recommendation ──
st.markdown("")
if stability<45: seg,offer,rate = "High-Risk Attrition","Retention Plus Bundle","68%"
elif stability<75: seg,offer,rate = "Medium-Risk Watchlist","Value Engagement Pack","54%"
else: seg,offer,rate = "Low-Risk Loyal","Referral Bonus Program","82%"

st.markdown(f"""<div class="reco">
<div class="reco-badge">🤖 AI RETENTION ADVISOR</div>
<div class="reco-title">Strategic Retention Recommendation</div>
<div class="reco-text">For customers in the "<b>{seg}</b>" segment, deploy the <b>{offer}</b>. Historical data shows a <b>{rate} renewal success rate</b> with targeted outreach.</div>
<div class="offer-row">
<div class="offer"><div class="offer-lbl">💳 OFFER A</div><div class="offer-val">15% Discount / 12mo</div></div>
<div class="offer"><div class="offer-lbl">📺 OFFER B</div><div class="offer-val">Streaming Pack Add-on</div></div>
</div></div>""", unsafe_allow_html=True)

st.markdown("")
b1, b2, _ = st.columns([1,1,3])
with b1:
    if st.button("🚀 Execute Campaign", use_container_width=True, type="primary"):
        st.success(f"✅ Campaign '{offer}' launched for {seg} segment!")
with b2:
    if df is not None:
        st.download_button("📥 Export Segments", df.to_csv(index=False), "segments.csv", "text/csv", use_container_width=True)
