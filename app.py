from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
import pickle, json, os, io, traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

CAT_COLS = ['Gender', 'Education_Level', 'Branch', 'Department', 'Job_Level', 'Contract_Type']

REQUIRED_COLS = [
    'Age', 'Gender', 'Education_Level', 'Branch', 'Department', 'Job_Level',
    'Contract_Type', 'Experience_Years', 'Salary_EGP', 'Last_Raise_Pct',
    'Performance_Score', 'Manager_Satisfaction', 'Training_Hours_Per_Year',
    'Absence_Days_Per_Year', 'Projects_Completed', 'Promotions_Count',
    'WFH_Days_Per_Week', 'Overtime_Hours_Per_Month', 'Team_Size_Managed',
    'Skills_Count', 'Certifications'
]

# ─── Model loading ────────────────────────────────────────────────
def load_models():
    """Load كل مودل مع الـ scaler الخاص بيه"""
    models, scalers = {}, {}
    for name in ['performance', 'attrition', 'offboarding', 'promotion', 'training_need']:
        model_path  = f'{MODELS_DIR}/model_{name}.pkl'
        # per-model scaler لو موجود، fallback للـ shared scaler
        scaler_path = f'{MODELS_DIR}/scaler_{name}.pkl'
        if not os.path.exists(scaler_path):
            scaler_path = f'{MODELS_DIR}/scaler.pkl'

        with open(model_path, 'rb') as f:  models[name]  = pickle.load(f)
        with open(scaler_path, 'rb') as f: scalers[name] = pickle.load(f)

    with open(f'{MODELS_DIR}/encoders.pkl', 'rb') as f: encoders = pickle.load(f)

    # feature_cols_by_model.pkl لو موجود (بعد إضافة موديلات Promotion/Training Need)
    # كل موديل ليه features مختلفة عشان نمنع الـ data leakage
    feat_cols_path = f'{MODELS_DIR}/feature_cols_by_model.pkl'
    if os.path.exists(feat_cols_path):
        with open(feat_cols_path, 'rb') as f: feat_cols_by_model = pickle.load(f)
    else:
        with open(f'{MODELS_DIR}/feature_cols.pkl', 'rb') as f: generic = pickle.load(f)
        feat_cols_by_model = {name: generic for name in models}

    return models, scalers, encoders, feat_cols_by_model

def preprocess(df, encoders, feat_cols):
    """Encode + extract features - بيتعامل مع unseen labels"""
    df = df.copy()
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
        df[col] = le.transform(df[col])
    return df[feat_cols].values

def get_offboard_reason(row):
    r = []
    if row.get('Performance_Score', 5) < 2.5:      r.append('Low Performance')
    if row.get('Absence_Days_Per_Year', 0) > 20:   r.append('Chronic Absence')
    if row.get('Projects_Completed', 99) < 3:       r.append('Low Productivity')
    if row.get('Manager_Satisfaction', 5) <= 2:     r.append('Manager Conflict')
    if row.get('Skills_Count', 99) < 3:             r.append('Skill Gap')
    return ' | '.join(r) if r else 'General Underperformance'

# ─── Data loading ──────────────────────────────────────────────────
def load_data():
    path = f'{MODELS_DIR}/predictions.json'
    if not os.path.exists(path):
        raise FileNotFoundError("predictions.json not found. Run train_models.py first.")
    return pd.read_json(path)

def get_stats(df):
    total = len(df)
    stats = {
        'total_employees': total,
        'attrition_risk':  int(df['Pred_Attrition'].sum()),
        'attrition_pct':   round(df['Pred_Attrition'].mean() * 100, 1),
        'to_offboard':     int(df['Pred_Offboard'].sum()),
        'offboard_pct':    round(df['Pred_Offboard'].mean() * 100, 1),
        'top_performers':  int(df['Pred_Performance'].sum()),
        'performers_pct':  round(df['Pred_Performance'].mean() * 100, 1),
        'avg_performance': round(float(df['Performance_Score'].mean()), 2),
        'avg_salary':      int(df['Salary_EGP'].mean()),
        'avg_absence':     round(float(df['Absence_Days_Per_Year'].mean()), 1),
    }
    if 'Pred_Promotion' in df.columns:
        stats['promotion_likely'] = int(df['Pred_Promotion'].sum())
        stats['promotion_pct']    = round(df['Pred_Promotion'].mean() * 100, 1)
    if 'Pred_TrainingNeed' in df.columns:
        stats['training_need']     = int(df['Pred_TrainingNeed'].sum())
        stats['training_need_pct'] = round(df['Pred_TrainingNeed'].mean() * 100, 1)
    return stats

# ─── Routes ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/overview')
def api_overview():
    df = load_data()
    stats = get_stats(df)

    df['Age_Group'] = pd.cut(df['Age'], bins=[20,30,40,50,60], labels=['20-30','30-40','40-50','50+'])

    with open(f'{MODELS_DIR}/results.json') as f:     model_results = json.load(f)
    with open(f'{MODELS_DIR}/feature_importance.json') as f: feat_imp = json.load(f)

    salary_by_level = df.groupby('Job_Level')['Salary_EGP'].mean().round(0).astype(int).to_dict()

    exp_bins = df.groupby(pd.cut(df['Experience_Years'], bins=range(0,12,1)))['Pred_Attrition_Prob'].mean().round(3)
    exp_trend = {str(k): round(float(v), 3) for k, v in exp_bins.items() if not np.isnan(v)}

    return jsonify({
        'stats':          stats,
        'branch_counts':  df['Branch'].value_counts().to_dict(),
        'dept_counts':    df['Department'].value_counts().to_dict(),
        'perf_by_branch': df.groupby('Branch')['Performance_Score'].mean().round(2).to_dict(),
        'attr_by_dept':   df.groupby('Department')['Pred_Attrition'].mean().multiply(100).round(1).to_dict(),
        'salary_by_level': salary_by_level,
        'age_dist':       df['Age_Group'].value_counts().to_dict(),
        'edu_dist':       df['Education_Level'].value_counts().to_dict(),
        'gender_dist':    df['Gender'].value_counts().to_dict(),
        'exp_trend':      exp_trend,
        'model_results':  model_results,
        'top_features':   dict(list(feat_imp.items())[:10]),
    })

@app.route('/api/attrition')
def api_attrition():
    df = load_data()
    at_risk = df[df['Pred_Attrition'] == 1].sort_values('Pred_Attrition_Prob', ascending=False)

    cols = ['Employee_ID', 'Branch', 'Department', 'Job_Level', 'Performance_Score',
            'Manager_Satisfaction', 'Absence_Days_Per_Year', 'Last_Raise_Pct',
            'Overtime_Hours_Per_Month', 'Experience_Years', 'Salary_EGP',
            'Pred_Attrition_Prob', 'Attrition_Risk']
    available_cols = [c for c in cols if c in at_risk.columns]

    attr_branch = df.groupby('Branch').agg(
        total=('Employee_ID','count'), at_risk=('Pred_Attrition','sum')
    ).reset_index()
    attr_branch['pct'] = (attr_branch['at_risk'] / attr_branch['total'] * 100).round(1)

    flagged = df[df['Pred_Attrition'] == 1]
    reasons = {
        'Low Manager Satisfaction (≤2)': int((flagged['Manager_Satisfaction'] <= 2).sum()),
        'No Recent Raise (<5%)':         int((flagged['Last_Raise_Pct'] < 5).sum()),
        'High Absence (>15 days)':       int((flagged['Absence_Days_Per_Year'] > 15).sum()),
        'High Overtime (>25 hrs)':       int((flagged['Overtime_Hours_Per_Month'] > 25).sum()),
        'Low Performance (<2.5)':        int((flagged['Performance_Score'] < 2.5).sum()),
    }

    risk_dist = df['Attrition_Risk'].value_counts().to_dict() if 'Attrition_Risk' in df.columns else {}

    return jsonify({
        'employees':     at_risk[available_cols].head(500).to_dict('records'),
        'risk_dist':     risk_dist,
        'attr_by_branch': attr_branch.to_dict('records'),
        'reasons':       reasons,
        'total_at_risk': int(df['Pred_Attrition'].sum()),
    })

@app.route('/api/offboarding')
def api_offboarding():
    df = load_data()
    to_off = df[df['Pred_Offboard'] == 1].sort_values('Pred_Offboard_Prob', ascending=False)

    cols = ['Employee_ID', 'Branch', 'Department', 'Job_Level', 'Performance_Score',
            'Absence_Days_Per_Year', 'Projects_Completed', 'Manager_Satisfaction',
            'Skills_Count', 'Experience_Years', 'Salary_EGP', 'Offboard_Reason', 'Pred_Offboard_Prob']
    available_cols = [c for c in cols if c in to_off.columns]

    off_dept = df.groupby('Department').agg(
        total=('Employee_ID','count'), to_offboard=('Pred_Offboard','sum')
    ).reset_index()
    off_dept['pct'] = (off_dept['to_offboard'] / off_dept['total'] * 100).round(1)

    from collections import Counter
    all_reasons = []
    if 'Offboard_Reason' in df.columns:
        for r in df[df['Pred_Offboard']==1]['Offboard_Reason'].dropna():
            all_reasons.extend([x.strip() for x in str(r).split('|')])

    return jsonify({
        'employees':    to_off[available_cols].to_dict('records'),
        'off_by_dept':  off_dept.to_dict('records'),
        'reason_counts': dict(Counter(all_reasons).most_common(8)),
        'total':        int(df['Pred_Offboard'].sum()),
    })

@app.route('/api/performance')
def api_performance():
    df = load_data()
    top = df[df['Pred_Performance'] == 1].sort_values('Pred_Performance_Prob', ascending=False)

    cols = ['Employee_ID', 'Branch', 'Department', 'Job_Level', 'Performance_Score',
            'Training_Hours_Per_Year', 'Projects_Completed', 'Certifications',
            'Skills_Count', 'Experience_Years', 'Salary_EGP', 'Promotions_Count',
            'Pred_Performance_Prob', 'Best_Branch_Fit']
    available_cols = [c for c in cols if c in top.columns]

    best_per_branch = []
    for branch in df['Branch'].unique():
        sub = df[df['Branch'] == branch].sort_values('Pred_Performance_Prob', ascending=False)
        if len(sub) == 0:
            continue
        emp = sub.iloc[0]
        best_per_branch.append({
            'branch':           branch,
            'employee_id':      emp.get('Employee_ID', '—'),
            'department':       emp.get('Department', '—'),
            'job_level':        emp.get('Job_Level', '—'),
            'performance_score': round(float(emp['Performance_Score']), 2),
            'performance_prob': round(float(emp['Pred_Performance_Prob']), 3),
            'projects':         int(emp.get('Projects_Completed', 0)),
            'training_hours':   int(emp.get('Training_Hours_Per_Year', 0)),
        })

    top_by_branch = top['Branch'].value_counts().to_dict()

    return jsonify({
        'employees':      top[available_cols].head(500).to_dict('records'),
        'best_per_branch': best_per_branch,
        'perf_by_dept':   df.groupby('Department')['Performance_Score'].mean().round(2).to_dict(),
        'top_by_branch':  top_by_branch,
        'total_top':      len(top),
    })

@app.route('/api/promotion')
def api_promotion():
    df = load_data()
    if 'Pred_Promotion' not in df.columns:
        return jsonify({'error': 'Promotion predictions not available — retrain models first'}), 200

    top = df[df['Pred_Promotion'] == 1].sort_values('Pred_Promotion_Prob', ascending=False)

    cols = ['Employee_ID', 'Branch', 'Department', 'Job_Level', 'Performance_Score',
            'Experience_Years', 'Promotions_Count', 'Manager_Satisfaction',
            'Salary_EGP', 'Pred_Promotion_Prob']
    available_cols = [c for c in cols if c in top.columns]

    promo_by_dept = df.groupby('Department').agg(
        total=('Employee_ID', 'count'), likely=('Pred_Promotion', 'sum')
    ).reset_index()
    promo_by_dept['pct'] = (promo_by_dept['likely'] / promo_by_dept['total'] * 100).round(1)

    promo_by_branch = top['Branch'].value_counts().to_dict()

    return jsonify({
        'employees':       top[available_cols].head(500).to_dict('records'),
        'promo_by_dept':   promo_by_dept.to_dict('records'),
        'promo_by_branch': promo_by_branch,
        'total':           int(df['Pred_Promotion'].sum()),
    })

@app.route('/api/training_need')
def api_training_need():
    df = load_data()
    if 'Pred_TrainingNeed' not in df.columns:
        return jsonify({'error': 'Training Need predictions not available — retrain models first'}), 200

    top = df[df['Pred_TrainingNeed'] == 1].sort_values('Pred_TrainingNeed_Prob', ascending=False)

    cols = ['Employee_ID', 'Branch', 'Department', 'Job_Level', 'Performance_Score',
            'Training_Hours_Per_Year', 'Skills_Count', 'Certifications',
            'Manager_Satisfaction', 'Pred_TrainingNeed_Prob']
    available_cols = [c for c in cols if c in top.columns]

    train_by_dept = df.groupby('Department').agg(
        total=('Employee_ID', 'count'), needed=('Pred_TrainingNeed', 'sum')
    ).reset_index()
    train_by_dept['pct'] = (train_by_dept['needed'] / train_by_dept['total'] * 100).round(1)

    return jsonify({
        'employees':      top[available_cols].head(500).to_dict('records'),
        'train_by_dept':  train_by_dept.to_dict('records'),
        'total':          int(df['Pred_TrainingNeed'].sum()),
    })

@app.route('/api/employees')
def api_employees():
    df = load_data()
    page       = int(request.args.get('page', 1))
    per_page   = int(request.args.get('per_page', 50))
    search     = request.args.get('search', '').strip()
    branch_f   = request.args.get('branch', '')
    dept_f     = request.args.get('dept', '')
    risk_f     = request.args.get('risk', '')

    filtered = df.copy()
    if search and 'Employee_ID' in filtered.columns:
        filtered = filtered[filtered['Employee_ID'].str.contains(search, case=False, na=False)]
    if branch_f:
        filtered = filtered[filtered['Branch'] == branch_f]
    if dept_f:
        filtered = filtered[filtered['Department'] == dept_f]
    if risk_f and 'Attrition_Risk' in filtered.columns:
        filtered = filtered[filtered['Attrition_Risk'] == risk_f]

    total = len(filtered)
    paginated = filtered.iloc[(page-1)*per_page : page*per_page]

    cols = ['Employee_ID', 'Gender', 'Age', 'Branch', 'Department', 'Job_Level',
            'Performance_Score', 'Salary_EGP', 'Experience_Years',
            'Attrition_Risk', 'Pred_Attrition_Prob', 'Pred_Performance',
            'Pred_Offboard', 'Best_Branch_Fit', 'Absence_Days_Per_Year']
    available_cols = [c for c in cols if c in paginated.columns]

    return jsonify({
        'employees': paginated[available_cols].to_dict('records'),
        'total':     total,
        'page':      page,
        'pages':     max(1, (total + per_page - 1) // per_page),
        'branches':  sorted(df['Branch'].unique().tolist()),
        'depts':     sorted(df['Department'].unique().tolist()),
    })

# ─── Upload ───────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        fname = file.filename or ''
        if not fname.endswith(('.xlsx', '.xls', '.csv')):
            return jsonify({'success': False, 'error': 'File must be .xlsx, .xls, or .csv'}), 400

        df = pd.read_csv(file) if fname.endswith('.csv') else pd.read_excel(file)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing {len(missing)} columns: {", ".join(missing[:5])}{"…" if len(missing)>5 else ""}',
                'missing_cols': missing
            }), 400

        if len(df) == 0:
            return jsonify({'success': False, 'error': 'File is empty'}), 400

        # Load models & run predictions
        models, scalers, encoders, feat_cols_by_model = load_models()

        for col_name, m_key in [('Performance','performance'), ('Attrition','attrition'), ('Offboard','offboarding'),
                                 ('Promotion','promotion'), ('TrainingNeed','training_need')]:
            X_raw = preprocess(df, encoders, feat_cols_by_model[m_key])
            X_sc = scalers[m_key].transform(X_raw)
            df[f'Pred_{col_name}']      = models[m_key].predict(X_sc)
            df[f'Pred_{col_name}_Prob'] = models[m_key].predict_proba(X_sc)[:, 1].round(3)

        df['Attrition_Risk'] = pd.cut(
            df['Pred_Attrition_Prob'],
            bins=[0, 0.35, 0.60, 1.0],
            labels=['Low', 'Medium', 'High']
        ).astype(str)

        if 'Employee_ID' not in df.columns:
            df.insert(0, 'Employee_ID', [f'EMP{i:05d}' for i in range(1, len(df)+1)])

        if 'Best_Branch_Fit' not in df.columns:
            df['Best_Branch_Fit'] = df.get('Branch', '—')

        df['Offboard_Reason'] = df.apply(get_offboard_reason, axis=1)

        df.to_json(f'{MODELS_DIR}/predictions.json', orient='records')
        df.to_excel(f'{MODELS_DIR}/predictions.xlsx', index=False)

        return jsonify({
            'success': True,
            'rows': len(df),
            'stats': {
                'total':          len(df),
                'top_performers': int(df['Pred_Performance'].sum()),
                'attrition_risk': int(df['Pred_Attrition'].sum()),
                'to_offboard':    int(df['Pred_Offboard'].sum()),
                'attrition_pct':  round(df['Pred_Attrition'].mean() * 100, 1),
                'offboard_pct':   round(df['Pred_Offboard'].mean() * 100, 1),
                'performers_pct': round(df['Pred_Performance'].mean() * 100, 1),
                'promotion_likely':   int(df['Pred_Promotion'].sum()),
                'promotion_pct':      round(df['Pred_Promotion'].mean() * 100, 1),
                'training_need':      int(df['Pred_TrainingNeed'].sum()),
                'training_need_pct':  round(df['Pred_TrainingNeed'].mean() * 100, 1),
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/template')
def api_template():
    output = io.BytesIO()
    pd.DataFrame(columns=['Employee_ID'] + REQUIRED_COLS).to_excel(output, index=False)
    output.seek(0)
    return send_file(output,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True, download_name='hr_template.xlsx')

# ─── Chat (Claude API) ────────────────────────────────────────────
def build_data_context(df):
    """بيبني ملخص مضغوط للداتا يتبعت لـ Claude"""
    total = len(df)
    ctx = {}

    # إحصائيات عامة
    ctx['اجمالي_الموظفين'] = total
    ctx['متوسط_الاداء'] = round(float(df['Performance_Score'].mean()), 2)
    ctx['متوسط_الراتب_EGP'] = int(df['Salary_EGP'].mean())
    ctx['متوسط_الغياب_يوم'] = round(float(df['Absence_Days_Per_Year'].mean()), 1)

    # Predictions summary
    ctx['توقع_مغادرة_12_شهر'] = int(df['Pred_Attrition'].sum())
    ctx['توقع_اداء_عالي_12_شهر'] = int(df['Pred_Performance'].sum())
    ctx['مرشح_للانهاء'] = int(df['Pred_Offboard'].sum())
    if 'Pred_Promotion' in df.columns:
        ctx['مرشح_للترقية'] = int(df['Pred_Promotion'].sum())
    if 'Pred_TrainingNeed' in df.columns:
        ctx['محتاج_تدريب'] = int(df['Pred_TrainingNeed'].sum())

    # توزيع الفروع
    ctx['توزيع_الفروع'] = df['Branch'].value_counts().to_dict()

    # أداء كل فرع
    ctx['متوسط_الاداء_بالفرع'] = df.groupby('Branch')['Performance_Score'].mean().round(2).to_dict()

    # أفضل موظف في كل فرع
    best = {}
    for branch in df['Branch'].unique():
        sub = df[df['Branch'] == branch].sort_values('Pred_Performance_Prob', ascending=False)
        if len(sub):
            e = sub.iloc[0]
            best[branch] = {
                'id': e.get('Employee_ID', '—'),
                'قسم': e.get('Department', '—'),
                'مستوى': e.get('Job_Level', '—'),
                'اداء': round(float(e['Performance_Score']), 2),
                'احتمال_اداء_عالي': round(float(e['Pred_Performance_Prob']), 2),
            }
    ctx['افضل_موظف_لكل_فرع'] = best

    # أعلى خطر مغادرة
    at_risk = df[df['Pred_Attrition'] == 1].sort_values('Pred_Attrition_Prob', ascending=False).head(10)
    ctx['اعلى_10_خطر_مغادرة'] = [{
        'id': r.get('Employee_ID','—'),
        'فرع': r.get('Branch','—'),
        'قسم': r.get('Department','—'),
        'احتمال_المغادرة': round(float(r['Pred_Attrition_Prob']), 2),
        'رضا_المدير': r.get('Manager_Satisfaction','—'),
        'غياب': r.get('Absence_Days_Per_Year','—'),
    } for _, r in at_risk.iterrows()]

    # مرشحون للإنهاء
    offboard = df[df['Pred_Offboard'] == 1].sort_values('Pred_Offboard_Prob', ascending=False).head(10)
    ctx['اعلى_10_مرشح_للانهاء'] = [{
        'id': r.get('Employee_ID','—'),
        'فرع': r.get('Branch','—'),
        'قسم': r.get('Department','—'),
        'سبب': r.get('Offboard_Reason','—'),
        'احتمال': round(float(r['Pred_Offboard_Prob']), 2),
    } for _, r in offboard.iterrows()]

    # توزيع الأقسام والمستويات
    ctx['توزيع_الاقسام'] = df['Department'].value_counts().to_dict()
    ctx['توزيع_المستويات'] = df['Job_Level'].value_counts().to_dict()
    ctx['نسبة_خطر_المغادرة_بالقسم'] = df.groupby('Department')['Pred_Attrition'].mean().round(3).multiply(100).round(1).to_dict()

    return ctx

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        body = request.get_json()
        question = (body.get('question') or '').strip()
        history  = body.get('history', [])   # [{role, content}, ...]

        if not question:
            return jsonify({'error': 'السؤال فاضي'}), 400

        # بناء context من الداتا الحالية
        df = load_data()
        data_ctx = build_data_context(df)

        system_prompt = f"""أنت مساعد ذكاء اصطناعي متخصص في تحليل بيانات الموارد البشرية.
عندك وصول كامل لبيانات {data_ctx['اجمالي_الموظفين']} موظف في الشركة مع تنبؤات من نماذج Machine Learning.

بيانات الشركة الحالية:
{json.dumps(data_ctx, ensure_ascii=False, indent=2)}

قواعد الإجابة:
- دايماً اجاوب بالعربي
- استند على البيانات الفعلية بالأرقام
- لو السؤال عن موظف معين ابحث في البيانات وأعطِ معلومات دقيقة
- لو السؤال عن "أفضل موظف" أو "أسوأ" استخدم الـ predictions
- كن محدداً وعملي - اقترح توصيات للـ HR لو مناسب
- الأرقام في الإجابة تكون واضحة ومنسقة"""

        # بناء messages مع history
        messages = []
        for h in history[-10:]:   # آخر 10 رسايل بس (context window)
            messages.append({'role': h['role'], 'content': h['content']})
        messages.append({'role': 'user', 'content': question})

        # API key: header من الـ browser أو env variable
        api_key = request.headers.get('X-API-Key') or os.environ.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            return jsonify({'error': 'محتاج ANTHROPIC_API_KEY — حطه في الـ input أو في البيئة', 'success': False}), 200

        # Claude API call
        import urllib.request
        payload = json.dumps({
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 1024,
            'system': system_prompt,
            'messages': messages
        }).encode('utf-8')

        req = urllib.request.Request(
            'https://api.anthropic.com/v1/messages',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
                'x-api-key': api_key
            },
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        answer = result['content'][0]['text']
        return jsonify({'answer': answer, 'success': True})

    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        if e.code == 401:
            return jsonify({'error': 'API Key غلط أو مش موجود — اضبط ANTHROPIC_API_KEY', 'success': False}), 200
        return jsonify({'error': f'Claude API Error {e.code}: {err_body}', 'success': False}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc(), 'success': False}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5050)