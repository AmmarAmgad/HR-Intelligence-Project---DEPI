"""
HR Intelligence - Model Training Pipeline
==========================================
الترتيب الصح:
  1. Label Encoding
  2. Train/Test Split (مع stratify)
  3. Scaling (fit على train بس)
  4. SMOTE (على train بس)
  5. Cross-Validation
  6. Training
  7. Evaluation على test
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle, json, os, warnings
warnings.filterwarnings('ignore')


DATA_PATH = 'models/hr_dataset_10000.xlsx'
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

CAT_COLS = ['Gender', 'Education_Level', 'Branch', 'Department', 'Job_Level', 'Contract_Type']

FEATURE_COLS = [
    'Age', 'Gender', 'Education_Level', 'Branch', 'Department', 'Job_Level',
    'Contract_Type', 'Experience_Years', 'Salary_EGP', 'Last_Raise_Pct',
    'Performance_Score', 'Manager_Satisfaction', 'Training_Hours_Per_Year',
    'Absence_Days_Per_Year', 'Projects_Completed', 'Promotions_Count',
    'WFH_Days_Per_Week', 'Overtime_Hours_Per_Month', 'Team_Size_Managed',
    'Skills_Count', 'Certifications'
]

# ── Features خاصة بكل موديل (بنستبعد أي عمود استُخدم لاشتقاق التارجت بتاعه، منعًا للـ data leakage) ──
# ── Features خاصة بكل موديل ──
# ملحوظة: Promotion و Training Need بقوا composite scores مبنية من الفيتشرز نفسها
# (نفس أسلوب Will_Perform_Next_12M/Should_Be_Offboarded)، فمفيش عمود لازم نستبعده.
FEATURE_COLS_PROMOTION = FEATURE_COLS
FEATURE_COLS_TRAINING_NEED = FEATURE_COLS

# ─── STEP 1: Label Encoding ──────────────────────────
def fit_encoders(df):
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders

def apply_encoders(df, encoders):
    df = df.copy()
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
        df[col] = le.transform(df[col])
    return df

# ─── CORE TRAINING FUNCTION ───────────────────────────
def train_model(X_all, y, model, model_name, cv_folds=5):
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Classes: { dict(zip(unique.tolist(), counts.tolist())) }")

    # STEP 2: Split مع stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # STEP 3: Scaling - fit على train بس ✅
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # STEP 4: SMOTE على train بس ✅
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train_sc, y_train)
    u2, c2 = np.unique(y_res, return_counts=True)
    print(f"  After SMOTE: {len(X_res):,}  |  Classes: { dict(zip(u2.tolist(), c2.tolist())) }")

    # STEP 5: Cross-Validation
    print(f"\n  {cv_folds}-Fold Stratified CV...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='f1_weighted', n_jobs=-1)
    print(f"  CV Scores : {[round(s,3) for s in cv_scores]}")
    print(f"  CV Mean   : {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")
    if cv_scores.std() > 0.05:
        print(f"  ⚠️  High variance — model might be unstable")

    # STEP 6: Final training
    model.fit(X_res, y_res)

    # STEP 7: Evaluation على test set
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    test_f1  = f1_score(y_test, y_pred, average='weighted')
    test_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Test F1  : {test_f1:.4f}")
    print(f"  Test AUC : {test_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred))

    gap = cv_scores.mean() - test_f1
    if gap > 0.05:
        print(f"  ⚠️  Possible overfitting (CV={cv_scores.mean():.3f} vs Test={test_f1:.3f})")
    else:
        print(f"  ✅ No major overfitting (gap={gap:.3f})")

    return model, scaler, {
        'f1':      round(test_f1, 4),
        'auc':     round(test_auc, 4),
        'cv_mean': round(float(cv_scores.mean()), 4),
        'cv_std':  round(float(cv_scores.std()), 4),
        'cv_scores': [round(s,4) for s in cv_scores.tolist()],
        'confusion_matrix': cm.tolist(),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

# ─── MAIN ─────────────────────────────────────────────
def train_all():
    print("Loading data...")
    df_raw = pd.read_excel(DATA_PATH)
    print(f"Shape: {df_raw.shape}")

    missing = [c for c in FEATURE_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # STEP 1: Encode
    print("\nStep 1: Label Encoding...")
    encoders = fit_encoders(df_raw)
    df_enc   = apply_encoders(df_raw, encoders)
    X_all    = df_enc[FEATURE_COLS].values

    # ── اشتقاق التارجتس اللي مش موجودة جاهزة في الداتا ──
    # ملحوظة مهمة: جربنا الأول نتنبأ بـ Promotions_Count و Training_Hours_Per_Year مباشرة،
    # لكن تحليل الـ correlation أثبت إنهم شبه مستقلين عن باقي الأعمدة (AUC ~0.48 = عشوائي تمامًا).
    # الحل: نبني composite score من نفس الفيتشرز القوية اللي بتبني بيها Will_Perform_Next_12M
    # (نفس أسلوب تصميم التارجتس الناجحة التلاتة التانية) — ده بيدّي موديل بمعنى حقيقي: "درجة جاهزية"
    # مبنية ومرجّحة بدل رقم عشوائي.
    def zscore(s):
        return (s - s.mean()) / s.std()

    # Promotion Readiness: أداء عالي + رضا مدير عالي + مشاريع كتير + شهادات + خبرة أطول + غياب أقل
    promo_score = (
        zscore(df_raw['Performance_Score']) + zscore(df_raw['Manager_Satisfaction']) +
        zscore(df_raw['Projects_Completed']) + zscore(df_raw['Certifications']) +
        zscore(df_raw['Experience_Years']) - zscore(df_raw['Absence_Days_Per_Year'])
    )
    df_raw['Promotion_Ready'] = (promo_score >= promo_score.quantile(0.75)).astype(int)  # أعلى 25%

    # Training Need: أداء ضعيف + رضا مدير منخفض + مهارات قليلة + شهادات قليلة + غياب أعلى
    train_score = (
        zscore(-df_raw['Performance_Score']) + zscore(-df_raw['Manager_Satisfaction']) +
        zscore(-df_raw['Skills_Count']) + zscore(-df_raw['Certifications']) +
        zscore(df_raw['Absence_Days_Per_Year'])
    )
    df_raw['Needs_Training'] = (train_score >= train_score.quantile(0.75)).astype(int)  # أعلى 25% احتياج

    print("\nDerived targets:")
    for col in ['Promotion_Ready', 'Needs_Training']:
        vc = df_raw[col].value_counts().to_dict()
        min_class = min(vc.values())
        print(f"  {col}: {vc}")
        if min_class < 6:
            print(f"  ⚠️  Minority class has only {min_class} rows — SMOTE (k_neighbors=5) may fail after the 80/20 split")

    X_promotion     = df_enc[FEATURE_COLS_PROMOTION].values
    X_training_need = df_enc[FEATURE_COLS_TRAINING_NEED].values

    results = {}
    models  = {}
    scalers = {}

    # Model 1: Performance
    m1 = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    models['performance'], scalers['performance'], results['performance'] = \
        train_model(X_all, df_raw['Will_Perform_Next_12M'].values, m1, "Performance (XGBoost)")

    # Model 2: Attrition
    m2 = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    models['attrition'], scalers['attrition'], results['attrition'] = \
        train_model(X_all, df_raw['Will_Leave_Next_12M'].values, m2, "Attrition (Random Forest)")

    # Model 3: Offboarding
    m3 = LogisticRegression(
        max_iter=1000, C=1.0, class_weight='balanced',
        solver='lbfgs', random_state=42
    )
    models['offboarding'], scalers['offboarding'], results['offboarding'] = \
        train_model(X_all, df_raw['Should_Be_Offboarded'].values, m3, "Offboarding (Logistic Regression)")

    # Model 4: Promotion
    m4 = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    models['promotion'], scalers['promotion'], results['promotion'] = \
        train_model(X_promotion, df_raw['Promotion_Ready'].values, m4, "Promotion Readiness (XGBoost)")

    # Model 5: Training Need
    m5 = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    models['training_need'], scalers['training_need'], results['training_need'] = \
        train_model(X_training_need, df_raw['Needs_Training'].values, m5, "Training Need (Random Forest)")

    # ── Save ──
    print("\nSaving models...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name in models:
        with open(f'{MODELS_DIR}/model_{name}.pkl', 'wb') as f: pickle.dump(models[name], f)
        with open(f'{MODELS_DIR}/scaler_{name}.pkl', 'wb') as f: pickle.dump(scalers[name], f)

    # backward compat: single scaler.pkl للـ app.py القديم
    with open(f'{MODELS_DIR}/scaler.pkl', 'wb') as f: pickle.dump(scalers['attrition'], f)
    with open(f'{MODELS_DIR}/encoders.pkl', 'wb') as f: pickle.dump(encoders, f)
    with open(f'{MODELS_DIR}/feature_cols.pkl', 'wb') as f: pickle.dump(FEATURE_COLS, f)
    with open(f'{MODELS_DIR}/feature_cols_by_model.pkl', 'wb') as f:
        pickle.dump({
            'performance':    FEATURE_COLS,
            'attrition':      FEATURE_COLS,
            'offboarding':    FEATURE_COLS,
            'promotion':      FEATURE_COLS_PROMOTION,
            'training_need':  FEATURE_COLS_TRAINING_NEED,
        }, f)
    with open(f'{MODELS_DIR}/results.json', 'w') as f: json.dump(results, f, indent=2)

    feat_imp = dict(zip(FEATURE_COLS, m2.feature_importances_))
    with open(f'{MODELS_DIR}/feature_importance.json', 'w') as f:
        json.dump(dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)), f, indent=2)

    # ── Predictions على full dataset للـ Dashboard ──
    print("\nGenerating predictions on full dataset...")
    df_out = df_raw.copy()

    if 'Employee_ID' not in df_out.columns:
        df_out.insert(0, 'Employee_ID', [f'EMP{i:05d}' for i in range(1, len(df_out)+1)])

    targets = [('Performance',   models['performance'],   scalers['performance'],   X_all),
               ('Attrition',     models['attrition'],     scalers['attrition'],     X_all),
               ('Offboard',      models['offboarding'],   scalers['offboarding'],   X_all),
               ('Promotion',     models['promotion'],     scalers['promotion'],     X_promotion),
               ('TrainingNeed',  models['training_need'], scalers['training_need'], X_training_need)]

    for col_name, model, scaler, X_model in targets:
        X_sc = scaler.transform(X_model)
        df_out[f'Pred_{col_name}']      = model.predict(X_sc)
        df_out[f'Pred_{col_name}_Prob'] = model.predict_proba(X_sc)[:, 1].round(3)

    df_out['Attrition_Risk'] = pd.cut(
        df_out['Pred_Attrition_Prob'],
        bins=[0, 0.35, 0.60, 1.0],
        labels=['Low', 'Medium', 'High']
    ).astype(str)

    if 'Best_Branch_Fit' not in df_out.columns:
        df_out['Best_Branch_Fit'] = df_out['Branch']

    df_out.to_excel(f'{MODELS_DIR}/predictions.xlsx', index=False)
    df_out.to_json(f'{MODELS_DIR}/predictions.json', orient='records')

    print("\n" + "="*55)
    print("✅ All done!")
    for name, res in results.items():
        print(f"  {name:15} | F1={res['f1']:.4f} | AUC={res['auc']:.4f} | CV={res['cv_mean']:.4f}±{res['cv_std']:.4f}")

if __name__ == '__main__':
    train_all()