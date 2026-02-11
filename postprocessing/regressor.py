import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             average_precision_score, f1_score, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold

def measure_logistic_predictor(X, y, adducts, feature_names=None, do_cross_val=True, k_folds=5):
    """
    Logistic regression with L1 penalty; 
    adducts is passed as separate column to make it easier to OHE.
    Returns:
        tprs: list of true positive rates for each fold
        mean_fpr: mean false positive rates for each fold
        metrics: dictionary of metrics
        coefs_list: list of coefficients for each fold
    """
    # 1. Prepare Data
    adducts_arr = np.array(adducts, dtype=str)
    
    mask = (~np.isnan(X).any(axis=1)) & \
           (~np.isnan(y)) & \
           (adducts_arr != 'None') & (adducts_arr != 'nan')
           
    Xc, yc, adducts_c = X[mask], y[mask], adducts_arr[mask]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(Xc.shape[1])]
    
    df_full = pd.DataFrame(Xc, columns=feature_names)
    df_full['adduct_ohe'] = adducts_c
    
    # 2. Setup Plotting & Metrics
    
    metrics = {
        'roc_auc': [], 
        'pr_auc': [], 
        'bal_acc_best': [], 
        'f1_best': [], 
        'prec_at_best_f1': [],
        # --- NEW METRICS ---
        'fpr_at_best_acc': [], 
        'fdr_at_best_acc': [],
        'recall_at_best_acc': [],
        'prec_at_best_acc': []
    }
    coefs_list = []
    final_feature_names = None 

    # 3. Setup CV
    if do_cross_val:
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = list(cv.split(df_full, yc))
        print(f"Running {k_folds}-fold cross-validation...")
    else:
        all_indices = np.arange(len(yc))
        splits = [(all_indices, all_indices)]
        print("Running on full dataset (no CV)...")

    # For Mean ROC
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 4. Define Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_names),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['adduct_ohe'])
        ],
        verbose_feature_names_out=False 
    )

    # 5. Main Loop
    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = df_full.iloc[train_idx], yc[train_idx]
        X_test, y_test = df_full.iloc[test_idx], yc[test_idx]

        # Use class_weight='balanced' to help with the F1/Precision issues you saw earlier
        clf = make_pipeline(
            preprocessor, 
            LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', class_weight='balanced')
        )
        
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # --- Extract Coefficients ---
        if final_feature_names is None:
            final_feature_names = clf[0].get_feature_names_out()
            
        coefs_list.append(clf[-1].coef_.flatten())
        
        # --- Metrics --
        # ROC
        fpr, tpr, thresh = roc_curve(y_test, y_prob)
        metrics['roc_auc'].append(auc(fpr, tpr))
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # --- Threshold Tuning (Best Balanced Accuracy) ---
        j = tpr - fpr
        best_idx_roc = j.argmax()
        best_thresh_roc = thresh[best_idx_roc]
        
        # Create hard predictions based on this specific threshold
        pred_best_roc = (y_prob >= best_thresh_roc).astype(int)
        
        metrics['bal_acc_best'].append(balanced_accuracy_score(y_test, pred_best_roc))

        # --- NEW: Calculate FPR & FDR at this specific threshold ---
        # Confusion Matrix: tn, fp, fn, tp
        tn, fp, fn, tp = confusion_matrix(y_test, pred_best_roc, labels=[0, 1]).ravel()
        
        # False Positive Rate: FP / (FP + TN)  (How many negatives did we wrongly call positive?)
        current_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Discovery Rate: FP / (FP + TP) (Of the ones we called positive, how many were wrong?)
        # This is 1 - Precision
        pred_positives = fp + tp
        current_fdr = fp / pred_positives if pred_positives > 0 else 0.0

        current_recall = tp / (fn + tp) if (fn + tp) > 0 else 0.0
        current_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        metrics['fpr_at_best_acc'].append(current_fpr)
        metrics['fdr_at_best_acc'].append(current_fdr)
        metrics['recall_at_best_acc'].append(current_recall)
        metrics['prec_at_best_acc'].append(current_prec)

        # Precision / Recall / F1 (Standard logic)
        precision, recall, pr_thresh = precision_recall_curve(y_test, y_prob)
        metrics['pr_auc'].append(average_precision_score(y_test, y_prob))
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        best_idx_f1 = f1_scores.argmax()
        metrics['f1_best'].append(f1_scores[best_idx_f1])
        metrics['prec_at_best_f1'].append(precision[best_idx_f1])

    return tprs, mean_fpr, metrics, coefs_list, final_feature_names

    

def plot_roc_curve(tprs, mean_fpr, metrics, coefs_list, final_feature_names, PALETTE):

    # 6. Aggregation & Reporting
    fig, ax = plt.subplots(figsize=(3,3))
    if len(tprs) > 1:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(metrics['roc_auc'])
        
        ax.plot(mean_fpr, mean_tpr, color=PALETTE[4], lw=2,
                label=f'Mean AUROC\n(AUC: {mean_auc:.3f} $\pm$ {std_auc:.3f})')
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=PALETTE[4], alpha=.2)

        print("-" * 40)
        print("Cross Validation Results:")
        print(f"AUROC:               {np.mean(metrics['roc_auc']):.3f} +/- {np.std(metrics['roc_auc']):.3f}")
        print(f"AUPRC:               {np.mean(metrics['pr_auc']):.3f} +/- {np.std(metrics['pr_auc']):.3f}")
        print("-" * 40)
        print("Metrics at 'Best Balanced Accuracy' threshold:")
        print(f"Balanced Acc.:          {np.mean(metrics['bal_acc_best']):.3f} +/- {np.std(metrics['bal_acc_best']):.3f}")
        print(f"False Positive rate:   {np.mean(metrics['fpr_at_best_acc']):.3f} +/- {np.std(metrics['fpr_at_best_acc']):.3f}")
        print(f"False Discovery rate:  {np.mean(metrics['fdr_at_best_acc']):.3f} +/- {np.std(metrics['fdr_at_best_acc']):.3f}")
        print(f"Recall rate:  {np.mean(metrics['recall_at_best_acc']):.3f} +/- {np.std(metrics['recall_at_best_acc']):.3f}")
        print(f"Precision rate:  {np.mean(metrics['prec_at_best_acc']):.3f} +/- {np.std(metrics['prec_at_best_acc']):.3f}")
        
        print("-" * 40)
        
    else:
        print(f"AUROC:               {metrics['roc_auc'][0]:.3f}")
        print(f"False Positive rate:   {metrics['fpr_at_best_acc'][0]:.3f}")
        print(f"False Discovery rate:  {metrics['fdr_at_best_acc'][0]:.3f}")

    # --- Feature Coefficient Table ---
    coefs_array = np.array(coefs_list)
    mean_coefs = np.mean(coefs_array, axis=0)
    std_coefs = np.std(coefs_array, axis=0)
    
    coef_df = pd.DataFrame({
        'Feature': final_feature_names,
        'Mean coef.': mean_coefs,
        'Std coef.': std_coefs,
        'Abs. importance': np.abs(mean_coefs)
    }).sort_values(by='Abs. importance', ascending=False)

    print("\n=== Feature coefficients (top 10 coeffs) ===")
    print(coef_df.head(10).to_string(index=False))

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color=PALETTE[5], label='Random', alpha=.8)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return fig, coef_df
