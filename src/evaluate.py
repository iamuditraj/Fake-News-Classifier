import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    roc_curve
)

def compute_metrics(y_true, y_pred, y_prob, model_name="baseline"):
    """
    Computes evaluation metrics and saves a confusion matrix and ROC curve.
    """
    # 1. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_prob)
    loss = log_loss(y_true, y_prob)
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'roc_auc': roc_auc,
        'log_loss': loss
    }
    
    print(f"--- {model_name.upper()} METRICS ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Log Loss:  {loss:.4f}\n")
    
    # 2. Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # 3. Generate and Save Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {model_name}')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # ROC Curve Plot
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='darkorange', lw=2)
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve - {model_name}')
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    
    # Save the figure before showing it
    fig.savefig(f'figures/evaluation_{model_name}.png', dpi=300)
    plt.show()
    
    return metrics
