#Data comparison
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Calculate the distribution
counts = df['label'].value_counts()
percentages = df['label'].value_counts(normalize=True) * 100

# 2. Print the text summary
print("--- Data Balance Summary ---")
for label, count in counts.items():
    label_name = "Dark Pattern (1)" if label == 1 else "Not Dark Pattern (0)"
    print(f"{label_name}: {count} samples ({percentages[label]:.2f}%)")

# 3. Visualize with a Count Plot
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df, palette='viridis')

# Adding labels and styling
plt.title('Distribution of Labels (Data Balance)', fontsize=14)
plt.xlabel('Label (0: Not Dark, 1: Dark Pattern)', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Not Dark Pattern (0)', 'Dark Pattern (1)'])

# Add text labels on top of the bars
for i, count in enumerate(counts.sort_index()):
    plt.text(i, count + (max(counts)*0.01), f'{count}', ha='center', fontweight='bold')

plt.show()
#model comparison
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


results = []
sklearn_models = {"Logistic Regression": model_binary, "Support Vector Machine": svm_model, "Random Forest": rf_model}

for name, model in sklearn_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    results.append({
        "Model": name, "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred), "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred), "AUC Score": roc_auc_score(y_test, y_prob)
    })

if 'all_labels' in locals():
    results.append({
        "Model": "DistilBERT (Transformer)", "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds), "Recall": recall_score(all_labels, all_preds),
        "F1 Score": f1_score(all_labels, all_preds), "AUC Score": roc_auc_score(all_labels, all_probs)
    })

final_df = pd.DataFrame(results)

def highlight_rf_light(row):
    is_rf = row['Model'] == 'Random Forest'

    return ['background-color: #c6f6d5; color: #22543d; font-weight: bold;' if is_rf else '' for _ in row]


styled_table = final_df.style.format({
    "Accuracy": "{:.2%}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}", "AUC Score": "{:.4f}"
}).apply(highlight_rf_light, axis=1).set_properties(
    
    subset=['Model'], **{'background-color': '#ebf8ff', 'color': '#2c5282', 'font-weight': '600'}
).set_table_styles([
    
    {'selector': 'th', 'props': [
        ('background-color', '#bee3f8'), ('color', '#2a4365'), ('font-family', 'Segoe UI, sans-serif'),
        ('border', '1px solid #a0aec0'), ('padding', '12px'), ('text-align', 'center')
    ]},
    
    {'selector': 'td', 'props': [
        ('background-color', '#ffffff'), ('color', '#2d3748'), ('font-family', 'Segoe UI, sans-serif'),
        ('border', '1px solid #e2e8f0'), ('padding', '10px'), ('text-align', 'center')
    ]},
    
    {'selector': 'tr:hover', 'props': [('background-color', '#f7fafc')]}
]).hide(axis='index')

styled_table
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Prepare data for the graph
comparison_metrics = []
models_to_plot = {
    "Logistic Regression": model_binary,
    "SVM": svm_model,
    "Random Forest": rf_model
}

for name, model in models_to_plot.items():
    y_pred = model.predict(X_test)
    comparison_metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

# Convert to a 'tidy' format for Seaborn
df_plot = pd.DataFrame(comparison_metrics)
df_melted = df_plot.melt(id_vars="Model", var_name="Metric", value_name="Score")

# 2. Create the Visual
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid") # Light mode background

# Custom Light Palette: Blue for LogReg, Purple for SVM, Success Green for RF
palette = {"Logistic Regression": "#90cdf4", "SVM": "#b794f4", "Random Forest": "#68d391"}

ax = sns.barplot(x="Metric", y="Score", hue="Model", data=df_melted, palette=palette)

# 3. Add Labels and Titles
plt.title("Performance Comparison: Dark Pattern Detection", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
plt.xlabel("Evaluation Metrics", fontsize=12)
plt.ylim(0.7, 1.05) # Zoom in to see the differences clearly
plt.legend(title="Classifier", bbox_to_anchor=(1.05, 1), loc='upper left')

# Add score labels on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=9, fontweight='bold')

sns.despine() # Remove top and right borders for a cleaner look
plt.tight_layout()
plt.show()
