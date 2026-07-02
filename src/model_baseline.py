import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Import additional models
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Please install: pip install catboost")

# Configuration
from config import OUTPUT_DIR, PROCESSED_DATASET

INPUT_PATH = PROCESSED_DATASET
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(sample_size=50000):
    print("Loading data...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Original Data Shape: {df.shape}")
    
    # Sample data for faster processing during testing
    if len(df) > sample_size:
        print(f"Sampling {sample_size} records for testing...")
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled Data Shape: {df.shape}")
    
    return df

def preprocess_data(df):
    print("Preprocessing...")
    print(f"Input data shape: {df.shape}")
    print(f"Data year range: {df['Year'].min()}-{df['Year'].max()}")
    
    # Validation of Lags (if CSBID and Year exist)
    if 'CSBID' in df.columns and 'Year' in df.columns:
        print("Validating Lag structure on sample...")
        sample_ids = df['CSBID'].unique()[:100]
        valid_count = 0
        total_checks = 0
        
        # Create a lookup for faster checking
        # (CSBID, Year) -> Crop_Type
        lookup = df.set_index(['CSBID', 'Year'])['Crop_Type'].to_dict()
        
        for idx, row in df[df['CSBID'].isin(sample_ids)].iterrows():
            csbid = row['CSBID']
            year = row['Year']
            if (csbid, year - 1) in lookup:
                total_checks += 1
                if row['Crop_Type_Lag1'] == lookup[(csbid, year - 1)]:
                    valid_count += 1
        
        if total_checks > 0:
            print(f"Lag validation: {valid_count}/{total_checks} matches ({valid_count/total_checks:.2%})")
        else:
            print("Lag validation: No consecutive years found in sample.")

    # Use the 5-class crop classification already in the data
    # Map crop types to numeric labels for modeling
    crop_type_mapping = {
        'Corn': 0,
        'Soybeans': 1, 
        'Alfalfa': 2,
        'Combined Hay/Grass': 3,
        'Other': 4
    }
    
    df['Target'] = df['Crop_Type'].map(crop_type_mapping)
    df['Lag1'] = df['Crop_Type_Lag1'].map(crop_type_mapping)
    df['Lag2'] = df['Crop_Type_Lag2'].map(crop_type_mapping)
    
    # Define Annual vs Perennial for hierarchical model
    # Annual: Corn (0), Soybeans (1), Other (4)
    # Perennial: Alfalfa (2), Combined Hay/Grass (3)
    annual_crops = [0, 1, 4]  # Corn, Soybeans, Other
    perennial_crops = [2, 3]  # Alfalfa, Combined Hay/Grass
    
    def get_crop_category(c):
        if pd.isna(c):
            return 0  # Default to Annual
        if c in annual_crops: 
            return 0  # Annual
        if c in perennial_crops: 
            return 1  # Perennial
        return 0  # Default to Annual
        
    df['Crop_Category'] = df['Target'].apply(get_crop_category)
    
    # Features - include coordinate and county-level features
    numeric_features = [
        'Planting_Precip', 'Growing_GDD', 
        'Planting_Precip_Lag1', 'Growing_GDD_Lag1',
        'Lag1', 'Lag2',
        'Longitude_Norm', 'Latitude_Norm',
        'County_Crop_Diversity', 'County_Avg_Field_Size',
        'CNTYFIPS'  # Treat county as numeric for simplicity
    ]

    # Soil numeric features — added only when the column exists and has
    # at least one non-null value (i.e. after generate_soil_mapping.py has run
    # and feature_engineering.py has been re-run to populate the parquet).
    soil_numeric_candidates = [
        'slopegradwta',    # slope gradient (%)
        'aws050wta',       # available water storage 0-50 cm
        'aws0100wta',      # available water storage 0-100 cm
        'drainage_numeric' # ordinal drainage class (0=very poorly drained, 6=excessively drained)
    ]
    for col in soil_numeric_candidates:
        if col in df.columns and df[col].notna().any():
            numeric_features.append(col)

    # Categorical features — hydro group and county dominant crop handled
    # natively by CatBoost; OHE applied for KNN via the preprocessor pipeline.
    categorical_candidates = ['hydro_group_primary', 'County_Dominant_Crop']
    categorical_features = [
        c for c in categorical_candidates
        if c in df.columns and df[c].notna().any()
    ]

    if soil_numeric_candidates[0] in numeric_features:
        print(f"  Soil features active: {[c for c in soil_numeric_candidates if c in numeric_features]}")
    else:
        print("  Soil features not available (run generate_soil_mapping.py first).")

    if categorical_features:
        print(f"  Categorical features: {categorical_features}")
    
    # Check for missing values in required features
    required_features = numeric_features + categorical_features + ['Target']
    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Remove missing features from the lists
        numeric_features = [col for col in numeric_features if col in df.columns]
        categorical_features = [col for col in categorical_features if col in df.columns]
    
    # Drop rows with missing values in critical features
    critical_features = ['Target'] + [col for col in numeric_features if col in df.columns]
    df = df.dropna(subset=critical_features)
    
    print(f"Final preprocessed data shape: {df.shape}")
    print(f"Target distribution:")
    print(df['Target'].value_counts().sort_index())
    
    return df, numeric_features, categorical_features

def train_evaluate(df, numeric_features, categorical_features):
    # Split Train/Test
    train_mask = df['Year'] <= 2022
    test_mask = df['Year'] > 2022
    
    # Include Crop_Category in y for hierarchical
    X_train = df.loc[train_mask, numeric_features + categorical_features]
    y_train = df.loc[train_mask, 'Target']
    y_train_category = df.loc[train_mask, 'Crop_Category']
    
    X_test = df.loc[test_mask, numeric_features + categorical_features]
    y_test = df.loc[test_mask, 'Target']
    # y_test_category = df.loc[test_mask, 'Crop_Category'] # Not needed for prediction
    
    print(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")
    
    # Preprocessing Pipeline for KNN (scale numerics, OHE categoricals)
    transformers = [('num', StandardScaler(), numeric_features)]
    if categorical_features:
        transformers.append((
            'cat',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            categorical_features
        ))
    preprocessor_knn = ColumnTransformer(transformers=transformers)
    
    # Create crop name mapping for better interpretability
    crop_names = {0: 'Corn', 1: 'Soybeans', 2: 'Alfalfa', 3: 'Combined Hay/Grass', 4: 'Other'}
    
    # Results storage
    results = {}
    
    # 1. KNN with n=5
    print("\n" + "="*50)
    print("TRAINING K-NEAREST NEIGHBORS (k=5)")
    print("="*50)
    
    # Stratified subsampling for KNN (computational efficiency)
    if len(X_train) > 30000:
        print("Subsampling for KNN training (Stratified)...")
        X_train_knn, _, y_train_knn, _ = train_test_split(
            X_train, y_train, train_size=min(30000, len(X_train)), stratify=y_train, random_state=42
        )
    else:
        X_train_knn = X_train
        y_train_knn = y_train
    
    knn_pipeline = Pipeline([
        ('preprocessor', preprocessor_knn),
        ('classifier', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    ])
    
    knn_pipeline.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn_pipeline.predict(X_test)
    y_proba_knn = knn_pipeline.predict_proba(X_test)
    
    results['KNN'] = {
        'predictions': y_pred_knn,
        'probabilities': y_proba_knn,
        'model': knn_pipeline
    }
    
    print(f"KNN Training completed. Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
    
    # 2. CatBoost 
    print("\n" + "="*50)
    print("TRAINING CATBOOST")
    print("="*50)
    
    if CATBOOST_AVAILABLE:
        # Subsampling for CatBoost if dataset is too large
        if len(X_train) > 30000:
            print("Subsampling for CatBoost training (Stratified)...")
            X_train_cat, _, y_train_cat, _ = train_test_split(
                X_train, y_train, train_size=min(30000, len(X_train)), stratify=y_train, random_state=42
            )
        else:
            X_train_cat = X_train
            y_train_cat = y_train
        
        # Fill NaN in categorical columns with a placeholder string so
        # CatBoost can handle them without errors.
        if categorical_features:
            X_train_cat = X_train_cat.copy()
            X_test      = X_test.copy()
            for col in categorical_features:
                if col in X_train_cat.columns:
                    X_train_cat[col] = X_train_cat[col].fillna("Unknown").astype(str)
                    X_test[col]      = X_test[col].fillna("Unknown").astype(str)

        # Determine integer indices of categorical columns for CatBoost
        all_features   = numeric_features + categorical_features
        cat_feat_indices = [all_features.index(c) for c in categorical_features]

        catboost_model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=4,
            verbose=False,
            random_state=42,
            auto_class_weights='Balanced',
            cat_features=cat_feat_indices if cat_feat_indices else None
        )

        catboost_model.fit(X_train_cat, y_train_cat)
        y_pred_cat = catboost_model.predict(X_test)
        y_proba_cat = catboost_model.predict_proba(X_test)
        
        results['CatBoost'] = {
            'predictions': y_pred_cat,
            'probabilities': y_proba_cat,
            'model': catboost_model
        }
        
        print(f"CatBoost Training completed. Accuracy: {accuracy_score(y_test, y_pred_cat):.4f}")
    else:
        print("CatBoost not available, skipping...")
    
    # Evaluation and Visualization
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    evaluate_models(results, y_test, crop_names, numeric_features, categorical_features)

def evaluate_models(results, y_test, crop_names, numeric_features=None, categorical_features=None):
    """Comprehensive evaluation of all models"""
    
    # 1. Print detailed classification reports
    for model_name, result in results.items():
        print(f"\n{'-'*20} {model_name} DETAILED RESULTS {'-'*20}")
        y_pred = result['predictions']
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report with crop names
        print(f"\nClassification Report:")
        report = classification_report(
            y_test, y_pred, 
            target_names=[crop_names[i] for i in sorted(crop_names.keys())],
            digits=4
        )
        print(report)
        
        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, crop_names, model_name)
    
    # 2. ROC AUC Analysis (multiclass)
    plot_roc_curves(results, y_test, crop_names)
    
    # 3. Model comparison summary
    print_model_comparison(results, y_test, crop_names, numeric_features, categorical_features)


def plot_confusion_matrix(y_test, y_pred, crop_names, model_name):
    """Plot confusion matrix with crop names"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Create labels from crop names
    labels = [crop_names[i] for i in sorted(crop_names.keys())]
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Crop Type', fontsize=12)
    plt.ylabel('True Crop Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    filename = f"{model_name.lower()}_confusion_matrix.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix: {filepath}")
    plt.show()

def plot_roc_curves(results, y_test, crop_names):
    """Plot ROC curves for all models and classes"""
    
    # Convert to binary format for multiclass ROC
    lb = LabelBinarizer()
    y_test_binary = lb.fit_transform(y_test)
    
    n_classes = len(crop_names)
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    # Create subplots for each class
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        
        for model_idx, (model_name, result) in enumerate(results.items()):
            y_proba = result['probabilities']
            
            # Calculate ROC curve for this class
            fpr, tpr, _ = roc_curve(y_test_binary[:, class_idx], y_proba[:, class_idx])
            auc_score = roc_auc_score(y_test_binary[:, class_idx], y_proba[:, class_idx])
            
            ax.plot(fpr, tpr, color=colors[model_idx], lw=2,
                   label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {crop_names[class_idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if n_classes < 6:
        fig.delaxes(axes[5])
    
    plt.suptitle('ROC Curves by Crop Type and Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    filepath = os.path.join(OUTPUT_DIR, "roc_curves_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves: {filepath}")
    plt.show()

def print_model_comparison(results, y_test, crop_names, numeric_features=None, categorical_features=None):
    """Print summary comparison table"""
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Create comparison dataframe
    comparison_data = []
    
    for model_name, result in results.items():
        y_pred = result['predictions']
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        
        # Multiclass ROC AUC
        try:
            lb = LabelBinarizer()
            y_test_binary = lb.fit_transform(y_test)
            y_proba = result['probabilities']
            auc_macro = roc_auc_score(y_test_binary, y_proba, average='macro', multi_class='ovr')
        except:
            auc_macro = np.nan
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{accuracy:.4f}",
            'Precision (Macro)': f"{precision:.4f}",
            'Recall (Macro)': f"{recall:.4f}",
            'F1-Score (Macro)': f"{f1:.4f}",
            'ROC AUC (Macro)': f"{auc_macro:.4f}" if not np.isnan(auc_macro) else "N/A"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    filepath = os.path.join(OUTPUT_DIR, "model_comparison_summary.csv")
    comparison_df.to_csv(filepath, index=False)
    print(f"\nSaved model comparison: {filepath}")
    
    # Feature importance analysis (for tree-based models)
    analyze_feature_importance(results, numeric_features, categorical_features)

def analyze_feature_importance(results, numeric_features=None, categorical_features=None):
    """Analyze and plot feature importance for tree-based models"""
    
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Use the actual feature lists passed from training so the labels
    # on the importance chart always match what the model saw.
    # CatBoost uses ALL features: numeric + categorical
    if numeric_features is not None and categorical_features is not None:
        feature_names = list(numeric_features) + list(categorical_features)
    elif numeric_features is not None:
        feature_names = list(numeric_features)
    else:
        # Fallback to common feature set
        feature_names = [
            'Planting_Precip', 'Growing_GDD',
            'Planting_Precip_Lag1', 'Growing_GDD_Lag1',
            'Lag1', 'Lag2',
            'Longitude_Norm', 'Latitude_Norm',
            'County_Crop_Diversity', 'County_Avg_Field_Size',
            'CNTYFIPS',
        ]
    
    # Plot feature importance for tree-based models
    tree_models = ['CatBoost', 'LightGBM']
    available_tree_models = [name for name in tree_models if name in results]
    
    if available_tree_models:
        fig, axes = plt.subplots(1, len(available_tree_models), figsize=(15, 6))
        if len(available_tree_models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(available_tree_models):
            model = results[model_name]['model']
            
            try:
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    print(f"  {model_name}: {len(importances)} feature importances, {len(feature_names)} feature names")
                else:
                    continue
                
                # Create feature importance dataframe
                # Ensure feature names and importances match exactly
                n_features = len(importances)
                if len(feature_names) >= n_features:
                    features_to_use = feature_names[:n_features]
                else:
                    # If we have fewer feature names than importances, pad with generic names
                    features_to_use = feature_names + [f'Feature_{i}' for i in range(len(feature_names), n_features)]
                
                feat_imp_df = pd.DataFrame({
                    'Feature': features_to_use,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                # Plot horizontal bar chart
                ax = axes[idx]
                ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'{model_name} Feature Importance')
                ax.grid(True, alpha=0.3)
                
                # Highlight different feature types with colors
                soil_features = ['slopegradwta', 'aws050wta', 'aws0100wta', 'drainage_numeric', 'hydro_group_primary']
                geometric_features = ['Longitude_Norm', 'Latitude_Norm', 'County_Crop_Diversity', 
                                    'County_Avg_Field_Size', 'CNTYFIPS']
                
                for i, feature in enumerate(feat_imp_df['Feature']):
                    if feature in soil_features:
                        ax.get_yticklabels()[i].set_color('green')
                        ax.get_yticklabels()[i].set_fontweight('bold')
                    elif feature in geometric_features:
                        ax.get_yticklabels()[i].set_color('red')
                        ax.get_yticklabels()[i].set_fontweight('bold')
                
            except Exception as e:
                print(f"Could not plot feature importance for {model_name}: {e}")
        
        plt.tight_layout()
        filepath = os.path.join(OUTPUT_DIR, "feature_importance_analysis.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance analysis: {filepath}")
        plt.show()
    



def main():
    df = load_data()
    df, num_feats, cat_feats = preprocess_data(df)
    train_evaluate(df, num_feats, cat_feats)

if __name__ == "__main__":
    main()
