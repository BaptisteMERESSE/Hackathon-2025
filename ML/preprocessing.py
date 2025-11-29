"""
Advanced Preprocessing Pipeline for PISA Math Score Prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class PISAPreprocessor:
    """
    Comprehensive preprocessing pipeline for PISA data
    """
    
    def __init__(self, 
                 missing_threshold=0.99,
                 correlation_threshold=1,
                 use_knn_imputer=False,
                 scale_method='robust',
                 feature_selection_k=None):
        """
        Args:
            missing_threshold: Drop features with more than this fraction of missing values
            correlation_threshold: Remove features with correlation above this threshold
            use_knn_imputer: Use KNN imputer for numeric features (slower but better)
            scale_method: 'standard' or 'robust'
            feature_selection_k: Number of features to select (None = keep all)
        """
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.use_knn_imputer = use_knn_imputer
        self.scale_method = scale_method
        self.feature_selection_k = feature_selection_k
        
        self.numeric_features = None
        self.categorical_features = None
        self.features_to_drop = []
        self.label_encoders = {}
        self.imputer_numeric = None
        self.imputer_categorical = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
    def fit_transform(self, X, y=None):
        """Fit and transform training data"""
        X = X.copy()
        
        print("Starting preprocessing pipeline...")
        print(f"Initial shape: {X.shape}")
        
        # Step 1: Identify feature types
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        print(f"Numeric features: {len(self.numeric_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        
        # Step 2: Drop features with too many missing values
        missing_pct = X.isnull().sum() / len(X)
        high_missing = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        self.features_to_drop.extend(high_missing)
        print(f"Dropping {len(high_missing)} features with >{self.missing_threshold*100}% missing values")
        
        # Step 3: Drop constant or near-constant features
        for col in X.columns:
            if col not in self.features_to_drop:
                nunique = X[col].nunique()
                if nunique == 1:
                    self.features_to_drop.append(col)
        
        X = X.drop(columns=self.features_to_drop)
        self.numeric_features = [f for f in self.numeric_features if f not in self.features_to_drop]
        self.categorical_features = [f for f in self.categorical_features if f not in self.features_to_drop]
        print(f"Shape after dropping low-quality features: {X.shape}")
        
        # Step 4: Handle categorical features
        if self.categorical_features:
            print(f"Encoding {len(self.categorical_features)} categorical features...")
            for col in self.categorical_features:
                le = LabelEncoder()
                # Handle missing values
                X[col] = X[col].fillna('missing')
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Step 5: Impute missing values
        print("Imputing missing values...")
        
        if self.numeric_features:
            if self.use_knn_imputer and len(X) <= 100000:
                print("Using KNN imputer (this may take a while)...")
                self.imputer_numeric = KNNImputer(n_neighbors=5, weights='distance')
            else:
                self.imputer_numeric = SimpleImputer(strategy='median')
            
            X[self.numeric_features] = self.imputer_numeric.fit_transform(X[self.numeric_features])
        
        # Step 6: Remove highly correlated features
        if self.numeric_features and len(self.numeric_features) > 1:
            print("Removing highly correlated features...")
            corr_matrix = X[self.numeric_features].corr().abs()
            upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            to_drop = [column for column in corr_matrix.columns 
                      if any(corr_matrix[column][upper_triangle[:, corr_matrix.columns.get_loc(column)]] > self.correlation_threshold)]
            X = X.drop(columns=to_drop)
            self.numeric_features = [f for f in self.numeric_features if f not in to_drop]
            print(f"Dropped {len(to_drop)} highly correlated features")
            print(f"Shape after correlation filter: {X.shape}")
        
        # Step 7: Scale numeric features
        if self.numeric_features:
            print(f"Scaling features using {self.scale_method} scaler...")
            if self.scale_method == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            
            X[self.numeric_features] = self.scaler.fit_transform(X[self.numeric_features])
        
        # Step 8: Feature selection
        if self.feature_selection_k and y is not None and self.feature_selection_k < X.shape[1]:
            print(f"Selecting top {self.feature_selection_k} features...")
            self.feature_selector = SelectKBest(score_func=mutual_info_regression, 
                                               k=self.feature_selection_k)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            X = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
            print(f"Final shape: {X.shape}")
        
        print("Preprocessing complete!")
        return X
    
    def transform(self, X):
        """Transform test/validation data"""
        X = X.copy()
        
        # Drop identified features
        X = X.drop(columns=[f for f in self.features_to_drop if f in X.columns])
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].fillna('missing')
                # Handle unseen categories
                X[col] = X[col].apply(lambda x: x if x in self.label_encoders[col].classes_ else 'missing')
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Impute numeric features
        if self.numeric_features:
            X[self.numeric_features] = self.imputer_numeric.transform(X[self.numeric_features])
        
        # Scale numeric features
        if self.numeric_features and self.scaler:
            X[self.numeric_features] = self.scaler.transform(X[self.numeric_features])
        
        # Feature selection
        if self.selected_features:
            X = X[self.selected_features]
        
        return X


def create_advanced_features(X, y=None):
    """
    Create advanced engineered features
    """
    X_new = X.copy()
    
    print("Creating advanced features...")
    
    # Look for PISA-related features (reading, science scores)
    pisa_cols = [col for col in X.columns if 'pisa' in col.lower() or 'score' in col.lower() or 'read' in col.lower()]
    
    if len(pisa_cols) >= 2:
        print(f"Found {len(pisa_cols)} PISA-related columns")
        # Average of other PISA scores
        X_new['pisa_avg'] = X[pisa_cols].mean(axis=1)
        X_new['pisa_std'] = X[pisa_cols].std(axis=1)
        X_new['pisa_max'] = X[pisa_cols].max(axis=1)
        X_new['pisa_min'] = X[pisa_cols].min(axis=1)
    
    # Look for socioeconomic features
    ses_cols = [col for col in X.columns if any(keyword in col.lower() 
                for keyword in ['income', 'education', 'parent', 'escs', 'wealth', 'homepos'])]
    
    if len(ses_cols) >= 2:
        print(f"Found {len(ses_cols)} socioeconomic columns")
        X_new['ses_avg'] = X[ses_cols].mean(axis=1)
        X_new['ses_std'] = X[ses_cols].std(axis=1)
    
    # Polynomial features for key predictors (if reading score exists)
    reading_cols = [col for col in X.columns if 'read' in col.lower() and 'score' in col.lower()]
    if reading_cols:
        print(f"Creating polynomial features for reading scores")
        for col in reading_cols[:3]:  # Limit to avoid explosion
            X_new[f'{col}_squared'] = X[col] ** 2
            X_new[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
    
    # Interaction features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        # Create interactions between highly correlated features with target
        if y is not None:
            correlations = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(5).index.tolist()
            
            if len(top_features) >= 2:
                print(f"Creating interactions for top features")
                X_new[f'{top_features[0]}_x_{top_features[1]}'] = X[top_features[0]] * X[top_features[1]]
                if len(top_features) >= 3:
                    X_new[f'{top_features[0]}_x_{top_features[2]}'] = X[top_features[0]] * X[top_features[2]]
    
    print(f"Feature engineering complete. New shape: {X_new.shape}")
    return X_new


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("="*80)
    print("DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Check if already preprocessed
    if Path('../data/X_train_processed.csv').exists():
        print("\n  X_train_processed.csv already exists!")
        response = input("Do you want to reprocess the data? (y/n): ")
        if response.lower() != 'y':
            print("Skipping preprocessing. Using existing files.")
            sys.exit(0)
    
    # Test preprocessing
    print("\nLoading raw data...")
    X_train = pd.read_csv('../data/X_train.csv', index_col=0)
    y_train = pd.read_csv('../data/y_train.csv', index_col=0)['MathScore']
    
    print(f"Raw data loaded: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # Initialize preprocessor
    preprocessor = PISAPreprocessor(
        missing_threshold=0.99,
        correlation_threshold=1,
        use_knn_imputer=False,  # Faster for large datasets
        scale_method='robust',
        feature_selection_k=None  # Keep all features initially
    )
    
    # Fit and transform
    print("\n" + "="*80)
    print("STEP 1: PREPROCESSING")
    print("="*80)
    X_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Feature engineering
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    X_engineered = create_advanced_features(X_processed, y_train)
    
    print(f"\nFinal preprocessed data shape: {X_engineered.shape}")
    
    # Save preprocessed data
    print("\n" + "="*80)
    print("STEP 3: SAVING")
    print("="*80)
    print("Saving preprocessed data...")
    from pathlib import Path
    Path('../data').mkdir(parents=True, exist_ok=True)
    X_engineered.to_csv('../data/X_train_processed.csv')
    
    # Save preprocessor
    import joblib
    joblib.dump(preprocessor, '../data/preprocessor.pkl')
    
    print("✓ X_train_processed.csv saved")
    print("✓ preprocessor.pkl saved")
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print("You can now train models using the preprocessed data.")


