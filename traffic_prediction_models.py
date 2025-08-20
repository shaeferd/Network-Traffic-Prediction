#!/usr/bin/env python3
"""
Network Traffic Classification and Intrusion Detection
Automated model training and saving script extracted from Traffic_Prediction.ipynb

This script:
1. Loads and preprocesses the NSL-KDD dataset
2. Trains K-means clustering model
3. Trains XGBoost classification models
4. Saves all models and necessary data files
5. Handles label encoding/decoding for categorical variables
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def load_data():
    """Load and preprocess the NSL-KDD dataset"""
    print("Loading NSL-KDD dataset...")
    
    # File paths
    train_file = 'nsl-kdd/KDDTrain+.txt'
    test_file = 'nsl-kdd/KDDTest+.txt'
    
    # Column names
    header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
                   'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
                   'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
                   'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
                   'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
                   'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
                   'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                   'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
                   'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']
    
    # Load attack category mapping
    category = defaultdict(list)
    category['benign'].append('normal')
    
    with open('nsl-kdd/training_attack_types.txt', 'r') as f:
        for line in f.readlines():
            attack, cat = line.strip().split(' ')
            category[cat].append(attack)
    
    attack_mapping = dict((v,k) for k in category for v in category[k])
    
    # Load data
    df_train_master = pd.read_csv(train_file, names=header_names)
    df_train_master['attack_category'] = df_train_master['attack_type'].map(lambda x: attack_mapping[x])
    df_train_master.drop(['success_pred'], axis=1, inplace=True)
    
    df_test = pd.read_csv(test_file, names=header_names)
    df_test['attack_category'] = df_test['attack_type'].map(lambda x: attack_mapping[x])
    df_test.drop(['success_pred'], axis=1, inplace=True)
    
    return df_train_master, df_test, attack_mapping

def preprocess_data(df_train, df_test):
    """Preprocess the data by handling missing values and encoding categorical variables"""
    print("Preprocessing data...")
    
    # Define column types
    continuous_cols = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                      'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations',
                      'num_shells', 'num_access_files', 'count', 'srv_count', 'serror_rate',
                      'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                      'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                      'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                      'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                      'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    
    binary_cols = ['root_shell', 'su_attempted', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    nominal_cols = ['flag', 'protocol_type', 'service']
    
    # Remove columns with only one value (num_outbound_cmds was already excluded from continuous_cols)
    df_train = df_train.drop(columns=['num_outbound_cmds'])
    df_test = df_test.drop(columns=['num_outbound_cmds'])
    
    # Fix su_attempted values
    df_train['su_attempted'] = df_train['su_attempted'].replace(2, 0)
    df_test['su_attempted'] = df_test['su_attempted'].replace(2, 0)
    
    # One-hot encoding for categorical variables
    df_total = pd.concat([df_train.drop(columns=['attack_category', 'attack_type']), 
                         df_test.drop(columns=['attack_category', 'attack_type'])])
    df_total = pd.get_dummies(df_total, columns=nominal_cols, drop_first=True)
    
    # Split back into train and test
    X = df_total[:len(df_train)]
    X_test = df_total[-len(df_test):]
    
    y = df_train['attack_category']
    y_test = df_test['attack_category']
    
    return X, X_test, y, y_test, continuous_cols, df_total.columns.tolist()

def resample_data(X, y):
    """Resample the data to balance classes"""
    print("Resampling data to balance classes...")
    
    # SMOTE oversampling
    sm = SMOTE(sampling_strategy='auto', random_state=1)
    X_sm, y_sm = sm.fit_resample(X, y)
    
    # Random undersampling
    mean_len = int(pd.Series(y).value_counts().sum() / 5)
    ratio = {'benign': mean_len, 'dos': mean_len, 'probe': mean_len, 'r2l': mean_len, 'u2r': mean_len}
    
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=1, replacement=True)
    X_rus, y_rus = rus.fit_resample(X_sm, y_sm)
    
    return X_rus, y_rus

def train_xgboost_models(X_train, y_train, continuous_cols):
    """Train XGBoost models for binary and multi-class classification"""
    print("Training XGBoost models...")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), continuous_cols)
    ], remainder='passthrough')
    
    # Binary classification pipeline
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('select', SelectPercentile(percentile=33)),
        ('model', XGBClassifier(random_state=2))
    ])
    
    # Multi-class classification pipeline
    xgb_pipeline2 = Pipeline([
        ('preprocessor', preprocessor),
        ('select', SelectPercentile(percentile=33)),
        ('model', XGBClassifier(random_state=2))
    ])
    
    # Convert labels to binary for binary classification
    y_train_benign = y_train.map(lambda x: 0.0 if x == 'benign' else 1.0)
    
    # Convert labels to numeric for multi-class classification
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Train models
    print("Training binary classification model...")
    xgb_pipeline.fit(X_train, y_train_benign)
    
    print("Training multi-class classification model...")
    xgb_pipeline2.fit(X_train, y_train_encoded)
    
    return xgb_pipeline, xgb_pipeline2, label_encoder

def train_kmeans_model(X_train, y_train, continuous_cols):
    """Train K-means clustering model"""
    print("Training K-means clustering model...")
    
    # Create preprocessor for continuous features only
    preprocessor = ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), continuous_cols)
    ], remainder='passthrough')
    
    # Transform training data
    X_train_cont = X_train[continuous_cols]
    X_train_scaled = preprocessor.fit_transform(X_train_cont)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=31)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Train K-means
    kmeans = KMeans(n_clusters=27, random_state=2)
    kmeans.fit(X_train_pca)
    
    return kmeans, preprocessor, pca

def save_models_and_data(xgb_pipeline, xgb_pipeline2, kmeans, preprocessor, pca, 
                         X_train, X_test, y_train, y_test, continuous_cols, 
                         attack_mapping, col_names, label_encoder):
    """Save all models and necessary data files"""
    print("Saving models and data files...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Save XGBoost models
    pickle.dump(xgb_pipeline, open("models/xgb_pipeline.sav", 'wb'))
    pickle.dump(xgb_pipeline2, open("models/xgb_pipeline2.sav", 'wb'))
    
    # Save K-means model
    pickle.dump(kmeans, open("models/kmeans.sav", 'wb'))
    
    # Save preprocessor and PCA
    pickle.dump(preprocessor, open("models/preprocessor.sav", 'wb'))
    pickle.dump(pca, open("models/pca.sav", 'wb'))
    
    # Save attack mapping for label decoding
    pickle.dump(attack_mapping, open("models/attack_mapping.sav", 'wb'))
    
    # Save label encoder for multi-class classification
    pickle.dump(label_encoder, open("models/label_encoder.sav", 'wb'))
    
    # Save continuous columns list
    df_cont = pd.DataFrame({"continuous_cols": continuous_cols})
    df_cont.to_csv('data/continuous_cols.csv', index=False)
    
    # Save column names
    df_cols = pd.DataFrame({"column_names": col_names})
    df_cols.to_csv('data/column_names.csv', index=False)
    
    # Save training and test data
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    
    # Save labels
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print("All models and data files saved successfully!")

def create_label_encoders(y_train, y_test):
    """Create and save label encoders for categorical variables"""
    print("Creating label encoders...")
    
    # Create label encoder for attack categories
    attack_encoder = LabelEncoder()
    attack_encoder.fit(pd.concat([y_train, y_test]))
    
    # Save encoder
    pickle.dump(attack_encoder, open("models/attack_encoder.sav", 'wb'))
    
    # Create reverse mapping for decoding
    attack_categories = attack_encoder.classes_
    attack_decoder = {i: category for i, category in enumerate(attack_categories)}
    pickle.dump(attack_decoder, open("models/attack_decoder.sav", 'wb'))
    
    print("Label encoders created and saved!")

def main():
    """Main function to run the entire pipeline"""
    print("Starting Network Traffic Prediction Model Training...")
    print("=" * 60)
    
    try:
        # Load data
        df_train_master, df_test, attack_mapping = load_data()
        
        # Preprocess data
        X, X_test, y, y_test, continuous_cols, col_names = preprocess_data(df_train_master, df_test)
        
        # Resample data
        X_rus, y_rus = resample_data(X, y)
        
        # Split data for training
        X_train, X_val, y_train, y_val = train_test_split(X_rus, y_rus, test_size=0.2, random_state=2)
        
        # Train XGBoost models
        xgb_pipeline, xgb_pipeline2, label_encoder = train_xgboost_models(X_train, y_train, continuous_cols)
        
        # Train K-means model
        kmeans, preprocessor, pca = train_kmeans_model(X_train, y_train, continuous_cols)
        
        # Create label encoders
        create_label_encoders(y_train, y_test)
        
        # Save all models and data
        save_models_and_data(xgb_pipeline, xgb_pipeline2, kmeans, preprocessor, pca,
                           X_train, X_test, y_train, y_test, continuous_cols,
                           attack_mapping, col_names, label_encoder)
        
        print("\n" + "=" * 60)
        print("SUCCESS: All models trained and saved!")
        print("=" * 60)
        print("\nSaved files:")
        print("- models/xgb_pipeline.sav (Binary classification)")
        print("- models/xgb_pipeline2.sav (Multi-class classification)")
        print("- models/kmeans.sav (Clustering)")
        print("- models/preprocessor.sav (Data preprocessing)")
        print("- models/pca.sav (Dimensionality reduction)")
        print("- models/attack_mapping.sav (Attack type mapping)")
        print("- models/label_encoder.sav (Multi-class label encoder)")
        print("- models/attack_encoder.sav (Label encoder)")
        print("- models/attack_decoder.sav (Label decoder)")
        print("- data/continuous_cols.csv (Continuous features)")
        print("- data/column_names.csv (All feature names)")
        print("- data/X_train.csv, data/X_test.csv (Training/test features)")
        print("- data/y_train.csv, data/y_test.csv (Training/test labels)")
        
    except Exception as e:
        print(f"ERROR: An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
