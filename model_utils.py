#!/usr/bin/env python3
"""
Utility functions for loading and using the trained models
with proper label encoding and decoding
"""

import pickle
import pandas as pd
import numpy as np

def load_models():
    """Load all saved models and encoders"""
    print("Loading saved models...")
    
    # Load XGBoost models
    xgb_pipeline = pickle.load(open("models/xgb_pipeline.sav", 'rb'))
    xgb_pipeline2 = pickle.load(open("models/xgb_pipeline2.sav", 'rb'))
    
    # Load K-means model
    kmeans = pickle.load(open("models/kmeans.sav", 'rb'))
    
    # Load preprocessor and PCA
    preprocessor = pickle.load(open("models/preprocessor.sav", 'rb'))
    pca = pickle.load(open("models/pca.sav", 'rb'))
    
    # Load label encoders and decoders
    attack_encoder = pickle.load(open("models/attack_encoder.sav", 'rb'))
    attack_decoder = pickle.load(open("models/attack_decoder.sav", 'rb'))
    attack_mapping = pickle.load(open("models/attack_mapping.sav", 'rb'))
    
    # Load continuous columns
    continuous_cols_df = pd.read_csv('data/continuous_cols.csv')
    continuous_cols = continuous_cols_df['continuous_cols'].tolist()
    
    # Load column names
    column_names_df = pd.read_csv('data/column_names.csv')
    column_names = column_names_df['column_names'].tolist()
    
    print("All models loaded successfully!")
    
    return {
        'xgb_pipeline': xgb_pipeline,
        'xgb_pipeline2': xgb_pipeline2,
        'kmeans': kmeans,
        'preprocessor': preprocessor,
        'pca': pca,
        'attack_encoder': attack_encoder,
        'attack_decoder': attack_decoder,
        'attack_mapping': attack_mapping,
        'continuous_cols': continuous_cols,
        'column_names': column_names
    }

def predict_binary_classification(models, X_new):
    """Predict binary classification (benign vs malicious)"""
    xgb_pipeline = models['xgb_pipeline']
    
    # Make prediction
    prediction = xgb_pipeline.predict(X_new)
    probabilities = xgb_pipeline.predict_proba(X_new)
    
    # Convert back to categorical names
    prediction_labels = ['benign' if pred == 0.0 else 'malicious' for pred in prediction]
    
    return prediction_labels, probabilities

def predict_multi_class_classification(models, X_new):
    """Predict multi-class classification (attack types)"""
    xgb_pipeline2 = models['xgb_pipeline2']
    
    # Make prediction
    prediction = xgb_pipeline2.predict(X_new)
    probabilities = xgb_pipeline2.predict_proba(X_new)
    
    # Convert encoded labels back to categorical names
    attack_decoder = models['attack_decoder']
    prediction_labels = [attack_decoder.get(pred, 'unknown') for pred in prediction]
    
    return prediction_labels, probabilities

def predict_clustering(models, X_new):
    """Predict clustering labels"""
    kmeans = models['kmeans']
    preprocessor = models['preprocessor']
    pca = models['pca']
    
    # Get continuous features only
    continuous_cols = models['continuous_cols']
    X_cont = X_new[continuous_cols]
    
    # Preprocess and apply PCA
    X_scaled = preprocessor.transform(X_cont)
    X_pca = pca.transform(X_scaled)
    
    # Predict clusters
    cluster_labels = kmeans.predict(X_pca)
    
    return cluster_labels

def combine_predictions(models, X_new):
    """Combine XGBoost and K-means predictions for ensemble approach"""
    # Get individual predictions
    binary_pred, binary_proba = predict_binary_classification(models, X_new)
    multi_pred, multi_proba = predict_multi_class_classification(models, X_new)
    cluster_pred = predict_clustering(models, X_new)
    
    # For demonstration, we'll use the binary classification as primary
    # and enhance with clustering insights
    combined_pred = binary_pred.copy()
    
    # You could implement more sophisticated ensemble logic here
    # For example, if clustering suggests outlier behavior, flag as suspicious
    
    return {
        'binary_classification': binary_pred,
        'multi_class_classification': multi_pred,
        'clustering': cluster_pred,
        'combined_prediction': combined_pred,
        'binary_probabilities': binary_proba,
        'multi_class_probabilities': multi_proba
    }

def decode_attack_type(attack_name, models):
    """Decode attack type to get more detailed information"""
    attack_mapping = models['attack_mapping']
    
    # Find the category for this attack
    for category, attacks in attack_mapping.items():
        if attack_name in attacks:
            return category
    
    return 'unknown'

def example_usage():
    """Example of how to use the models"""
    print("Loading models...")
    models = load_models()
    
    # Load some test data
    print("\nLoading test data...")
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    
    # Make predictions on a few samples
    print("\nMaking predictions on first 5 test samples...")
    X_sample = X_test.head(5)
    
    # Get combined predictions
    results = combine_predictions(models, X_sample)
    
    print("\nPrediction Results:")
    print("=" * 50)
    
    for i in range(len(X_sample)):
        print(f"\nSample {i+1}:")
        print(f"  Binary Classification: {results['binary_classification'][i]}")
        print(f"  Multi-class Classification: {results['multi_class_classification'][i]}")
        print(f"  Clustering Cluster: {results['clustering'][i]}")
        print(f"  Combined Prediction: {results['combined_prediction'][i]}")
        
        # Decode attack type if it's an attack
        if results['multi_class_classification'][i] != 'benign':
            attack_category = decode_attack_type(results['multi_class_classification'][i], models)
            print(f"  Attack Category: {attack_category}")
        
        # Show confidence scores
        binary_conf = max(results['binary_probabilities'][i])
        print(f"  Binary Classification Confidence: {binary_conf:.3f}")

if __name__ == "__main__":
    example_usage()
