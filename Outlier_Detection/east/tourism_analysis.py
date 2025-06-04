import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_data(filepath):
    dtype_map = {
        'DGSTFN': 'float32',
        'REVISIT_INTENTION': 'float32',
        'RCMDTN_INTENTION': 'float32',
        'VISIT_ORDER': 'Int16',
        'RESIDENCE_TIME_MIN': 'Int32'
    }
    usecols = [
        'VISIT_AREA_ID', 'TRAVEL_ID', 'DGSTFN',
        'REVISIT_INTENTION', 'RCMDTN_INTENTION',
        'VISIT_AREA_NM', 'VISIT_START_YMD'
    ]
    data = pd.read_csv(
        filepath,
        usecols=usecols,
        dtype=dtype_map,
        parse_dates=['VISIT_START_YMD'],
    )
    return data

def create_composite_features(data):
    data = data.copy()
    data['satisfaction_revisit_ratio'] = data['DGSTFN'] / (data['REVISIT_INTENTION'] + 0.1)
    data['satisfaction_recommend_ratio'] = data['DGSTFN'] / (data['RCMDTN_INTENTION'] + 0.1)
    features = ['DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION', 
                'satisfaction_revisit_ratio', 'satisfaction_recommend_ratio']
    return data[features].dropna()

def detect_outliers_improved(data):
    filtered = data.dropna(subset=['DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']).copy()
    filtered.loc[:, 'DGSTFN'] = filtered['DGSTFN'].astype('float32')

    # IQR Method
    Q1 = filtered['DGSTFN'].quantile(0.25)
    Q3 = filtered['DGSTFN'].quantile(0.75)
    IQR = Q3 - Q1
    statistical_outliers = (filtered['DGSTFN'] < Q1 - 2*IQR) | (filtered['DGSTFN'] > Q3 + 2*IQR)

    # Isolation Forest
    features = create_composite_features(filtered)
    contamination_rate = 0.05
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination_rate,
        max_samples='auto',
        random_state=42
    )
    iso_outliers = iso.fit_predict(features) == -1

    # Combine methods
    combined_outliers = statistical_outliers | iso_outliers

    return {
        'statistical': statistical_outliers.sum(),
        'isolation': iso_outliers.sum(),
        'combined': combined_outliers.sum(),
        'filtered_data': filtered,
        'combined_mask': combined_outliers,
        'cleaned_data': filtered[~combined_outliers]  # Added cleaned data
    }

def create_transactions(data):
    data = data.dropna(subset=['VISIT_AREA_NM', 'TRAVEL_ID']).copy()
    transactions = data.groupby('TRAVEL_ID').agg({
        'VISIT_AREA_NM': list,
        'DGSTFN': 'mean',
        'REVISIT_INTENTION': 'mean',
        'RCMDTN_INTENTION': 'mean',
        'VISIT_START_YMD': ['min', 'max']
    }).reset_index()

    transactions.columns = [
        'travel_id', 'visit_sequence', 'avg_satisfaction',
        'avg_revisit_intent', 'avg_recommend_intent',
        'trip_start', 'trip_end'
    ]
    
    transactions['trip_duration'] = (transactions['trip_end'] - transactions['trip_start']).dt.days + 1
    return transactions

def visualize_outlier_detection(filtered_data, outliers_mask):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0,0].scatter(filtered_data['REVISIT_INTENTION'], filtered_data['DGSTFN'], 
                     c=outliers_mask, cmap='RdYlBu', alpha=0.6)
    axes[0,0].set_title('Satisfaction vs Revisit Intention')
    
    axes[0,1].scatter(filtered_data['RCMDTN_INTENTION'], filtered_data['DGSTFN'],
                     c=outliers_mask, cmap='RdYlBu', alpha=0.6)
    axes[0,1].set_title('Satisfaction vs Recommendation Intention')
    
    axes[1,0].scatter(filtered_data['REVISIT_INTENTION'], filtered_data['RCMDTN_INTENTION'],
                     c=outliers_mask, cmap='RdYlBu', alpha=0.6)
    axes[1,0].set_title('Revisit vs Recommendation Intention')
    
    axes[1,1].hist(outliers_mask.astype(int), bins=2, alpha=0.7)
    axes[1,1].set_title('Outlier Distribution')

    plt.tight_layout()
    plt.savefig('enhanced_outlier_analysis.png')

if __name__ == "__main__":
    # Load data
    raw_data = load_data('tn_visit_area_info_east.csv')

    # Outlier detection
    outliers = detect_outliers_improved(raw_data)

    # Save original data with outliers
    transactions_with_outliers = create_transactions(raw_data)
    transactions_with_outliers.to_csv('transactions_with_outliers.csv', index=False)

    # Save cleaned data without outliers
    cleaned_transactions = create_transactions(outliers['cleaned_data'])
    cleaned_transactions.to_csv('transactions_cleaned.csv', index=False)
    
    # Save full cleaned dataset
    outliers['cleaned_data'].to_csv('cleaned_dataset.csv', index=False)

    # Generate visualizations
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=raw_data['DGSTFN'].dropna())
    plt.title('Satisfaction Distribution')
    
    plt.subplot(1, 2, 2)
    sns.histplot(raw_data['DGSTFN'].dropna(), kde=True)
    plt.title('Satisfaction Histogram')
    plt.savefig('satisfaction_distribution.png')

    # Enhanced visualization
    visualize_outlier_detection(outliers['filtered_data'], outliers['combined_mask'])

    # Save results
    with open('outliers.json', 'w') as f:
        json.dump({
            'statistical_outliers': int(outliers['statistical']),
            'isolation_outliers': int(outliers['isolation']),
            'combined_outliers': int(outliers['combined'])
        }, f)

    print("Analysis complete. Check output files:")
    print("- transactions_with_outliers.csv (original data)")
    print("- transactions_cleaned.csv (outliers removed)")
    print("- cleaned_dataset.csv (full cleaned dataset)")
    print("- *.png visualizations")
