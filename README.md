# TravelTide - Loyalty Program Developing

## Overview

This repository contains a data analytics project aimed at segmenting TravelTide users based on their booking behaviors and demographic characteristics. The project involves data extraction, preprocessing, and clustering techniques to provide insights for personalizing loyalty perks. The analysis was conducted using Python and several data science libraries, focusing on leveraging machine learning for user segmentation.

## Project Structure

- **Data Source**: PostgreSQL database with multiple tables, including user profiles, hotel bookings, flight bookings, and user sessions.
- **Clustering Model**: K-Means clustering algorithm was used to segment users into distinct clusters based on their behavior and demographics.
- **Evaluation**: The model performance was evaluated using metrics like the Silhouette Score to assess the quality of the clustering.

## Data Pipeline

### 1. Data Extraction
The data was extracted from a PostgreSQL database using SQLAlchemy for database connection and querying:

```python
import sqlalchemy as sa
import pandas as pd

# Database connection
engine = sa.create_engine("postgresql://username:password@host/database")
connection = engine.connect()

# Data extraction
users = pd.read_sql_table('users', connection)
hotels = pd.read_sql_table('hotels', connection)
flights = pd.read_sql_table('flights', connection)
sessions = pd.read_sql_table('sessions', connection)
```

### 2. Data Preprocessing
Data preprocessing steps included cleaning, handling missing values, and scaling numerical features:

```python
from sklearn.preprocessing import StandardScaler

# Example: Scaling numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['feature1', 'feature2', 'feature3']])
```

### 3. Dimensionality Reduction
Principal Component Analysis (PCA) was employed to reduce the dimensionality of the data and prepare it for clustering:

```python
from sklearn.decomposition import PCA

# Applying PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
```

### 4. Clustering
The K-Means clustering algorithm was applied to segment users into distinct clusters based on their behavior and demographics:

```python
from sklearn.cluster import KMeans

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(pca_data)
```

### 5. Evaluation
Model performance was evaluated using the Silhouette Score, which assesses how well the data points fit within their assigned clusters:

```python
from sklearn.metrics import silhouette_score

# Calculate Silhouette Score
score = silhouette_score(pca_data, clusters)
print(f'Silhouette Score: {score}')
```

## Results

### Clustering Insights
The analysis revealed four distinct clusters of users based on their booking behavior and demographics:

- **Cluster 0**: High-frequency travelers who prefer perks like free checked baggage.
- **Cluster 1**: Low-engagement users who value booking flexibility (e.g., no cancellation fees).
- **Cluster 2**: Moderately engaged users who respond well to combined travel offers, such as "1 Night Free Hotel with Flight."
- **Cluster 3**: Price-sensitive users who are drawn to exclusive discounts.

### Evaluation Metrics
- **Silhouette Score**: The clustering yielded a Silhouette Score of 0.175, indicating that the clustering captures meaningful patterns despite the complexity of user behavior.

## Key Libraries

- **Pandas**: Used for data manipulation and analysis.
- **SQLAlchemy**: Used for database connection and querying.
- **Scikit-learn**: Used for machine learning algorithms, including K-Means clustering, PCA, and evaluation metrics.
- **Matplotlib/Seaborn**: Used for data visualization.

## Conclusion and Recommendations

The clustering analysis provided valuable insights into user behavior that can be leveraged to personalize the TravelTide loyalty program. By tailoring perks to specific user segments, the platform can increase engagement and improve customer retention. Future work includes continuous monitoring of user preferences and refining the clustering model to adapt to changing behaviors.
