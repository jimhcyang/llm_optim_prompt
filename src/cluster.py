import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.manifold import TSNE
import random

# Global parameters
NUM_CLUSTERS = 5  # Adjustable number of clusters
RANDOM_STATE = 42  # For reproducibility
VISUALIZATION = True  # Set to True to visualize clusters

def load_data(file_path):
    """Load data from CSV file"""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} query pairs")
    return df

def preprocess_data(df):
    """Preprocess the data for clustering"""

    df['combined'] = df['natural_language_query'] + ' ' + df['sql_query']
    
    # Convert text to numerical features using TF-IDF with the SQL query only
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    features = vectorizer.fit_transform(df['sql_query'])
    
    # Reduce dimensionality for visualization and potentially better clustering
    pca = PCA(n_components=min(10, features.shape[1]))
    scaled_features = StandardScaler().fit_transform(features.toarray())
    reduced_features = pca.fit_transform(scaled_features)
    
    print(f"Reduced features shape: {reduced_features.shape}")
    return reduced_features, vectorizer

def apply_kmeans(features, n_clusters):
    """Apply K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(features)
    score = silhouette_score(features, clusters) if n_clusters > 1 else 0
    print(f"K-means clustering completed with silhouette score: {score:.4f}")
    return clusters, score

def apply_hierarchical(features, n_clusters):
    """Apply Hierarchical clustering"""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = hierarchical.fit_predict(features)
    score = silhouette_score(features, clusters) if n_clusters > 1 else 0
    print(f"Hierarchical clustering completed with silhouette score: {score:.4f}")
    return clusters, score

def apply_dbscan(features):
    """Apply DBSCAN clustering"""
    # Automatically determine eps parameter based on data
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)
    distances = np.sort(distances[:, 4])  # Distance to 5th nearest neighbor
    
    # Use the "elbow" in the distance graph to estimate epsilon
    knee_point = np.diff(distances).argmax() + 1
    eps = distances[knee_point]
    
    # Apply DBSCAN with estimated eps
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(features)
    
    # Handle potential noise points (cluster -1)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"DBSCAN found {n_clusters} clusters with eps={eps:.4f}")
    
    # Assign noise points to the nearest cluster
    if -1 in clusters:
        noise_points = features[clusters == -1]
        if len(noise_points) > 0 and n_clusters > 0:
            non_noise_clusters = list(set(clusters) - {-1})
            for i, point_idx in enumerate(np.where(clusters == -1)[0]):
                # Find closest cluster for each noise point
                distances = []
                for cluster_id in non_noise_clusters:
                    cluster_points = features[clusters == cluster_id]
                    min_dist = np.min(np.linalg.norm(cluster_points - features[point_idx].reshape(1, -1), axis=1))
                    distances.append(min_dist)
                closest_cluster = non_noise_clusters[np.argmin(distances)]
                clusters[point_idx] = closest_cluster
    
    # Recalculate number of clusters
    n_clusters = len(set(clusters))
    
    # Calculate silhouette score if more than one cluster
    if n_clusters > 1:
        score = silhouette_score(features, clusters)
        print(f"DBSCAN clustering completed with silhouette score: {score:.4f}")
    else:
        score = 0
        print("DBSCAN found only one cluster, silhouette score not applicable")
    
    return clusters, score, n_clusters

def get_cluster_representatives(df, clusters, method_name):
    """Get representative samples from each cluster"""
    df_with_clusters = df.copy()
    df_with_clusters[f'{method_name}_cluster'] = clusters
    
    representatives = []
    cluster_counts = {}
    
    for cluster_id in sorted(set(clusters)):
        cluster_df = df_with_clusters[df_with_clusters[f'{method_name}_cluster'] == cluster_id]
        cluster_counts[cluster_id] = len(cluster_df)
        
        # Select a representative sample (center point of the cluster)
        # For simplicity, we'll randomly select a point from the cluster
        # In a more sophisticated approach, you could select the point closest to the centroid
        representative = cluster_df.sample(1, random_state=RANDOM_STATE)
        representatives.append(representative)
    
    representatives_df = pd.concat(representatives).reset_index(drop=True)
    return representatives_df, cluster_counts

def visualize_clusters(features, clusters, method_name, folder, display_vis=True):
    """Visualize clusters using t-SNE"""
    if not display_vis:
        return
        
    # Use t-SNE for better visualization in 2D
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Clusters using {method_name}')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.savefig(folder+"/"+f'{method_name}_clusters.png')
    print(f"Saved visualization to {method_name}_clusters.png")

def cluster(file = 'src/split_data/test.csv', path=True, n_clusters=NUM_CLUSTERS, display_vis=VISUALIZATION):    
    if not path:
        df = file
    else:
        df = load_data(file)
    
    # Preprocess data
    features, vectorizer = preprocess_data(df)
    
    # Apply clustering algorithms
    print("\n1. K-means clustering:")
    kmeans_clusters, kmeans_score = apply_kmeans(features, n_clusters)
    
    print("\n2. Hierarchical clustering:")
    hierarchical_clusters, hierarchical_score = apply_hierarchical(features, n_clusters)
    
    print("\n3. DBSCAN clustering:")
    dbscan_clusters, dbscan_score, actual_dbscan_clusters = apply_dbscan(features)
        
    # Get and save representatives
    kmeans_representatives, kmeans_counts = get_cluster_representatives(df, kmeans_clusters, 'kmeans')
    hierarchical_representatives, hierarchical_counts = get_cluster_representatives(df, hierarchical_clusters, 'hierarchical')
    dbscan_representatives, dbscan_counts = get_cluster_representatives(df, dbscan_clusters, 'dbscan')
    
    # Print cluster statistics
    print("\nCluster sizes:")
    print("K-means:", kmeans_counts)
    print("Hierarchical:", hierarchical_counts)
    print("DBSCAN:", dbscan_counts)
        
    # Return the best performing method based on silhouette score
    scores = {
        'K-means': kmeans_score,
        'Hierarchical': hierarchical_score,
        'DBSCAN': dbscan_score
    }
    best_method = max(scores, key=scores.get)
    print(f"\nBest performing method: {best_method} with silhouette score: {scores[best_method]:.4f}")
    
    # Save combined representatives from the best method
    if best_method == 'K-means':
        best_representatives = kmeans_representatives
    elif best_method == 'Hierarchical':
        best_representatives = hierarchical_representatives
    else:
        best_representatives = dbscan_representatives
    output_dir = 'cluster_data'
    os.makedirs(output_dir, exist_ok=True)

    # Save the split datasets to CSV files
    kmeans_path = os.path.join(output_dir, 'kmeans_representatives.csv')
    hierachical_path = os.path.join(output_dir, 'hierarchical_representatives.csv')
    dbscan_path = os.path.join(output_dir, 'dbscan_representatives.csv')
    best_path = os.path.join(output_dir, 'best_representatives.csv')

    # Save representative dataframes
    kmeans_representatives.to_csv(kmeans_path, index=False)
    hierarchical_representatives.to_csv(hierachical_path, index=False)
    dbscan_representatives.to_csv(dbscan_path, index=False)
    best_representatives.to_csv(best_path, index=False)
    print("\nRepresentative samples saved to CSV files.")
    # Visualize clusters
    visualize_clusters(features, kmeans_clusters, 'KMeans', output_dir, display_vis)
    visualize_clusters(features, hierarchical_clusters, 'Hierarchical', output_dir, display_vis)
    visualize_clusters(features, dbscan_clusters, 'DBSCAN', output_dir, display_vis)
    
    return best_representatives
