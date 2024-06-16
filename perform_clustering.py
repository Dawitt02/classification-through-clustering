from kmodes.kmodes import KModes
from sklearn.cluster import HDBSCAN
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    SpectralClustering,
    MeanShift,
    estimate_bandwidth,
)
from minisom import MiniSom
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS

from import_export_format_data import export_scatter_plot, export_dendrogram, export_data_csv
from visualize_data import visualize_clustering_scatter, visualize_clustering_dendrogram

'------------------------------------ Information about perform clustering --------------------------------------------'
'''
This module will calculate, visualize and exports the results of the clustering.
In this module additional cluster methods can be added.
'''
'----------------------------------------------------------------------------------------------------------------------'


# This functions performs the clustering of the selected Method.
def perform_clustering(formatted_data, cluster_method, raw_data, save_path, data_format):
    """
        This function will execute:
         1. The calculation of the cluster label for each data point
         2. The visualizations of cluster
         3. The export of these plots
        """

    if cluster_method not in ['affinity_propagation', 'mean_shift', 'optics', 'som', 'hdbscan', 'dbscan']:
        # Selection of the desired number of clusters
        num_clusters = int(input("Enter the desired number of clusters:"))
        print('')
    else:
        num_clusters = None

    # Calculate the labels for the clustering
    cluster_labels = calc_clustering(formatted_data, cluster_method, num_clusters, data_format)

    # Visualize and export cluster result for scatter plot
    fig_scatter = visualize_clustering_scatter(formatted_data, raw_data, cluster_labels, cluster_method, data_format)
    export_scatter_plot(fig_scatter, save_path, cluster_method)

    # Visualize and export cluster result for dendrogram (only for hierarchical clustering)
    fig_dendrogram = visualize_clustering_dendrogram(formatted_data, cluster_method, num_clusters)
    export_dendrogram(fig_dendrogram, save_path, cluster_method)

    # Export the final CSV with added Labels
    export_data_csv(raw_data, cluster_labels, save_path, cluster_method)


# Function to calculate clustering labels
def calc_clustering(formatted_data, cluster_method, num_clusters, data_format):
    """
     This function will calculate the cluster label for each data point und return a list with the cluster labels
     """
    # Select the appropriate clustering algorithm based on the input
    if cluster_method == 'affinity_propagation':
        clustering = AffinityPropagation()
    elif cluster_method == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=num_clusters)
    elif cluster_method == 'birch':
        clustering = Birch(n_clusters=num_clusters, threshold=0.1)
    elif cluster_method == 'ward':
        clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    elif cluster_method == 'hdbscan':
        clustering = HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, max_cluster_size=None,
                             metric='euclidean', metric_params=None, alpha=1.0, algorithm='auto', leaf_size=40,
                             n_jobs=None, cluster_selection_method='eom', allow_single_cluster=False,
                             store_centers=None, copy=False)
    elif cluster_method == 'dbscan':
        clustering = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto',
                            leaf_size=30, p=None, n_jobs=None)
    elif cluster_method == 'gmm':
        clustering = GaussianMixture(n_components=num_clusters)
    elif cluster_method == 'mean_shift':
        bandwidth = estimate_bandwidth(formatted_data, quantile=0.2, n_samples=formatted_data.shape[0])
        clustering = MeanShift(bandwidth=bandwidth)
    elif cluster_method == 'optics':
        clustering = OPTICS(min_samples=5)
    elif cluster_method == 'kmeans':
        clustering = KMeans(n_clusters=num_clusters, random_state=30, n_init='auto')
    elif cluster_method == 'kmedoids':
        clustering = KMedoids(n_clusters=num_clusters)
    elif cluster_method == 'spectral':
        clustering = SpectralClustering(n_clusters=num_clusters)
    elif cluster_method == 'som':
        clustering = calc_som_clustering(formatted_data, data_format)
    elif cluster_method == 'kmodes':
        clustering = KModes(n_clusters=num_clusters, init='Huang', n_init=5, verbose=1)

    # Fit Data to clustering
    if cluster_method == 'som':
        cluster_labels = clustering
    else:
        cluster_labels = clustering.fit_predict(formatted_data)
    return cluster_labels


def calc_som_clustering(formatted_data, data_format):
    if data_format == 'bool':
        # Initialize the SOM clustering algorithm with the specified parameters
        som = MiniSom(7, 7, formatted_data.shape[1], sigma=0.1, learning_rate=0.1)
        som.random_weights_init(formatted_data)
        som.train_random(formatted_data, 100)

        # Compute cluster assignments for each data point
        labels = []
        for i in range(formatted_data.shape[0]):
            x = formatted_data[i]
            winner = som.winner(x)
            labels.append(winner[0])

        return labels
    else:
        # Initialize the SOM clustering algorithm with the specified parameters
        som = MiniSom(7, 7, formatted_data.shape[1], sigma=0.5, learning_rate=0.5)
        som.random_weights_init(formatted_data)
        som.train_random(formatted_data, 100)

        # Compute cluster assignments for each data point
        labels = []
        for i in range(formatted_data.shape[0]):
            x = formatted_data[i]
            winner = som.winner(x)
            labels.append(winner[0])

        return labels
