import scipy.spatial.distance
from gap_statistic import OptimalK
from kmodes.kmodes import KModes
from sklearn.neighbors import NearestCentroid
from s_dbw import SD
from c_index import (calc_cindex_clusterSim_implementation, pdist_array)
from scipy.spatial.distance import pdist, cdist

from import_export_format_data import export_scores_plt, export_gap_stat_plt
from perform_clustering import *
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    Birch,
    SpectralClustering,
)
from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from visualize_data import visualize_scores, visualize_gap_stat_score

'--------------------------------- Information about perform scoring -----------------------------------------------'
'''
This module will calculate, visualize and exports the optimal number of scores for all indexes.
In this module additional indexes can be added. 
'''
'----------------------------------------------------------------------------------------------------------------------'


def perform_scoring(formatted_data, cluster_method, range_of_clusters_for_score, save_path):
    """
    This function will execute:
     1. The calculation of the optimal number of clusters
     2. The visualizations of each index
     3. The export of these plots
     4. Print the status on the console
    """
    score_names = ['Calinski-Harabasz Score', 'Davies-Bouldin Score', 'Silhouette Score',
                   'Dunn Index', 'Hartigan Index', 'Krzanowski-Lai Index',
                   'Hubert and Levin Index', 'McClain-Rao Index', 'Milligan Index', 'Ball-Hall Index',
                   'Halkidi et al. Index', 'Elbow Method Index']

    if cluster_method not in ['affinity_propagation', 'mean_shift', 'optics', 'som', 'hdbscan', 'dbscan']:

        # List of cluster numbers to evaluate
        scores = []

        # Calculate the optimal number of clusters for all scores except gap statistic.
        for score_name in score_names:
            score_list = calculate_scores(formatted_data, cluster_method, range_of_clusters_for_score, score_name)
            # Visualize scores
            if score_list is not None and any(score is not None for score in score_list):
                scores = visualize_scores(range_of_clusters_for_score, score_list, score_name, scores)

        # Export the plt of each score except gap statistic score
        export_scores_plt(scores, save_path)

        # Calculate, Visualize and Export gap statistic
        optimal_clusters_gap_stat = 0
        if cluster_method in ['kmeans', 'kmedoids', 'mean_shift']:
            optimal_clusters_gap_stat, optimalK = calc_gap_stat_score(cluster_method, formatted_data,
                                                                      range_of_clusters_for_score)
            gap_stat_plt = visualize_gap_stat_score(cluster_method, optimal_clusters_gap_stat, optimalK)
            export_gap_stat_plt(gap_stat_plt, save_path)

        # Print the highest score and corresponding clusters.
        print_score_clusters(scores, cluster_method, optimal_clusters_gap_stat)

    return


def calculate_scores(formatted_data, cluster_method, range_of_clusters_for_score, score_name):
    """
     This function will calculate the score of each index and return a list with the scores
     """
    # Ensure data is 2-dimensional
    data_2d = np.atleast_2d(formatted_data)

    score_list = []
    # Iterate over different numbers of clusters
    for i in range_of_clusters_for_score:
        if cluster_method == 'agglomerative':
            clustering = AgglomerativeClustering(n_clusters=i)
        elif cluster_method == 'birch':
            clustering = Birch(n_clusters=i, threshold=0.1)
        elif cluster_method == 'ward':
            clustering = AgglomerativeClustering(n_clusters=i, linkage='ward')
        elif cluster_method == 'kmeans':
            clustering = KMeans(n_clusters=i, random_state=30, n_init='auto')
        elif cluster_method == 'kmodes':
            clustering = KModes(n_clusters=i, init="random", n_init=5, verbose=1)
        elif cluster_method == 'kmedoids':
            clustering = KMedoids(n_clusters=i)
        elif cluster_method == 'gmm':
            clustering = GaussianMixture(n_components=i)
        elif cluster_method == 'spectral':
            clustering = SpectralClustering(n_clusters=i)
        else:
            print('')
            print('No score will be calculated for this cluster method')
            return

        # Calculate Cluster center and labels
        clustering.fit(formatted_data)

        # Returns Cluster to the Data
        cluster_labels = clustering.fit_predict(formatted_data)

        if score_name == 'Calinski-Harabasz Score':
            score = calinski_harabasz_score(formatted_data, cluster_labels)
        elif score_name == 'Davies-Bouldin Score':
            score = davies_bouldin_score(formatted_data, cluster_labels)
        elif score_name == 'Silhouette Score':
            score = silhouette_score(formatted_data, cluster_labels)
        elif score_name == 'Dunn Index':
            score = calculate_dunn_index(data_2d, cluster_labels)
        elif score_name == 'Hartigan Index':
            score = calculate_hartigan_index(formatted_data, cluster_labels)
        elif score_name == 'Krzanowski-Lai Index':
            score = calculate_krzanowski_lai_index(formatted_data, cluster_labels, i)
        elif score_name == 'Hubert and Levin Index':
            score = calculate_hubert_index(formatted_data, cluster_labels)
        elif score_name == 'McClain-Rao Index':
            score = calculate_mcclain_rao_index(formatted_data, cluster_labels)
        elif score_name == 'Milligan Index':
            score = calculate_milligan_index(formatted_data, cluster_labels)
        elif score_name == 'Ball-Hall Index':
            score = calculate_ball_hall_index(formatted_data, cluster_labels, i)
        elif score_name == 'Elbow Method Index':
            score = calculate_elbow_method(formatted_data, cluster_method, i, cluster_labels)
        elif score_name == 'Halkidi et al. Index':
            score = SD(formatted_data, cluster_labels, k=1.0, centers_id=None, alg_noise='bind', centr='mean',
                       nearest_centr=True, metric='euclidean')

        # creates a list of scores
        score_list.append(score)

    return score_list


def print_score_clusters(scores, cluster_method, optimal_clusters_gap_stat):
    """
     This function will print the results on the console
     """
    if scores is not None:
        print('')
        print('-------- Score Results --------')
        for fig, score_name in scores:
            if score_name in ['Elbow Method Index', 'Frey and van Groenewoud Index', 'Gap Statistic Index']:
                print(f"{score_name} is saved as Plot and can be investigated")
            else:
                # Search and print the highest score of each index
                score_list = fig.get_axes()[0].lines[0].get_ydata()

                # Filter out None values from score_list
                filtered_scores = [score for score in score_list if score is not None]

                if len(filtered_scores) == 0:
                    continue

                if score_name in ['Davies-Bouldin Score', 'Halkidi et al. Index', 'Hubert and Levin Index',
                                  'Ball-Hall Index']:
                    # Find the index of the smallest score
                    best_score_index = np.argmin(filtered_scores)
                else:
                    # Find the index of the highest score
                    best_score_index = np.argmax(filtered_scores)

                best_num_clusters = 2 + best_score_index
                print(f"Best {score_name} achieved with {best_num_clusters} clusters. "
                      f"See the Plot for further investigation.")

        if cluster_method in ['kmeans', 'kmedoids', 'mean_shift']:
            # Call gaps statistic function because it was implemented separately
            print(f"Best Gap Statistic Index achieved with {optimal_clusters_gap_stat} clusters. "
                  f"See the Plot for further investigation.")

        print('-------------------------------')
        print('')
        print('Please see the plots ')


def calculate_dunn_index(data, cluster_labels):
    # Calculate cluster centers
    cluster_centers = []
    for i in np.unique(cluster_labels):
        cluster_centers.append(np.mean(data[cluster_labels == i], axis=0))

    # Calculate maximum cluster diameter
    max_diameter = 0
    for i in range(len(cluster_centers)):
        for j in range(i + 1, len(cluster_centers)):
            diameter = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            if diameter > max_diameter:
                max_diameter = diameter

    # Calculate minimum distance between clusters
    min_distance = np.inf
    for i in range(len(cluster_centers)):
        for j in range(i + 1, len(cluster_centers)):
            distance = np.min(cdist(data[cluster_labels == i], data[cluster_labels == j], metric='euclidean'))
            if distance < min_distance:
                min_distance = distance

    # Calculate Dunn Index
    dunn_index = min_distance / max_diameter
    return dunn_index


def calculate_hartigan_index(data, labels):
    # Calculate the sum of within-cluster sum of squares (WSS)
    wss = 0
    for i, label in enumerate(np.unique(labels)):
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        wss += np.sum(np.square(cdist(cluster_points, [centroid])))

    # Calculate the sum of between-cluster sum of squares (BSS)
    bss = np.sum(np.square(cdist(data, [np.mean(data, axis=0)]))) - wss

    # Calculate the Hartigan Index
    hartigan_index = np.log(wss / bss)

    return hartigan_index


def calculate_krzanowski_lai_index(data, labels, num_clusters):
    # Calculate the sum of squared distances within each cluster
    wss = 0
    for i, label in enumerate(np.unique(labels)):
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        wss += np.sum(np.square(cdist(cluster_points, [centroid])))

    # Calculate the sum of squared distances between cluster centroids
    centroids = []
    for i, label in enumerate(np.unique(labels)):
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    bss = np.sum(np.square(cdist(centroids, [np.mean(centroids, axis=0)])))

    # Calculate the Krzanowski-Lai Index
    kl_index = (wss / bss) * (len(data) - num_clusters) / (num_clusters - 1)

    return kl_index


def calculate_hubert_index(data, labels):
    # nbclust implementation takes an array of pointwise differences
    distances_array = pdist_array(data)

    # clusterSim C Index python equivalent
    cindex = calc_cindex_clusterSim_implementation(distances_array, labels)

    return cindex


def calculate_mcclain_rao_index(data, labels):
    # Calculate the sum of squared distances between all pairs of objects
    sum_distances = np.sum(pdist(data) ** 2)

    # Calculate the sum of squared distances within each cluster
    sum_within_cluster_distances = 0
    for label in np.unique(labels):
        cluster_points = data[labels == label]
        sum_within_cluster_distances += np.sum(pdist(cluster_points) ** 2)

    # Calculate the McClain and Rao Index
    mcclain_rao_index = sum_within_cluster_distances / sum_distances

    return mcclain_rao_index


def calculate_milligan_index(data, labels):
    # Calculate the sum of squared distances between all pairs of objects
    sum_distances = np.sum(pdist(data) ** 2)

    # Calculate the sum of squared distances within each cluster
    sum_within_cluster_distances = 0
    for label in np.unique(labels):
        cluster_points = data[labels == label]
        sum_within_cluster_distances += np.sum(pdist(cluster_points) ** 2)

    # Calculate the sum of squared distances between pairs of objects in different clusters
    sum_between_cluster_distances = sum_distances - sum_within_cluster_distances

    # Calculate the number of pairs of objects in different clusters
    num_pairs_between_clusters = len(data) * (len(data) - 1) / 2 - np.sum(np.unique(labels, return_counts=True)[1] ** 2)

    # Calculate the Milligan Index
    milligan_index = sum_between_cluster_distances / num_pairs_between_clusters

    return milligan_index


def calculate_ball_hall_index(data, labels, num_clusters):
    # Calculate the sum of squares within the clusters
    sum_squares_within_clusters = np.sum(
        [np.sum(np.square(pdist(data[labels == label]))) for label in np.unique(labels)])

    # Calculate the Ball Index
    ball_index = sum_squares_within_clusters / num_clusters

    return ball_index


def calculate_elbow_method(data, cluster_method, i, cluster_labels):
    if cluster_method == 'kmeans':
        kmean_model = KMeans(n_clusters=i, random_state=30, n_init='auto')
        kmean_model.fit_predict(data)
        distortion = kmean_model.inertia_
        return distortion
    elif cluster_method == 'kmedoids':
        kmedoids_model = KMedoids(n_clusters=i)
        kmedoids_model.fit_predict(data)
        distortion = kmedoids_model.inertia_
        return distortion
    elif cluster_method == 'kmodes':
        kmodes_model = KModes(n_clusters=i, init="random", n_init=5, verbose=1)
        kmodes_model.fit_predict(data)
        distortion = kmodes_model.cost_
        return distortion
    else:
        if len(np.unique(cluster_labels)) == i:
            # Calculate cluster centers
            cluster_centers = []
            for label in np.unique(cluster_labels):
                cluster_centers.append(np.mean(data[cluster_labels == label], axis=0))
            # Calculate the Sum of Squared Errors
            distortion = 0
            for j in range(i):
                cluster_center = np.array(cluster_centers[j])
                cluster_data = data[cluster_labels == j]
                squared_distances = np.sum(np.square(cluster_data - cluster_center), axis=1)
                distortion += np.sum(squared_distances)
            return distortion
        else:
            return None


def special_clustering_func_kmeans(formatted_data, k):
    m = KMeans(random_state=30, n_init='auto')
    m.fit(formatted_data)

    # Return the location of each cluster center,
    # and the labels for each point.
    return m.cluster_centers_, m.predict(formatted_data)


def special_clustering_func_kmedoids(formatted_data, k):
    m = KMedoids()
    m.fit(formatted_data)

    # Return the location of each cluster center,
    # and the labels for each point.
    return m.cluster_centers_, m.predict(formatted_data)


def special_clustering_func_meanshift(formatted_data, k):
    m = MeanShift()
    m.fit(formatted_data)

    # Return the location of each cluster center,
    # and the labels for each point.
    return m.cluster_centers_, m.predict(formatted_data)


def calc_gap_stat_score(cluster_method, formatted_data, range_of_clusters_for_score):
    optimalK = None
    if cluster_method in ['kmeans', 'kmedoids', 'mean_shift']:
        if cluster_method == 'kmeans':
            optimalK = OptimalK(clusterer=special_clustering_func_kmeans, parallel_backend='joblib')
        elif cluster_method == 'kmedoids':
            optimalK = OptimalK(clusterer=special_clustering_func_kmedoids, parallel_backend='joblib')
        elif cluster_method == 'mean_shift':
            optimalK = OptimalK(clusterer=special_clustering_func_meanshift, parallel_backend='joblib')
    if optimalK is not None:
        # Use the callable instance as normal.
        n_clusters = optimalK(formatted_data, n_refs=3, cluster_array=range_of_clusters_for_score)
        return n_clusters, optimalK
    else:
        # Handle the case where optimalk is not assigned
        return None
