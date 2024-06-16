import pandas as pd
from gap_statistic import optimalK, OptimalK
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift
from sklearn_extra.cluster import KMedoids

'---------------------------------------- Information about visualize data --------------------------------------------'
'''
This module will visualize the results of the scoring and clustering.
'''
'----------------------------------------------------------------------------------------------------------------------'


def visualize_clustering_scatter(formatted_data, raw_data, cluster_labels, cluster_method, data_format):
    """
        This function creates a scatter plot
        """
    # Delete ID row from raw data set
    raw_data = raw_data.iloc[:, 1:]

    # Create Pandas data set for boolean data
    if data_format == 'bool':
        formatted_data = pd.DataFrame(formatted_data, columns=raw_data.iloc[0])

    # Create Pandas data set for numerical data
    else:
        formatted_data = pd.DataFrame(formatted_data, columns=raw_data.columns)

    # Visualization
    if formatted_data.shape[1] == 1:  # Check if data has 1 feature for 1D scatter plot
        fig_scatter, ax = plt.subplots()
        scatter = ax.scatter(range(len(formatted_data)), formatted_data.iloc[:, 0], c=cluster_labels)
        ax.set_title('Clustering Visualization: ' + cluster_method)
        ax.set_xlabel(formatted_data.columns[0])
        ax.set_ylabel(formatted_data.columns[0])  # corrected indexing for ylabel
        # Create a legend for the cluster cluster_labels
        legend = ax.legend(*scatter.legend_elements(),
                           title="Number of Clusters",
                           loc="upper right", bbox_to_anchor=(1.4, 1))
        ax.add_artist(legend)
        fig_scatter = plt.gcf()

    elif formatted_data.shape[1] == 2:  # Check if data has 2 features for 2D scatter plot
        fig_scatter, ax = plt.subplots()  # added fig, ax
        scatter = ax.scatter(formatted_data.iloc[:, 0], formatted_data.iloc[:, 1], c=cluster_labels)
        ax.set_title('Clustering Visualization: ' + cluster_method)
        ax.set_xlabel(formatted_data.columns[0])
        ax.set_ylabel(formatted_data.columns[1])
        # Create a legend for the cluster cluster_labels
        legend = ax.legend(*scatter.legend_elements(),
                           title="Number of Clusters",
                           loc="upper right", bbox_to_anchor=(1.4, 1))
        ax.add_artist(legend)
        fig_scatter = plt.gcf()

    elif formatted_data.shape[1] == 3:  # Check if data has at least 3 features for 3D scatter plot
        fig_scatter = plt.figure()
        ax = fig_scatter.add_subplot(111, projection='3d')
        scatter = ax.scatter(formatted_data.iloc[:, 0], formatted_data.iloc[:, 1],
                             formatted_data.iloc[:, 2], c=cluster_labels)
        ax.set_title('Clustering Visualization: ' + cluster_method)
        ax.set_xlabel(formatted_data.columns[0])
        ax.set_ylabel(formatted_data.columns[1])
        ax.set_zlabel(formatted_data.columns[2])
        # Create a legend for the cluster cluster_labels
        legend = ax.legend(*scatter.legend_elements(),
                           title="Number of Clusters",
                           loc="upper right", bbox_to_anchor=(1.4, 1))
        ax.add_artist(legend)
        fig_scatter = plt.gcf()

    else:
        print('Dimension is', formatted_data.shape[1], ' --> A scatter plot can only be used until dimension 3')
        return

    return fig_scatter


def visualize_clustering_dendrogram(formatted_data, cluster_method, num_clusters):
    """
    This function creates a dendrogram
    """
    # Visualization
    if cluster_method in ['agglomerative', 'ward', 'birch', 'hdbscan']:
        # Compute the linkage matrix
        linkage_matrix = linkage(formatted_data, method='ward')

        # Plot the dendrogram
        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.tight_layout()
        fig_dendrogram = plt.gcf()

        # Add legend for the number of clusters
        if cluster_method not in ['hdbscan']:
            plt.axhline(y=num_clusters, color='r', linestyle='--', label='Number of Clusters')
            plt.legend()

        return fig_dendrogram
    else:
        return None


def visualize_scores(range_of_clusters_for_score, score_list, score_name, scores):
    if score_list is not None:
        fig, ax = plt.subplots()
        ax.plot(range_of_clusters_for_score, score_list)
        ax.set_xlabel('Number of Clusters')
        if score_name != 'Elbow Method Index':
            ax.set_ylabel(score_name)
        else:
            ax.set_ylabel('Distortions')

        ax.set_title(f'{score_name} for Different Numbers of Clusters')

        scores.append((fig, score_name))  # Tuple of Figure and score name
    else:
        scores.append((None, score_name))  # Append (None, score_name) when score_list is None

    return scores


# The following functions will only be needed for the visualization of Gap Statistic

def visualize_gap_stat_score(cluster_method, n_clusters, optimalK):

    # Define the OptimalK instance, but pass in our own clustering function
    if cluster_method in ['kmeans', 'kmedoids', 'mean_shift']:

        plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
        plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
                    optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
        plt.grid(True)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Gap Value')
        plt.title('Gap Statistic Index')
        fig_gap_stat = plt.gcf()
        return fig_gap_stat


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