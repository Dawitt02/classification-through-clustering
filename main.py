from perform_scoring import *
from perform_clustering import *
from program_assistant import *

'--------------------------------- General Information ----------------------------------------------------------------'
'''
This program is designed to determine the optimal number of clusters from a given or random data set and
perform clustering using a selected method.
The file is structured in such a way that you can set all the parameters necessary for the functionality in
the main window.
Moreover, the structure is such
 that the program can be extended with the desired algorithms/scores. 
In order to get the program running, you have to set the following parameters and run this module:
'''

'--------------------------------- Set your parameters ----------------------------------------------------------------'

# --- Read data from the CSV file. Specify the path to the data.---
# Example: source_path = '/Users/dawittvoss/Library/Mobile Documents/com~apple~CloudDocs/David/SC2Test/rawdata_test.csv'
# source_path = '/Users/dawittvoss/Documents/UNI/Masterarbeit/Masterarbeit_Latex/Python/Data_set_47_items.csv'
source_path = '/Users/dawittvoss/Documents/UNI/Masterarbeit/Masterarbeit_Latex/Umfrage/Python_csv.csv'

# --- Where should the results be saved? Specify your desired path. ---
# Example: save_path = '/Users/dawittvoss/Library/Mobile Documents/com~apple~CloudDocs/David/SC2Test'
save_path = '/Users/dawittvoss/Library/Mobile Documents/com~apple~CloudDocs/Johanna & David/SC2Test'

# --- Do you want to use test data? Specify 'yes' or 'no'. ---
test_data = 'no'

# --- Do you use boolean or numeric data? Specify 'bool' or 'num'. ---
data_format = 'num'

# --- For which range of clusters do you want the get a score? ---
range_of_clusters_for_score = list(range(2, 9))

# ---- Please select one clustering method. ----
# cluster_method = 'affinity_propagation'
# cluster_method = 'agglomerative'
# cluster_method = 'birch'
# cluster_method = 'ward'
# cluster_method = 'hdbscan'
# cluster_method = 'dbscan'
# cluster_method = 'gmm'
# cluster_method = 'mean_shift'
# cluster_method = 'optics'
cluster_method = 'kmeans'
# cluster_method = 'kmodes'
# cluster_method = 'kmedoids'
# cluster_method = 'spectral'
# cluster_method = 'som'

'--------------------------------------------Now you are all set-------------------------------------------------------'

# Check, if conditions for execution are valid
check_conditions(source_path, data_format, cluster_method, range_of_clusters_for_score, save_path, test_data)

# Import raw data/test data
raw_data = import_data(source_path, data_format, test_data)

# Format the data.
formatted_data = format_data(raw_data, data_format)

# General information about your Execution
display_overview(source_path, data_format, cluster_method, save_path, formatted_data)

# Perform calculation of optimal number of clusters
perform_scoring(formatted_data, cluster_method, range_of_clusters_for_score, save_path)

# Perform clustering.
perform_clustering(formatted_data, cluster_method, raw_data, save_path, data_format)
