import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from generate_test_data import create_test_dataset

'------------------------------------ Information about import export format data -------------------------------------'
'''
This module will import the test dataset or your dataset with real data.
It will also format the data according to a standard.
Finally, it will export all results and plots.
'''
'----------------------------------------------------------------------------------------------------------------------'


def import_data(source_path, data_format, test_data):
    """
    It is important to consider the format of the CSV file because if the format does not meet these requirements,
    it cannot be imported.
    The CSV file must have the following format
    One value per cell. Each column represents one dimension and each row represents one data point.
    """
    # Import your data set
    if test_data.lower() == 'no':
        try:
            raw_data = pd.read_csv(source_path, sep=';', decimal=',')

            return raw_data
        except FileNotFoundError:
            print(f"ERROR: File '{source_path}' not found.")
            sys.exit()

    # Import test data
    if test_data.lower() == 'yes':
        raw_data = create_test_dataset(data_format)
        return raw_data


def format_data(raw_data, data_format):
    if data_format == 'num':
        # Delete first column with ID´s
        prepared_data = raw_data.iloc[:, 1:]

        # Create an imputation transformer
        imputer = SimpleImputer(strategy="mean")

        # Impute the missing values
        raw_data_imputer = imputer.fit_transform(prepared_data)

        # Normalize the numeric dataset
        scaler = MinMaxScaler()
        normalized_numeric_data = scaler.fit_transform(raw_data_imputer)

        return normalized_numeric_data

    elif data_format == 'bool':

        # Delete the second row (index 0)
        prepared_data_1 = raw_data.drop(0)

        # Delete first column with IDs
        prepared_data_2 = prepared_data_1.iloc[:, 1:].copy()

        # Fill missing values with 0
        prepared_data_2.fillna(0, inplace=True)

        # Convert the DataFrame to a 2D array
        flattened_bool_data = prepared_data_2.values
        flattened_bool_data = flattened_bool_data.astype(int)

        return flattened_bool_data


def export_scores_plt(scores, save_path):
    if scores is not None:
        for fig, score_name in scores:
            if fig is not None:
                plot_path = save_path + '/' + score_name + '.png'
                fig.savefig(plot_path)
                print("The Scoring plot was saved in the file", plot_path)
        fig.clf()


def export_gap_stat_plt(gap_stat_plt, save_path):
    if gap_stat_plt is not None:
        plot_path = save_path + '/Gap Statistics Index.png'
        gap_stat_plt.savefig(plot_path)
        print("The Scoring plot was saved in the file", plot_path)
    print('')


def export_data_csv(raw_data, cluster_labels, save_path, cluster_method):
    if not isinstance(raw_data, pd.DataFrame):
        df = pd.DataFrame(raw_data)
    else:
        df = raw_data.copy()

    # Check the difference in the number of rows
    row_difference = len(df) - len(cluster_labels)

    if row_difference == 0:
        # If the number of rows is identical, do nothing
        # Adds the column Labels to the DataSet
        df['Labels'] = cluster_labels
        pass
    elif row_difference == 1:
        # If the difference is 1, shift 'Labels' column downwards by one position
        df_with_labels = pd.DataFrame(index=df.index[:-1])
        df_with_labels['Labels'] = cluster_labels
        df_with_labels.index += 1  # Increase the index of 'df_with_labels' by 1 to make space for an additional value
        df = pd.concat([df, df_with_labels], axis=1)
    elif row_difference == 2:
        # If the difference is 2, shift 'Labels' column downwards by two positions
        df_with_labels = pd.DataFrame(index=df.index[:-2])
        df_with_labels['Labels'] = cluster_labels
        df_with_labels.index += 2  # Increase the index of 'df_with_labels' by 1 to make space for an additional value
        df = pd.concat([df, df_with_labels], axis=1)
    else:
        # If the difference is more than 2, raise an error or handle as needed
        raise ValueError("Difference in the number of rows is greater than 2, handling not specified.")

    # Save the cluster labels as a CSV file.
    labels_path = save_path + '/' + cluster_method + '.csv'
    print("CSV with added cluster labels were saved in the file", labels_path)

    # Export the DataFrame as a CSV file
    df.to_csv(labels_path, index=False)


def export_scatter_plot(fig_scatter, save_path, cluster_method):
    if fig_scatter is not None:
        plot_path = save_path + '/' + cluster_method + '_scatter.png'
        fig_scatter.savefig(plot_path)
        print("The scatter plot was saved in the file", plot_path)


def export_dendrogram(fig_dendrogram, save_path, cluster_method):
    if fig_dendrogram is not None:
        plot_path = save_path + '/' + cluster_method + '_dendrogram.png'
        fig_dendrogram.savefig(plot_path)
        print("The dendrogram was saved in the file", plot_path)
