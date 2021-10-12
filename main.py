import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall  # To execute a function with execution time recorded

warnings.filterwarnings('ignore')
medianHouseValue = []


# Automatically test with
# various scaled and encoded dataset,
# various subsets of the features of the dataset,
# various model parameters values, and
# various hyperparameters values.
def auto_ml(dataset, model):
    # Selecting features randomly
    feature_combination_list = []
    numeric_cols = list(dataset.columns)
    numeric_cols.remove('ocean_proximity')
    for i in range(4):
        selected_features = random.sample(numeric_cols, i + 2)
        feature_combination_list.append(selected_features)

    # Run algorithms with every combination
    for combination in feature_combination_list:
        data_combination = scale_encode_combination(dataset, combination, ['ocean_proximity'])
        for data_name, data in data_combination.items():
            data = data[combination]
            model_dict = {'kmeans': test_kmeans(data),
                          'em': test_gaussian(data),
                          'clarans': test_clarans(data),
                          'dbscan': test_dbscan(data),
                          'meanshift': test_mean_shift(data)}
            model_dict[model]


# Dataset scaling and encoding function
def scale_encode_combination(dataset, numerical_feature_list, categorical_feature_list):
    # scalers
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    scaler_robust = RobustScaler()
    scaler_maxabs = MaxAbsScaler()
    scaler_normalize = Normalizer()
    scalers_list = [scaler_standard, scaler_minmax, scaler_robust, scaler_maxabs, scaler_normalize]
    scalers_name = ["standard", "minmax", "robust", "maxabs", "normalize"]

    # encoders
    encoder_ordinal = OrdinalEncoder()
    encoder_onehot = OneHotEncoder()
    encoder_label = LabelEncoder()
    encoders_list = [encoder_ordinal, encoder_onehot, encoder_label]
    encoders_name = ["ordinal", "onehot", "label"]

    result = []
    result_dict = {}
    k = 0

    # scalers x encoders = 15 combinations
    for i, scaler in enumerate(scalers_list):
        for j, encoder in enumerate(encoders_list):
            result.append(dataset.copy())
            # scaling and encoding
            if len(numerical_feature_list) != 0:
                result[k][numerical_feature_list] = scaler.fit_transform(dataset[numerical_feature_list])
            if len(categorical_feature_list) != 0 and len(categorical_feature_list) > 1:
                result[k][categorical_feature_list] = encoder.fit_transform(dataset[categorical_feature_list])
            elif len(categorical_feature_list) == 1:
                result[k][categorical_feature_list[0]] = encoder.fit_transform(dataset[categorical_feature_list])

            # save in dictionary
            dataset_type = scalers_name[i] + "_" + encoders_name[j]
            result_dict[dataset_type] = result[k]

            k = k + 1

    return result_dict


# K-means
def test_kmeans(x):
    pca = PCA(n_components=2)  # for the feature reduction and plotting
    x = pca.fit_transform(x)
    n_clusters = [2, 3, 4, 5, 6, 7, 8]  # k
    n_init = [10, 20, 30, 40, 50]
    algorithm = ['auto', 'full', 'elkan']  # algorithm list

    # the combination of kmeans
    for n in n_clusters:
        for algo in algorithm:
            for ni in n_init:
                kmeans = KMeans(n_clusters=n,n_init=ni, algorithm=algo)
                kmeans.fit(x)
                y = kmeans.predict(x)
                print_result('K-Means', pd.DataFrame(x), y, 5)


# EM(GMM)
def test_gaussian(x):
    pca = PCA(n_components=2)
    x = pd.DataFrame(pca.fit_transform(x))

    # Parameters
    n_components = range(2, 13)
    covariance_type = ['full', 'tied', 'diag', 'spherical']
    init_params = ['kmeans', 'random']

    for n in n_components:
        for cov in covariance_type:
            for init in init_params:
                model_gaussian = GaussianMixture(n_components=n, covariance_type=cov, init_params=init)
                model_gaussian.fit(x)
                y = model_gaussian.predict(x)
                for q in range(4, 7):
                    title = 'EM(GMM) K:' + str(n) + ', Covariance:' + str(cov) + ', q:' + str(q)
                    print_result(title, x, y, q)


# CLARANS
def test_clarans(x):
    pca = PCA(n_components=2)
    x = pd.DataFrame(pca.fit_transform(x))

    # Parameters
    h_data = x.values.tolist()
    number_clusters = range(2, 12)
    numlocal = 1
    maxneighbor = 1

    for n in number_clusters:
        # h_data = random.sample(h_data, 100)
        clarans_obj = clarans(h_data, number_clusters=n, numlocal=numlocal, maxneighbor=maxneighbor)
        (tks, res) = timedcall(clarans_obj.process)
        print("number of clusters: {}, Execution time : {}\n".format(n, tks))
        clusters = clarans_obj.get_clusters()
        num = len(h_data)
        y = np.zeros(num, dtype=int).tolist()
        for cluster_no in range(np.shape(clusters)[0]):
            for idx in clusters[cluster_no]:
                y[idx] = cluster_no
        print_result('CLARANS', x, y, 5)


# DBSCAN
def test_dbscan(x):
    pca = PCA(n_components=2)
    df_new = pd.DataFrame(pca.fit_transform(x))

    # Parameters of DBSCAN
    eps = [0.05, 0.1, 0.5, 1, 2]
    min_samples = [5, 30, 500]

    for i in range(len(min_samples)):
        plt.figure(figsize=(25, 5))
        for j in range(len(eps)):
            dbscan = DBSCAN(min_samples=min_samples[i], eps=eps[j])
            y = dbscan.fit_predict(df_new)

            # Plotting the results comparing with 'Median house value'
            for q in range(3, 4):
                title = 'DBSCAN(MinPts:' + str(min_samples[i]) + ', Eps:' + str(eps[j]) + ', q:' + str(q) + ')'
                print_result(title, df_new, y, q)

            # Plotting the results of clustering
            plt.subplot(1, len(eps), j + 1)
            plt.scatter(df_new.iloc[:, 0], df_new.iloc[:, 1], c=y, alpha=0.7)
            plt.title("eps = {:.2f}".format(eps[j]))

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.8, wspace=0.4, hspace=0.4)
        plt.suptitle("DBSCAN: min_samples = {}".format(min_samples[i]))
        plt.savefig('./DBSCAN/dbscan_minsamples_' + str(min_samples[i]) + '.png', dpi=300)

    # Elbow curve
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(df_new)
    distances, indices = neigh.kneighbors(df_new)

    plt.figure(figsize=(12, 6))
    plt.plot(np.sort(distances[:, 4]))
    plt.title("Elbow curve of DBSCAN")
    plt.show()


# Mean Shift
def test_mean_shift(x):
    min_bin_freq = [1, 3, 5, 7, 9, 11]
    pca = PCA(n_components=2)
    sample_list = [100, 1000, 5000, 10000]
    x = pca.fit_transform(x)

    # the combination of meanshift
    for nsam in sample_list:
        bandwidth = estimate_bandwidth(x, n_samples=nsam)
        for min in min_bin_freq:
            model = MeanShift(bandwidth=bandwidth, cluster_all=True, max_iter=500, min_bin_freq=min)
            x = pd.DataFrame(x)
            model.fit(x)
            y = model.predict(x)
            print_result('Mean Shift', x, y, 5)


# Result printing function
def print_result(model_name, x, y, quantile):
    # Set the data for plotting a "median house value" distribution
    labels_median_house_value = list(map(str, np.arange(0, quantile)))
    labeled_median_house_value = pd.cut(medianHouseValue, quantile, labels=labels_median_house_value, include_lowest=True)
    new_x = pd.concat([x, labeled_median_house_value], axis=1)

    # Plot the results of the clustering and "median house value" distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    axes[0].set_title(model_name + ' Model Clustering')
    axes[0].set_xlabel(x.columns[0] if x.columns[0] != 0 else 'x')
    axes[0].set_ylabel(x.columns[1] if x.columns[1] != 1 else 'y')
    axes[0].scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, s=40, cmap='viridis')
    axes[1].set_title('Median house value distribution')
    axes[1].set_xlabel(x.columns[0] if x.columns[0] != 0 else 'x')
    axes[1].set_ylabel(x.columns[1] if x.columns[1] != 1 else 'y')
    sns.scatterplot(ax=axes[1], data=new_x, x=new_x.iloc[:, 0], y=new_x.iloc[:, 1], hue='median_house_value')
    plt.show()

    # Print the measurement results using purity and silhouette
    new_x['median_house_value'] = pd.to_numeric(new_x['median_house_value'])  # Change the type to string to int
    print("Purity Score: ", purity_score(new_x['median_house_value'], y))
    print('Euclidian Silhouette Score: ', silhouette_score(x, y, metric='euclidean'))
    print('Manhattan Silhouette Score: ', silhouette_score(x, y, metric='manhattan'))
    print('L2 Silhouette Score:', silhouette_score(x, y, metric='l2'))
    print('L1 Silhouette Score:', silhouette_score(x, y, metric='l1'))


def purity_score(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)  # compute confusion matrix

    return np.sum(np.amax(cf_matrix, axis=0)) / np.sum(cf_matrix)


######################################################################################################
# Read dataset
df = pd.read_csv('housing.csv')
print('Dateset info')
print(df.info(), end='\n\n')
print('Dateset head', df.head(), sep='\n', end='\n\n')

# Dirty value detection
print('Before preprocessing dirty values')
print(df.isnull().sum(), end='\n\n')

# Replace dirty values with mean
df.fillna(df.mean(), inplace=True)
print('After preprocessing dirty values')
print(df.isnull().sum(), end='\n\n')
print('Shape of the dataset')
print(df.shape, end='\n\n')

# Drop the feature 'median_house_value'
medianHouseValue = df['median_house_value']
df.drop(['median_house_value'], axis=1, inplace=True)

# Test all combinations
auto_ml(df, 'kmeans')
