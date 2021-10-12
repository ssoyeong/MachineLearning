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
def auto_ml(dataset):
    # TODO: Minmax scaled & Ordinal encoded 데이터 사용 테스트용입니다.
    # TODO: Feature 조합 선택하시고 사용할 데이터와 함수 주석 해제하셔서 사용하시면 됩니다.
    feature_selection1 = ['total_rooms', 'households']
    feature_selection2 = ['longitude', 'latitude']
    data_combination = scale_encode_combination(dataset, feature_selection1, ['ocean_proximity'])
    data_combination = data_combination['minmax_ordinal']
    # data1 = data_combination[feature_selection1]
    #data2 = data_combination[['longitude', 'latitude', 'ocean_proximity']]
    # test_kmeans(data1)
    #test_gaussian(data1)
    #test_clarans(data1)
    #test_dbscan(data1)
    #test_mean_shift(data1)

    """
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
            test_kmeans(data)
            test_gaussian(data)
            test_clarans(data)
            test_dbscan(data)
            test_mean_shift(data)
    """


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
    pca = PCA(n_components=2) # for the feature reduction and plotting
    n_clusters = [2, 3, 4, 5, 6, 7, 8] # k
    n_init = [10, 20, 30, 40, 50]
    algorithm = ['auto', 'full', 'elkan'] # algorithm list
    distortions = [] # for elbow method
    x = pca.fit_transform(x)
    # the combination of kmeans
    for n in n_clusters:
        for algo in algorithm:
            for ni in n_init:
                kmeans = KMeans(n_clusters=n,n_init=ni, algorithm=algo)
                kmeans.fit(x)
                y = kmeans.predict(x)
                print_result('K-Means', pd.DataFrame(x), y, 5)

    """
    # elbow method
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        distortions.append(kmeans.inertia_)

    
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 9), distortions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()
    list={}

    for i in range(0, len(n_clusters)):
        plt.figure(figsize=(16, 4))
        plt.suptitle("K-Means: N_CLUSTERS={0}".format(n_clusters[i]))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.rc("font", size=5)

        # SUBPLOT POSITION
        position = 1
        for j in range(0, len(algorithm)):
            x = pca.fit_transform(x)
            model = KMeans(random_state=0, n_clusters=n_clusters[i], init='k-means++', max_iter=max_iter[2],algorithm=algorithm[j])
            model.fit(x)
            label = model.labels_

            cluster_id = pd.DataFrame(label)
            kx = pd.DataFrame(x)
            k1 = pd.concat([kx, cluster_id], axis=1)
            k1.columns = ['p1', 'p2', "cluster"]
            labeled = k1.groupby("cluster")
            score = silhouette_score(x, label, metric="euclidean")

            plt.subplot(1, 3, position)
            plt.title("Algorithm={algo}Score={score}".format(algo=algorithm[j],score=round(score,3)))
            position += 1

            for cluster, pos in labeled:
                if cluster == -1:
                    # NOISE WITH COLOR BLACK
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='', color='black')
                else:
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='')

        plt.show()
        """


# EM(GMM)
def test_gaussian(x):
    pca = PCA(n_components=2)
    x = pd.DataFrame(pca.fit_transform(x))

    # Parameters
    n_components = range(2, 13)
    covariance_type = ['full', 'tied', 'diag', 'spherical']
    init_params = ['kmeans', 'random']

    for k in range(4, 7):
        plt.figure(figsize=(25, 5))

        for idx, cov in enumerate(covariance_type):
            model_gaussian = GaussianMixture(n_components=k, covariance_type=cov, init_params='kmeans')
            y = model_gaussian.fit_predict(x)

            # Plotting the results of clustering
            plt.subplot(1, 4, idx + 1)
            plt.title("Covariance = {}".format(cov))
            plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, alpha=0.7)
            """
            # Plotting the results comparing with 'Median house value'
            for q in range(6, 7):
                title = 'EM(GMM) K:' + str(k) + ', Covariance:' + str(cov) + ', q:' + str(q)
                print_result(title, x, y, q)
            """

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.8, wspace=0.4, hspace=0.4)
        plt.suptitle("EM(GMM): K = {}".format(k))
        plt.savefig('./Figure_' + str(k) + '.png', dpi=300)
    """
    for n in n_components:
        for covariance in covariance_type:
            for init in init_params:
                model_gaussian = GaussianMixture(n_components=n, covariance_type=covariance, init_params=init)
                model_gaussian.fit(x)
                y = model_gaussian.predict(x)
                for q in range(4, 7):
                    print_result('Gaussian Mixture', x, y, q)
    """


# CLARANS
def test_clarans(x):
    pca = PCA(n_components=2)
    x = pd.DataFrame(pca.fit_transform(x))

    # Parameters
    h_data = x.values.tolist()
    number_clusters = range(2, 12)
    numlocal = 2
    maxneighbor = 3

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
            for q in range(6, 7):
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
    bin_seeding = [True, False]
    min_bin_freq = [1, 3, 5, 7, 9, 11]
    cluster_all = [True, False] # the option
    pca = PCA(n_components=2)
    sample_list = [100, 1000, 5000, 10000]
    x = pca.fit_transform(x)
    # the combination of meanshift
    for nsam in sample_list:
        for min in min_bin_freq:
            bandwidth = estimate_bandwidth(x,n_samples=nsam,min_bin_freq=min)
            model = MeanShift(bandwidth=bandwidth, cluster_all=True, max_iter=500, min_bin_freq=min_bin_freq[2])
            x = pd.DataFrame(x)
            model.fit(x)
            y = model.predict(x)
            print_result('Mean Shift', x, y, 5)

    """
    for i in range(0, len(sample_list)):
        plt.figure(figsize=(16, 4))
        plt.suptitle("MeansShift: N_samples={0}".format(sample_list[i]))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.rc("font", size=5)

        position = 1
        bandwidth = estimate_bandwidth(x, n_samples=sample_list[i])

        for j in range(0, len(min_bin_freq)):
            x = pca.fit_transform(x)
            model = MeanShift(bandwidth=bandwidth, cluster_all=True, max_iter=max_iter[2], min_bin_freq=min_bin_freq[j])
            model.fit(x)

            labels = model.labels_
            cluster_id = pd.DataFrame(labels)
            kx = pd.DataFrame(x)
            k1 = pd.concat([kx, cluster_id], axis=1)
            k1.columns = ['p1', 'p2', "cluster"]
            labeled = k1.groupby("cluster")

            score = silhouette_score(x, labels, metric="euclidean")
            plt.subplot(1, 6, position)
            plt.title("Min_bin_freq={maxiter} Score={score}".format(maxiter=min_bin_freq[j], score=round(score, 3)))
            position += 1

            for cluster, pos in labeled:
                if cluster == -1:
                    # NOISE WITH COLOR BLACK
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='', color='black')
                else:
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='')

        plt.show()
        """


# Result printing function
def print_result(model_name, x, y, quantile):
    # Set the data for plotting a "median house value" distribution
    labels_median_house_value = list(map(str, np.arange(0, quantile)))
    labeled_median_house_value = pd.cut(medianHouseValue, quantile, labels=labels_median_house_value, include_lowest=True)
    new_x = pd.concat([x, labeled_median_house_value], axis=1)

    # Plot the results of the clustering and "median house value" distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    axes[0].set_title(model_name + 'Model Clustering')
    axes[0].set_xlabel(x.columns[0] if x.columns[0] != 0 else 'x')
    axes[0].set_ylabel(x.columns[1] if x.columns[1] != 1 else 'y')
    axes[0].scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, s=40, cmap='viridis')
    axes[1].set_title('Median house value distribution')
    axes[1].set_xlabel(x.columns[0] if x.columns[0] != 0 else 'x')
    axes[1].set_ylabel(x.columns[1] if x.columns[1] != 1 else 'y')
    sns.scatterplot(ax=axes[1], data=new_x, x=new_x.iloc[:, 0], y=new_x.iloc[:, 1], hue='median_house_value')
    plt.show()
    new_x['median_house_value']=pd.to_numeric(new_x['median_house_value']) # change the median value to string to int
    print("Purity Score: ",purity_score(new_x['median_house_value'],y))

    # Print the measurement results using Silhouette and purity
    print('Euclidian Silhouette Score: ', silhouette_score(x, y, metric='euclidean'))
    print('Manhattan Silhouette Score: ', silhouette_score(x, y, metric='manhattan'))
    print('L2 Silhouette Score:', silhouette_score(x, y, metric='l2'))
    print('L1 Silhouette Score:', silhouette_score(x, y, metric='l1'))


def purity_score(y_true, y_pred):
    # compute confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    return np.sum(np.amax(cf_matrix, axis=0)) / np.sum(cf_matrix)


######################################################################################################
# Read dataset
df = pd.read_csv('housing.csv')
# print('Dateset info')
# print(df.info(), end='\n\n')
# print('Dateset head', df.head(), sep='\n', end='\n\n')

# Dirty value detection
# print('Before preprocessing dirty values')
# print(df.isnull().sum(), end='\n\n')
df.fillna(df.mean(), inplace=True)
# print('After preprocessing dirty values')
# print(df.isnull().sum(), end='\n\n')
# print('Shape of the dataset')
# print(df.shape, end='\n\n')

# Drop the feature 'median_house_value'
medianHouseValue = df['median_house_value']
df.drop(['median_house_value'], axis=1, inplace=True)

# Draw heat map
# heatmap_data = df
# colormap = plt.cm.PuBu
# plt.figure(figsize=(15, 15))
# plt.title("Correlation of Features", y=1.05, size=15)
# sns.heatmap(heatmap_data.corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
#             annot=True, annot_kws={"size": 8})
# plt.show()

# Test all combinations
auto_ml(df)
