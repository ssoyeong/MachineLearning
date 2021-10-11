import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint as pp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
from pyclustering.cluster.clarans import clarans
from pyclustering.cluster import cluster_visualizer_multidim  # Modules for implementing CLARANS algorithm and plotting multi-dimensional data
from pyclustering.utils import timedcall                               # To execute a function with execution time recorded

warnings.filterwarnings('ignore')
medianHouseValue = []


# Automatically test with
# various scaled and encoded dataset,
# various subsets of the features of the dataset,
# various model parameters values, and
# various hyperparameters values.
def auto_ml(dataset):
    # Selecting features randomly
    feature_combination_list = []
    numeric_cols = list(dataset.columns)
    numeric_cols.remove('ocean_proximity')

    for i in range(4):
        selected_features = random.sample(numeric_cols, i + 2)
        feature_combination_list.append(selected_features)

    # TODO: 테스트해보면 출력 잘되는거 볼 수 있으실 거예요
    print("=> combination dataset! (15 types)")
    data_combination = scale_encode_combination(dataset, ['total_rooms', 'households'], ['ocean_proximity'])
    pp(data_combination)
    test_gaussian(data_combination['standard_ordinal'])

    for combination in feature_combination_list.items():
        data_combination = scale_encode_combination(dataset, combination, ['ocean_proximity'])
        for data_name, data in data_combination:
            test_kmeans(data)
            test_gaussian(data)
            test_clarans(data)
            test_dbscan(data)
            test_mean_shift(data)


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


def encode_scale_temp(dataframe, col):
    # Encode the dataset
    df_ordinal = dataframe.copy()
    df_label = dataframe.copy()

    x = pd.DataFrame(df[col])
    # Convert categorical features to numeric values using ordinalEncoder
    ordinalEncoder = OrdinalEncoder()
    ordinalEncoder.fit(x)
    df_ordinal[col] = ordinalEncoder.transform(x)

    """
    # Convert categorical features to numeric values using labelEncoder
    labelEncoder = LabelEncoder()
    labelEncoder.fit(x)
    df_label[col] = labelEncoder.transform(x)

    # Scaling the dataset
    df_ordinal_standard = df_ordinal.copy()
    df_ordinal_robust = df_ordinal.copy()
    df_ordinal_minmax = df_ordinal.copy()
    df_ordinal_maxabs = df_ordinal.copy()
    df_label_standard = df_label.copy()
    df_label_robust = df_label.copy()
    df_label_minmax = df_label.copy()
    df_label_maxabs = df_label.copy()

    # Scaling the dataset using StandardScaler
    scaler = StandardScaler()
    df_ordinal_standard = scaler.fit_transform(df_ordinal)
    df_ordinal_standard = pd.DataFrame(df_ordinal_standard, columns=df_ordinal.columns)
    df_label_standard = scaler.fit_transform(df_label)
    df_label_standard = pd.DataFrame(df_label_standard, columns=df_label.columns)

    # Scaling the dataset using RobustScaler
    scaler = RobustScaler()
    df_ordinal_robust = scaler.fit_transform(df_ordinal)
    df_ordinal_robust = pd.DataFrame(df_ordinal_robust, columns=df_ordinal.columns)
    df_label_robust = scaler.fit_transform(df_label)
    df_label_robust = pd.DataFrame(df_label_robust, columns=df_label.columns)
    """
    # Scaling the dataset using MinMaxScaler
    scaler = MinMaxScaler()
    df_ordinal_minmax = scaler.fit_transform(df_ordinal)
    df_ordinal_minmax = pd.DataFrame(df_ordinal_minmax, columns=df_ordinal.columns)
    """
    df_label_minmax = scaler.fit_transform(df_label)
    df_label_minmax = pd.DataFrame(df_label_minmax, columns=df_label.columns)
    
    # Scaling the dataset using MaxAbsScaler
    scaler = MaxAbsScaler()
    df_ordinal_maxabs = scaler.fit_transform(df_ordinal)
    df_ordinal_maxabs = pd.DataFrame(df_ordinal_maxabs, columns=df_ordinal.columns)
    df_label_maxabs = scaler.fit_transform(df_label)
    df_label_maxabs = pd.DataFrame(df_label_maxabs, columns=df_label.columns)

    # Show the results using OrdinalEncoder and MinMaxScaler
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 15))
    ax1.set_title('Before Scaling the OrdinalEncoded dataset')
    ax2.set_title('After MinMax Scaling the OrdinalEncoded dataset')

    for i in df_ordinal_minmax.columns:
        sns.kdeplot(df_ordinal[i], ax=ax1)
        sns.kdeplot(df_ordinal_minmax[i], ax=ax2)

    # plt.show()
    """

    # Return the one of the encoded and scaled datasets
    return df_ordinal_minmax

# K-means
def test_kmeans(x):
    pca = PCA(n_components=2)
    n_clusters = [2, 3, 4, 5, 6, 7, 8]
    init = ['k-means++', 'random']
    n_init = [10, 20, 30, 40, 50]
    max_iter = [100, 300, 500, 700, 900]
    algorithm = ['auto', 'full', 'elkan']
    distortions = []

    x = pca.fit_transform(x)
    kmeans = KMeans(n_clusters=4)
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
    n_components = range(2, 12)
    covariance_type = ['full', 'tied', 'diag', 'spherical']
    init_params = ['kmeans', 'random']

    for n in n_components:
        for covariance in covariance_type:
            for init in init_params:
                model_gaussian = GaussianMixture(n_components=n, covariance_type=covariance, init_params=init)
                model_gaussian.fit(x)
                y = model_gaussian.predict(x)
                for q in range(4, 7):
                    print_result('Gaussian Mixture', x, y, q)


# CLARANS
def test_clarans(x, k, a):  # x: dataframe, k: num of cluster, a: feature name that want to look
    h_data = x.values.tolist()
    # clarans parameters:
    # dataset, number of cluster, numlocal(amount of iterations for solving the problem), maxneighbor
    clarans_obj = clarans(random.sample(h_data, 300), k, 3, 5)  # 프로그램 실행하는 데 풀 데이터를 사용하면 시간이 너무 오래 걸려서 random으로 300개 뽑아내는 걸로 바꿨습니다
    (tks, res) = timedcall(clarans_obj.process)
    print("Execution time : ", tks, "\n")
    clst = clarans_obj.get_clusters()
    med = clarans_obj.get_medoids()
    print("Index of clusters' points :\n", clst)
    print("\nIndex of the best medoids : ", med)
    vis = cluster_visualizer_multidim()
    vis.append_clusters(clst, h_data, markersize=5)
    # Display the clusters formed in multiple dimensions
    pair = []
    for i in range(x.shape[1]):
        pair.append([a, i])

    vis.show(pair_filter=pair, max_row_size=3)


# DBSCAN
def test_dbscan(x):

    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    df_new = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    # Parameters of DBSCAN
    # eps = [0.05, 0.1, 0.5, 1, 2]
    # min_samples = [5, 10, 15, 30, 50, 100]
    eps = [0.05, 0.1]
    min_samples = [5, 10]
    '''
    for i in range(len(min_samples)):
        plt.figure(figsize=(10, 5))
        for j in range(len(eps)):
            dbscan = DBSCAN(min_samples=min_samples[i], eps=eps[j])
            clusters = dbscan.fit_predict(df_new)

            # Show scatter
            plt.subplot(1, len(eps), j + 1)
            plt.scatter(df_new.iloc[:, 0], df_new.iloc[:, 1], c=clusters, alpha=0.7)
            plt.title("eps = {:.2f}".format(eps[j]))

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.95,
                            top=0.8,
                            wspace=0.4,
                            hspace=0.4)
        plt.suptitle("DBSCAN: min_samples = {}".format(min_samples[i]))
        plt.savefig('./DBSCAN/dbscan_minsamples_' + str(min_samples[i]) + '.png', dpi=300)
    '''
    neigh = NearestNeighbors(n_neighbors=50)
    neigh_fit = neigh.fit(df_new)
    distances, indices = neigh_fit.kneighbors(df_new)

    # Elbow method
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.title("Knee method of DBSCAN determining min_samples")
    plt.show()

    # Print result
    x = pca.fit_transform(x)
    dbscan = DBSCAN(min_samples=50)
    y = dbscan.fit_predict(x)
    print_result('DBSCAN', pd.DataFrame(x), y, 5)


# Mean Shift
def test_mean_shift(x):
    bin_seeding = [True, False]
    min_bin_freq = [1, 3, 5, 7, 9, 11]
    cluster_all = [True, False]
    n_jobs = [1, 10, 100, 1000, 2000]
    max_iter = [100, 300, 500, 700, 900, 1000]
    pca = PCA(n_components=2)
    sample_list = [100, 1000, 5000, 10000]

    x = pca.fit_transform(x)
    bandwidth = estimate_bandwidth(x)
    model = MeanShift(bandwidth=bandwidth, cluster_all=True, max_iter=max_iter[2], min_bin_freq=min_bin_freq[2])
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

    # Print the measurement results using Silhouette, knee, and purity
    silhouette_score(x, y, metric='euclidean')
    silhouette_score(x, y, metric='manhattan')


######################################################################################################
# Read dataset
df = pd.read_csv('housing.csv')
# print('Dateset info')
# print(df.info(), end='\n\n')
# print('Dateset head', df.head(), sep='\n', end='\n\n')

# Dirty value detection
# print('Before preprocessing dirty values')
# print(df.isnull().sum(), end='\n\n')
df.dropna(axis=0, inplace=True)
# print('After preprocessing dirty values')
# print(df.isnull().sum(), end='\n\n')
# print('Shape of the dataset')
# print(df.shape, end='\n\n')

# Drop the feature 'median_house_value'
medianHouseValue = df['median_house_value']
df.drop(['median_house_value'], axis=1, inplace=True)

# Test all combinations
# auto_ml(df)


########## 임시 테스트용 ############
# Encoding and Scaling the dataset
df_encoded_scaled = encode_scale_temp(df, 'ocean_proximity')

# Draw heat map
# heatmap_data = df
# colormap = plt.cm.PuBu
# plt.figure(figsize=(15, 15))
# plt.title("Correlation of Features", y=1.05, size=15)
# sns.heatmap(heatmap_data.corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
#             annot=True, annot_kws={"size": 8})
# plt.show()

# Combinations of features
col1 = ['total_rooms', 'total_bedrooms', 'population', 'households']
col2 = ['total_rooms', 'households']
col3 = ['longitude', 'latitude', 'ocean_proximity']
col4 = ['median_income', 'households', 'total_rooms']
col5 = ['latitude', 'total_rooms', 'households', 'median_income']

# Dataframes with various combination
df1 = df_encoded_scaled[col1]
df2 = df_encoded_scaled[col2]
df3 = df_encoded_scaled[col3]
df4 = df_encoded_scaled[col4]
df5 = df_encoded_scaled[col5]

# print("\n-------The result---------\n")
# test_kmeans(df3)
# test_gaussian(df3)
# test_clarans(df3, 5, 5)
# test_dbscan(df3)
# test_mean_shift(df3)
