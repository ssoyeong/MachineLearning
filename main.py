import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from sklearn.cluster import DBSCAN
from pyclustering.cluster.clarans import clarans  #Class for implementing CLARANS algorithm
from pyclustering.utils import timedcall          #To execute a function with execution time recorded
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

medianHouseValue = []


def auto_ml(dataset):
    feature_combination_list = []

    # TODO: 여기서 feature combination 자동으로 만들어서
    # TODO: 위에 리스트에 넣어주는 코드 짜야할 것 같아요

    for combination in feature_combination_list:
        data_combination = encode_scale_combination(dataset, combination, ['ocean_proximity'])
        for data in data_combination:
            # print(data.head(10))
            test_gaussian(data)
            doDBSCAN(data)


def encode_scale_combination(dataset, numerical_feature_list, categorical_feature_list):
    # encoders
    encoder_ordinal = OrdinalEncoder()
    encoder_onehot = OneHotEncoder()
    encoder_label = LabelEncoder()
    encoders_list = [encoder_ordinal, encoder_onehot, encoder_label]
    encoders_name = ["ordinal", "onehot", "label"]

    # scalers
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    scaler_robust = RobustScaler()
    scaler_maxabs = MaxAbsScaler()
    scaler_normalize = Normalizer()
    scalers_list = [scaler_standard, scaler_minmax, scaler_robust, scaler_maxabs, scaler_normalize]
    scalers_name = ["standard", "minmax", "robust", "maxabs", "normalize"]

    result = []
    result_dict = {}
    i = 0

    # scalers x encoders = 10 combination
    for scaler in scalers_list:
        for encoder in encoders_list:
            result.append(dataset.copy())
            # scaling
            if len(numerical_feature_list) != 0:
                result[i][numerical_feature_list] = scaler.fit_transform(dataset[numerical_feature_list])
            elif len(categorical_feature_list) != 0:
                result[i][categorical_feature_list] = encoder.fit_transform(dataset[categorical_feature_list])
            for k in [3, 5, 10]:
                # save in dictionary
                dataset_type = scalers_name[int(i / 2)] + "_" + encoders_name[i % 2]
                result_dict[dataset_type] = result[i]

                # EM(GMM) test
                test_gaussian()
                print_result()

            i = i + 1

    return result


def doKmeans(X):
    pca=PCA(n_components=2)
    n_clusters=[2,3,4,5,6,7,8]
    init=['k-means++','random']
    n_init=[10,20,30,40,50]
    max_iter=[100,300,500,700,900]
    algorithm=['auto','full','elkan']
    distortions = []
    # elbow method
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(2, 9), distortions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()
    list={}
    for i in range(0,len(n_clusters)):
        plt.figure(figsize=(16, 4))
        plt.rc("font",size=5)
        # SUBPLOT POSITION
        position = 1
        for j in range(0, len(max_iter)):
                    X=pca.fit_transform(X)
                    model = KMeans(random_state=0,n_clusters=n_clusters[i],init='k-means++',max_iter=max_iter[j])
                    model.fit(X)
                    label=model.labels_
                    cluster_id=pd.DataFrame(label)
                    kx=pd.DataFrame(X)
                    k1=pd.concat([kx,cluster_id],axis=1)
                    k1.columns=['p1','p2',"cluster"]
                    labeled=k1.groupby("cluster")
                    plt.subplot(1, 5, position)
                    score = silhouette_score(X, label, metric="euclidean")
                    plt.title("MAX_ITER={maxiter}Score={score}".format(maxiter=max_iter[j],score=round(score,3)))
                    for cluster, pos in labeled:
                        if cluster == -1:
                            # NOISE WITH COLOR BLACK
                            plt.plot(pos.p1, pos.p2, marker='o', linestyle='', color='black')
                        else:
                            plt.plot(pos.p1, pos.p2, marker='o', linestyle='')
                    position += 1
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle("K-Means: N_CLUSTERS={0}".format(n_clusters[i]))
        plt.show()


def domeanShift(X):
    bin_seeding=[True,False]
    min_bin_freq=[1,3,5,7,9,11]
    cluster_all=[True, False]
    n_jobs:[1,10,100,1000,2000]
    max_iter=[100,300,500,700, 900,1000]
    pca = PCA(n_components=2)
    sampleList=[100,1000,5000,10000]
    for i in range(0,len(sampleList)):
        bandwidth=estimate_bandwidth(X,n_samples=sampleList[i])
        plt.figure(figsize=(16, 4))
        plt.rc("font", size=5)
        position = 1
        for j in range(0, len(min_bin_freq)):
            model=MeanShift(bandwidth=bandwidth, cluster_all=True,max_iter=max_iter[2],min_bin_freq=min_bin_freq[j])
            X=pca.fit_transform(X)
            model.fit(X)
            labels=model.labels_
            cluster_id = pd.DataFrame(labels)
            kx = pd.DataFrame(X)
            k1 = pd.concat([kx, cluster_id], axis=1)
            k1.columns = ['p1', 'p2', "cluster"]
            labeled = k1.groupby("cluster")
            plt.subplot(1, 6, position)
            score = silhouette_score(X, labels, metric="euclidean")
            plt.title("Min_bin_freq={maxiter} Score={score}".format(maxiter=min_bin_freq[j], score=round(score, 3)))
            for cluster, pos in labeled:
                if cluster == -1:
                    # NOISE WITH COLOR BLACK
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='', color='black')
                else:
                    plt.plot(pos.p1, pos.p2, marker='o', linestyle='')
            position += 1
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.suptitle("MeansShift: N_samples={0}".format(sampleList[i]))
        plt.show()


# EM(GMM)
def test_gaussian(x):
    for i in range(2, 12):
        model_gaussian = GaussianMixture(i, init_params='random')
        model_gaussian.fit(x)
        y = model_gaussian.predict(x)
        print_result(x, y, i)


def doDBSCAN(dataset):
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(dataset)
    df_new = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    # Parameters of DBSCAN
    eps = [0.05, 0.1, 0.5, 1, 2]
    min_samples = [5, 10, 15, 30, 50, 100]

    for i in range(len(min_samples)):
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


def doCLARANS(dataset, k):
    h_data = dataset.tolist()
    # clarans(dataset, number of cluster, numlocal(amount of iterations for solving the problem, maxneighbor)
    clarans_obj = clarans(h_data[0:50], k, 3, 5)  # 프로그램 실행하는 데 풀 데이터를 사용하면 시간이 너무 오래 걸려서 그냥 예시로 데이터 셋 50개만 해봤어요! 바꾸셔도 되요!
    (tks, res) = timedcall(clarans_obj.process)
    print("Execution time : ", tks, "\n")
    clst = clarans_obj.get_clusters()
    med = clarans_obj.get_medoids()
    print("Index of clusters' points :\n", clst)
    print("\nIndex of the best medoids : ", med)


# Result printing function
def print_result(x, y, i):
    # Set the data for plotting a "median house value" distribution
    labels_median_house_value = list(map(str, np.arange(0, i)))
    labeled_median_house_value = pd.cut(medianHouseValue, i, labels=labels_median_house_value, include_lowest=True)
    new_x = pd.concat([x, labeled_median_house_value], axis=1)

    # Plot the results of the clustering and "median house value" distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    axes[0].set_title('Gaussian Mixture Model Clustering')
    axes[0].set_xlabel(x.columns[0])
    axes[0].set_ylabel(x.columns[1])
    axes[0].scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, s=40, cmap='viridis')
    axes[1].set_title('Median house value distribution')
    axes[1].set_xlabel(x.columns[0])
    axes[1].set_ylabel(x.columns[1])
    sns.scatterplot(ax=axes[1], data=new_x, x=new_x.iloc[:, 0], y=new_x.iloc[:, 1], hue='median_house_value')
    plt.show()

    # Print the measurement results using Silhouette, knee, and purity
    silhouette_score(x, y)


######################################################################################################
# Read dataset
df = pd.read_csv('housing.csv')
# print(df.info())
# print(df.head())
# print(df.describe())

# Dirty value detection
# print(df.isnull().sum())
df.dropna(axis=0, inplace=True)
# print(df.isnull().sum())
# print(df.shape)

# ########## 임시 테스트용 ############
# # Draw heat map
# heatmap_data = df
# colormap = plt.cm.PuBu
# plt.figure(figsize=(15, 15))
# plt.title("Correlation of Features", y=1.05, size=15)
# sns.heatmap(heatmap_data.corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
#             annot=True, annot_kws={"size": 8})
# # plt.show()
#

# print("\n-------the result of CLARANS---------\n")
# doCLARANS(df_encoded_scaled.values, 5)

auto_ml(df)
