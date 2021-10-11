import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from pyclustering.cluster.clarans import clarans  #Class for implementing CLARANS algorithm
from pyclustering.utils import timedcall          #To execute a function with execution time recorded

warnings.filterwarnings('ignore')

medianHouseValue = []


def auto_ml(dataset):
    feature_combination_list = []
    # TODO: 여기서 feature combination 자동으로 만들어서
    # TODO: 위에 리스트에 넣어주는 코드 짜야할 것 같아요

    for combination in feature_combination_list:
        data_combination = encode_scale_combination(dataset, combination, ['ocean_proximity'])
        for data in data_combination:
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

def encode_scale_temp(dataframe, col):
    # Encode the dataset
    df_ordinal = dataframe.copy()
    df_label = dataframe.copy()

    X = pd.DataFrame(df[col])
    # Convert categorical features to numeric values using ordinalEncoder
    ordinalEncoder = OrdinalEncoder()
    ordinalEncoder.fit(X)
    df_ordinal[col] = ordinalEncoder.transform(X)
    print(df_ordinal.isna().sum())

    # Convert categorical features to numeric values using labelEncoder
    labelEncoder = LabelEncoder()
    labelEncoder.fit(X)
    df_label[col] = labelEncoder.transform(X)
    print(df_label.isna().sum())

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

    # Scaling the dataset using MinMaxScaler
    scaler = MinMaxScaler()
    df_ordinal_minmax = scaler.fit_transform(df_ordinal)
    df_ordinal_minmax = pd.DataFrame(df_ordinal_minmax, columns=df_ordinal.columns)
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

    # Return the one of the encoded and scaled datasets
    return df_ordinal_minmax


# EM(GMM)
def test_gaussian(x):
    for i in range(2, 12):
        model_gaussian = GaussianMixture(i, init_params='random')
        model_gaussian.fit(x)
        y = model_gaussian.predict(x)
        print_result(x, y, i)


def doDBSCAN(dataset):
    print(dataset)

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
print(df.isnull().sum())
df.dropna(axis=0, inplace=True)
print(df.isnull().sum())
# print(df.shape)

########## 임시 테스트용 ############

# Encoding and Scaling the dataset
df_encoded_scaled = encode_scale_temp(df, 'ocean_proximity')

# Draw heat map
heatmap_data = df
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 15))
plt.title("Correlation of Features", y=1.05, size=15)
sns.heatmap(heatmap_data.corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
            annot=True, annot_kws={"size": 8})
# plt.show()

# Feature selection
df_median_house_value = df_encoded_scaled['median_house_value']
df_median_house_value.columns = ['median_house_value']

# Combinations of features
# TODO: 일단 제 마음대로 했는데 바꾸셔도 돼요
col1 = ['total_rooms', 'total_bedrooms', 'population', 'households']
col2 = ['total_rooms', 'households']
col3 = ['longitude', 'latitude', 'ocean_proximity']
col4 = ['median_income', 'households', 'total_rooms']
col5 = ['latitude', 'total_rooms', 'households', 'median_income']

# Make dataframes with various combination
df1 = df_encoded_scaled[col1]
df1.columns = col1
df2 = df_encoded_scaled[col2]
df2.columns = col2
df3 = df_encoded_scaled[col3]
df3.columns = col3
df4 = df_encoded_scaled[col4]
df4.columns = col4
df5 = df_encoded_scaled[col5]
df5.columns = col5

print("\n-------the result of CLARANS---------\n")
doCLARANS(df_encoded_scaled.values, 5)

auto_ml(df1)
# autoML(df2)
# autoML(df3)
# autoML(df4)
# autoML(df5)