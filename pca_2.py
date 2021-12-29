import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv('GDSClung_rna.csv')
x = df.values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=95)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))