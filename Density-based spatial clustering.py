import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pandas import DataFrame
from sklearn.cluster import DBSCAN
X,_=make_blobs(n_samples=100,centers=3,n_features=2,random_state=20)

#%%

df= DataFrame(dict(x=X[:,0], y=X[:,1]))
fig, ax=plt.subplots(figsize=(3,3))
df.plot(ax=ax,kind='scatter',x='x',y='y')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()

#%%

CL=DBSCAN(eps=1,min_samples=5).fit(X)
cluster=CL.labels_
len(set(cluster))

#%%

def show_clustters(X, cluster):
    df=DataFrame(dict(x=X[:,0], y=X[:,1], label=cluster))
    colors={-1:'red', 0:'blue', 1:'orange', 2:'green'}
    fig, ax=plt.subplots(figsize=(3,3))
    grouped= df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.show()
show_clustters(X, cluster)

#%%