
df = pd.read_csv('U_Pu_comp.csv')
print(f'\nShape of the dataset : {df.shape}\n')
print(f'\n First 5 rows : {df.head(5)} \n')
print(f'\n Last 5 rows : {df.tail(5)} \n')
print(f' \n Data info : {df.info()}')
print(f' \n All Columns : {df.columns.tolist()}')
print(f'\n Summary stats : {df.describe()} \n')
print(f'\n Mean : {df.mean()} \n')
print(f'\n Median : {df.median()} \n')
print(f'\n Mode : {df.mode().iloc[0]} \n')



plt.hist([df['Pu-239 Impurity, %'], df['U-235 Enrichment, %']], bins=10,label=['Pu Impurity', 'U Enrichment'], alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Frequency Distribution for Pu Impurity % and U Enrichment %')
plt.legend()
plt.show()

plt.hist([df['Pu-239 Impurity, %']], bins=10,label=['Pu Impurity', 'U Enrichment'], alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Frequency Distribution for Pu Impurity %')
plt.legend()
plt.show()

plt.hist([df['U-235 Enrichment, %']], bins=10,label=['Pu Impurity', 'U Enrichment'], alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Frequency Distribution U Enrichment %')
plt.legend()
plt.show()


x1=np.array(df['U-235 Enrichment, %'])
x2=np.array(df['Pu-239 Impurity, %'])
X=np.array(list(zip(x1,x2)))

distortions=[]
inertias=[]
mapping1={}
mapping2={}
K = range(1,11)

for k in K:
  kmeanModel=KMeans(n_clusters=k, random_state=42).fit(X)
  distortions.append(sum(np.min(cdist(X,kmeanModel.cluster_centers_,'euclidean'), axis=1)**2)/X.shape[0])
  inertias.append(kmeanModel.inertia_)
  mapping1[k]=distortions[-1]
  mapping2[k]=inertias[-1]

print('Distortion Values')

for key, val in mapping1.items():
  print(f'Key : {key}, Value : {val}')

plt.plot(K,distortions, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method using Distortion')
plt.show()

plt.plot(K,inertias,'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method using Inertia')
plt.show()

X=np.array(list(zip(x1,x2)))

kmeans=KMeans(n_clusters=10,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=y_kmeans,cmap='viridis',marker='*',s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],marker='x',s=300,c='purple',label='Centroids', edgecolor='black')
plt.title(f'k.means Clustering (k=10)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()

labels=kmeans.predict(X)
ClusterCount=np.bincount(labels)
print(f'\n List of the number of data points per cluster : {ClusterCount} \n')
mean=df['Pu-239 Impurity, %'].mean()
mean1=df['U-235 Enrichment, %'].mean()
print(f' \n Mean percentage of U-235 Enrichment : {mean}')
print(f' \n Mean percentage of Pu - 239 Impurity : {mean1}')
