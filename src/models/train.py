import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.covariance import EmpiricalCovariance, MinCovDet

#######################################################################

training_data_set1=pd.read_csv(r'C:\Users\juli\Dropbox\My PC (DESKTOP-J5HOA3A)\Desktop\AI codes\Anomaly_detection_machinelearning\Data\training.csv',header=None)
testing_data_set1=pd.read_csv(r'C:\Users\juli\Dropbox\My PC (DESKTOP-J5HOA3A)\Desktop\AI codes\Anomaly_detection_machinelearning\Data\validation.csv',header=None)
############################################################
training_data_set=training_data_set1.T
testing_data_set=testing_data_set1.T
fig, axs = plt.subplots(3,1,figsize=(15,15))
axs[0].plot(training_data_set.iloc[:,np.arange(0,17,3)])
#axs[0].set_xlim(0,training_data_set[0])
axs[0].set_title("Plot of ozone at all locations over time")
axs[0].set_ylabel('Signal magnitude')

axs[1].plot(training_data_set.iloc[:,np.arange(1,17,3)])
#axs[1].set_xlim(0,training_data_set.shape[0])
axs[1].set_title("Plot of nitrogen oxyde at all locations over time")
axs[1].set_ylabel('Signal magnitude')

axs[2].plot(training_data_set.iloc[:,np.arange(2,17,3)])
#axs[2].set_xlim(0,training_data_set.shape[0])
axs[2].set_title("Plot of dioxide at all locatons over time")
axs[2].set_ylabel('Signal magnitude')
plt.show()

    
x_scaled = StandardScaler().fit_transform(training_data_set)
df = pd.DataFrame(x_scaled)
#print(df)

x_scaled2= preprocessing.normalize(training_data_set)
df2 = pd.DataFrame(x_scaled2)
#print(df2)

cov_mat1=np.cov(x_scaled,rowvar=False)
cov_mat2=np.cov(x_scaled2,rowvar=False)

'''plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
hm = sns.heatmap(cov_mat1,cbar=True,annot=True,square=True,fmt='.1f',annot_kws={'size':12})
plt.title('Covariance matrix showing correlation coefficients')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
hm = sns.heatmap(cov_mat2,cbar=True,annot=True,square=True,fmt='.1f',annot_kws={'size':12})
plt.title('Covariance matrix showing correlation coefficients')
plt.tight_layout()
plt.show()'''

U, S, V = np.linalg.svd(cov_mat2,full_matrices=0)
print(f'u={U};Singular_V={S}, V={V}')

plt.figure(figsize=(15,8))
plt.plot(np.cumsum(S)/np.sum(S),'-o',color='k')
plt.xticks(np.arange(0,18,step=1))
plt.xlabel('Singular Values')
plt.ylabel('Magnitude')
plt.title('Variance captured by singular vectors')
plt.show()

pca = PCA()
principalcomponents = pca.fit_transform(x_scaled)
principalDf = pd.DataFrame(data=principalcomponents[:,:2],
                           columns=['principal component 1', 'principal component 2'])


fig = plt.figure(figsize=(6,8))
plt.scatter(principalDf['principal component 1'],
            principalDf['principal component 2'],marker='o',color='r',s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim((-8,8))
plt.ylim((-8,8))
plt.show()


robust_covariance = MinCovDet().fit(principalcomponents[:,:12])

# Get the Mahalanobis distance
Mahalanobis_distance = robust_covariance.mahalanobis(principalcomponents[:,:12])
#plot fault detection distance from center threshold
fig = plt.figure(figsize=(15,8))
plt.plot(Mahalanobis_distance)
plt.ylabel('Distance from gravity center of training set')
plt.title('Mahalanobis Distance')
plt.show()
################################################################
threshold_mahalanobis = np.mean(Mahalanobis_distance) + (3 * np.std(Mahalanobis_distance))
fig = plt.figure(figsize=(15,8))
plt.plot(Mahalanobis_distance)
plt.plot(threshold_mahalanobis  + 0 * Mahalanobis_distance,label='decision threshold')
plt.ylabel('Distance from gravity center of training set')
plt.title('Mahalanobis Distance with Decision threshold')
plt.xlim(0,558)
plt.legend(loc='best')
plt.show()




#################################################################


euclidean_distance = np.zeros(x_scaled.shape[0])

for i in range(12):
    euclidean_distance += (principalcomponents[:,i] - np.mean(principalcomponents[:,:12]))**2/np.var(principalcomponents[:,1])
    
threshold_euclidean = np.mean(euclidean_distance) + (3 * np.std(euclidean_distance))
fig = plt.figure(figsize=(15,8))
plt.plot(euclidean_distance)
plt.plot(threshold_euclidean  + 0 * euclidean_distance,label='decision threshold')
plt.ylabel('Distance from gravity center of training set')
plt.title('Euclidean Distance with Decision threshold')
plt.xlim(0,558)
plt.legend(loc='best')
plt.show()

# Apply the PCA model on second data set ; Faulty operation
# First normalize the data



# noramlize the features

standardized_data = StandardScaler().fit_transform(testing_data_set)

#implement PCA using scikit-learn
pca = PCA()
principalcomponents = pca.fit_transform(standardized_data)
principalDf = pd.DataFrame(data=principalcomponents[:,:2],
                           columns=['principal component 1', 'principal component 2'])


#Visualize 2D projection
fig = plt.figure(figsize=(6,8))
plt.scatter(principalDf['principal component 1'],
            principalDf['principal component 2'],marker='o',color='r',s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


robust_covariance = MinCovDet().fit(principalcomponents[:,:12])

# Get the Mahalanobis distance
Mahalanobis_distance = robust_covariance.mahalanobis(principalcomponents[:,:12])
#plot fault detection distance from center threshold
fig = plt.figure(figsize=(15,8))
plt.plot(Mahalanobis_distance)
plt.ylabel('Distance from gravity center of training set')
plt.title('Mahalanobis Distance')
plt.show()

fig = plt.figure(figsize=(15,8))
plt.plot(Mahalanobis_distance)
plt.plot(threshold_mahalanobis  + 0 * Mahalanobis_distance,label='decision threshold')
plt.ylabel('Distance from gravity center of training set')
plt.title('Mahalanobis Distance with Decision threshold')
plt.xlim(0,558)
plt.legend(loc='best')
plt.show()

colors = [plt.cm.jet(float(i)/max(Mahalanobis_distance)) for i in Mahalanobis_distance]
fig = plt.figure(figsize=(6,8))
with plt.style.context(('ggplot')):
    plt.scatter(principalDf['principal component 1'],
            principalDf['principal component 2'], c = colors, edgecolors = 'k', s = 60)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim((-8,8))
    plt.ylim((-8,8))
    plt.title('Mahalanobis Distance Score Plot')
    plt.show()



'''
fig = plt.figure(figsize=(6,8))
plt.scatter(principalDf['principal component 1'],
            principalDf['principal component 2'],marker='o',color='r',s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim((-8,8))
plt.ylim((-8,8))
plt.show()
###################################################################
x=file.values
print(x.shape)

#data1=x[0:3,:]
#plt.plot(data1[0,:])
#plt.plot(data1[1,:])
#plt.plot(data1[2,:])
#plt.show()

x_scaled = preprocessing.normalize(x)
df = pd.DataFrame(x_scaled)
#print(df)
cov=np.cov(df)
print(cov)
plt.imshow(cov, extent=[-1, 1, -1, 1])
plt.show()
U, D, V = np.linalg.svd(cov)
print(f'u={U};D={D}, V={V}')
print(U)'''

'''for i in range(5):
    plt.plot(training_data_set.iloc[np.arange(3*i,17,3),:])
plt.show()'''
    
    
    #################################################################################
    