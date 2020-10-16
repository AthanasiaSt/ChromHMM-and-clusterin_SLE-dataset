#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:54:32 2020

@author: labuser
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import operator
from sklearn.decomposition import PCA
from itertools import compress
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster, leaves_list, cophenet, dendrogram, linkage
import re
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from scipy.stats import chi2_contingency, stats

#___CLustering of the lupus samples (~212 totally but only 200 are considered in the analysis) based on the emission probabilities resulting from the 10 state Chrom_HMM____# 

#loading the data in a dataframe 
emission_df = pd.read_csv("emissions_10.txt",sep='\t') #read the text file as dataframes

columns = emission_df.columns.tolist() #save the column names-->sample names 

emission_df = emission_df.drop([columns[0]], axis=1)#drop the first column which shows the states

columns = columns[1::] #delete also the first column name

emission_df = emission_df.T #Transope, rows --> samples and columns --> features

N = emission_df.shape[0] #keeping the dimensions
D = emission_df.shape[1]


#----------exctracting the labels and deleting the samples not considered in the analysis---------------
labels_final = pd.read_csv("/home/labuser/Documents/info_SLE_RNA/info.csv",sep='\,', engine = 'python') #read the csv files as dataframes

#change the custom made sample names to be the same as in labels_final. Also subtract the ones not considered in the analysis
new_labels = []
for elem in columns:
    pattern = '^GB([0-9]+)'
    if len(elem) == 3:
        x = re.search(pattern, elem)
        new_labels+=['GB00'+x.group(1)]
        
    elif len(elem) == 4:
        x = re.search(pattern, elem)
        new_labels+=['GB0'+x.group(1)]
        
    else:
        new_labels+=[elem]


emission_df.index = new_labels #replace the sample names of the dataframe with the new ones that correspond to the labels exctracted from info.csv
emission_df = emission_df.loc[labels_final['Sample ID'].tolist()] #final dataframe 

#making a copy of the dataframe that includes the real labels and other information on the samples
emission_df_labels = emission_df.copy() #make a copy of the dataframe
emission_df_labels['Emission'] = labels_final['Disease'].tolist() # save the labels Healthy/SLE
emission_df_labels['DAI'] = labels_final['CLUSTER_DAI'].tolist() #save the labels of the clinical SLE DAI as three groups [0-2], [3-8], [9--)
emission_df_labels['DAI']=emission_df_labels['DAI'].astype(str) #make them as string 
emission_df_labels['merge'] = emission_df_labels[['Emission', 'DAI']].apply(lambda x: '--'.join(x), axis=1) #merge two labels

#--------------------------------------------------------------------------------------------------------------------------
#------------initial plotting of the raw data---------
sns.set()
plt.figure()
    
sns_plot=sns.scatterplot(x=emission_df[3], y=emission_df[4], hue=emission_df_labels['Emission']).set_title('Emission probabilities of all the samples, State 4 vs State 5')

plt.xlabel("state4")
plt.ylabel("state5")

# fig = sns_plot.get_figure()
# fig.savefig('4_6_emissions.png', dpi=300)

#--------------------------------------------------------------------------------------------------------------------------
#-----------PCA------------

def PCA_(emission_df, emission_df_labels): #input the dataframe and the one with the labels 
    
#-----find the number of PCs that return approximately 0.95% of the total variance -----------

    # print('-----------Conducting Conventional PCA----------------')
    
    # #-----deciding the number of components based on the explained variance 
    
    # i=min(N,D)
    # while i != 0 :
        
    #     pca = PCA(n_components=i)
    #     pca.fit(emission_df)
    #     principalComponents = pca.transform(emission_df)
    
    #     explained_var = pca.explained_variance_ratio_
    #     variance = np.sum(explained_var)
        
    #     if variance <= 0.95:
    #         components=i
    #         break 
        
    #     i = i-1
        
    # print('\n',components,'PCs were kept, explaining',variance,'% of the total variance\n')
    
    #conducting PCA
    # pca = PCA(n_components=components)
    # pca.fit(emission_df)
    # principalComponents = pca.transform(emission_df)
    
    print('-----------Conducting Conventional PCA, keeping three components----------------')

    #conducting conventional PCA with 3 components
    pca = PCA(n_components=3)
    pca.fit(emission_df)
    principalComponents = pca.transform(emission_df)
    
    
    ##--------2D PCA plot----------
    sns.set()
    plt.figure()
        
    sns.set_palette('colorblind')
    sns_plot=sns.scatterplot(x=principalComponents[:,0], y=principalComponents[:,1], hue=emission_df_labels['Emission']).set_title('The first two principal components')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # fig = sns_plot.get_figure()
    # fig.savefig('2_pca.png', dpi=300)

    #-------------3D PCA plot-----------------    
    sns.set()
    plt.figure()
    
    colors = {'SLE' : 'b',
              'Healthy' : 'r'}
    
    ax = plt.axes(projection='3d')
    sns_plot=ax.scatter(principalComponents[:,0], principalComponents[:,1], principalComponents[:,2], c=[colors[val] for val in emission_df_labels['Emission']], cmap='gist_stern')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title('Principal components')
    ax.view_init(0, 100) #change the viewpoint of the plot
    plt.show()
    
    # ax = sns_plot.get_figure()
    # ax.savefig('3_30_50_PCA.png', dpi=300)
    
    
    #-------making a dataframe with the principal components---------------
    pcs=[]
    for j in range (1,3+1):
        pcs.append('PC'+str(j))    
    principalDF = pd.DataFrame(data = principalComponents, columns = pcs)
    
    return principalDF, principalComponents


principalDF, principalComponents= PCA_(emission_df,emission_df_labels) #output the principal components and a corresponding dataframe

#-----------------------------------------------------------------------------------------------------------------

def clustering(emission_df, emission_df_labels, num_clusters):
#----------------Kmeans---------------------------
#-------find the optimal number of clusters-------

   #   print('------------Conducting Kmeans------------')
   #   silhouette_scores=[]
   #   n_clust = []
    
   #   for n_clusters in list(range(2,int(N/2))):
        
   #       clusterer = KMeans(n_clusters=n_clusters)
   #       preds = clusterer.fit_predict(emission_df)
   #       # centers = clusterer.cluster_centers_
    
   #       score = silhouette_score(emission_df, preds)
   #       print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
   #       silhouette_scores.append(score)
   #       n_clust += [n_clusters]
        
   #   max_index, max_value = max(enumerate(silhouette_scores), key=operator.itemgetter(1))
        
   #   print('\n The max silhouette score is accomplished for', n_clust[max_index],'clusters and is equal to',max_value)
        
   #   #-------plotting the Silhouette values--------------------
   #   sns.set()
   #   fig_sil = plt.figure(figsize = (7,6))
   #   ax_sil = fig_sil.add_subplot(1,1,1)
    
   #   ax_sil.plot(range(2,int(N/2)), silhouette_scores, marker = 'o', c = "black")
   #   ax_sil.set_title('Silhouette values vs number of clusters')
   #   ax_sil.set_xlabel('Number of Clusters')
   #   ax_sil.set_ylabel('Silhouette score')
   #   plt.show()
    
   #   fig = ax_sil.get_figure()
   #   fig.savefig('silhouette.png', dpi=300)
  
   #   #-----------____conducting k_means____-------
    
   #   kmeans = KMeans(n_clusters=n_clust[max_index]).fit(emission_df)
   #   predicted_labels = kmeans.labels_
   #   centroids = kmeans.cluster_centers_
    
   #   #--------exctracting the samples for each resulting cluster-----------
    
   #   group1_kmeans = list(compress(columns,np.array([predicted_labels], dtype=bool)[0]))
   #   group2_kmeans = list(compress(columns,~np.array([predicted_labels], dtype=bool)[0]))
    
    
   #   #------------K-means plotting----------------------------------
    
   #   df_plot_kmeans = emission_df.copy()
   #   df_plot_kmeans['predicted_labels'] = predicted_labels

   #   df_plot = emission_df[emission_df.columns[0::]].copy()
   #   df_plot = df_plot.rename(columns={3:'state4',4:'state5'})
        
   #   sns.set()
   #   plt.figure(figsize = (7,6))    
   #   sns_plot = sns.scatterplot(x=df_plot['state4'], y=df_plot['state5'], data=df_plot, hue=df_plot_kmeans['predicted_labels'],palette='Dark2').set_title('Clustered data, K-means, without PCA')
    
   #   # fig = sns_plot.get_figure()
   #   # fig.savefig('clustred_k-means_pca.png', dpi=300)
    
    
   #    #---------------Mixture of Gaussian------------------

   #   gmm = GaussianMixture(n_components=n_clust[max_index]).fit(emission_df)
   #   weights = gmm.weights_
   #   predicted_labels = gmm.predict(emission_df)
    
   #   df_plot_MoG = emission_df.copy()
   #   df_plot_MoG['predicted_labels'] = predicted_labels
        
   #   df_plot = emission_df[emission_df.columns[7::]].copy()
   #   df_plot = df_plot.rename(columns={3:'state8',4:'state10'})
       
   #   sns.set()
   #   plt.figure(figsize = (7,6))         
   #   sns_plot = sns.scatterplot(x=df_plot['state4'], y=df_plot['state5'], data=df_plot, hue=df_plot_MoG['predicted_labels'],palette='Dark2').set_title('Clustered data, Mixture of Gaussians, without PCA')
    
   #   # fig = sns_plot.get_figure()
   #   # fig.savefig('clustred_MOG_pca.png', dpi=300)

   # #--------exctracting the samples for each resulting cluster-----------
    
   #   group1_MoG = list(compress(columns,np.array([predicted_labels], dtype=bool)[0]))
   #   group2_MoG = list(compress(columns,~np.array([predicted_labels], dtype=bool)[0]))

#-------------------------------------------------------------------------------------------------------------

     #-----------hierarchical clustering-------------
  
    ##--------SciPy method---------------------------
    Z = linkage(emission_df, method = 'ward', metric = 'euclidean')#performing agglomerative clustering with distance metric--> ward, try: 'cityblock' and other metrics or methods
    #z dimensions (N-1)x4, in the first two columns are the elements that are merged, in the third column is shown the distance between them and the last column
    #represents the original observations of this cluster

    c, coph_dists = cophenet(Z, pdist(emission_df)) #compares (correlates) the actual pairwise distances of all your samples to those implied by the hierarchical clustering --> close to 1
    
    print('The cophenetic correlation coeficient is:',c)    
    
    #looking at the last 4 distances of the merged samples/ trying to find any major difference among the values
    # Z[-4:,2]
    
    # leaves = leaves_list(Z) #200 samples as expected
    
    #showing all the samples 
    # calculate full dendrogram
    plt.figure(figsize=(50, 20))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        labels=emission_df_labels['Emission'],
        color_threshold = 0.7*max(Z[:,2]),
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=20.,  # font size for the x axis labels
    )
    plt.show()
    
    # plt.savefig('Dendrogram_emission.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', papertype='letter', format=None, transparent=True, bbox_inches=None, pad_inches=0.1, frameon=None) 

    #The above shows a truncated dendrogram, which only shows the last p=12 out of our 199 merges.
    
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()

    # plt.savefig('Dendrogram_truncated.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', papertype='letter', format=None, transparent=True, bbox_inches=None, pad_inches=0.1, frameon=None) 

    #choose the cut-off distance 
    max_d = 3 #three clusters
    clusters = fcluster(Z, max_d, criterion='distance')
        
    emission_df_labels['three_clusters'] = clusters  #add to the dataframe a column with the resuliing clusters
    emission_df_labels['three_clusters'].value_counts()

    grouped_emission = emission_df_labels.groupby(['merge','three_clusters']).size().reset_index().rename(columns={0:'count'}) #group the results to make the contigency table
    
    N=len(grouped_emission['three_clusters'].unique())
    D=len(grouped_emission['merge'])
    
    #making the contingency table
    arr = np.zeros((N+1, D+1), dtype=object)
    arr[0,1:] = grouped_emission['merge']
    arr[1:,0] = [i for i in range (1, N+1)]
    arr[1,list(np.asarray(grouped_emission[grouped_emission['three_clusters'] == 1].index + 1))] = grouped_emission[grouped_emission['three_clusters'] == 1]['count']
    arr[2,list(np.asarray(grouped_emission[grouped_emission['three_clusters'] == 2].index + 1))] = grouped_emission[grouped_emission['three_clusters'] == 2]['count']
    arr[3,list(np.asarray(grouped_emission[grouped_emission['three_clusters'] == 3].index + 1))] = grouped_emission[grouped_emission['three_clusters'] == 3]['count']
    # arr[4, list(np.asarray(grouped_emission[grouped_emission['three_clusters']==4].index +1))] =grouped_emission[grouped_emission['three_clusters']==4]['count']
    array = arr[1:,1:][arr[1:,1:] != 0] #making a compress table
    matrix = np.reshape(array, (N, 4)) #and them make it at least 2Dimensional
    
    test=chi2_contingency(matrix) #chi2 testing
    # stats.fisher_exact (matrix)
    
    #finding the percentages of each label belonging to each cluster 
    summ_col = np.sum(matrix, axis=0)
    matrix_percentage=matrix/summ_col
    
    return  emission_df_labels, test


emission_df_info, chi2_test = clustering(emission_df, emission_df_labels)




   
    
    