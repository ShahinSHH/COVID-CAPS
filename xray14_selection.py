"""
@author: Shahin Heidarian
This code is available at https://github.com/ShahinSHH/COVID-CAPS
"""
#%% Libraries
import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
from itertools import chain
from numpy import save
#%% Data Selection

all_xray_df = pd.read_csv('Data_Entry_2017.csv') #loading the csv file
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('.', 'data', 'images*', '*.png'))}


all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]

#Converting Labels to 0 and 1
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)


print('Clean Labels ({})'.format(len(all_labels)), 
      [(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])


X = all_xray_df.iloc[:,:].values # Converting datafram to numpy array

Y = X[:,12::] #Selecting Labels as Y
X_paths = X[:,0] #File names

Y_cat = np.zeros((Y.shape[0],5)) #Empty array for new categories

#Creating new categories
Y_cat[:,0] = Y[:,10] #No finding
Y_cat[:,1] = Y[:,8] + Y[:,9] + Y[:,11] #Tumor
Y_cat[:,2] = Y[:,4] + Y[:,12] + Y[:,14] #Pleural disease
Y_cat[:,3] = Y[:,2] + Y[:,13] #Lung infection
Y_cat[:,4] = Y[:,0] + Y[:,1] + Y[:,3] + Y[:,5] + Y[:,6] + Y[:,7] #Others

Y_cat = (Y_cat>=1)*1

#Removing cases with multiple labels
cat_sum = np.sum(Y_cat,axis=1)
nonselect_id = []
for i in range(len(cat_sum)):
    if cat_sum[i]>1:
        nonselect_id.append(i) 

#Reduced dataset without multi-labels        
X_r = np.delete(X,nonselect_id,0)
Y_cat_r =  np.delete(Y_cat,nonselect_id,0) #final labels
X_paths_r = X_r[:,0] #final paths


#%%Converting Train Images/Labels into numpy array
# Note : Running this section may take several hours
data_path = r'./database_preprocessed/'

X_image = cv2.imread(data_path + X_paths_r[0])
X_image = np.expand_dims(X_image, axis = 0)
Y_labels = Y_cat_r[0]
Y_labels = np.expand_dims(Y_labels, axis = 0)
k = 0 #counter
for i in range(1,len(X_paths_r)):
    next_image = cv2.imread(data_path + X_paths_r[i])
    next_image = np.expand_dims(next_image, axis = 0)
    next_label =  Y_cat_r[i]
    next_label = np.expand_dims(next_label, axis = 0)
    X_image = np.concatenate((X_image,next_image), axis=0)
    Y_labels = np.concatenate((Y_labels,next_label), axis=0)
    k+=1 #Counter to check the progress
    if (k%1000==0):
        print(k,'/',len(X_paths_r))

#Saving the created numpy array
save('X_image',X_image)
save('Y_labels',Y_labels)



