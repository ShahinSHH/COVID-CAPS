"""
@author: Shahin Heidarian
This code is available at https://github.com/ShahinSHH/COVID-CAPS
"""
#%% Libraries
import pandas as pd
import numpy as np
import os
from skimage import transform
import matplotlib.pyplot as plt

#%%Resizeing Images from (1024,1024) to (224,224)

def replaceMultiple(mainString, toBeReplaces, newString): #replacement of multiple characters to use in the createPostName function
    # Iterate over the strings to be replaced
    for elem in toBeReplaces :
        # Check if string is in the main string
        if elem in mainString :
            # Replace the string
            mainString = mainString.replace(elem, newString)
    
    return  mainString 


def preprocess_data(n_to_process=-1, img_shape=(224,224)):

	os.makedirs(f'./database_preprocessed/', exist_ok=True)

	train_data = pd.read_csv(r'./dataset//train_1.txt', header=None, index_col=None)[0].str.split(' ', 1)
	val_data   = pd.read_csv(r'./dataset/val_1.txt', header=None, index_col=None)[0].str.split(' ', 1)
	test_data  = pd.read_csv(r'./dataset/test_1.txt', header=None, index_col=None)[0].str.split(' ', 1)

	# number of samples to process
	train_data = train_data if (n_to_process==-1 or n_to_process>len(train_data)) else train_data[:n_to_process]
	val_data   = val_data if (n_to_process ==-1 or n_to_process>len(val_data)) else val_data[:n_to_process]
	test_data  = test_data if (n_to_process ==-1 or n_to_process>len(test_data)) else test_data[:n_to_process]


	train_paths = train_data.apply(lambda x: r'./database/' + x[0]).to_numpy()
	val_paths   = val_data.apply(lambda x: r'./database/' + x[0]).to_numpy()
	test_paths  = test_data.apply(lambda x: r'./database/' + x[0]).to_numpy()
	all_paths   = np.hstack((train_paths, val_paths, test_paths))

	i=0
	for img_path in all_paths:
		i += 1
		if  i % max(1, int(len(all_paths)/1000))==0: print(i, '/', len(all_paths))
		new_path = img_path.replace('database', 'database_preprocessed'); new_path = replaceMultiple(new_path,['images_001/', 'images_002/', 'images_003/','images_004/', 'images_005/','images_006/', 'images_007/','images_008/', 'images_009/', 'images_010/', 'images_011/', 'images_012/'],'')
		img = plt.imread(img_path)
		img = transform.resize(img, img_shape, anti_aliasing=True)
		plt.imsave(fname=new_path, arr=img, cmap='gray')

preprocess_data(n_to_process=-1, img_shape=(224,224))
