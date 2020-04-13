#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: parnian 
"""

from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import numpy as np
import keras
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

K.set_image_data_format('channels_last')

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)


class Capsule(Layer):
   

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,
        
       
           
        })
        return config

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)




batch_size = 16  
num_classes = 2
epochs = 100     



x_test=  np.load("x_test.npy")
y_test=  np.load("y_test.npy")>=3







#model: model without pre-training

input_image = Input(shape=(None, None, 3))
x = Conv2D(64, (3, 3), activation='relu')(input_image)
x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = AveragePooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)




x = Reshape((-1, 128))(x)
x = Capsule(32, 8, 3, True)(x)  
x = Capsule(32, 8, 3, True)(x)   
capsule = Capsule(2, 16, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)


#model2: model with pre-training
input_image2 = Input(shape=(None, None, 3))
x2 = Conv2D(64, (3, 3), activation='relu')(input_image2)
x2=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x2)
x2 = Conv2D(64, (3, 3), activation='relu')(x2)
x2 = AveragePooling2D((2, 2))(x2)
x2 = Conv2D(128, (3, 3), activation='relu')(x2)
x2 = Conv2D(128, (3, 3), activation='relu')(x2)




x2 = Reshape((-1, 128))(x2)
x2 = Capsule(32, 8, 3, True)(x2)  
x2 = Capsule(32, 8, 3, True)(x2)   
capsule2 = Capsule(2, 16, 3, True)(x2)
output2 = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule2)




model = Model(inputs=[input_image], outputs=[output])
model2 = Model(inputs=[input_image2], outputs=[output2])



model.load_weights('weights-improvement-binary-86.h5')
model2.load_weights('weights-improvement-binary-after-44.h5')

predict=model.predict([x_test])
predict=np.argmax(predict,axis=1)

predict2=model2.predict([x_test])
predict2=np.argmax(predict2,axis=1)


summation1=0


for i in range(len(x_test)):
    if predict[i]==y_test[i]:
        summation1=summation1+1
        
accuracy_before=summation1/len(x_test)



summation1=0
summation2=0


for i in range(len(x_test)):
    if predict[i]==y_test[i] and y_test[i]==0:
        summation1=summation1+1
        
specificity_before=summation1/np.count_nonzero(y_test==0)

for i in range(len(x_test)):
    if predict[i]==y_test[i] and y_test[i]==1:
        summation2=summation2+1
        
sensitivity_before=summation2/np.count_nonzero(y_test==1)


summation1=0


for i in range(len(x_test)):
    if predict2[i]==y_test[i]:
        summation1=summation1+1
        
accuracy_after=summation1/len(x_test)



summation1=0
summation2=0


for i in range(len(x_test)):
    if predict2[i]==y_test[i] and y_test[i]==0:
        summation1=summation1+1
        
specificity_after=summation1/np.count_nonzero(y_test==0)

for i in range(len(x_test)):
    if predict2[i]==y_test[i] and y_test[i]==1:
        summation2=summation2+1
        
sensitivity_after=summation2/np.count_nonzero(y_test==1)

y_test = utils.to_categorical(y_test, num_classes)
y_score_before=model.predict([x_test])
y_score_after=model2.predict([x_test])


fpr_before = dict()
tpr_before = dict()
roc_auc_before = dict()
for i in range(num_classes):
    fpr_before[i], tpr_before[i], _ = roc_curve(y_test[:, i], y_score_before[:, i])
    roc_auc_before[i] = auc(fpr_before[i], tpr_before[i])

fpr_before["micro"], tpr_before["micro"], _ = roc_curve(y_test.ravel(), y_score_before.ravel())
roc_auc_before["micro"] = auc(fpr_before["micro"], tpr_before["micro"])


fpr_after = dict()
tpr_after = dict()
roc_auc_after = dict()
for i in range(num_classes):
    fpr_after[i], tpr_after[i], _ = roc_curve(y_test[:, i], y_score_after[:, i])
    roc_auc_after[i] = auc(fpr_after[i], tpr_after[i])

fpr_after["micro"], tpr_after["micro"], _ = roc_curve(y_test.ravel(), y_score_after.ravel())
roc_auc_after["micro"] = auc(fpr_after["micro"], tpr_after["micro"])

plt.rcParams.update({'font.size': 13})


plt.figure()
lw = 3
plt.plot(fpr_before[1], tpr_before[1], color='darkorange',
         lw=lw, label='ROC curve without pre-train (area = %0.2f)' % roc_auc_before[1])
plt.plot(fpr_after[1], tpr_after[1], color='green',
         lw=lw, label='ROC curve with pre-train (area = %0.2f)' % roc_auc_after[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()




 
