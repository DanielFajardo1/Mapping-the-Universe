#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.io import fits
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import convolve, Gaussian1DKernel
import pyspeckit
from scipy.ndimage import convolve
from scipy.integrate import trapz
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.automl import H2OAutoML
from scipy.stats import f_oneway
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import BatchNormalization

import requests
from bs4 import BeautifulSoup
import os
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from tqdm import tqdm
import time
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import chisquare
from scipy.interpolate import interp1d


from sklearn.metrics import confusion_matrix
import seaborn as sns


# ### We will attempt to do it automatically for all standardized spectra

# In[ ]:





# In[2]:


wave= fits.open('B_R_Z_wavelenght.fits')
Bwave = wave[1].data
Rwave = wave[2].data
Zwave = wave[3].data
wavelenght = np.hstack((Bwave, Rwave, Zwave)) #Contiene la cadena completa de longitudes de onda B+Z+R para cada espectro


# In[3]:


archivos= ['DataDESI_76.fits', 'DataDESI_152.fits', 'DataDESI_213.fits', 'DataDESI_284.fits', 'DataDESI_351.fits', 'DataDESI_438.fits'
             , 'DataDESI_530.fits', 'DataDESI_606.fits', 'DataDESI_690.fits', 'DataDESI_752.fits']
#archivos= ['DataDESI_76.fits']


#Generamos las listas con los datos:
Spectra_set= None #Este tensor contiene los elementos de flujo completo R+Z+B
y=np.array([]) #Esta lsita contiene las etiquetas para el ejercicio de clasificacion
z=np.array([]) #Esta matriz contiene los corrimientos z para el ejercicio de regresion

for h in range(len(archivos)):
    espc = fits.open(archivos[h]) #open file
    len_espc= len(espc[2].data)
    
    #leemos la informacion
    Bflux=espc[2].data
    Zflux=espc[4].data
    Rflux=espc[3].data
    
    spectra=np.hstack((Bflux, Rflux, Zflux)) #Contiene la cadena completa de flujo B+Z+R para cada espectro
#    spectra=spectra.reshape(spectra.shape[0], spectra.shape[1], 1)
    
    if Spectra_set is None:
        Spectra_set = spectra
    else:
        Spectra_set = np.concatenate((Spectra_set, spectra), axis=0)
    
    y=np.append(y, Table.read(espc, hdu=1)['SPECTYPE'].data)
    
#    z=np.append(z, Table.read(espc, hdu=1)['Z'].data)
#    z=z.reshape(-1,1)

spectra= Spectra_set


# ### Haremos MAD en rangos de a 1000 A, para encontrar espectros defectuosos
spectra
wavelenght
# In[4]:


# Calculamos los límites de los intervalos
num_intervals = 6
#num_intervals = 6
interval_limits = np.linspace(wavelenght.min(), wavelenght.max(), num_intervals + 1)

#Creamos las listas en las que guardaremos 
mad_parameters = [[] for _ in range(num_intervals)]

for i in range(len(interval_limits)-1):
    mascara = (wavelenght >= interval_limits[i]) & (wavelenght <= interval_limits[i+1])
    wavelenght_filtrado= wavelenght[mascara]
    for spec in spectra:
        spec_filtered= spec[mascara]
        mad = np.median(np.abs(spec_filtered-np.median(spec_filtered)))
        mad_parameters[i].append(mad)
    print('listo')


# In[5]:


interval_limits


# In[6]:


np.array(mad_parameters).shape


# In[7]:


#Barrido horizontal
# Inicializar la lista de índices a comprobar
check = []

# Iterar sobre las seis posiciones de las listas en mad_parameters
for j in range(len(mad_parameters[0])):
    # Calcular el máximo de las seis listas en la posición j
    max_value = max(mad_list[j] for mad_list in mad_parameters)
    
    # Calcular el umbral como el 1% del valor máximo en esta posición
#    threshold = max_value * 0.15
    threshold = max_value * 0.1
    
    # Verificar si algún valor en la posición j cumple con el criterio
    for i in range(len(mad_parameters)):
        if mad_parameters[i][j] <= threshold:
            check.append((i, j))  # Añadir el índice a la lista check
            break  # Salir del bucle interno si se cumple el criterio


# In[8]:


len(check)


# In[10]:


len(spectra)


# In[11]:


# Iterar sobre los datos
for k in check:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Gráfico del espectro
    axs[0].plot(wavelenght, spectra[k[1]], color='black', label='Espectro ' + str(k))
    axs[0].set_title('Spectra_checking'+str(k))
    axs[0].set_xlabel('Wavelength')
    axs[0].set_ylabel('Flux')

    mean_flux = np.mean(spectra[k[1]])
    median_flux = np.median(spectra[k[1]])
    standard_dev = np.std(spectra[k[1]])
    skewness = 3*(mean_flux - median_flux)/ (standard_dev)

    # Histograma del espectro
    axs[1].hist(spectra[k[1]], bins=100, color='blue', alpha=0.7, label='skewness: '+str(round(skewness,3)))
    axs[1].set_title('Histogram')
    axs[1].set_xlabel('Flux')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
# Ajustar el espacio entre subgráficos
    plt.tight_layout()
# Mostrar la figura
    plt.show()


# In[12]:


threshold


# In[13]:


# Lista para almacenar los índices de los espectros que cumplen con el criterio
espectros_a_graficar = []

# Iterar sobre los datos
for k in check:
    # Histograma del espectro
    frecuencias, bins, _ = axs[1].hist(spectra[k[1]], bins=150, color='blue', alpha=0.7) #150
    # Verificar si hay alguna barra en el rango dado que tenga una altura mayor a 1400
    if any((bins[:-1] >= -0.5) & (bins[1:] <= -0.25) & (frecuencias > 1400)):
        espectros_a_graficar.append(k[1])

print(len(espectros_a_graficar))
        
# Iterar sobre los espectros a graficar
for k in espectros_a_graficar:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Gráfico del espectro
    axs[0].plot(wavelenght, spectra[k], color='black')
    axs[0].set_title('Spectra_checking'+str(k))
    axs[0].set_xlabel('Wavelength')
    axs[0].set_ylabel('Flux')
# Histograma del espectro
    mean_flux = np.mean(spectra[k])
    median_flux = np.median(spectra[k])
    standard_dev = np.std(spectra[k])
    skewness = 3*(mean_flux - median_flux)/ (standard_dev)
    
    axs[1].hist(spectra[k], bins=200, color='blue', alpha=0.7, label='skewness'+str(round(skewness,3))) #150
    axs[1].set_title('Histogram')
    axs[1].set_xlabel('Flux')
    axs[1].set_ylabel('Frecuency')
    axs[1].legend()
# Ajustar el espacio entre subgráficos
    plt.tight_layout()
# Mostrar la figura
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


MAD= []
MEAN= []
MEDIAN= []
STD= []

for spec in spectra:
    mad = np.median(np.abs(spec-np.median(spec)))
    mean = np.mean(spec)
    median = np.median(spec)
    std = np.std(spec)
    
    MAD.append(mad)
    MEAN.append(mean)
    MEDIAN.append(median)
    STD.append(std)

MAD = np.array(MAD)
MEAN = np.array(MEAN)
MEDIAN = np.array(MEDIAN)
STD = np.array(STD)

MEAN_MEAN = np.mean(MEAN)
MEAN_MAD = np.mean(MAD)
MEAN_MEDIAN = np.mean(MEDIAN)
MEAN_STD = np.mean(STD)


# In[20]:


# Crear una figura con subfiguras de 2x2
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# Histograma de MEAN
axs[0, 0].hist(MEAN, bins=150, color='red', edgecolor='black')
axs[0, 0].set_title('Histograma de la distribución de MEAN')
axs[0, 0].set_xlabel('Valor de MEAN')
axs[0, 0].set_ylabel('Frecuencia')
# Histograma de MAD
axs[0, 1].hist(MAD, bins=150, color='green', edgecolor='black')
axs[0, 1].set_title('Histograma de la distribución de MAD')
axs[0, 1].set_xlabel('Valor de MAD')
axs[0, 1].set_ylabel('Frecuencia')
# Histograma de MEDIAN
axs[1, 0].hist(MEDIAN, bins=150, color='blue', edgecolor='black')
axs[1, 0].set_title('Histograma de la distribución de MEDIAN')
axs[1, 0].set_xlabel('Valor de MEDIAN')
axs[1, 0].set_ylabel('Frecuencia')
# Histograma de STD
axs[1, 1].hist(STD, bins=150, color='orange', edgecolor='black')
axs[1, 1].set_title('Histograma de la distribución de STD')
axs[1, 1].set_xlabel('Valor de STD')
axs[1, 1].set_ylabel('Frecuencia')
# Ajustar el espacio entre subfiguras
plt.tight_layout()
# Mostrar la figura
plt.show()

# Crear una figura con subfiguras de 1x2
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# Primer subplot: scatter plot de MAD vs. MEDIAN
axs[0].scatter(MAD, MEDIAN, color='black')
axs[0].set_xlabel('MAD')
axs[0].set_ylabel('MEDIAN')
axs[0].axhline(y=MEAN_MEDIAN, color='r', linestyle='--')
axs[0].axvline(x=MEAN_MAD, color='r', linestyle='--')
axs[0].set_title('Scatter plot de MEDIAN vs. MAD')
# Segundo subplot: scatter plot de MEAN vs. STD
axs[1].scatter(MEAN, STD, color='black')
axs[1].set_xlabel('MEAN')
axs[1].set_ylabel('STD')
axs[1].axhline(y=MEAN_STD, color='b', linestyle='--')  # Cambia el color a azul
axs[1].axvline(x=MEAN_MEAN, color='b', linestyle='--')  # Cambia el color a azul
axs[1].set_title('Scatter plot de STD vs. MEAN')
# Ajustar el espacio entre subfiguras
plt.tight_layout()
# Mostrar la figura
plt.show()


# In[21]:


# Calcula el rango intercuartil (IQR) para MEAN
Q1_MEAN = np.percentile(MEAN, 25)
Q3_MEAN = np.percentile(MEAN, 75)
IQR_MEAN = Q3_MEAN - Q1_MEAN

# Calcula los límites para detectar valores atípicos para MEAN
lower_bound_MEAN = Q1_MEAN - 1.5 * IQR_MEAN
upper_bound_MEAN = Q3_MEAN + 1.5 * IQR_MEAN

# Identifica los índices de los valores atípicos para MEAN
outliers_MEAN_indices = np.where((MEAN < lower_bound_MEAN) | (MEAN > upper_bound_MEAN))[0]

# Calcula el rango intercuartil (IQR) para MAD
Q1_MAD = np.percentile(MAD, 25)
Q3_MAD = np.percentile(MAD, 75)
IQR_MAD = Q3_MAD - Q1_MAD

# Calcula los límites para detectar valores atípicos para MAD
lower_bound_MAD = Q1_MAD - 1.5 * IQR_MAD
upper_bound_MAD = Q3_MAD + 1.5 * IQR_MAD

# Identifica los índices de los valores atípicos para MAD
outliers_MAD_indices = np.where((MAD < lower_bound_MAD) | (MAD > upper_bound_MAD))[0]

# Calcula el rango intercuartil (IQR) para MEDIAN
Q1_MEDIAN = np.percentile(MEDIAN, 25)
Q3_MEDIAN = np.percentile(MEDIAN, 75)
IQR_MEDIAN = Q3_MEDIAN - Q1_MEDIAN

# Calcula los límites para detectar valores atípicos para MEDIAN
lower_bound_MEDIAN = Q1_MEDIAN - 1.5 * IQR_MEDIAN
upper_bound_MEDIAN = Q3_MEDIAN + 1.5 * IQR_MEDIAN

# Identifica los índices de los valores atípicos para MEDIAN
outliers_MEDIAN_indices = np.where((MEDIAN < lower_bound_MEDIAN) | (MEDIAN > upper_bound_MEDIAN))[0]

# Calcula el rango intercuartil (IQR) para STD
Q1_STD = np.percentile(STD, 25)
Q3_STD = np.percentile(STD, 75)
IQR_STD = Q3_STD - Q1_STD

# Calcula los límites para detectar valores atípicos para STD
lower_bound_STD = Q1_STD - 1.5 * IQR_STD
upper_bound_STD = Q3_STD + 1.5 * IQR_STD

# Identifica los índices de los valores atípicos para STD
outliers_STD_indices = np.where((STD < lower_bound_STD) | (STD > upper_bound_STD))[0]


# In[23]:


len(outliers_MEAN_indices)


# In[24]:


len(outliers_MAD_indices)


# In[25]:


len(outliers_MEDIAN_indices)


# In[26]:


len(outliers_STD_indices)


# In[30]:


# Encuentra los índices que están en todas las listas de outliers
outliers_indices_all = np.intersect1d(outliers_MEAN_indices, outliers_MAD_indices)
outliers_indices_all = np.intersect1d(outliers_indices_all, outliers_MEDIAN_indices)
outliers_indices_both = np.intersect1d(outliers_indices_all, outliers_STD_indices)


# In[31]:


len(outliers_indices_both)


# In[32]:


outliers_indices_both


# In[33]:


for index in outliers_indices_both:
    plt.figure()
    plt.plot(wavelenght, spectra[index], color='black')
    plt.title('Spectra index: '+str(index))
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.show()


# In[ ]:





# In[ ]:




