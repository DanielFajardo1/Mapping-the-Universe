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


#archivos= ['DataDESI_76.fits', 'DataDESI_152.fits', 'DataDESI_213.fits', 'DataDESI_284.fits', 'DataDESI_351.fits', 'DataDESI_438.fits'
#             , 'DataDESI_530.fits', 'DataDESI_606.fits', 'DataDESI_690.fits', 'DataDESI_752.fits']
archivos= ['DataDESI_76.fits']


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


# In[4]:


len(spectra)


# In[5]:


# Crear una lista para almacenar los índices donde se encuentra 'GALAXY', 'STAR', 'QSO'
indices = [index for index, value in enumerate(y) if value == 'GALAXY']
spectra_galaxy = np.array([spectra[index] for index in indices])


# In[6]:


indices = [index for index, value in enumerate(y) if value == 'QSO']
spectra_qso = np.array([spectra[index] for index in indices])


# In[7]:


indices = [index for index, value in enumerate(y) if value == 'STAR']
spectra_star = np.array([spectra[index] for index in indices])


# In[8]:


len(spectra_galaxy)


# In[9]:


len(spectra_qso)


# In[10]:


len(spectra_star)


# ### We calculate the Fluxes under the color bands

# In[ ]:





# In[11]:


FLUX_U_GALAXY=[]
FLUX_U_QSO=[]
FLUX_U_STAR=[]

# Definir una máscara booleana para las longitudes de onda entre 3055.11 y 4030.64 (Banda u con lambda_eff=3608.04)
mascara = (wavelenght >= 3055.11) & (wavelenght <= 4030.64)
wavelenght_filtrado= wavelenght[mascara]

for galaxy in spectra_galaxy:
    espectro_filtrado= galaxy[mascara]
#    area= trapz(espectro_filtrado, wavelenght_filtrado)
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_U_GALAXY.append(area)

for qso in spectra_qso:
    espectro_filtrado= qso[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_U_QSO.append(area)

for star in spectra_star:
    espectro_filtrado= star[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_U_STAR.append(area)


# In[13]:


FLUX_V_G_GALAXY=[]
FLUX_V_G_QSO=[]
FLUX_V_G_STAR=[]

# Definir una máscara booleana para las longitudes de onda entre 3797.64 y 5553.04 (Banda g con lambda_eff=4671.82)
mascara = (wavelenght >= 3797.64) & (wavelenght <= 5553.04)
wavelenght_filtrado= wavelenght[mascara]

for galaxy in spectra_galaxy:
    espectro_filtrado= galaxy[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_V_G_GALAXY.append(area)

for qso in spectra_qso:
    espectro_filtrado= qso[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_V_G_QSO.append(area)

for star in spectra_star:
    espectro_filtrado= star[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_V_G_STAR.append(area)


# In[ ]:


FLUX_R_GALAXY=[]
FLUX_R_QSO=[]
FLUX_R_STAR=[]

# Definir una máscara booleana para las longitudes de onda entre 5418.23 y 6994.42 (Banda r con lambda_eff=6141.12)
mascara = (wavelenght >= 5418.23) & (wavelenght <= 6994.42)
wavelenght_filtrado= wavelenght[mascara]

for galaxy in spectra_galaxy:
    espectro_filtrado= galaxy[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_R_GALAXY.append(area)

for qso in spectra_qso:
    espectro_filtrado= qso[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_R_QSO.append(area)

for star in spectra_star:
    espectro_filtrado= star[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_R_STAR.append(area)


# In[ ]:


FLUX_I_GALAXY=[]
FLUX_I_QSO=[]
FLUX_I_STAR=[]

# Definir una máscara booleana para las longitudes de onda entre 6692.41 y 8400.32 (Banda i con lambda_eff=7457.89)
mascara = (wavelenght >= 6692.41) & (wavelenght <= 8400.32)
wavelenght_filtrado= wavelenght[mascara]

for galaxy in spectra_galaxy:
    espectro_filtrado= galaxy[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_I_GALAXY.append(area)

for qso in spectra_qso:
    espectro_filtrado= qso[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_I_QSO.append(area)

for star in spectra_star:
    espectro_filtrado= star[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_I_STAR.append(area)


# In[ ]:


FLUX_z_GALAXY=[]
FLUX_z_QSO=[]
FLUX_z_STAR=[]

# Definir una máscara booleana para las longitudes de onda entre 7964.70 y 10873.33 (Banda z con lambda_eff=8922.78)
mascara = (wavelenght >= 8385) & (wavelenght <= 9875)
wavelenght_filtrado= wavelenght[mascara]

for galaxy in spectra_galaxy:
    espectro_filtrado= galaxy[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_z_GALAXY.append(area)

for qso in spectra_qso:
    espectro_filtrado= qso[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_z_QSO.append(area)

for star in spectra_star:
    espectro_filtrado= star[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_z_STAR.append(area)


# In[57]:


# Crear una figura con subgráficos para cada filtro
fig, axs = plt.subplots(5, 1, figsize=(10, 30))

# Lista de etiquetas para las clases
clases = ['GALAXY', 'STAR', 'QSO']

# Lista de listas de valores para cada filtro y clase
flux_values = [
    [FLUX_U_GALAXY, FLUX_U_STAR, FLUX_U_QSO],
    [FLUX_V_G_GALAXY, FLUX_V_G_STAR, FLUX_V_G_QSO],
    [FLUX_R_GALAXY, FLUX_R_STAR, FLUX_R_QSO],
    [FLUX_I_GALAXY, FLUX_I_STAR, FLUX_I_QSO],
    [FLUX_z_GALAXY, FLUX_z_STAR, FLUX_z_QSO]
]

# Lista para almacenar los datos filtrados
filtered_flux_values = []

# Iterar sobre cada subgráfico y crear un boxplot para cada filtro
for i in range(5):  # Cambiando el rango a 5
    ax = axs[i]
    flux_values_current = flux_values[i]

    # Crear un boxplot y obtener los límites de los bigotes
    boxplot = ax.boxplot(flux_values_current, labels=clases, notch=True, patch_artist=True)
    whiskers = boxplot['whiskers']
    
    # Obtener los límites de los bigotes
    lower_whisker, upper_whisker = whiskers[0].get_ydata()[1], whiskers[1].get_ydata()[1]
    
    # Filtrar los datos que están dentro de los bigotes
    filtered_data = [val for sublist in flux_values_current for val in sublist if lower_whisker <= val <= upper_whisker]
    filtered_flux_values.append(filtered_data)

    ax.set_xlabel('Class')
    ax.set_ylabel(f'FLUX_{["U", "G", "R", "I", "z"][i]}')
    ax.set_title(f'Distribution of FLUX_{["U", "G", "R", "I", "z"][i]} values')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[58]:


# Crear una figura con subgráficos para cada filtro
fig, axs = plt.subplots(5, 1, figsize=(10, 30))

# Lista de etiquetas para las clases
clases = ['GALAXY', 'STAR', 'QSO']

# Lista de listas de valores para cada filtro y clase
flux_values = [
    [FLUX_U_GALAXY, FLUX_U_STAR, FLUX_U_QSO],
    [FLUX_V_G_GALAXY, FLUX_V_G_STAR, FLUX_V_G_QSO],
    [FLUX_R_GALAXY, FLUX_R_STAR, FLUX_R_QSO],
    [FLUX_I_GALAXY, FLUX_I_STAR, FLUX_I_QSO],
    [FLUX_z_GALAXY, FLUX_z_STAR, FLUX_z_QSO]
]

# Iterar sobre cada subgráfico y crear un boxplot para cada filtro
for i in range(5):  # Cambiando el rango a 5
    ax = axs[i]
    flux_values_current = flux_values[i]

    # Crear un boxplot sin mostrar los outliers
    boxplot = ax.boxplot(flux_values_current, labels=clases, notch=True, patch_artist=True, showfliers=False)
    
    ax.set_xlabel('Class')
    ax.set_ylabel(f'FLUX_{["U", "G", "R", "I", "z"][i]}')
    ax.set_title(f'Distribution of FLUX_{["U", "G", "R", "I", "z"][i]} values')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[39]:


# Create a figure with subplots for each filter
fig, axs = plt.subplots(5, 1, figsize=(10, 30))

# List of labels for the classes
classes = ['GALAXY', 'STAR', 'QSO']

# List of lists of values for each filter and class
flux_values = [
    [FLUX_U_GALAXY, FLUX_U_STAR, FLUX_U_QSO],
    [FLUX_V_G_GALAXY, FLUX_V_G_STAR, FLUX_V_G_QSO],
    [FLUX_R_GALAXY, FLUX_R_STAR, FLUX_R_QSO],
    [FLUX_I_GALAXY, FLUX_I_STAR, FLUX_I_QSO],
    [FLUX_z_GALAXY, FLUX_z_STAR, FLUX_z_QSO]
]

# Iterate over each subplot and create a histogram for each filter
for i in range(5):
    ax = axs[i]
    flux_values_current = flux_values[i]
    
    # Plot histogram for each class with different colors
    for j, class_flux in enumerate(flux_values_current):
        ax.hist(class_flux, bins=10, label=classes[j], color=['blue', 'green', 'red'][j], alpha=0.7)

    # Set labels and title
    ax.set_xlabel(f'FLUX_{["U", "G", "R", "I", "z"][i]} values')
    ax.set_ylabel('Ln(Number of objects)')
    ax.set_title(f'FLUX_{["U", "G", "R", "I", "z"][i]} distribution')
    ax.legend(loc='upper right', title='Classes')  # Move the legend to each subplot

    # Apply natural logarithm to the y-axis values
    ax.set_yscale('log')

# Adjust subplots layout
plt.tight_layout()

# Show the plot
plt.show()


# ### Now, we plot the correlations between the Flux values that we will enter as parameters.

# In[40]:


# Crear una figura con subgráficos para cada filtro
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# Scatter Plot 1
axs[0, 0].scatter(FLUX_R_GALAXY, FLUX_V_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[0, 0].scatter(FLUX_R_STAR, FLUX_V_G_STAR, label='STAR', color='green', alpha=0.7)
axs[0, 0].scatter(FLUX_R_QSO, FLUX_V_G_QSO, label='QSO', color='red', alpha=0.7)
axs[0, 0].set_xlabel('FLUX_R')
axs[0, 0].set_ylabel('FLUX_G')
axs[0, 0].set_title('FLUX_R vs FLUX_G')
axs[0, 0].legend()
# Scatter Plot 2
axs[0, 1].scatter(FLUX_R_GALAXY, FLUX_z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[0, 1].scatter(FLUX_R_STAR, FLUX_z_STAR, label='STAR', color='green', alpha=0.7)
axs[0, 1].scatter(FLUX_R_QSO, FLUX_z_QSO, label='QSO', color='red', alpha=0.7)
axs[0, 1].set_xlabel('FLUX_R')
axs[0, 1].set_ylabel('FLUX_Z')
axs[0, 1].set_title('FLUX_R vs FLUX_Z')
axs[0, 1].legend()
# Scatter Plot 3
axs[1, 0].scatter(FLUX_R_GALAXY, FLUX_U_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[1, 0].scatter(FLUX_R_STAR, FLUX_U_STAR, label='STAR', color='green', alpha=0.7)
axs[1, 0].scatter(FLUX_R_QSO, FLUX_U_QSO, label='QSO', color='red', alpha=0.7)
axs[1, 0].set_xlabel('FLUX_R')
axs[1, 0].set_ylabel('FLUX_U')
axs[1, 0].set_title('FLUX_R vs FLUX_U')
axs[1, 0].legend()
# Scatter Plot 4
axs[1, 1].scatter(FLUX_R_GALAXY, FLUX_I_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[1, 1].scatter(FLUX_R_STAR, FLUX_I_STAR, label='STAR', color='green', alpha=0.7)
axs[1, 1].scatter(FLUX_R_QSO, FLUX_I_QSO, label='QSO', color='red', alpha=0.7)
axs[1, 1].set_xlabel('FLUX_R')
axs[1, 1].set_ylabel('FLUX_I')
axs[1, 1].set_title('FLUX_R vs FLUX_I')
axs[1, 1].legend()
# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[41]:


# Crear una figura con subgráficos para cada filtro
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Scatter Plot 1
axs[0].scatter(FLUX_U_GALAXY, FLUX_V_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[0].scatter(FLUX_U_STAR, FLUX_V_G_STAR, label='STAR', color='green', alpha=0.7)
axs[0].scatter(FLUX_U_QSO, FLUX_V_G_QSO, label='QSO', color='red', alpha=0.7)
axs[0].set_xlabel('FLUX_U')
axs[0].set_ylabel('FLUX_G')
axs[0].set_title('FLUX_U vs FLUX_G')
axs[0].legend()

# Scatter Plot 2
axs[1].scatter(FLUX_U_GALAXY, FLUX_I_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[1].scatter(FLUX_U_STAR, FLUX_I_STAR, label='STAR', color='green', alpha=0.7)
axs[1].scatter(FLUX_U_QSO, FLUX_I_QSO, label='QSO', color='red', alpha=0.7)
axs[1].set_xlabel('FLUX_U')
axs[1].set_ylabel('FLUX_I')
axs[1].set_title('FLUX_U vs FLUX_I')
axs[1].legend()

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Crear un scatter plot
plt.scatter(FLUX_U_GALAXY, FLUX_z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
plt.scatter(FLUX_U_STAR, FLUX_z_STAR, label='STAR', color='green', alpha=0.7)
plt.scatter(FLUX_U_QSO, FLUX_z_QSO, label='QSO', color='red', alpha=0.7)
# Añadir etiquetas y título
plt.xlabel('FLUX_U')
plt.ylabel('FLUX_Z')
plt.title('FLUX_U vs FLUX_Z')
# Añadir leyenda
plt.legend()
# Mostrar el gráfico
plt.show()


# In[42]:


# Crear una figura con subgráficos para cada filtro
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
# Scatter Plot 1
axs[0].scatter(FLUX_V_G_GALAXY, FLUX_I_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[0].scatter(FLUX_V_G_STAR, FLUX_I_STAR, label='STAR', color='green', alpha=0.7)
axs[0].scatter(FLUX_V_G_QSO, FLUX_I_QSO, label='QSO', color='red', alpha=0.7)
axs[0].set_xlabel('FLUX_G')
axs[0].set_ylabel('FLUX_I')
axs[0].set_title('FLUX_V_G vs FLUX_I')
axs[0].legend()
# Scatter Plot 2
axs[1].scatter(FLUX_V_G_GALAXY, FLUX_z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[1].scatter(FLUX_V_G_STAR, FLUX_z_STAR, label='STAR', color='green', alpha=0.7)
axs[1].scatter(FLUX_V_G_QSO, FLUX_z_QSO, label='QSO', color='red', alpha=0.7)
axs[1].set_xlabel('FLUX_G')
axs[1].set_ylabel('FLUX_Z')
axs[1].set_title('FLUX_V_G vs FLUX_z')
axs[1].legend()
# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[43]:


# Crear un scatter plot
plt.scatter(FLUX_I_GALAXY, FLUX_z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
plt.scatter(FLUX_I_STAR, FLUX_z_STAR, label='STAR', color='green', alpha=0.7)
plt.scatter(FLUX_I_QSO, FLUX_z_QSO, label='QSO', color='red', alpha=0.7)
# Añadir etiquetas y título
plt.xlabel('FLUX_I')
plt.ylabel('FLUX_Z')
plt.title('FLUX_I vs FLUX_Z')
# Añadir leyenda
plt.legend()
# Mostrar el gráfico
plt.show()


# ### We extract the features: u, g, r, i, z, (u-r), (u-g), (g-z), (r-z), (i-z)

# In[16]:


#Features con los los flujos
FLUX_U_GALAXY= np.array(FLUX_U_GALAXY)
FLUX_U_STAR= np.array(FLUX_U_STAR)
FLUX_U_QSO= np.array(FLUX_U_QSO)
FLUX_U= np.concatenate([FLUX_U_GALAXY, FLUX_U_STAR, FLUX_U_QSO])

FLUX_G_GALAXY= np.array(FLUX_V_G_GALAXY)
FLUX_G_STAR= np.array(FLUX_V_G_STAR)
FLUX_G_QSO= np.array(FLUX_V_G_QSO)
FLUX_G= np.concatenate([FLUX_G_GALAXY, FLUX_G_STAR, FLUX_G_QSO])

FLUX_R_GALAXY= np.array(FLUX_R_GALAXY)
FLUX_R_STAR= np.array(FLUX_R_STAR)
FLUX_R_QSO= np.array(FLUX_R_QSO)
FLUX_R= np.concatenate([FLUX_R_GALAXY, FLUX_R_STAR, FLUX_R_QSO])

FLUX_I_GALAXY= np.array(FLUX_I_GALAXY)
FLUX_I_STAR= np.array(FLUX_I_STAR)
FLUX_I_QSO= np.array(FLUX_I_QSO)
FLUX_I= np.concatenate([FLUX_I_GALAXY, FLUX_I_STAR, FLUX_I_QSO])

FLUX_Z_GALAXY= np.array(FLUX_z_GALAXY)
FLUX_Z_STAR= np.array(FLUX_z_STAR)
FLUX_Z_QSO= np.array(FLUX_z_QSO)
FLUX_Z= np.concatenate([FLUX_Z_GALAXY, FLUX_Z_STAR, FLUX_Z_QSO])

#Features con indices de color
U_R_GALAXY= FLUX_U_GALAXY-FLUX_R_GALAXY
U_R_STAR= FLUX_U_STAR-FLUX_R_STAR
U_R_QSO= FLUX_U_QSO-FLUX_R_QSO
U_R= np.concatenate([U_R_GALAXY, U_R_STAR, U_R_QSO])

U_G_GALAXY= FLUX_U_GALAXY-FLUX_G_GALAXY
U_G_STAR= FLUX_U_STAR-FLUX_G_STAR
U_G_QSO= FLUX_U_QSO-FLUX_G_QSO
U_G= np.concatenate([U_G_GALAXY, U_G_STAR, U_G_QSO])

G_Z_GALAXY= FLUX_G_GALAXY-FLUX_Z_GALAXY
G_Z_STAR= FLUX_G_STAR-FLUX_Z_STAR
G_Z_QSO= FLUX_G_QSO-FLUX_Z_QSO
G_Z= np.concatenate([G_Z_GALAXY, G_Z_STAR, G_Z_QSO])

R_Z_GALAXY= FLUX_R_GALAXY-FLUX_Z_GALAXY
R_Z_STAR= FLUX_R_STAR-FLUX_Z_STAR
R_Z_QSO= FLUX_R_QSO-FLUX_Z_QSO
R_Z= np.concatenate([R_Z_GALAXY, R_Z_STAR, R_Z_QSO])

I_Z_GALAXY= FLUX_I_GALAXY-FLUX_Z_GALAXY
I_Z_STAR= FLUX_I_STAR-FLUX_Z_STAR
I_Z_QSO= FLUX_I_QSO-FLUX_Z_QSO
I_Z= np.concatenate([I_Z_GALAXY, I_Z_STAR, I_Z_QSO])

#Features con razones entres flujos
RU_GALAXY= FLUX_R_GALAXY/FLUX_U_GALAXY
RU_STAR= FLUX_R_STAR/FLUX_U_STAR
RU_QSO= FLUX_R_QSO/FLUX_U_QSO
RU= np.concatenate([RU_GALAXY, RU_STAR, RU_QSO])

GZ_GALAXY= FLUX_G_GALAXY/FLUX_Z_GALAXY
GZ_STAR= FLUX_G_STAR/FLUX_Z_STAR
GZ_QSO= FLUX_G_QSO/FLUX_Z_QSO
GZ= np.concatenate([GZ_GALAXY, GZ_STAR, GZ_QSO])

RZ_GALAXY= FLUX_R_GALAXY/FLUX_Z_GALAXY
RZ_STAR= FLUX_R_STAR/FLUX_Z_STAR
RZ_QSO= FLUX_R_QSO/FLUX_Z_QSO
RZ= np.concatenate([RZ_GALAXY, RZ_STAR, RZ_QSO])

IZ_GALAXY= FLUX_I_GALAXY/FLUX_Z_GALAXY
IZ_STAR= FLUX_I_STAR/FLUX_Z_STAR
IZ_QSO= FLUX_I_QSO/FLUX_Z_QSO
IZ= np.concatenate([IZ_GALAXY, IZ_STAR, IZ_QSO])

UG_GALAXY= FLUX_U_GALAXY/FLUX_G_GALAXY
UG_STAR= FLUX_U_STAR/FLUX_G_STAR
UG_QSO= FLUX_U_QSO/FLUX_G_QSO
UG= np.concatenate([UG_GALAXY, UG_STAR, UG_QSO])

UZ_GALAXY= FLUX_U_GALAXY/FLUX_Z_GALAXY
UZ_STAR= FLUX_U_STAR/FLUX_Z_STAR
UZ_QSO= FLUX_U_QSO/FLUX_Z_QSO
UZ= np.concatenate([UZ_GALAXY, UZ_STAR, UZ_QSO])

STYPE=[]
for i in range(len(FLUX_U_GALAXY)):
    STYPE.append('GALAXY')
for j in range(len(FLUX_U_STAR)):
    STYPE.append('STAR')
for k in range(len(FLUX_U_QSO)):
    STYPE.append('QSO')
STYPE= np.array(STYPE)


# In[46]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [U_R_GALAXY, U_R_STAR, U_R_QSO],
    [U_G_GALAXY, U_G_STAR, U_G_QSO],
    [G_Z_GALAXY, G_Z_STAR, G_Z_QSO],
    [R_Z_GALAXY, R_Z_STAR, R_Z_QSO],
    [I_Z_GALAXY, I_Z_STAR, I_Z_QSO],
    [I_Z_GALAXY, I_Z_STAR, I_Z_QSO]
]

# Etiquetas para los boxplots
labels = ['U-R', 'U-G', 'G-Z', 'R-Z', 'I-Z', 'I-Z']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j])
        ax.set_xticklabels(['GALAXY', 'STAR', 'QSO'])
        ax.set_title('Distribution of '+labels[i * 3 + j]+' values')
        ax.set_xlabel('Class')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[47]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [U_R_GALAXY, U_R_STAR, U_R_QSO],
    [U_G_GALAXY, U_G_STAR, U_G_QSO],
    [G_Z_GALAXY, G_Z_STAR, G_Z_QSO],
    [R_Z_GALAXY, R_Z_STAR, R_Z_QSO],
    [I_Z_GALAXY, I_Z_STAR, I_Z_QSO],
    [I_Z_GALAXY, I_Z_STAR, I_Z_QSO]
]

# Etiquetas para los boxplots
labels = ['U-R', 'U-G', 'G-Z', 'R-Z', 'I-Z', 'I-Z']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j], showfliers=False)
        ax.set_xticklabels(['GALAXY', 'STAR', 'QSO'])
        ax.set_title('Distribution of '+labels[i * 3 + j]+' values')
        ax.set_xlabel('Class')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[48]:


# Datos para los histogramas (sustituye estos datos por los tuyos)
data = [
    [U_R_GALAXY, U_R_STAR, U_R_QSO],
    [U_G_GALAXY, U_G_STAR, U_G_QSO],
    [G_Z_GALAXY, G_Z_STAR, G_Z_QSO],
    [R_Z_GALAXY, R_Z_STAR, R_Z_QSO],
    [I_Z_GALAXY, I_Z_STAR, I_Z_QSO],
    [I_Z_GALAXY, I_Z_STAR, I_Z_QSO]
]

# Etiquetas para los histogramas
labels = ['U-R', 'U-G', 'G-Z', 'R-Z', 'I-Z', 'I-Z']
colors = ['blue', 'green', 'red']

# Crear la figura con subgráficos de 2 filas y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# Iterar sobre cada subgráfico y crear el histograma correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        for k in range(len(data[i * 3 + j])):
            ax.hist(data[i * 3 + j][k], bins=10, color=colors[k], alpha=0.7, label=['Galaxia', 'Estrella', 'QSO'][k])

        ax.set_xlabel(labels[i * 3 + j])
        ax.set_ylabel('Ln(Number of objects)')
        ax.set_title(labels[i * 3 + j]+ ' distribution')
        ax.legend()
        
        # Establecer la escala logarítmica en el eje y
        ax.set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[49]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [RU_GALAXY, RU_STAR, RU_QSO],
    [UG_GALAXY, UG_STAR, UG_QSO],
    [GZ_GALAXY, GZ_STAR, GZ_QSO],
    [RZ_GALAXY, RZ_STAR, RZ_QSO],
    [IZ_GALAXY, IZ_STAR, IZ_QSO],
    [UZ_GALAXY, UZ_STAR, UZ_QSO]
]

# Etiquetas para los boxplots
labels = ['R/U', 'U/G', 'G/Z', 'R/Z', 'I/Z', 'I/Z']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j])
        ax.set_xticklabels(['GALAXY', 'STAR', 'QSO'])
        ax.set_title('Distribution of '+labels[i * 3 + j]+' values')
        ax.set_xlabel('Class')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[50]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [RU_GALAXY, RU_STAR, RU_QSO],
    [UG_GALAXY, UG_STAR, UG_QSO],
    [GZ_GALAXY, GZ_STAR, GZ_QSO],
    [RZ_GALAXY, RZ_STAR, RZ_QSO],
    [IZ_GALAXY, IZ_STAR, IZ_QSO],
    [UZ_GALAXY, UZ_STAR, UZ_QSO]
]

# Etiquetas para los boxplots
labels = ['R/U', 'U/G', 'G/Z', 'R/Z', 'I/Z', 'I/Z']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j], showfliers=False)
        ax.set_xticklabels(['GALAXY', 'STAR', 'QSO'])
        ax.set_title('Distribution of '+labels[i * 3 + j]+' values')
        ax.set_xlabel('Class')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[52]:


# Datos para los histogramas (sustituye estos datos por los tuyos)
data = [
    [RU_GALAXY, RU_STAR, RU_QSO],
    [UG_GALAXY, UG_STAR, UG_QSO],
    [GZ_GALAXY, GZ_STAR, GZ_QSO],
    [RZ_GALAXY, RZ_STAR, RZ_QSO],
    [IZ_GALAXY, IZ_STAR, IZ_QSO],
    [UZ_GALAXY, UZ_STAR, UZ_QSO]
]

# Etiquetas para los histogramas
labels = ['R/U', 'U/G', 'G/Z', 'R/Z', 'I/Z', 'U/Z']
colors = ['blue', 'green', 'red']

# Crear la figura con subgráficos de 2 filas y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# Iterar sobre cada subgráfico y crear el histograma correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        for k in range(len(data[i * 3 + j])):
            ax.hist(data[i * 3 + j][k], bins=10, color=colors[k], alpha=0.7, label=['GALAXY', 'STAR', 'QSO'][k])

        ax.set_xlabel(labels[i * 3 + j])
        ax.set_ylabel('Ln(Number of objects)')
        ax.set_title(labels[i * 3 + j]+ ' distribution')
        ax.legend()
        
        # Establecer la escala logarítmica en el eje y
        ax.set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[53]:


# Crear una figura con subgráficos para cada combinación de índices de color
fig, axs = plt.subplots(5, 2, figsize=(12, 18))

# Scatter Plot 1: U_R vs G_Z
axs[0, 0].scatter(U_R_GALAXY, G_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[0, 0].scatter(U_R_STAR, G_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[0, 0].scatter(U_R_QSO, G_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[0, 0].set_xlabel('U-R')
axs[0, 0].set_ylabel('G-Z')
axs[0, 0].set_title('U-R vs G-Z')
axs[0, 0].legend()

# Scatter Plot 2: U_R vs R_Z
axs[0, 1].scatter(U_R_GALAXY, R_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[0, 1].scatter(U_R_STAR, R_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[0, 1].scatter(U_R_QSO, R_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[0, 1].set_xlabel('U-R')
axs[0, 1].set_ylabel('R-Z')
axs[0, 1].set_title('U-R vs R-Z')
axs[0, 1].legend()

# Scatter Plot 3: U_R vs I_Z
axs[1, 0].scatter(U_R_GALAXY, I_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[1, 0].scatter(U_R_STAR, I_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[1, 0].scatter(U_R_QSO, I_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[1, 0].set_xlabel('U-R')
axs[1, 0].set_ylabel('I-Z')
axs[1, 0].set_title('U-R vs I-Z')
axs[1, 0].legend()

# Scatter Plot 4: U_R vs U_G
axs[1, 1].scatter(U_R_GALAXY, U_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[1, 1].scatter(U_R_STAR, U_G_STAR, label='STAR', color='green', alpha=0.7)
axs[1, 1].scatter(U_R_QSO, U_G_QSO, label='QSO', color='red', alpha=0.7)
axs[1, 1].set_xlabel('U-R')
axs[1, 1].set_ylabel('U-G')
axs[1, 1].set_title('U-R vs U-G')
axs[1, 1].legend()

# Scatter Plot 5: G_Z vs R_Z
axs[2, 0].scatter(G_Z_GALAXY, R_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[2, 0].scatter(G_Z_STAR, R_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[2, 0].scatter(G_Z_QSO, R_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[2, 0].set_xlabel('G-Z')
axs[2, 0].set_ylabel('R-Z')
axs[2, 0].set_title('G-Z vs R-Z')
axs[2, 0].legend()

# Scatter Plot 6: G_Z vs I_Z
axs[2, 1].scatter(G_Z_GALAXY, I_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[2, 1].scatter(G_Z_STAR, I_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[2, 1].scatter(G_Z_QSO, I_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[2, 1].set_xlabel('G-Z')
axs[2, 1].set_ylabel('I-Z')
axs[2, 1].set_title('G-Z vs I-Z')
axs[2, 1].legend()

# Scatter Plot 7: G_Z vs U_G
axs[3, 0].scatter(G_Z_GALAXY, U_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[3, 0].scatter(G_Z_STAR, U_G_STAR, label='STAR', color='green', alpha=0.7)
axs[3, 0].scatter(G_Z_QSO, U_G_QSO, label='QSO', color='red', alpha=0.7)
axs[3, 0].set_xlabel('G-Z')
axs[3, 0].set_ylabel('U-G')
axs[3, 0].set_title('G-Z vs U-G')
axs[3, 0].legend()

# Scatter Plot 8: R_Z vs I_Z
axs[3, 1].scatter(R_Z_GALAXY, I_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[3, 1].scatter(R_Z_STAR, I_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[3, 1].scatter(R_Z_QSO, I_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[3, 1].set_xlabel('R-Z')
axs[3, 1].set_ylabel('I-Z')
axs[3, 1].set_title('R-Z vs I-Z')
axs[3, 1].legend()

# Scatter Plot 9: R_Z vs U_G
axs[4, 0].scatter(R_Z_GALAXY, U_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[4, 0].scatter(R_Z_STAR, U_G_STAR, label='STAR', color='green', alpha=0.7)
axs[4, 0].scatter(R_Z_QSO, U_G_QSO, label='QSO', color='red', alpha=0.7)
axs[4, 0].set_xlabel('R-Z')
axs[4, 0].set_ylabel('U-G')
axs[4, 0].set_title('R-Z vs U-G')
axs[4, 0].legend()

# Scatter Plot 10: I_Z vs U_G
axs[4, 1].scatter(I_Z_GALAXY, U_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[4, 1].scatter(I_Z_STAR, U_G_STAR, label='STAR', color='green', alpha=0.7)
axs[4, 1].scatter(I_Z_QSO, U_G_QSO, label='QSO', color='red', alpha=0.7)
axs[4, 1].set_xlabel('I-Z')
axs[4, 1].set_ylabel('U-G')
axs[4, 1].set_title('I-Z vs U-G')
axs[4, 1].legend()

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# In[ ]:





# ### Spectrum curvature parameter

# spectra_galaxy
# spectra_qso
# spectra_star

# In[54]:


wavelenght


# In[17]:


curv_gal=[]
curv_qso=[]
curv_star=[]

b_gal=[]
b_qso=[]
b_star=[]

c_gal=[]
c_qso=[]
c_star=[]

# Define una función para ajustar (aquí se usa un polinomio de segundo grado como ejemplo)
def func(x, a, b, c):
    return a * x**2 + b * x + c
    
for spec in spectra_galaxy:
   # Realiza el ajuste de curva a los datos suavizados
    popt, _ = curve_fit(func, wavelenght, spec)
    # Obtiene los parámetros ajustados
    a, b, c = popt
    # Calcula la curvatura (segunda derivada)
    curvature = 2 * a
    curv_gal.append(curvature)

    b_gal.append(b)
    c_gal.append(c)

for spec in spectra_qso:
   # Realiza el ajuste de curva a los datos suavizados
    popt, _ = curve_fit(func, wavelenght, spec)
    # Obtiene los parámetros ajustados
    a, b, c = popt
    # Calcula la curvatura (segunda derivada)
    curvature = 2 * a
    curv_qso.append(curvature)

    b_qso.append(b)
    c_qso.append(c)

for spec in spectra_star:
   # Realiza el ajuste de curva a los datos suavizados
    popt, _ = curve_fit(func, wavelenght, spec)
    # Obtiene los parámetros ajustados
    a, b, c = popt
    # Calcula la curvatura (segunda derivada)
    curvature = 2 * a
    curv_star.append(curvature)

    b_star.append(b)
    c_star.append(c)
    
#curv_gal=np.array(curv_gal)
#curv_qso=np.array(curv_qso)
#curv_star=np.array(curv_star)

#b_gal=np.array(b_gal)
#b_qso=np.array(b_qso)
#b_star=np.array(b_star)

#c_gal=np.array(c_gal)
#c_qso=np.array(c_qso)
#c_star=np.array(c_star)


# In[62]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [curv_gal, curv_star, curv_qso],
    [b_gal, b_star, b_qso],
    [c_gal, c_star, c_qso]
]

# Etiquetas para los boxplots
labels = ['Curvature', 'b parameter', 'c parameter']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i, ax in enumerate(axs):
    ax.boxplot(data[i])
    ax.set_xticklabels(['GALAXY', 'STAR', 'QSO'])
    ax.set_title(labels[i])
    ax.set_xlabel('Class')
    ax.set_ylabel(labels[i])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[61]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [curv_gal, curv_star, curv_qso],
    [b_gal, b_star, b_qso],
    [c_gal, c_star, c_qso]
]

# Etiquetas para los boxplots
labels = ['Curvature', 'b parameter', 'c parameter']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i, ax in enumerate(axs):
    ax.boxplot(data[i], showfliers=False)
    ax.set_xticklabels(['GALAXY', 'STAR', 'QSO'])
    ax.set_title('Distribution of '+labels[i]+' values')
    ax.set_xlabel('Class')
    ax.set_ylabel(labels[i])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[63]:


# Datos para los histogramas (sustituye estos datos por los tuyos)
data_curv = [curv_gal, curv_star, curv_qso]
data_b = [b_gal, b_star, b_qso]
data_c = [c_gal, c_star, c_qso]

# Etiquetas para los histogramas
labels = ['Curvature', 'b parameter', 'c parameter']
colors = ['blue', 'green', 'red']

# Crear una figura con subgráficos para cada clase
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase
for i, data in enumerate([data_curv, data_b, data_c]):
    for j, class_data in enumerate(data):
        axs[i].hist(class_data, bins=10, color=colors[j], alpha=0.7, label=['GALAXY', 'STAR', 'QSO'][j])

    # Agregar leyenda y título
    axs[i].legend()
    axs[i].set_title(labels[i])
    axs[i].set_xlabel(labels[i]+' values')
    axs[i].set_ylabel('Ln(Number of objects)')
    
    # Establecer la escala logarítmica en el eje y
    axs[i].set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# ### MAD Parameter (median absolute deviation)

# In[18]:


mad_gal=[]
mad_qso=[]
mad_star=[]
    
for spec in spectra_galaxy:
   # Calcula la desviación absoluta media (MAD)
    mad_value = np.median(np.abs(spec - np.median(spec)))
    mad_gal.append(mad_value)

for spec in spectra_qso:
   # Calcula la desviación absoluta media (MAD)
    mad_value = np.median(np.abs(spec - np.median(spec)))
    mad_qso.append(mad_value)

for spec in spectra_star:
   # Calcula la desviación absoluta media (MAD)
    mad_value = np.median(np.abs(spec - np.median(spec)))
    mad_star.append(mad_value)
    
#mad_gal=np.array(mad_gal)
#mad_qso=np.array(mad_qso)
#mad_star=np.array(mad_star)


# In[66]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [mad_gal, mad_star, mad_qso]

# Etiquetas para los boxplots
labels = ['MAD GALAXY', 'MAD STAR', 'MAD QSO']

# Crear la figura y los ejes de la figura
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Crear el boxplot con outliers
axs[0].boxplot(data)
axs[0].set_xticklabels(['GALAXY', 'STAR', 'QSO'])
axs[0].set_title('MAD distribution')
axs[0].set_xlabel('Class')
axs[0].set_ylabel('MAD')

# Crear el boxplot sin outliers
axs[1].boxplot(data, showfliers=False)
axs[1].set_xticklabels(['GALAXY', 'STAR', 'QSO'])
axs[1].set_title('MAD distribution')
axs[1].set_xlabel('Class')
axs[1].set_ylabel('MAD')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[67]:


# Datos para los histogramas
data = [mad_gal, mad_star, mad_qso]

# Etiquetas para los histogramas
labels = ['GALAXY', 'STAR', 'QSO']

# Colores para los histogramas
colors = ['blue', 'green', 'red']

# Crear una figura con subgráficos para cada clase
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase en la primera columna
for i, class_data in enumerate(data):
    axs[0].hist(class_data, bins=10, color=colors[i], alpha=0.7, label=labels[i])

# Agregar leyenda y título a la primera columna
axs[0].legend()
axs[0].set_title('MAD distribution')
axs[0].set_xlabel('MAD values')
axs[0].set_ylabel('Number of objects')

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase en la segunda columna
for i, class_data in enumerate(data):
    axs[1].hist(class_data, bins=10, color=colors[i], alpha=0.7, label=labels[i])

# Agregar leyenda y título a la segunda columna
axs[1].legend()
axs[1].set_title('MAD distribution')
axs[1].set_xlabel('MAD values')
axs[1].set_ylabel('Ln(Number of objects)')

# Establecer escala logarítmica en el eje y para la segunda columna
axs[1].set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[ ]:





# ### Synthetic filters

# In[19]:


FLUX_0_GALAXY=[]
FLUX_0_QSO=[]
FLUX_0_STAR=[]

# Definir una máscara booleana para las longitudes de onda entre 3055.11 y 4030.64 (Banda u con lambda_eff=3608.04)
mascara = (wavelenght >= 3055.11) & (wavelenght <= 3756)
wavelenght_filtrado= wavelenght[mascara]

for galaxy in spectra_galaxy:
    espectro_filtrado= galaxy[mascara]
#    area= trapz(espectro_filtrado, wavelenght_filtrado)
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_0_GALAXY.append(area)

for qso in spectra_qso:
    espectro_filtrado= qso[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_0_QSO.append(area)

for star in spectra_star:
    espectro_filtrado= star[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_0_STAR.append(area)


# In[20]:


FLUX_1_GALAXY=[]
FLUX_1_QSO=[]
FLUX_1_STAR=[]

# Definir una máscara booleana para las longitudes de onda entre 3055.11 y 4030.64 (Banda u con lambda_eff=3608.04)
mascara = (wavelenght >= 5578) & (wavelenght <= 5582)
wavelenght_filtrado= wavelenght[mascara]

for galaxy in spectra_galaxy:
    espectro_filtrado= galaxy[mascara]
#    area= trapz(espectro_filtrado, wavelenght_filtrado)
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_1_GALAXY.append(area)

for qso in spectra_qso:
    espectro_filtrado= qso[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_1_QSO.append(area)

for star in spectra_star:
    espectro_filtrado= star[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_1_STAR.append(area)


# In[21]:


FLUX_2_GALAXY=[]
FLUX_2_QSO=[]
FLUX_2_STAR=[]

# Definir una máscara booleana para las longitudes de onda entre 3055.11 y 4030.64 (Banda u con lambda_eff=3608.04)
mascara = (wavelenght >= 7601) & (wavelenght <= 7603)
wavelenght_filtrado= wavelenght[mascara]

for galaxy in spectra_galaxy:
    espectro_filtrado= galaxy[mascara]
#    area= trapz(espectro_filtrado, wavelenght_filtrado)
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_2_GALAXY.append(area)

for qso in spectra_qso:
    espectro_filtrado= qso[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_2_QSO.append(area)

for star in spectra_star:
    espectro_filtrado= star[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_2_STAR.append(area)


# In[22]:


FLUX_3_GALAXY=[]
FLUX_3_QSO=[]
FLUX_3_STAR=[]

# Definir una máscara booleana para las longitudes de onda entre 3055.11 y 4030.64 (Banda u con lambda_eff=3608.04)
mascara = (wavelenght >= 9310) & (wavelenght <= 9570)
wavelenght_filtrado= wavelenght[mascara]

for galaxy in spectra_galaxy:
    espectro_filtrado= galaxy[mascara]
#    area= trapz(espectro_filtrado, wavelenght_filtrado)
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_3_GALAXY.append(area)

for qso in spectra_qso:
    espectro_filtrado= qso[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_3_QSO.append(area)

for star in spectra_star:
    espectro_filtrado= star[mascara]
    area= trapz(espectro_filtrado, wavelenght_filtrado)
    FLUX_3_STAR.append(area)


# In[72]:


# Crear una figura con subgráficos para cada filtro
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

# Lista de etiquetas para las clases
clases = ['GALAXY', 'STAR', 'QSO']

# Lista de listas de valores para cada filtro y clase
flux_values = [
    [FLUX_0_GALAXY, FLUX_0_STAR, FLUX_0_QSO],
    [FLUX_1_GALAXY, FLUX_1_STAR, FLUX_1_QSO],
    [FLUX_2_GALAXY, FLUX_2_STAR, FLUX_2_QSO],
    [FLUX_3_GALAXY, FLUX_3_STAR, FLUX_3_QSO]
]

# Lista para almacenar los datos filtrados
filtered_flux_values = []

# Iterar sobre cada subgráfico y crear un boxplot para cada filtro
for i in range(4):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    flux_values_current = flux_values[i]

    # Crear un boxplot y obtener los límites de los bigotes
    boxplot = ax.boxplot(flux_values_current, labels=clases, notch=True, patch_artist=True)
    whiskers = boxplot['whiskers']
    
    # Obtener los límites de los bigotes
    lower_whisker, upper_whisker = whiskers[0].get_ydata()[1], whiskers[1].get_ydata()[1]
    
    # Filtrar los datos que están dentro de los bigotes
    filtered_data = [val for sublist in flux_values_current for val in sublist if lower_whisker <= val <= upper_whisker]
    filtered_flux_values.append(filtered_data)

    ax.set_xlabel('Class')
    ax.set_ylabel(f'FLUX_{i}')
    ax.set_title(f'Distribution of FLUX_{i} values')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[73]:


# Crear una figura con subgráficos para cada filtro
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

# Lista de etiquetas para las clases
clases = ['GALAXY', 'STAR', 'QSO']

# Lista de listas de valores para cada filtro y clase
flux_values = [
    [FLUX_0_GALAXY, FLUX_0_STAR, FLUX_0_QSO],
    [FLUX_1_GALAXY, FLUX_1_STAR, FLUX_1_QSO],
    [FLUX_2_GALAXY, FLUX_2_STAR, FLUX_2_QSO],
    [FLUX_3_GALAXY, FLUX_3_STAR, FLUX_3_QSO]
]

# Lista para almacenar los datos filtrados
filtered_flux_values = []

# Iterar sobre cada subgráfico y crear un boxplot para cada filtro
for i in range(4):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    flux_values_current = flux_values[i]

    # Crear un boxplot y obtener los límites de los bigotes
    boxplot = ax.boxplot(flux_values_current, labels=clases, notch=True, patch_artist=True, showfliers=False)
    whiskers = boxplot['whiskers']
    
    # Obtener los límites de los bigotes
    lower_whisker, upper_whisker = whiskers[0].get_ydata()[1], whiskers[1].get_ydata()[1]
    
    # Filtrar los datos que están dentro de los bigotes
    filtered_data = [val for sublist in flux_values_current for val in sublist if lower_whisker <= val <= upper_whisker]
    filtered_flux_values.append(filtered_data)

    ax.set_xlabel('Class')
    ax.set_ylabel(f'FLUX_{i}')
    ax.set_title(f'Distribution of FLUX_{i} values')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[76]:


import matplotlib.pyplot as plt

# Create a figure with subplots for each filter
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

# List of labels for the classes
classes = ['GALAXY', 'STAR', 'QSO']

# List of lists of values for each filter and class
flux_values = [
    [FLUX_0_GALAXY, FLUX_0_STAR, FLUX_0_QSO],
    [FLUX_1_GALAXY, FLUX_1_STAR, FLUX_1_QSO],
    [FLUX_2_GALAXY, FLUX_2_STAR, FLUX_2_QSO],
    [FLUX_3_GALAXY, FLUX_3_STAR, FLUX_3_QSO]
]

# Colors for each class
colors = ['blue', 'green', 'red']

# Iterate over each subplot and create a histogram for each filter
for i in range(4):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    flux_values_current = flux_values[i]

    # Plot histogram for each class with specified colors
    for j, class_flux in enumerate(flux_values_current):
        ax.hist(class_flux, bins=20, label=classes[j], color=colors[j], alpha=0.7)

    ax.set_xlabel(f'FLUX_{i} values')
    ax.set_ylabel('Ln(Number of objects)')
    ax.set_title(f'FLUX_{i} distribution')
    ax.legend()

    # Apply natural logarithm to the y-axis values
    ax.set_yscale('log')

# Adjust subplots layout
plt.tight_layout()

# Show the plot
plt.show()


# In[77]:


# Lista de combinaciones específicas que quieres comparar
combinaciones_especificas = [('FLUX_0', 'FLUX_U'), ('FLUX_0', 'FLUX_V_G'), ('FLUX_0', 'FLUX_R'), ('FLUX_0', 'FLUX_I'), ('FLUX_0', 'FLUX_z'),
                             ('FLUX_1', 'FLUX_U'), ('FLUX_1', 'FLUX_V_G'), ('FLUX_1', 'FLUX_R'), ('FLUX_1', 'FLUX_I'), ('FLUX_1', 'FLUX_z'),
                             ('FLUX_2', 'FLUX_U'), ('FLUX_2', 'FLUX_V_G'), ('FLUX_2', 'FLUX_R'), ('FLUX_2', 'FLUX_I'), ('FLUX_2', 'FLUX_z'),
                             ('FLUX_3', 'FLUX_U'), ('FLUX_3', 'FLUX_V_G'), ('FLUX_3', 'FLUX_R'), ('FLUX_3', 'FLUX_I'), ('FLUX_3', 'FLUX_z')]

# Crear una figura con subgráficos para cada combinación específica
fig, axs = plt.subplots(len(combinaciones_especificas), 1, figsize=(10, 5*len(combinaciones_especificas)))

# Iterar sobre las combinaciones específicas y crear los gráficos correspondientes
for i, (flujo, filtro) in enumerate(combinaciones_especificas):
    # Scatter Plot para la combinación actual
    axs[i].scatter(eval(f'{filtro}_GALAXY'), eval(flujo + '_GALAXY'), label='GALAXY', color='blue', alpha=0.7)
    axs[i].scatter(eval(f'{filtro}_STAR'), eval(flujo + '_STAR'), label='STAR', color='green', alpha=0.7)
    axs[i].scatter(eval(f'{filtro}_QSO'), eval(flujo + '_QSO'), label='QSO', color='red', alpha=0.7)
    axs[i].set_xlabel(f'{filtro}')
    axs[i].set_ylabel(f'{flujo}')
    axs[i].set_title(f'{flujo} vs {filtro}')
    axs[i].legend()

# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[80]:


import itertools

# Lista de todos los flujos
flujos = ['FLUX_0', 'FLUX_1', 'FLUX_2', 'FLUX_3']

# Lista de todos los filtros
filtros = ['FLUX_U', 'FLUX_V_G', 'FLUX_R', 'FLUX_I', 'FLUX_z', 'FLUX_0', 'FLUX_1', 'FLUX_2', 'FLUX_3']

# Generar todas las combinaciones posibles entre flujos y filtros
combinaciones = list(itertools.product(flujos, filtros))

# Crear una figura con subgráficos para cada combinación específica
fig, axs = plt.subplots(len(combinaciones), 1, figsize=(10, 5*len(combinaciones)))

# Iterar sobre todas las combinaciones y crear los gráficos correspondientes
for i, (flujo, filtro) in enumerate(combinaciones):
    # Scatter Plot para la combinación actual
    axs[i].scatter(eval(f'{filtro}_GALAXY'), eval(flujo + '_GALAXY'), label='GALAXY', color='blue', alpha=0.7)
    axs[i].scatter(eval(f'{filtro}_STAR'), eval(flujo + '_STAR'), label='STAR', color='green', alpha=0.7)
    axs[i].scatter(eval(f'{filtro}_QSO'), eval(flujo + '_QSO'), label='QSO', color='red', alpha=0.7)
    axs[i].set_xlabel(f'{filtro}')
    axs[i].set_ylabel(f'{flujo}')
    axs[i].set_title(f'{flujo} vs {filtro}')
    axs[i].legend()

# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[23]:


import numpy as np

# Calcular las razones entre los flujos de los números 0, 1, 2 y 3
_0U_GALAXY = np.array(FLUX_0_GALAXY) / np.array(FLUX_U_GALAXY)
_0U_STAR = np.array(FLUX_0_STAR) / np.array(FLUX_U_STAR)
_0U_QSO = np.array(FLUX_0_QSO) / np.array(FLUX_U_QSO)

_0Z_GALAXY = np.array(FLUX_0_GALAXY) / np.array(FLUX_z_GALAXY)
_0Z_STAR = np.array(FLUX_0_STAR) / np.array(FLUX_z_STAR)
_0Z_QSO = np.array(FLUX_0_QSO) / np.array(FLUX_z_QSO)

_0G_GALAXY = np.array(FLUX_0_GALAXY) / np.array(FLUX_V_G_GALAXY)
_0G_STAR = np.array(FLUX_0_STAR) / np.array(FLUX_V_G_STAR)
_0G_QSO = np.array(FLUX_0_QSO) / np.array(FLUX_V_G_QSO)

_0R_GALAXY = np.array(FLUX_0_GALAXY) / np.array(FLUX_R_GALAXY)
_0R_STAR = np.array(FLUX_0_STAR) / np.array(FLUX_R_STAR)
_0R_QSO = np.array(FLUX_0_QSO) / np.array(FLUX_R_QSO)

_0I_GALAXY = np.array(FLUX_0_GALAXY) / np.array(FLUX_I_GALAXY)
_0I_STAR = np.array(FLUX_0_STAR) / np.array(FLUX_I_STAR)
_0I_QSO = np.array(FLUX_0_QSO) / np.array(FLUX_I_QSO)

_01_GALAXY = np.array(FLUX_0_GALAXY) / np.array(FLUX_1_GALAXY)
_01_STAR = np.array(FLUX_0_STAR) / np.array(FLUX_1_STAR)
_01_QSO = np.array(FLUX_0_QSO) / np.array(FLUX_1_QSO)

_02_GALAXY = np.array(FLUX_0_GALAXY) / np.array(FLUX_2_GALAXY)
_02_STAR = np.array(FLUX_0_STAR) / np.array(FLUX_2_STAR)
_02_QSO = np.array(FLUX_0_QSO) / np.array(FLUX_2_QSO)

_03_GALAXY = np.array(FLUX_0_GALAXY) / np.array(FLUX_3_GALAXY)
_03_STAR = np.array(FLUX_0_STAR) / np.array(FLUX_3_STAR)
_03_QSO = np.array(FLUX_0_QSO) / np.array(FLUX_3_QSO)

# Calcular las razones entre los flujos de los números 0, 1, 2 y 3
_1U_GALAXY = np.array(FLUX_1_GALAXY) / np.array(FLUX_U_GALAXY)
_1U_STAR = np.array(FLUX_1_STAR) / np.array(FLUX_U_STAR)
_1U_QSO = np.array(FLUX_1_QSO) / np.array(FLUX_U_QSO)

_1Z_GALAXY = np.array(FLUX_1_GALAXY) / np.array(FLUX_z_GALAXY)
_1Z_STAR = np.array(FLUX_1_STAR) / np.array(FLUX_z_STAR)
_1Z_QSO = np.array(FLUX_1_QSO) / np.array(FLUX_z_QSO)

_1G_GALAXY = np.array(FLUX_1_GALAXY) / np.array(FLUX_V_G_GALAXY)
_1G_STAR = np.array(FLUX_1_STAR) / np.array(FLUX_V_G_STAR)
_1G_QSO = np.array(FLUX_1_QSO) / np.array(FLUX_V_G_QSO)

_1R_GALAXY = np.array(FLUX_1_GALAXY) / np.array(FLUX_R_GALAXY)
_1R_STAR = np.array(FLUX_1_STAR) / np.array(FLUX_R_STAR)
_1R_QSO = np.array(FLUX_1_QSO) / np.array(FLUX_R_QSO)

_1I_GALAXY = np.array(FLUX_1_GALAXY) / np.array(FLUX_I_GALAXY)
_1I_STAR = np.array(FLUX_1_STAR) / np.array(FLUX_I_STAR)
_1I_QSO = np.array(FLUX_1_QSO) / np.array(FLUX_I_QSO)

_12_GALAXY = np.array(FLUX_1_GALAXY) / np.array(FLUX_2_GALAXY)
_12_STAR = np.array(FLUX_1_STAR) / np.array(FLUX_2_STAR)
_12_QSO = np.array(FLUX_1_QSO) / np.array(FLUX_2_QSO)

_13_GALAXY = np.array(FLUX_1_GALAXY) / np.array(FLUX_3_GALAXY)
_13_STAR = np.array(FLUX_1_STAR) / np.array(FLUX_3_STAR)
_13_QSO = np.array(FLUX_1_QSO) / np.array(FLUX_3_QSO)


# Razones entre el flujo 2 y el flujo 0
_2U_GALAXY = np.array(FLUX_2_GALAXY) / np.array(FLUX_U_GALAXY)
_2U_STAR = np.array(FLUX_2_STAR) / np.array(FLUX_U_STAR)
_2U_QSO = np.array(FLUX_2_QSO) / np.array(FLUX_U_QSO)

_2Z_GALAXY = np.array(FLUX_2_GALAXY) / np.array(FLUX_z_GALAXY)
_2Z_STAR = np.array(FLUX_2_STAR) / np.array(FLUX_z_STAR)
_2Z_QSO = np.array(FLUX_2_QSO) / np.array(FLUX_z_QSO)

_2G_GALAXY = np.array(FLUX_2_GALAXY) / np.array(FLUX_V_G_GALAXY)
_2G_STAR = np.array(FLUX_2_STAR) / np.array(FLUX_V_G_STAR)
_2G_QSO = np.array(FLUX_2_QSO) / np.array(FLUX_V_G_QSO)

_2R_GALAXY = np.array(FLUX_2_GALAXY) / np.array(FLUX_R_GALAXY)
_2R_STAR = np.array(FLUX_2_STAR) / np.array(FLUX_R_STAR)
_2R_QSO = np.array(FLUX_2_QSO) / np.array(FLUX_R_QSO)

_2I_GALAXY = np.array(FLUX_2_GALAXY) / np.array(FLUX_I_GALAXY)
_2I_STAR = np.array(FLUX_2_STAR) / np.array(FLUX_I_STAR)
_2I_QSO = np.array(FLUX_2_QSO) / np.array(FLUX_I_QSO)

_23_GALAXY = np.array(FLUX_2_GALAXY) / np.array(FLUX_3_GALAXY)
_23_STAR = np.array(FLUX_2_STAR) / np.array(FLUX_3_STAR)
_23_QSO = np.array(FLUX_2_QSO) / np.array(FLUX_3_QSO)


# Razones entre el flujo 3 y el flujo 0
_3U_GALAXY = np.array(FLUX_3_GALAXY) / np.array(FLUX_U_GALAXY)
_3U_STAR = np.array(FLUX_3_STAR) / np.array(FLUX_U_STAR)
_3U_QSO = np.array(FLUX_3_QSO) / np.array(FLUX_U_QSO)

_3Z_GALAXY = np.array(FLUX_3_GALAXY) / np.array(FLUX_z_GALAXY)
_3Z_STAR = np.array(FLUX_3_STAR) / np.array(FLUX_z_STAR)
_3Z_QSO = np.array(FLUX_3_QSO) / np.array(FLUX_z_QSO)

_3G_GALAXY = np.array(FLUX_3_GALAXY) / np.array(FLUX_V_G_GALAXY)
_3G_STAR = np.array(FLUX_3_STAR) / np.array(FLUX_V_G_STAR)
_3G_QSO = np.array(FLUX_3_QSO) / np.array(FLUX_V_G_QSO)

_3R_GALAXY = np.array(FLUX_3_GALAXY) / np.array(FLUX_R_GALAXY)
_3R_STAR = np.array(FLUX_3_STAR) / np.array(FLUX_R_STAR)
_3R_QSO = np.array(FLUX_3_QSO) / np.array(FLUX_R_QSO)

_3I_GALAXY = np.array(FLUX_3_GALAXY) / np.array(FLUX_I_GALAXY)
_3I_STAR = np.array(FLUX_3_STAR) / np.array(FLUX_I_STAR)
_3I_QSO = np.array(FLUX_3_QSO) / np.array(FLUX_I_QSO)


# In[84]:


ratios = [
    [_0U_GALAXY, _0U_STAR, _0U_QSO],
    [_0Z_GALAXY, _0Z_STAR, _0Z_QSO],
    [_0G_GALAXY, _0G_STAR, _0G_QSO],
    [_0R_GALAXY, _0R_STAR, _0R_QSO],
    [_0I_GALAXY, _0I_STAR, _0I_QSO],
    [_01_GALAXY, _01_STAR, _01_QSO],
    [_02_GALAXY, _02_STAR, _02_QSO],
    [_03_GALAXY, _03_STAR, _03_QSO],
    [_1U_GALAXY, _1U_STAR, _1U_QSO],
    [_1Z_GALAXY, _1Z_STAR, _1Z_QSO],
    [_1G_GALAXY, _1G_STAR, _1G_QSO],
    [_1R_GALAXY, _1R_STAR, _1R_QSO],
    [_1I_GALAXY, _1I_STAR, _1I_QSO],
    [_12_GALAXY, _12_STAR, _12_QSO],
    [_13_GALAXY, _13_STAR, _13_QSO],
    [_2U_GALAXY, _2U_STAR, _2U_QSO],
    [_2Z_GALAXY, _2Z_STAR, _2Z_QSO],
    [_2G_GALAXY, _2G_STAR, _2G_QSO],
    [_2R_GALAXY, _2R_STAR, _2R_QSO],
    [_2I_GALAXY, _2I_STAR, _2I_QSO],
    [_23_GALAXY, _23_STAR, _23_QSO],
    [_3U_GALAXY, _3U_STAR, _3U_QSO],
    [_3Z_GALAXY, _3Z_STAR, _3Z_QSO],
    [_3G_GALAXY, _3G_STAR, _3G_QSO],
    [_3R_GALAXY, _3R_STAR, _3R_QSO],
    [_3I_GALAXY, _3I_STAR, _3I_QSO]
]

# Iterar sobre cada grupo de variables en ratios
for grupo in ratios:
    # Datos de las variables para cada clase
    datos_galaxia = grupo[0]
    datos_estrella = grupo[1]
    datos_cuasar = grupo[2]

    # Etiquetas y colores para las clases
    clases = ['GALAXY', 'STAR', 'QSO']
    colores = ['blue', 'green', 'red']

    # Crear una figura y un conjunto de ejes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Combinar los datos de las tres clases
    datos_combinados = [datos_galaxia, datos_estrella, datos_cuasar]

    # Crear el boxplot
    boxplot = ax.boxplot(datos_combinados, labels=clases, patch_artist=True, notch=True, showfliers=False)

    # Personalizar los colores de las cajas
    for patch, color in zip(boxplot['boxes'], colores):
        patch.set_facecolor(color)

    # Configuración del gráfico
    titulo = f'Ratios distributions'
    ax.set_title(titulo)
    ax.set_xlabel('Class')
    ax.set_ylabel('Ratios values')

    # Mostrar el boxplot
    plt.show()


# In[85]:


# Iterar sobre cada grupo de variables en ratios
for grupo in ratios:
    # Datos de las variables para cada clase
    datos_galaxia = grupo[0]
    datos_estrella = grupo[1]
    datos_cuasar = grupo[2]

    # Etiquetas y colores para las clases
    clases = ['GALAXY', 'STAR', 'QSO']
    colores = ['blue', 'green', 'red']

    # Crear una figura y un conjunto de ejes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Iterar sobre cada clase y trazar un histograma para ella
    for i, datos in enumerate([datos_galaxia, datos_estrella, datos_cuasar]):
        ax.hist(datos, bins=20, color=colores[i], alpha=0.7, label=clases[i], log=True)

    # Configuración del gráfico
    titulo = f'Ratios distributions'
    ax.set_title(titulo)
    ax.set_xlabel('Ratios values')
    ax.set_ylabel('Ln(Number of objects)')
    ax.legend()

    # Mostrar el histograma
    plt.show()


# In[86]:


# Datos de las variables
datos_galaxia = _3I_GALAXY
datos_estrella = _3I_STAR
datos_cuasar = _3I_QSO

# Crear una figura y un conjunto de ejes con 1 fila y 3 columnas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Etiquetas y colores para las clases
clases = ['GALAXY', 'STAR', 'QSO']
colores = ['blue', 'green', 'red']

# Iterar sobre cada conjunto de datos y crear un histograma en su respectivo eje
for i, datos in enumerate([datos_galaxia, datos_estrella, datos_cuasar]):
    axs[i].hist(datos, bins=30, color=colores[i], alpha=0.7, log=True)
    axs[i].set_title(f'{clases[i]} distribution')
    axs[i].set_xlabel('_3I values')
    axs[i].set_ylabel('Ln(Number of objects)')

# Ajustar el espacio entre los subgráficos
plt.tight_layout()

# Mostrar los histogramas
plt.show()


# In[ ]:





# ### Transformation with logarithms, Temperature
# λ2

# mλ1 − mλ2 = −2.5 lg(Fλ1/Fλ2)
#           = a + b/Tc
# El logaritmo no acepta valores negativos, asi que lo haremos para el fljo total recibido, es decir, el absoluto

# In[24]:


U_R_GALAXY_T= -2.5*np.log10(np.abs(FLUX_U_GALAXY/FLUX_R_GALAXY))
U_R_STAR_T= -2.5*np.log10(np.abs(FLUX_U_STAR/FLUX_R_STAR))
U_R_QSO_T= -2.5*np.log10(np.abs(FLUX_U_QSO/FLUX_R_QSO))

U_G_GALAXY_T= -2.5*np.log10(np.abs(FLUX_U_GALAXY/FLUX_G_GALAXY))
U_G_STAR_T= -2.5*np.log10(np.abs(FLUX_U_STAR/FLUX_G_STAR))
U_G_QSO_T= -2.5*np.log10(np.abs(FLUX_U_QSO/FLUX_G_QSO))

G_Z_GALAXY_T= -2.5*np.log10(np.abs(FLUX_G_GALAXY/FLUX_Z_GALAXY))
G_Z_STAR_T= -2.5*np.log10(np.abs(FLUX_G_STAR/FLUX_Z_STAR))
G_Z_QSO_T= -2.5*np.log10(np.abs(FLUX_G_QSO/FLUX_Z_QSO))

R_Z_GALAXY_T= -2.5*np.log10(np.abs(FLUX_R_GALAXY/FLUX_Z_GALAXY))
R_Z_STAR_T= -2.5*np.log10(np.abs(FLUX_R_STAR/FLUX_Z_STAR))
R_Z_QSO_T= -2.5*np.log10(np.abs(FLUX_R_QSO/FLUX_Z_QSO))

I_Z_GALAXY_T= -2.5*np.log10(np.abs(FLUX_I_GALAXY/FLUX_Z_GALAXY))
I_Z_STAR_T= -2.5*np.log10(np.abs(FLUX_I_STAR/FLUX_Z_STAR))
I_Z_QSO_T= -2.5*np.log10(np.abs(FLUX_I_QSO/FLUX_Z_QSO))


# In[92]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [U_R_GALAXY_T, U_R_STAR_T, U_R_QSO_T],
    [U_G_GALAXY_T, U_G_STAR_T, U_G_QSO_T],
    [G_Z_GALAXY_T, G_Z_STAR_T, G_Z_QSO_T],
    [R_Z_GALAXY_T, R_Z_STAR_T, R_Z_QSO_T],
    [I_Z_GALAXY_T, I_Z_STAR_T, I_Z_QSO_T],
    [I_Z_GALAXY_T, I_Z_STAR_T, I_Z_QSO_T]
]

# Etiquetas para los boxplots
labels = ['U/R T', 'U/G T', 'G/Z T', 'R/Z T', 'I/Z T', 'I/Z T']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j])
        ax.set_xticklabels(['GALAXY', 'STAR', 'QSO'])
        ax.set_title(labels[i * 3 + j]+' distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[93]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [U_R_GALAXY_T, U_R_STAR_T, U_R_QSO_T],
    [U_G_GALAXY_T, U_G_STAR_T, U_G_QSO_T],
    [G_Z_GALAXY_T, G_Z_STAR_T, G_Z_QSO_T],
    [R_Z_GALAXY_T, R_Z_STAR_T, R_Z_QSO_T],
    [I_Z_GALAXY_T, I_Z_STAR_T, I_Z_QSO_T],
    [I_Z_GALAXY_T, I_Z_STAR_T, I_Z_QSO_T]
]

# Etiquetas para los boxplots
labels = ['U/R T', 'U/G T', 'G/Z T', 'R/Z T', 'I/Z T', 'I/Z T']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j], showfliers=False)
        ax.set_xticklabels(['GALAXY', 'STAR', 'QSO'])
        ax.set_title(labels[i * 3 + j]+' distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[94]:


# Datos para los histogramas (sustituye estos datos por los tuyos)
data = [
    [U_R_GALAXY_T, U_R_STAR_T, U_R_QSO_T],
    [U_G_GALAXY_T, U_G_STAR_T, U_G_QSO_T],
    [G_Z_GALAXY_T, G_Z_STAR_T, G_Z_QSO_T],
    [R_Z_GALAXY_T, R_Z_STAR_T, R_Z_QSO_T],
    [I_Z_GALAXY_T, I_Z_STAR_T, I_Z_QSO_T],
    [I_Z_GALAXY_T, I_Z_STAR_T, I_Z_QSO_T]
]

# Etiquetas para los histogramas
labels = ['U/R T', 'U/G T', 'G/Z T', 'R/Z T', 'I/Z T', 'I/Z T']
colors = ['blue', 'green', 'red']

# Crear la figura con subgráficos de 2 filas y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# Iterar sobre cada subgráfico y crear el histograma correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        for k in range(len(data[i * 3 + j])):
            ax.hist(data[i * 3 + j][k], bins=10, color=colors[k], alpha=0.7, label=['GALAXY', 'STAR', 'QSO'][k])

        ax.set_xlabel(labels[i * 3 + j]+' values')
        ax.set_ylabel('Ln(Number of objects)')
        ax.set_title(labels[i * 3 + j]+' distribution')
        ax.legend()
        
        # Establecer la escala logarítmica en el eje y
        ax.set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[91]:


# Crear una figura con subgráficos para cada combinación de índices de color
fig, axs = plt.subplots(5, 2, figsize=(12, 18))

# Scatter Plot 1: U_R vs G_Z
axs[0, 0].scatter(U_R_GALAXY_T, G_Z_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[0, 0].scatter(U_R_STAR_T, G_Z_STAR_T, label='STAR', color='green', alpha=0.7)
axs[0, 0].scatter(U_R_QSO_T, G_Z_QSO_T, label='QSO', color='red', alpha=0.7)
axs[0, 0].set_xlabel('U_R T')
axs[0, 0].set_ylabel('G_Z T')
axs[0, 0].set_title('U_R T vs G_Z T')
axs[0, 0].legend()

# Scatter Plot 2: U_R vs R_Z
axs[0, 1].scatter(U_R_GALAXY_T, R_Z_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[0, 1].scatter(U_R_STAR_T, R_Z_STAR_T, label='STAR', color='green', alpha=0.7)
axs[0, 1].scatter(U_R_QSO_T, R_Z_QSO_T, label='QSO', color='red', alpha=0.7)
axs[0, 1].set_xlabel('U_R T')
axs[0, 1].set_ylabel('R_Z T')
axs[0, 1].set_title('U_R T vs R_Z T')
axs[0, 1].legend()

# Scatter Plot 3: U_R vs I_Z
axs[1, 0].scatter(U_R_GALAXY_T, I_Z_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[1, 0].scatter(U_R_STAR_T, I_Z_STAR_T, label='STAR', color='green', alpha=0.7)
axs[1, 0].scatter(U_R_QSO_T, I_Z_QSO_T, label='QSO', color='red', alpha=0.7)
axs[1, 0].set_xlabel('U_R T')
axs[1, 0].set_ylabel('I_Z T')
axs[1, 0].set_title('U_R T vs I_Z T')
axs[1, 0].legend()

# Scatter Plot 4: U_R vs U_G
axs[1, 1].scatter(U_R_GALAXY_T, U_G_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[1, 1].scatter(U_R_STAR_T, U_G_STAR_T, label='STAR', color='green', alpha=0.7)
axs[1, 1].scatter(U_R_QSO_T, U_G_QSO_T, label='QSO', color='red', alpha=0.7)
axs[1, 1].set_xlabel('U_R T')
axs[1, 1].set_ylabel('U_G T')
axs[1, 1].set_title('U_R T vs U_G T')
axs[1, 1].legend()

# Scatter Plot 5: G_Z vs R_Z
axs[2, 0].scatter(G_Z_GALAXY_T, R_Z_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[2, 0].scatter(G_Z_STAR_T, R_Z_STAR_T, label='STAR', color='green', alpha=0.7)
axs[2, 0].scatter(G_Z_QSO_T, R_Z_QSO_T, label='QSO', color='red', alpha=0.7)
axs[2, 0].set_xlabel('G_Z T')
axs[2, 0].set_ylabel('R_Z T')
axs[2, 0].set_title('G_Z T vs R_Z T')
axs[2, 0].legend()

# Scatter Plot 6: G_Z vs I_Z
axs[2, 1].scatter(G_Z_GALAXY_T, I_Z_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[2, 1].scatter(G_Z_STAR_T, I_Z_STAR_T, label='STAR', color='green', alpha=0.7)
axs[2, 1].scatter(G_Z_QSO_T, I_Z_QSO_T, label='QSO', color='red', alpha=0.7)
axs[2, 1].set_xlabel('G_Z T')
axs[2, 1].set_ylabel('I_Z T')
axs[2, 1].set_title('G_Z T vs I_Z T')
axs[2, 1].legend()

# Scatter Plot 7: G_Z vs U_G
axs[3, 0].scatter(G_Z_GALAXY_T, U_G_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[3, 0].scatter(G_Z_STAR_T, U_G_STAR_T, label='STAR', color='green', alpha=0.7)
axs[3, 0].scatter(G_Z_QSO_T, U_G_QSO_T, label='QSO', color='red', alpha=0.7)
axs[3, 0].set_xlabel('G_Z T')
axs[3, 0].set_ylabel('U_G T')
axs[3, 0].set_title('G_Z T vs U_G T')
axs[3, 0].legend()

# Scatter Plot 8: R_Z vs I_Z
axs[3, 1].scatter(R_Z_GALAXY_T, I_Z_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[3, 1].scatter(R_Z_STAR_T, I_Z_STAR_T, label='STAR', color='green', alpha=0.7)
axs[3, 1].scatter(R_Z_QSO_T, I_Z_QSO_T, label='QSO', color='red', alpha=0.7)
axs[3, 1].set_xlabel('R_Z T')
axs[3, 1].set_ylabel('I_Z T')
axs[3, 1].set_title('R_Z T vs I_Z T')
axs[3, 1].legend()

# Scatter Plot 9: R_Z vs U_G
axs[4, 0].scatter(R_Z_GALAXY_T, U_G_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[4, 0].scatter(R_Z_STAR_T, U_G_STAR_T, label='STAR', color='green', alpha=0.7)
axs[4, 0].scatter(R_Z_QSO_T, U_G_QSO_T, label='QSO', color='red', alpha=0.7)
axs[4, 0].set_xlabel('R_Z T')
axs[4, 0].set_ylabel('U_G T')
axs[4, 0].set_title('R_Z T vs U_G T')
axs[4, 0].legend()

# Scatter Plot 10: I_Z vs U_G
axs[4, 1].scatter(I_Z_GALAXY_T, U_G_GALAXY_T, label='GALAXY', color='blue', alpha=0.7)
axs[4, 1].scatter(I_Z_STAR_T, U_G_STAR_T, label='STAR', color='green', alpha=0.7)
axs[4, 1].scatter(I_Z_QSO_T, U_G_QSO_T, label='QSO', color='red', alpha=0.7)
axs[4, 1].set_xlabel('I_Z T')
axs[4, 1].set_ylabel('U_G T')
axs[4, 1].set_title('I_Z T vs U_G T')
axs[4, 1].legend()

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# In[95]:


import itertools

# Lista de todos los índices de color
indices_color = ['U_R', 'U_G', 'G_Z', 'R_Z', 'I_Z']

# Lista de todos los filtros
filtros = ['FLUX_U', 'FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z']

# Generar todas las combinaciones posibles entre índices de color y filtros
combinaciones = list(itertools.product(indices_color, filtros))

# Crear una figura con subgráficos para cada combinación de índices de color y filtros
fig, axs = plt.subplots(len(combinaciones), 1, figsize=(10, 5*len(combinaciones)))

# Iterar sobre todas las combinaciones y crear los gráficos correspondientes
for i, (indice_color, filtro) in enumerate(combinaciones):
    # Scatter Plot para la combinación actual
    axs[i].scatter(eval(filtro + '_GALAXY'), eval(indice_color + '_GALAXY_T'), label='GALAXY', color='blue', alpha=0.7)
    axs[i].scatter(eval(filtro + '_STAR'), eval(indice_color + '_STAR_T'), label='STAR', color='green', alpha=0.7)
    axs[i].scatter(eval(filtro + '_QSO'), eval(indice_color + '_QSO_T'), label='QSO', color='red', alpha=0.7)
    axs[i].set_xlabel(filtro)
    axs[i].set_ylabel(indice_color)
    axs[i].set_title(f'{indice_color} vs {filtro}')
    axs[i].legend()

# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[ ]:





# ### We examine the total spectrum of each type to determine if the flux saturates at certain wavelengths

# In[25]:


# Sumar los flujos para cada clase de objeto celeste
espectro_star = np.sum(spectra_star, axis=0)
espectro_qso = np.sum(spectra_qso, axis=0) 
espectro_gal = np.sum(spectra_galaxy, axis=0)


# In[99]:


plt.figure()
plt.plot(wavelenght, espectro_gal, color='black')
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Total flux of galaxy spectra')
plt.show()


# In[100]:


plt.figure()
plt.plot(wavelenght, espectro_qso, color='black')
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Total flux of quasar spectra')
plt.show()


# In[101]:


plt.figure()
plt.plot(wavelenght, espectro_star, color='black')
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Total flux of star spectra')
plt.show()


# In[45]:


# Crea una instancia de la clase Spectrum
sp = pyspeckit.Spectrum(data=espectro_gal, xarr=wavelenght, xarrkwargs={'unit':'angstroms'})
# Realiza un suavizado utilizando el método smooth con un ancho de ventana específico
sp.smooth(5)  # Ajusta el valor según tus necesidades
# Encuentra los máximos locales en el espectro suavizado considerando solo los picos significativos
peaks, _ = find_peaks(sp.data, height=0.465 * np.max(sp.data))  # Ajusta el umbral según tus necesidades

# Define una función para encontrar el ancho medio
def half_width(x, y, peak_index):
    half_height = y[peak_index] / 2.0
    # Encuentra los índices donde el suavizado cruza la mitad de la altura del pico
    left_index = np.argmin(np.abs(y[:peak_index] - half_height))
    right_index = np.argmin(np.abs(y[peak_index:] - half_height)) + peak_index
    # Calcula la longitud de onda correspondiente a esos índices
    left_wavelength = x[left_index]
    right_wavelength = x[right_index]
    # Calcula el ancho medio
    width = (right_wavelength - left_wavelength)* ((max(wavelenght)-min(wavelenght))/len(wavelenght))
    return width

# Imprime información sobre los picos encontrados
for peak_index in peaks:
    amplitude = sp.data[peak_index]  # Amplitud del pico
    wavelength_at_peak = sp.xarr[peak_index]  # Longitud de onda en el pico
    width = half_width(sp.xarr, sp.data, peak_index)  # Ancho medio
    print(f"Pico - Amplitud: {amplitude}, Longitud de Onda: {wavelength_at_peak}, Ancho Medio: {width}")


# Grafica el espectro original, el suavizado y los picos encontrados
plt.figure(figsize=(10, 6))
plt.plot(wavelenght, espectro_gal, label='Espectro Original')
plt.plot(sp.xarr, sp.data, label='Espectro Suavizado')
plt.plot(sp.xarr[peaks], sp.data[peaks], 'ro', label='Picos encontrados')
plt.xlabel('Longitud de Onda (Angstroms)')
plt.ylabel('Intensidad')
plt.legend()
plt.show()


# 

# Los picos maximos para los filtros sinteticos son:

# Pico - Amplitud: 1010202.9701346895, Longitud de Onda: 5580.00000000045 Angstrom, Ancho Medio: 4735.182910279743 Angstrom

# In[ ]:





# In[31]:


# Crea una instancia de la clase Spectrum
sp = pyspeckit.Spectrum(data=espectro_qso, xarr=wavelenght, xarrkwargs={'unit':'angstroms'})
# Realiza un suavizado utilizando el método smooth con un ancho de ventana específico
sp.smooth(5)  # Ajusta el valor según tus necesidades
# Encuentra los máximos locales en el espectro suavizado considerando solo los picos significativos
peaks, _ = find_peaks(sp.data, height=0.2799 * np.max(sp.data))  # Ajusta el umbral según tus necesidades

# Define una función para encontrar el ancho medio
def half_width(x, y, peak_index):
    half_height = y[peak_index] / 2.0
    # Encuentra los índices donde el suavizado cruza la mitad de la altura del pico
    left_index = np.argmin(np.abs(y[:peak_index] - half_height))
    right_index = np.argmin(np.abs(y[peak_index:] - half_height)) + peak_index
    # Calcula la longitud de onda correspondiente a esos índices
    left_wavelength = x[left_index]
    right_wavelength = x[right_index]
    # Calcula el ancho medio
    width = (right_wavelength - left_wavelength)* ((max(wavelenght)-min(wavelenght))/len(wavelenght))
    return width

# Imprime información sobre los picos encontrados
for peak_index in peaks:
    amplitude = sp.data[peak_index]  # Amplitud del pico
    wavelength_at_peak = sp.xarr[peak_index]  # Longitud de onda en el pico
    width = half_width(sp.xarr, sp.data, peak_index)  # Ancho medio
    print(f"Pico - Amplitud: {amplitude}, Longitud de Onda: {wavelength_at_peak}, Ancho Medio: {width}")


# Grafica el espectro original, el suavizado y los picos encontrados
plt.figure(figsize=(10, 6))
plt.plot(wavelenght, espectro_qso, label='Espectro Original')
plt.plot(sp.xarr, sp.data, label='Espectro Suavizado')
plt.plot(sp.xarr[peaks], sp.data[peaks], 'ro', label='Picos encontrados')
plt.xlabel('Longitud de Onda (Angstroms)')
plt.ylabel('Intensidad')
plt.legend()
plt.show()


# ### Now we attempt chi-square, but with the mean flux value of the standardized spectra
spectra_galaxy
spectra_qso
spectra_star
# In[26]:


spectra_galaxy_mean = np.mean(spectra_galaxy, axis=0)
spectra_qso_mean = np.mean(spectra_qso, axis=0)
spectra_star_mean = np.mean(spectra_star, axis=0)


# In[105]:


# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, spectra_star_mean, 'k-', label='Datos')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Mean flux of star spectra')
plt.show()


# In[106]:


# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, spectra_galaxy_mean, 'k-', label='Datos')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Mean flux of galaxy spectra')
plt.show()


# In[107]:


# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, spectra_qso_mean, 'k-', label='Datos')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Mean flux of qso spectra')
plt.show()


# In[27]:


galaxy_chi_qso = []
galaxy_chi_galaxy = []
galaxy_chi_star = []

# Iterar sobre los espectros de galaxia
for spec in spectra_galaxy:
    # Calcular el chi-cuadrado para cada modelo utilizando la función chisquare
    chi_qso =  np.sum(((spec - spectra_qso_mean)**2)/spectra_qso_mean) 
    chi_galaxy = np.sum(((spec - spectra_galaxy_mean)**2)/spectra_galaxy_mean) 
    chi_star = np.sum(((spec - spectra_star_mean)**2)/spectra_star_mean) 
    
    galaxy_chi_qso.append(chi_qso)
    galaxy_chi_galaxy.append(chi_galaxy)
    galaxy_chi_star.append(chi_star)


# In[28]:


qso_chi_qso = []
qso_chi_galaxy = []
qso_chi_star = []

# Iterar sobre los espectros de galaxia
for spec in spectra_qso:
    # Calcular el chi-cuadrado para cada modelo utilizando la función chisquare
    chi_qso =  np.sum(((spec - spectra_qso_mean)**2)/spectra_qso_mean) 
    chi_galaxy = np.sum(((spec - spectra_galaxy_mean)**2)/spectra_galaxy_mean) 
    chi_star = np.sum(((spec - spectra_star_mean)**2)/spectra_star_mean) 
    
    qso_chi_qso.append(chi_qso)
    qso_chi_galaxy.append(chi_galaxy)
    qso_chi_star.append(chi_star)


# In[29]:


star_chi_qso = []
star_chi_galaxy = []
star_chi_star = []

# Iterar sobre los espectros de galaxia
for spec in spectra_star:
    # Calcular el chi-cuadrado para cada modelo utilizando la función chisquare
    chi_qso =  np.sum(((spec - spectra_qso_mean)**2)/spectra_qso_mean) 
    chi_galaxy = np.sum(((spec - spectra_galaxy_mean)**2)/spectra_galaxy_mean) 
    chi_star = np.sum(((spec - spectra_star_mean)**2)/spectra_star_mean) 
    
    star_chi_qso.append(chi_qso)
    star_chi_galaxy.append(chi_galaxy)
    star_chi_star.append(chi_star)


# In[125]:


# Crear subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para QSO
axs[0].hist(galaxy_chi_qso, bins=20, alpha=0.5, color='b')
axs[0].set_title('GALAXY as QSO distribution')
axs[0].set_xlabel('Chi-squared')
axs[0].set_ylabel('Ln(Number of objects)')
axs[0].set_yscale('log')

# Histograma para Galaxia
axs[1].hist(galaxy_chi_galaxy, bins=20, alpha=0.5, color='g')
axs[1].set_title('GALAXY as GALAXY distribution')
axs[1].set_xlabel('Chi-squared')
axs[1].set_ylabel('Ln(Number of objects)')
axs[1].set_yscale('log')

# Histograma para Estrella
axs[2].hist(galaxy_chi_star, bins=20, alpha=0.5, color='r')
axs[2].set_title('GALAXY as STAR distribution')
axs[2].set_xlabel('Chi-squared')
axs[2].set_ylabel('Ln(Number of objects)')
axs[2].set_yscale('log')

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Mostrar los subgráficos
plt.show()


# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Boxplot para GALAXY como QSO
ax.boxplot(galaxy_chi_qso, positions=[1], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='b'))
# Boxplot para GALAXY como GALAXY
ax.boxplot(galaxy_chi_galaxy, positions=[2], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='g'))
# Boxplot para GALAXY como STAR
ax.boxplot(galaxy_chi_star, positions=[3], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='r'))

# Configuraciones adicionales
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['GALAXY as QSO', 'GALAXY as GALAXY', 'GALAXY as STAR'])
ax.set_ylabel('Chi-squared value')
ax.set_title('GALAXY as QSO, GALAXY y STAR distributions')

# Mostrar la gráfica
plt.show()


# In[126]:


# Crear subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para QSO
axs[0].hist(qso_chi_qso, bins=20, alpha=0.5, color='b')
axs[0].set_title('QSO as QSO distribution')
axs[0].set_xlabel('Chi-squared value')
axs[0].set_ylabel('Ln(Number of objects)')
axs[0].set_yscale('log')

# Histograma para Galaxia
axs[1].hist(qso_chi_galaxy, bins=20, alpha=0.5, color='g')
axs[1].set_title('QSO as GALAXY distribution')
axs[1].set_xlabel('Chi-squared value')
axs[1].set_ylabel('Ln(Number of objects)')
axs[1].set_yscale('log')

# Histograma para Estrella
axs[2].hist(qso_chi_star, bins=20, alpha=0.5, color='r')
axs[2].set_title('QSO as STAR distribution')
axs[2].set_xlabel('Chi-squared value')
axs[2].set_ylabel('Ln(Number of objects)')
axs[2].set_yscale('log')

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Mostrar los subgráficos
plt.show()


# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Boxplot para GALAXY como QSO
ax.boxplot(qso_chi_qso, positions=[1], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='b'))
# Boxplot para GALAXY como GALAXY
ax.boxplot(qso_chi_galaxy, positions=[2], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='g'))
# Boxplot para GALAXY como STAR
ax.boxplot(qso_chi_star, positions=[3], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='r'))

# Configuraciones adicionales
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['QSO as QSO', 'QSO as GALAXY', 'QSO as STAR'])
ax.set_ylabel('Chi-squared value')
ax.set_title('QSO as QSO, GALAXY y STAR distributions')

# Mostrar la gráfica
plt.show()


# In[115]:


# Crear subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para QSO
axs[0].hist(star_chi_qso, bins=20, alpha=0.5, color='b')
axs[0].set_title('STAR as QSO distribution')
axs[0].set_xlabel('Chi-squared value')
axs[0].set_ylabel('Ln(Number of objects)')
axs[0].set_yscale('log')

# Histograma para Galaxia
axs[1].hist(star_chi_galaxy, bins=20, alpha=0.5, color='g')
axs[1].set_title('STAR as GALAXY distribution')
axs[1].set_xlabel('Chi-squared value')
axs[1].set_ylabel('Ln(Number of objects)')
axs[1].set_yscale('log')

# Histograma para Estrella
axs[2].hist(qso_chi_star, bins=20, alpha=0.5, color='r')
axs[2].set_title('STAR as STAR distribution')
axs[2].set_xlabel('Chi-squared value')
axs[2].set_ylabel('Ln(Number of objects)')
axs[2].set_yscale('log')

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Mostrar los subgráficos
plt.show()

# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Boxplot para GALAXY como QSO
ax.boxplot(star_chi_qso, positions=[1], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='b'))
# Boxplot para GALAXY como GALAXY
ax.boxplot(star_chi_galaxy, positions=[2], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='g'))
# Boxplot para GALAXY como STAR
ax.boxplot(star_chi_star, positions=[3], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='r'))

# Configuraciones adicionales
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['STAR as QSO', 'STAR as GALAXY', 'STAR as STAR'])
ax.set_ylabel('Chi-squared value')
ax.set_title('STAR as QSO, GALAXY y STAR distributions')

# Mostrar la gráfica
plt.show()


# Intentemos ver como se comparan los azules, los verdes y los rojos

# In[116]:


# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma para GALAXY como QSO
ax.hist(galaxy_chi_qso, bins=20, alpha=0.5, color='b', label='GALAXY as QSO')
# Histograma para QSO como QSO
ax.hist(qso_chi_qso, bins=20, alpha=0.5, color='r', label='QSO as QSO')
ax.hist(star_chi_qso, bins=20, alpha=0.5, color='g', label='STAR as QSO')

# Configuraciones adicionales
ax.set_xlabel('Chi-squared value')
ax.set_ylabel('Ln(Number of objects)')
ax.set_title('GALAXY, STAR y QSO as QSO distributions')
ax.legend()
ax.set_yscale('log')  # Establecer escala logarítmica en el eje y

# Mostrar la gráfica
plt.show()

# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))
# Boxplot para GALAXY como QSO
ax.boxplot(galaxy_chi_qso, positions=[1], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='b'))
# Boxplot para GALAXY como GALAXY
ax.boxplot(qso_chi_qso, positions=[2], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='b'))
# Boxplot para GALAXY como STAR
ax.boxplot(star_chi_qso, positions=[3], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='b'))
# Configuraciones adicionales
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['GALAXY as QSO', 'QSO as QSO', 'STAR as QSO'])
ax.set_ylabel('Chi-squared value')
ax.set_title('GALAXY, STAR, QSO as QSO distributions')

# Mostrar la gráfica
plt.show()



# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))
# Boxplot para GALAXY como QSO
ax.boxplot(galaxy_chi_qso, positions=[1], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='b'))
# Boxplot para GALAXY como GALAXY
ax.boxplot(qso_chi_qso, positions=[2], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='b'))
# Configuraciones adicionales
ax.set_xticks([1, 2])
ax.set_xticklabels(['GALAXY as QSO', 'QSO as QSO'])
ax.set_ylabel('Chi-squared value')
ax.set_title('GALAXY, QSO as QSO distributions')

# Mostrar la gráfica
plt.show()


# In[117]:


# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma para GALAXY como QSO
ax.hist(galaxy_chi_galaxy, bins=20, alpha=0.5, color='b', label='GALAXY as GALAXY')
# Histograma para QSO como QSO
ax.hist(qso_chi_galaxy, bins=20, alpha=0.5, color='r', label='QSO as GALAXY')
ax.hist(star_chi_galaxy, bins=20, alpha=0.5, color='g', label='STAR as GALAXY')

# Configuraciones adicionales
ax.set_xlabel('Chi-squared value')
ax.set_ylabel('Ln(Number of objects)')
ax.set_title('GALAXY, STAR y QSO as GALAXY distributions')
ax.legend()
ax.set_yscale('log')  # Establecer escala logarítmica en el eje y

# Mostrar la gráfica
plt.show()



# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))
# Boxplot para GALAXY como GALAXY
ax.boxplot(galaxy_chi_galaxy, positions=[1], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='g'))
# Boxplot para QSO como GALAXY
ax.boxplot(qso_chi_galaxy, positions=[2], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='g'))
# Boxplot para STAR como GALAXY
ax.boxplot(star_chi_galaxy, positions=[3], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='g'))
# Configuraciones adicionales
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['GALAXY as GALAXY', 'QSO as GALAXY', 'STAR as GALAXY'])
ax.set_ylabel('Chi-squared value')
ax.set_title('GALAXY, QSO y STAR as GALAXY distributions')

# Mostrar la gráfica
plt.show()



# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))
# Boxplot para GALAXY como GALAXY
ax.boxplot(galaxy_chi_galaxy, positions=[1], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='g'))
# Boxplot para QSO como GALAXY
ax.boxplot(qso_chi_galaxy, positions=[2], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='g'))
# Configuraciones adicionales
ax.set_xticks([1, 2])
ax.set_xticklabels(['GALAXY as GALAXY', 'QSO as GALAXY'])
ax.set_ylabel('Chi-squared value')
ax.set_title('GALAXY, QSO as GALAXY distributions')

# Mostrar la gráfica
plt.show()


# In[118]:


# Crear la figura y el subplot para el histograma
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma para GALAXY como STAR
ax.hist(galaxy_chi_star, bins=20, alpha=0.5, color='b', label='GALAXY as STAR')
# Histograma para QSO como STAR
ax.hist(qso_chi_star, bins=20, alpha=0.5, color='r', label='QSO as STAR')
# Histograma para STAR como STAR
ax.hist(star_chi_star, bins=20, alpha=0.5, color='g', label='STAR as STAR')

# Configuraciones adicionales
ax.set_xlabel('Chi-squared value')
ax.set_ylabel('Ln(Number of objects)')
ax.set_title('GALAXY, QSO y STAR as STAR distributions')
ax.legend()
ax.set_yscale('log')  # Establecer escala logarítmica en el eje y

# Mostrar la gráfica del histograma
plt.show()

# Crear la figura y el subplot para los boxplots
fig, ax = plt.subplots(figsize=(10, 6))
# Boxplot para GALAXY como STAR
ax.boxplot(galaxy_chi_star, positions=[1], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='r'))
# Boxplot para QSO como STAR
ax.boxplot(qso_chi_star, positions=[2], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='r'))
# Boxplot para STAR como STAR
ax.boxplot(star_chi_star, positions=[3], showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='r'))
# Configuraciones adicionales
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['GALAXY as STAR', 'QSO as STAR', 'STAR as STAR'])
ax.set_ylabel('Chi-squared value')
ax.set_title('GALAXY, QSO y STAR as STAR distributions')

# Mostrar la gráfica de los boxplots
plt.show()

Ahora hagamos un grafico 3d Donde cada las coordenadas son el chi_star, chi_galaxy, chi_qso
# In[119]:


from mpl_toolkits.mplot3d import Axes3D

# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, galaxy_chi_star, c='b', marker='o', label='GALAXY')
ax.scatter(qso_chi_qso, qso_chi_galaxy, qso_chi_star, c='r', marker='o', label='QSO')
ax.scatter(star_chi_qso, star_chi_galaxy, star_chi_star, c='g', marker='o', label='STAR')
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi GALAXY')
ax.set_zlabel('Chi STAR')
ax.set_title('Chi-squared 3D diagram')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()


# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, galaxy_chi_star, c='b', marker='o', label='GALAXY')
ax.scatter(qso_chi_qso, qso_chi_galaxy, qso_chi_star, c='r', marker='o', label='QSO')
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi GALAX')
ax.set_zlabel('Chi STAR')
ax.set_title('Chi-squared 3D diagram')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()

# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(qso_chi_qso, qso_chi_galaxy, qso_chi_star, c='r', marker='o', label='QSO')
ax.scatter(star_chi_qso, star_chi_galaxy, star_chi_star, c='g', marker='o', label='STAR')
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi GALAXY')
ax.set_zlabel('Chi STAR')
ax.set_title('Chi-squared 3D diagram')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()

# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, galaxy_chi_star, c='b', marker='o', label='GALAXY')
ax.scatter(star_chi_qso, star_chi_galaxy, star_chi_star, c='g', marker='o', label='STAR')
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi GALAXY')
ax.set_zlabel('Chi STAR')
ax.set_title('Chi-squared 3D diagram')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()


# In[127]:


# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, galaxy_chi_star, c='b', marker='o', label='GALAXY')
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, np.zeros_like(galaxy_chi_star), color='gray', alpha=0.3)  # Agregar sombra
ax.scatter(galaxy_chi_qso, np.zeros_like(galaxy_chi_galaxy), galaxy_chi_star, color='gray', alpha=0.3)  # Agregar sombra
ax.scatter(np.zeros_like(galaxy_chi_qso), galaxy_chi_galaxy, galaxy_chi_star, color='gray', alpha=0.3)  # Agregar sombra
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi GALAXY')
ax.set_zlabel('Chi STAR')
ax.set_title('Chi-squared 3D diagram')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()


#Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(qso_chi_qso, qso_chi_galaxy, qso_chi_star, c='r', marker='o', label='QSO')
ax.scatter(qso_chi_qso, qso_chi_galaxy, np.zeros_like(qso_chi_star), color='gray', alpha=0.1)  # Agregar sombra
ax.scatter(qso_chi_qso, np.zeros_like(qso_chi_galaxy), qso_chi_star, color='gray', alpha=0.1)  # Agregar sombra
ax.scatter(np.zeros_like(qso_chi_qso), qso_chi_galaxy, qso_chi_star, color='gray', alpha=0.1)  # Agregar sombra
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi GALAXY')
ax.set_zlabel('Chi STAR')
ax.set_title('Chi-squared 3D diagram')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()


# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D con sombra
ax.scatter(star_chi_qso, star_chi_galaxy, star_chi_star, c='g', marker='o', label='STAR')
ax.scatter(star_chi_qso, star_chi_galaxy, np.zeros_like(star_chi_star), color='gray', alpha=0.1)  # Agregar sombra
ax.scatter(star_chi_qso, np.zeros_like(star_chi_galaxy), star_chi_star, color='gray', alpha=0.1)  # Agregar sombra
ax.scatter(np.zeros_like(star_chi_qso), star_chi_galaxy, star_chi_star, color='gray', alpha=0.1)  # Agregar sombra
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi GALAXY')
ax.set_zlabel('Chi STAR')
ax.set_title('Chi-squared 3D diagram')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()


# Ahora, veremos si las pendientes pueden contener informacion util

# In[30]:


gal_galqso= np.array(galaxy_chi_galaxy)/np.array(galaxy_chi_qso)
gal_galstar= np.array(galaxy_chi_galaxy)/np.array(galaxy_chi_star)
gal_qsostar= np.array(galaxy_chi_qso)/np.array(galaxy_chi_star)


# In[31]:


qso_galqso= np.array(qso_chi_galaxy)/np.array(qso_chi_qso)
qso_galstar= np.array(qso_chi_galaxy)/np.array(qso_chi_star)
qso_qsostar= np.array(qso_chi_qso)/np.array(qso_chi_star)


# In[32]:


star_galqso= np.array(star_chi_galaxy)/np.array(star_chi_qso)
star_galstar= np.array(star_chi_galaxy)/np.array(star_chi_star)
star_qsostar= np.array(star_chi_qso)/np.array(star_chi_star)


# In[131]:


# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_galqso
plt.hist(gal_galqso, bins=20, color='b', alpha=0.5, label='GALAXY')
# Histograma para qso_galqso
plt.hist(qso_galqso, bins=20, color='r', alpha=0.5, label='QSO')
# Histograma para star_galqso
plt.hist(star_galqso, bins=20, color='g', alpha=0.5, label='STAR')
# Configuraciones adicionales
plt.title('(CHI^2 GALAXY/ CHI^2 QSO) distributions')
plt.xlabel('CHI^2 GALAXY/ CHI^2 QSO')
plt.ylabel('Ln(Number of objects)')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()


# In[132]:


# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_galstar
plt.hist(gal_galstar, bins=20, color='b', alpha=0.5, label='GAL')
# Histograma para qso_galstar
plt.hist(qso_galstar, bins=20, color='r', alpha=0.5, label='QSO')
# Histograma para star_galstar
plt.hist(star_galstar, bins=20, color='g', alpha=0.5, label='STAR')
# Configuraciones adicionales
plt.title('(CHI^2 GALAXY/ CHI^2 STAR) distributions')
plt.xlabel('CHI^2 GALAXY/ CHI^2 STAR')
plt.ylabel('Ln(Number of objects)')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()

# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_galstar
plt.hist(gal_galstar, bins=20, color='b', alpha=0.5, label='GAL')
# Histograma para qso_galstar
plt.hist(qso_galstar, bins=20, color='r', alpha=0.5, label='QSO')
plt.title('CHI^2 GALAXY/ CHI^2 STAR) distributions')
plt.xlabel('CHI^2 GALAXY/ CHI^2 STAR')
plt.ylabel('Ln(Number of objects)')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()


# In[133]:


# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_qsostar
plt.hist(gal_qsostar, bins=20, color='b', alpha=0.5, label='GAL')
# Histograma para qso_qsostar
plt.hist(qso_qsostar, bins=20, color='r', alpha=0.5, label='QSO')
# Histograma para star_qsostar
plt.hist(star_qsostar, bins=20, color='g', alpha=0.5, label='STAR')
# Configuraciones adicionales
plt.title('CHI^2 QSO/ CHI^2 STAR) distributions')
plt.xlabel('CHI^2 QSO/ CHI^2 STAR')
plt.ylabel('Ln(Number of objects)')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()

# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_qsostar
plt.hist(gal_qsostar, bins=20, color='b', alpha=0.5, label='GAL')
# Histograma para qso_qsostar
plt.hist(qso_qsostar, bins=20, color='r', alpha=0.5, label='QSO')
# Configuraciones adicionales
plt.title('CHI^2 QSO/ CHI^2 STAR) distributions')
plt.xlabel('CHI^2 QSO/ CHI^2 STAR')
plt.ylabel('Ln(Number of objects)')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()


# ### Abbe Parameter
spectra_galaxy
spectra_qso
spectra_star
# In[33]:


abbe_gal=[]
abbe_qso=[]
abbe_star=[]

for spec in spectra_galaxy:
    parte_1 = (len(spec)/(2*(len(spec)-1))) * (1/(np.sum((spec-np.mean(spec))**2)))
    parte_2 = 0
    for i in range(len(spec)-1):
        parte_2 += (spec[i+1]-spec[i])**2
    abbe_value = parte_1 * parte_2
    abbe_gal.append(abbe_value)

for spec in spectra_qso:
    parte_1 = (len(spec)/(2*(len(spec)-1))) * (1/(np.sum((spec-np.mean(spec))**2)))
    parte_2 = 0
    for i in range(len(spec)-1):
        parte_2 += (spec[i+1]-spec[i])**2
    abbe_value = parte_1 * parte_2
    abbe_qso.append(abbe_value)

for spec in spectra_star:
    parte_1 = (len(spec)/(2*(len(spec)-1))) * (1/(np.sum((spec-np.mean(spec))**2)))
    parte_2 = 0
    for i in range(len(spec)-1):
        parte_2 += (spec[i+1]-spec[i])**2
    abbe_value = parte_1 * parte_2
    abbe_star.append(abbe_value)

abbe_gal=np.array(abbe_gal)
abbe_qso=np.array(abbe_qso)
abbe_star=np.array(abbe_star)


# In[12]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [abbe_gal, abbe_star, abbe_qso]

# Etiquetas para los boxplots
labels = ['ABBE GALAXY', 'ABBE STAR', 'ABBE QSO']

# Crear la figura y los ejes de la figura
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Crear el boxplot con outliers
axs[0].boxplot(data)
axs[0].set_xticklabels(['GALAXY', 'STAR', 'QSO'])
axs[0].set_title('ABBE distributions')
axs[0].set_xlabel('Class')
axs[0].set_ylabel('ABBE')

# Crear el boxplot sin outliers
axs[1].boxplot(data, showfliers=False)
axs[1].set_xticklabels(['GALAXY', 'STAR', 'QSO'])
axs[1].set_title('ABBE distributions')
axs[1].set_xlabel('Class')
axs[1].set_ylabel('ABBE')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[13]:


# Datos para los histogramas
data = [abbe_gal, abbe_star, abbe_qso]

# Etiquetas para los histogramas
labels = ['ABBE GALAXY', 'ABBE STAR', 'ABBE QSO']

# Colores para los histogramas
colors = ['blue', 'green', 'red']

# Crear una figura con subgráficos para cada clase
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase en la primera columna
for i, class_data in enumerate(data):
    axs[0].hist(class_data, bins=10, color=colors[i], alpha=0.7, label=labels[i])

# Agregar leyenda y título a la primera columna
axs[0].legend()
axs[0].set_title('ABBE distributions')
axs[0].set_xlabel('ABBE')
axs[0].set_ylabel('Number of objects')

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase en la segunda columna
for i, class_data in enumerate(data):
    axs[1].hist(class_data, bins=10, color=colors[i], alpha=0.7, label=labels[i])

# Agregar leyenda y título a la segunda columna
axs[1].legend()
axs[1].set_title('ABBE distributions')
axs[1].set_xlabel('ABBE')
axs[1].set_ylabel('Ln(Number of objects)')

# Establecer escala logarítmica en el eje y para la segunda columna
axs[1].set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[ ]:





# ### Color-magnitude diagram

# In[ ]:


F_ref=1 #Este es el flujo de referencia como punto cero

#Features con los los flujos
U_COLOR_GALAXY= -2.5*np.log10(np.abs(FLUX_U_GALAXY)/F_ref)
U_COLOR_STAR= -2.5*np.log10(np.abs(FLUX_U_STAR)/F_ref)
U_COLOR_QSO= -2.5*np.log10(np.abs(FLUX_U_QSO)/F_ref)

G_COLOR_GALAXY= -2.5*np.log10(np.abs(FLUX_G_GALAXY)/F_ref)
G_COLOR_STAR= -2.5*np.log10(np.abs(FLUX_G_STAR)/F_ref)
G_COLOR_QSO= -2.5*np.log10(np.abs(FLUX_G_QSO)/F_ref)

R_COLOR_GALAXY= -2.5*np.log10(np.abs(FLUX_R_GALAXY)/F_ref)
R_COLOR_STAR= -2.5*np.log10(np.abs(FLUX_R_STAR)/F_ref)
R_COLOR_QSO= -2.5*np.log10(np.abs(FLUX_R_QSO)/F_ref)

I_COLOR_GALAXY= -2.5*np.log10(np.abs(FLUX_I_GALAXY)/F_ref)
I_COLOR_STAR= -2.5*np.log10(np.abs(FLUX_I_STAR)/F_ref)
I_COLOR_QSO= -2.5*np.log10(np.abs(FLUX_I_QSO)/F_ref)

Z_COLOR_GALAXY= -2.5*np.log10(np.abs(FLUX_Z_GALAXY)/F_ref)
Z_COLOR_STAR= -2.5*np.log10(np.abs(FLUX_Z_STAR)/F_ref)
Z_COLOR_QSO= -2.5*np.log10(np.abs(FLUX_Z_QSO)/F_ref)

U_COLOR_GALAXY= -2.5*np.log10(np.abs(FLUX_U_GALAXY)/F_ref)
U_COLOR_STAR= -2.5*np.log10(np.abs(FLUX_U_STAR)/F_ref)
U_COLOR_QSO= -2.5*np.log10(np.abs(FLUX_U_QSO)/F_ref)


# In[ ]:


#creamos los indices de color:

#Features con indices de color
U_R_COLOR_GALAXY= U_COLOR_GALAXY-R_COLOR_GALAXY
U_R_COLOR_STAR= U_COLOR_STAR-R_COLOR_STAR
U_R_COLOR_QSO= U_COLOR_QSO-R_COLOR_QSO

U_G_COLOR_GALAXY= U_COLOR_GALAXY-G_COLOR_GALAXY
U_G_COLOR_STAR= U_COLOR_STAR-G_COLOR_STAR
U_G_COLOR_QSO= U_COLOR_QSO-G_COLOR_QSO

G_Z_COLOR_GALAXY= G_COLOR_GALAXY-Z_COLOR_GALAXY
G_Z_COLOR_STAR= G_COLOR_STAR-Z_COLOR_STAR
G_Z_COLOR_QSO= G_COLOR_QSO-Z_COLOR_QSO

R_Z_COLOR_GALAXY= R_COLOR_GALAXY-Z_COLOR_GALAXY
R_Z_COLOR_STAR= R_COLOR_STAR-Z_COLOR_STAR
R_Z_COLOR_QSO= R_COLOR_QSO-Z_COLOR_QSO

I_Z_COLOR_GALAXY= I_COLOR_GALAXY-Z_COLOR_GALAXY
I_Z_COLOR_STAR= I_COLOR_STAR-Z_COLOR_STAR
I_Z_COLOR_QSO= I_COLOR_QSO-Z_COLOR_QSO


# In[ ]:


#Lista de todos los índices de color
indices_color = ['U_COLOR', 'G_COLOR', 'R_COLOR', 'I_COLOR', 'Z_COLOR']

# Lista de todos los filtros
colores = ['U_R_COLOR', 'U_G_COLOR', 'G_Z_COLOR', 'R_Z_COLOR', 'I_Z_COLOR']

# Generar todas las combinaciones posibles entre índices de color y filtros
combinaciones = list(itertools.product(indices_color, colores))

# Crear una figura con subgráficos para cada combinación de índices de color y filtros
fig, axs = plt.subplots(len(combinaciones), 3, figsize=(15, 5*len(combinaciones)))

# Iterar sobre todas las combinaciones y crear los gráficos correspondientes
for i, (indice_color, filtro) in enumerate(combinaciones):
    # Scatter Plot para GALAXY
    axs[i, 0].scatter(eval(filtro + '_GALAXY'), eval(indice_color + '_GALAXY'), label='GALAXY', color='blue', alpha=0.6)
    axs[i, 0].set_xlabel(filtro)
    axs[i, 0].set_ylabel(indice_color)
    axs[i, 0].set_title(f'{indice_color} vs {filtro}')
    axs[i, 0].legend()
    axs[i, 0].invert_yaxis()

    # Scatter Plot para STAR
    axs[i, 1].scatter(eval(filtro + '_STAR'), eval(indice_color + '_STAR'), label='STAR', color='green', alpha=0.6)
    axs[i, 1].set_xlabel(filtro)
    axs[i, 1].set_ylabel(indice_color)
    axs[i, 1].set_title(f'{indice_color} vs {filtro}')
    axs[i, 1].legend()
    axs[i, 1].invert_yaxis()

    # Scatter Plot para QSO
    axs[i, 2].scatter(eval(filtro + '_QSO'), eval(indice_color + '_QSO'), label='QSO', color='red', alpha=0.6)
    axs[i, 2].set_xlabel(filtro)
    axs[i, 2].set_ylabel(indice_color)
    axs[i, 2].set_title(f'{indice_color} vs {filtro}')
    axs[i, 2].legend()
    axs[i, 2].invert_yaxis()

# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[ ]:





# ### Haremos MAD en rangos de a 1000 A, para encontrar espectros defectuosos
spectra
wavelenght
# In[ ]:


# Calculamos los límites de los intervalos
num_intervals = 6
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


# In[9]:


interval_limits


# In[10]:


np.array(mad_parameters).shape


# In[64]:


#Barrido vertical
# Calcular el valor umbral para cada lista de mad_parameters
thresholds = [max(mad_list) * 0.001 for mad_list in mad_parameters]

# Inicializar la lista de índices a comprobar
check = []

# Iterar sobre las listas de MAD y encontrar los índices que cumplen con el criterio
for i, mad_list in enumerate(mad_parameters):
    for j, mad in enumerate(mad_list):
        if mad <= thresholds[i]:
            check.append((i, j))  # Añadir el índice a la lista check


# In[98]:


#Barrido horizontal
# Inicializar la lista de índices a comprobar
check = []

# Iterar sobre las seis posiciones de las listas en mad_parameters
for j in range(len(mad_parameters[0])):
    # Calcular el máximo de las seis listas en la posición j
    max_value = max(mad_list[j] for mad_list in mad_parameters)
    
    # Calcular el umbral como el 1% del valor máximo en esta posición
    threshold = max_value * 0.24
    
    # Verificar si algún valor en la posición j cumple con el criterio
    for i in range(len(mad_parameters)):
        if mad_parameters[i][j] <= threshold:
            check.append((i, j))  # Añadir el índice a la lista check
            break  # Salir del bucle interno si se cumple el criterio


# In[99]:


len(check)


# In[100]:


for k in check:
    plt.plot(wavelenght, spectra[k[1]], color='black')
    plt.title('Spectra_checking'+str(k))
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.show()


# In[69]:


threshold


# In[13]:





# ### Features aleatoria

# In[34]:


# Establecer la semilla
random.seed(42)  # Puedes cambiar este número por cualquier otro valor entero

# Generar números aleatorios
aleatorio = [random.random() for _ in spectra]
aleatorio = np.array(aleatorio)


# In[35]:


len(aleatorio)


# ### Construimos el modelo con las 55 features+ la aleatoria

# In[36]:


# Inicializa el cluster de H2O y mostrar info del cluster en uso
h2o.init()


# In[37]:


#Features con los los flujos
FLUX_U_GALAXY= np.array(FLUX_U_GALAXY)
FLUX_U_STAR= np.array(FLUX_U_STAR)
FLUX_U_QSO= np.array(FLUX_U_QSO)
FLUX_U= np.concatenate([FLUX_U_GALAXY, FLUX_U_STAR, FLUX_U_QSO])

FLUX_G_GALAXY= np.array(FLUX_V_G_GALAXY)
FLUX_G_STAR= np.array(FLUX_V_G_STAR)
FLUX_G_QSO= np.array(FLUX_V_G_QSO)
FLUX_G= np.concatenate([FLUX_G_GALAXY, FLUX_G_STAR, FLUX_G_QSO])

FLUX_R_GALAXY= np.array(FLUX_R_GALAXY)
FLUX_R_STAR= np.array(FLUX_R_STAR)
FLUX_R_QSO= np.array(FLUX_R_QSO)
FLUX_R= np.concatenate([FLUX_R_GALAXY, FLUX_R_STAR, FLUX_R_QSO])

FLUX_I_GALAXY= np.array(FLUX_I_GALAXY)
FLUX_I_STAR= np.array(FLUX_I_STAR)
FLUX_I_QSO= np.array(FLUX_I_QSO)
FLUX_I= np.concatenate([FLUX_I_GALAXY, FLUX_I_STAR, FLUX_I_QSO])

FLUX_Z_GALAXY= np.array(FLUX_z_GALAXY)
FLUX_Z_STAR= np.array(FLUX_z_STAR)
FLUX_Z_QSO= np.array(FLUX_z_QSO)
FLUX_Z= np.concatenate([FLUX_Z_GALAXY, FLUX_Z_STAR, FLUX_Z_QSO])

#Features con indices de color
U_R_GALAXY= FLUX_U_GALAXY-FLUX_R_GALAXY
U_R_STAR= FLUX_U_STAR-FLUX_R_STAR
U_R_QSO= FLUX_U_QSO-FLUX_R_QSO
U_R= np.concatenate([U_R_GALAXY, U_R_STAR, U_R_QSO])

U_G_GALAXY= FLUX_U_GALAXY-FLUX_G_GALAXY
U_G_STAR= FLUX_U_STAR-FLUX_G_STAR
U_G_QSO= FLUX_U_QSO-FLUX_G_QSO
U_G= np.concatenate([U_G_GALAXY, U_G_STAR, U_G_QSO])

G_Z_GALAXY= FLUX_G_GALAXY-FLUX_Z_GALAXY
G_Z_STAR= FLUX_G_STAR-FLUX_Z_STAR
G_Z_QSO= FLUX_G_QSO-FLUX_Z_QSO
G_Z= np.concatenate([G_Z_GALAXY, G_Z_STAR, G_Z_QSO])

R_Z_GALAXY= FLUX_R_GALAXY-FLUX_Z_GALAXY
R_Z_STAR= FLUX_R_STAR-FLUX_Z_STAR
R_Z_QSO= FLUX_R_QSO-FLUX_Z_QSO
R_Z= np.concatenate([R_Z_GALAXY, R_Z_STAR, R_Z_QSO])

I_Z_GALAXY= FLUX_I_GALAXY-FLUX_Z_GALAXY
I_Z_STAR= FLUX_I_STAR-FLUX_Z_STAR
I_Z_QSO= FLUX_I_QSO-FLUX_Z_QSO
I_Z= np.concatenate([I_Z_GALAXY, I_Z_STAR, I_Z_QSO])

#Features con razones entres flujos
UR_GALAXY= FLUX_U_GALAXY/FLUX_R_GALAXY
UR_STAR= FLUX_U_STAR/FLUX_R_STAR
UR_QSO= FLUX_U_QSO/FLUX_R_QSO
UR= np.concatenate([UR_GALAXY, UR_STAR, UR_QSO])

GZ_GALAXY= FLUX_G_GALAXY/FLUX_Z_GALAXY
GZ_STAR= FLUX_G_STAR/FLUX_Z_STAR
GZ_QSO= FLUX_G_QSO/FLUX_Z_QSO
GZ= np.concatenate([GZ_GALAXY, GZ_STAR, GZ_QSO])

RZ_GALAXY= FLUX_R_GALAXY/FLUX_Z_GALAXY
RZ_STAR= FLUX_R_STAR/FLUX_Z_STAR
RZ_QSO= FLUX_R_QSO/FLUX_Z_QSO
RZ= np.concatenate([RZ_GALAXY, RZ_STAR, RZ_QSO])

IZ_GALAXY= FLUX_I_GALAXY/FLUX_Z_GALAXY
IZ_STAR= FLUX_I_STAR/FLUX_Z_STAR
IZ_QSO= FLUX_I_QSO/FLUX_Z_QSO
IZ= np.concatenate([IZ_GALAXY, IZ_STAR, IZ_QSO])

UG_GALAXY= FLUX_U_GALAXY/FLUX_G_GALAXY
UG_STAR= FLUX_U_STAR/FLUX_G_STAR
UG_QSO= FLUX_U_QSO/FLUX_G_QSO
UG= np.concatenate([UG_GALAXY, UG_STAR, UG_QSO])

UZ_GALAXY= FLUX_U_GALAXY/FLUX_Z_GALAXY
UZ_STAR= FLUX_U_STAR/FLUX_Z_STAR
UZ_QSO= FLUX_U_QSO/FLUX_Z_QSO
UZ= np.concatenate([UZ_GALAXY, UZ_STAR, UZ_QSO])

#Razon entre indices de color
G_Z_R_Z_GALAXY= G_Z_GALAXY/R_Z_GALAXY #Esta features es (G-Z)/(R-Z)
G_Z_R_Z_STAR= G_Z_STAR/R_Z_STAR
G_Z_R_Z_QSO= G_Z_QSO/R_Z_QSO
G_Z_R_Z= np.concatenate([G_Z_R_Z_GALAXY, G_Z_R_Z_STAR, G_Z_R_Z_QSO])

#Feature con las curvas:
CURV_GALAXY= np.array(curv_gal)
CURV_STAR= np.array(curv_star)
CURV_QSO= np.array(curv_qso)
CURV= np.concatenate([CURV_GALAXY, CURV_STAR, CURV_QSO])

#Feature con MAD:
MAD_GALAXY= np.array(mad_gal)
MAD_STAR= np.array(mad_star)
MAD_QSO= np.array(mad_qso)
MAD= np.concatenate([MAD_GALAXY, MAD_STAR, MAD_QSO])

#Feature con FLUX_O:
F0_GALAXY= np.array(FLUX_0_GALAXY)
F0_STAR= np.array(FLUX_0_STAR)
F0_QSO= np.array(FLUX_0_QSO)
F0= np.concatenate([F0_GALAXY, F0_STAR, F0_QSO])

#Feature con FLUX_1:
F1_GALAXY= np.array(FLUX_1_GALAXY)
F1_STAR= np.array(FLUX_1_STAR)
F1_QSO= np.array(FLUX_1_QSO)
F1= np.concatenate([F1_GALAXY, F1_STAR, F1_QSO])

#Feature con FLUX_2:
F2_GALAXY= np.array(FLUX_2_GALAXY)
F2_STAR= np.array(FLUX_2_STAR)
F2_QSO= np.array(FLUX_2_QSO)
F2= np.concatenate([F2_GALAXY, F2_STAR, F2_QSO])

#Feature con FLUX_3:
F3_GALAXY= np.array(FLUX_3_GALAXY)
F3_STAR= np.array(FLUX_3_STAR)
F3_QSO= np.array(FLUX_3_QSO)
F3= np.concatenate([F3_GALAXY, F3_STAR, F3_QSO])

#Features con razones entre flujos sinteticos y otros parametros:
ratios = [
    [_0Z_GALAXY, _0Z_STAR, _0Z_QSO],
    [_0G_GALAXY, _0G_STAR, _0G_QSO],
    [_0R_GALAXY, _0R_STAR, _0R_QSO],
    [_0I_GALAXY, _0I_STAR, _0I_QSO],
    [_01_GALAXY, _01_STAR, _01_QSO],
    [_02_GALAXY, _02_STAR, _02_QSO],
    [_03_GALAXY, _03_STAR, _03_QSO],
    [_1U_GALAXY, _1U_STAR, _1U_QSO],
    [_1Z_GALAXY, _1Z_STAR, _1Z_QSO],
    [_1R_GALAXY, _1R_STAR, _1R_QSO],
    [_1I_GALAXY, _1I_STAR, _1I_QSO],
    [_12_GALAXY, _12_STAR, _12_QSO],
    [_2Z_GALAXY, _2Z_STAR, _2Z_QSO],
    [_2R_GALAXY, _2R_STAR, _2R_QSO],
    [_2I_GALAXY, _2I_STAR, _2I_QSO],
    [_23_GALAXY, _23_STAR, _23_QSO],
    [_3U_GALAXY, _3U_STAR, _3U_QSO],
    [_3Z_GALAXY, _3Z_STAR, _3Z_QSO],
    [_3R_GALAXY, _3R_STAR, _3R_QSO],
    [_3I_GALAXY, _3I_STAR, _3I_QSO]
]

nombres = [
    '_0Z',
    '_0G',
    '_0R',
    '_0I',
    '_01',
    '_02',
    '_03',
    '_1U',
    '_1Z',
    '_1R',
    '_1I',
    '_12',
    '_2Z',
    '_2R',
    '_2I',
    '_23',
    '_3U',
    '_3Z',
    '_3R',
    '_3I'
]

for nombre, ratio in zip(nombres, ratios):
    # Concatenar los valores de cada sublista en la lista de ratios
    valor_concatenado = np.concatenate([ratio[0], ratio[1], ratio[2]])
    # Asignar el valor concatenado a la variable con el nombre correspondiente
    globals()[nombre] = valor_concatenado

#Features de Temperatura o transformaciones con Log(f1/f2)
datas = [
    [U_R_GALAXY_T, U_R_STAR_T, U_R_QSO_T],
    [U_G_GALAXY_T, U_G_STAR_T, U_G_QSO_T],
    [G_Z_GALAXY_T, G_Z_STAR_T, G_Z_QSO_T],
    [R_Z_GALAXY_T, R_Z_STAR_T, R_Z_QSO_T],
    [I_Z_GALAXY_T, I_Z_STAR_T, I_Z_QSO_T]
]

names = [
    'T_UR',
    'T_UG',
    'T_GZ',
    'T_RZ',
    'T_IZ'
]

for name, data in zip(names, datas):
    # Concatenar los valores de cada sublista en la lista de ratios
    valor_concatenado = np.concatenate([data[0], data[1], data[2]])
    # Asignar el valor concatenado a la variable con el nombre correspondiente
    globals()[name] = valor_concatenado

#Feature con chi^2:
GALAXY_GALAXY= np.array(galaxy_chi_galaxy)
STAR_GALAXY= np.array(star_chi_galaxy)
QSO_GALAXY= np.array(qso_chi_galaxy)
CHI_GALAXY= np.concatenate([GALAXY_GALAXY, STAR_GALAXY, QSO_GALAXY])

GALAXY_QSO= np.array(galaxy_chi_qso)
STAR_QSO= np.array(star_chi_qso)
QSO_QSO= np.array(qso_chi_qso)
CHI_QSO= np.concatenate([GALAXY_QSO, STAR_QSO, QSO_QSO])

GALAXY_STAR= np.array(galaxy_chi_star)
STAR_STAR= np.array(star_chi_star)
QSO_STAR= np.array(qso_chi_star)
CHI_STAR= np.concatenate([GALAXY_STAR, STAR_STAR, QSO_STAR])

#Feature razon entre chi
CHI_GALAXYQSO= CHI_GALAXY/CHI_QSO

CHI_GALAXYSTAR= CHI_GALAXY/CHI_STAR

CHI_QSOSTAR= CHI_QSO/CHI_STAR

#Featre Abbe
ABBE= np.concatenate([abbe_gal, abbe_star, abbe_qso])

#Feature aleatoria
ALEATORIO= aleatorio

STYPE=[]
for i in range(len(FLUX_U_GALAXY)):
    STYPE.append('GALAXY')
for j in range(len(FLUX_U_STAR)):
    STYPE.append('STAR')
for k in range(len(FLUX_U_QSO)):
    STYPE.append('QSO')
STYPE= np.array(STYPE)


# In[38]:


#Creamos el H2O frame con una columna
datos = h2o.H2OFrame(python_obj=STYPE, column_names=['SPECTYPE'], column_types=["string"])

#Agregamos mas columnas al frame que hemos creado
flux_u_h2o= h2o.H2OFrame(python_obj=FLUX_U, column_names=['FLUX_U'], column_types=["float"])
datos= datos.cbind(flux_u_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_g_h2o= h2o.H2OFrame(python_obj=FLUX_G, column_names=['FLUX_G'], column_types=["float"])
datos= datos.cbind(flux_g_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_r_h2o= h2o.H2OFrame(python_obj=FLUX_R, column_names=['FLUX_R'], column_types=["float"])
datos= datos.cbind(flux_r_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_i_h2o= h2o.H2OFrame(python_obj=FLUX_I, column_names=['FLUX_I'], column_types=["float"])
datos= datos.cbind(flux_i_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_z_h2o= h2o.H2OFrame(python_obj=FLUX_Z, column_names=['FLUX_Z'], column_types=["float"])
datos= datos.cbind(flux_z_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_r_z_h2o= h2o.H2OFrame(python_obj=R_Z, column_names=['R-Z'], column_types=["float"])
datos= datos.cbind(flux_r_z_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_u_r_h2o= h2o.H2OFrame(python_obj=U_R, column_names=['U-R'], column_types=["float"])
datos= datos.cbind(flux_u_r_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_g_z_h2o= h2o.H2OFrame(python_obj=G_Z, column_names=['G-Z'], column_types=["float"])
datos= datos.cbind(flux_g_z_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_i_z_h2o= h2o.H2OFrame(python_obj=I_Z, column_names=['I-Z'], column_types=["float"])
datos= datos.cbind(flux_i_z_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_u_g_h2o= h2o.H2OFrame(python_obj=U_G, column_names=['U-G'], column_types=["float"])
datos= datos.cbind(flux_u_g_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_g_z_r_z_h2o= h2o.H2OFrame(python_obj=G_Z_R_Z, column_names=['(G-Z)/(R-Z)'], column_types=["float"])
datos= datos.cbind(flux_g_z_r_z_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_ur_h2o= h2o.H2OFrame(python_obj=RU, column_names=['R/U'], column_types=["float"])
datos= datos.cbind(flux_ur_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_gz_h2o= h2o.H2OFrame(python_obj=GZ, column_names=['G/Z'], column_types=["float"])
datos= datos.cbind(flux_gz_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_rz_h2o= h2o.H2OFrame(python_obj=RZ, column_names=['R/Z'], column_types=["float"])
datos= datos.cbind(flux_rz_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_iz_h2o= h2o.H2OFrame(python_obj=IZ, column_names=['I/Z'], column_types=["float"])
datos= datos.cbind(flux_iz_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_ug_h2o= h2o.H2OFrame(python_obj=UG, column_names=['U/G'], column_types=["float"])
datos= datos.cbind(flux_ug_h2o)

#Agregamos mas columnas al frame que hemos creado
flux_uz_h2o= h2o.H2OFrame(python_obj=UZ, column_names=['U/Z'], column_types=["float"])
datos= datos.cbind(flux_uz_h2o)

#Agregamos mas columnas al frame que hemos creado
curv_h2o= h2o.H2OFrame(python_obj=CURV, column_names=['CURV'], column_types=["float"])
datos= datos.cbind(curv_h2o)

#Agregamos mas columnas al frame que hemos creado
mad_h2o= h2o.H2OFrame(python_obj=MAD, column_names=['MAD'], column_types=["float"])
datos= datos.cbind(mad_h2o)

#Agregamos mas columnas al frame que hemos creado
f0_h2o= h2o.H2OFrame(python_obj=F0, column_names=['F0'], column_types=["float"])
datos= datos.cbind(f0_h2o)

#Agregamos mas columnas al frame que hemos creado
f1_h2o= h2o.H2OFrame(python_obj=F1, column_names=['F1'], column_types=["float"])
datos= datos.cbind(f1_h2o)

#Agregamos mas columnas al frame que hemos creado
f2_h2o= h2o.H2OFrame(python_obj=F2, column_names=['F2'], column_types=["float"])
datos= datos.cbind(f2_h2o)

#Agregamos mas columnas al frame que hemos creado
f3_h2o= h2o.H2OFrame(python_obj=F3, column_names=['F3'], column_types=["float"])
datos= datos.cbind(f3_h2o)

#Agregamos mas columnas al frame que hemos creado
_0z_h2o= h2o.H2OFrame(python_obj=_0Z, column_names=['F0/Z'], column_types=["float"])
datos= datos.cbind(_0z_h2o)

#Agregamos mas columnas al frame que hemos creado
_0g_h2o= h2o.H2OFrame(python_obj=_0G, column_names=['F0/G'], column_types=["float"])
datos= datos.cbind(_0g_h2o)

#Agregamos mas columnas al frame que hemos creado
_0r_h2o= h2o.H2OFrame(python_obj=_0R, column_names=['F0/R'], column_types=["float"])
datos= datos.cbind(_0r_h2o)

#Agregamos mas columnas al frame que hemos creado
_0i_h2o= h2o.H2OFrame(python_obj=_0I, column_names=['F0/I'], column_types=["float"])
datos= datos.cbind(_0i_h2o)

#Agregamos mas columnas al frame que hemos creado
_01_h2o= h2o.H2OFrame(python_obj=_01, column_names=['F0/F1'], column_types=["float"])
datos= datos.cbind(_01_h2o)

#Agregamos mas columnas al frame que hemos creado
_02_h2o= h2o.H2OFrame(python_obj=_02, column_names=['F0/F2'], column_types=["float"])
datos= datos.cbind(_02_h2o)

#Agregamos mas columnas al frame que hemos creado
_03_h2o= h2o.H2OFrame(python_obj=_03, column_names=['F0/F3'], column_types=["float"])
datos= datos.cbind(_03_h2o)

#Agregamos mas columnas al frame que hemos creado
_1U_h2o= h2o.H2OFrame(python_obj=_1U, column_names=['F1/U'], column_types=["float"])
datos= datos.cbind(_1U_h2o)

#Agregamos mas columnas al frame que hemos creado
_1Z_h2o= h2o.H2OFrame(python_obj=_1Z, column_names=['F1/Z'], column_types=["float"])
datos= datos.cbind(_1Z_h2o)

#Agregamos mas columnas al frame que hemos creado
_1R_h2o= h2o.H2OFrame(python_obj=_1R, column_names=['F1/R'], column_types=["float"])
datos= datos.cbind(_1R_h2o)

#Agregamos mas columnas al frame que hemos creado
_1i_h2o= h2o.H2OFrame(python_obj=_1I, column_names=['F1/I'], column_types=["float"])
datos= datos.cbind(_1i_h2o)

#Agregamos mas columnas al frame que hemos creado
_12_h2o= h2o.H2OFrame(python_obj=_12, column_names=['F1/F2'], column_types=["float"])
datos= datos.cbind(_12_h2o)

#Agregamos mas columnas al frame que hemos creado
_2z_h2o= h2o.H2OFrame(python_obj=_2Z, column_names=['F2/Z'], column_types=["float"])
datos= datos.cbind(_2z_h2o)

#Agregamos mas columnas al frame que hemos creado
_2r_h2o= h2o.H2OFrame(python_obj=_2R, column_names=['F2/R'], column_types=["float"])
datos= datos.cbind(_2r_h2o)

#Agregamos mas columnas al frame que hemos creado
_2i_h2o= h2o.H2OFrame(python_obj=_2I, column_names=['F2/I'], column_types=["float"])
datos= datos.cbind(_2i_h2o)

#Agregamos mas columnas al frame que hemos creado
_23_h2o= h2o.H2OFrame(python_obj=_23, column_names=['F2/F3'], column_types=["float"])
datos= datos.cbind(_23_h2o)

#Agregamos mas columnas al frame que hemos creado
_3u_h2o= h2o.H2OFrame(python_obj=_3U, column_names=['F3/U'], column_types=["float"])
datos= datos.cbind(_3u_h2o)

#Agregamos mas columnas al frame que hemos creado
_3z_h2o= h2o.H2OFrame(python_obj=_3Z, column_names=['F3/Z'], column_types=["float"])
datos= datos.cbind(_3z_h2o)

#Agregamos mas columnas al frame que hemos creado
_3r_h2o= h2o.H2OFrame(python_obj=_3R, column_names=['F3/R'], column_types=["float"])
datos= datos.cbind(_3r_h2o)

#Agregamos mas columnas al frame que hemos creado
_3i_h2o= h2o.H2OFrame(python_obj=_3I, column_names=['F3/I'], column_types=["float"])
datos= datos.cbind(_3i_h2o)

#Agregamos mas columnas al frame que hemos creado
t_ur_h2o= h2o.H2OFrame(python_obj=T_UR, column_names=['-2.5LOG(U/R)'], column_types=["float"])
datos= datos.cbind(t_ur_h2o)

#Agregamos mas columnas al frame que hemos creado
t_ug_h2o= h2o.H2OFrame(python_obj=T_UG, column_names=['-2.5LOG(U/G)'], column_types=["float"])
datos= datos.cbind(t_ug_h2o)

#Agregamos mas columnas al frame que hemos creado
t_gz_h2o= h2o.H2OFrame(python_obj=T_GZ, column_names=['-2.5LOG(G/Z)'], column_types=["float"])
datos= datos.cbind(t_gz_h2o)

#Agregamos mas columnas al frame que hemos creado
t_rz_h2o= h2o.H2OFrame(python_obj=T_RZ, column_names=['-2.5LOG(R/Z)'], column_types=["float"])
datos= datos.cbind(t_rz_h2o)

#Agregamos mas columnas al frame que hemos creado
t_iz_h2o= h2o.H2OFrame(python_obj=T_IZ, column_names=['-2.5LOG(I/Z)'], column_types=["float"])
datos= datos.cbind(t_iz_h2o)

#Agregamos mas columnas al frame que hemos creado
chi_gal_h2o= h2o.H2OFrame(python_obj=CHI_GALAXY, column_names=['CHI^2_GALAXY'], column_types=["float"])
datos= datos.cbind(chi_gal_h2o)

#Agregamos mas columnas al frame que hemos creado
chi_star_h2o= h2o.H2OFrame(python_obj=CHI_STAR, column_names=['CHI^2_STAR'], column_types=["float"])
datos= datos.cbind(chi_star_h2o)

#Agregamos mas columnas al frame que hemos creado
chi_qso_h2o= h2o.H2OFrame(python_obj=CHI_QSO, column_names=['CHI^2_QSO'], column_types=["float"])
datos= datos.cbind(chi_qso_h2o)

#Agregamos mas columnas al frame que hemos creado
chi_galqso_h2o= h2o.H2OFrame(python_obj=CHI_GALAXYQSO, column_names=['CHI^2_GALAXY/CHI^2_QSO'], column_types=["float"])
datos= datos.cbind(chi_galqso_h2o)

#Agregamos mas columnas al frame que hemos creado
chi_galstar_h2o= h2o.H2OFrame(python_obj=CHI_GALAXYSTAR, column_names=['CHI^2_GALAXY/CHI^2_STAR'], column_types=["float"])
datos= datos.cbind(chi_galstar_h2o)

#Agregamos mas columnas al frame que hemos creado
chi_qsostar_h2o= h2o.H2OFrame(python_obj=CHI_QSOSTAR, column_names=['CHI^2_QSO/CHI^2_STAR'], column_types=["float"])
datos= datos.cbind(chi_qsostar_h2o)

#Agregamos mas columnas al frame que hemos creado
abbe_h2o= h2o.H2OFrame(python_obj=ABBE, column_names=['ABBE'], column_types=["float"])
datos= datos.cbind(abbe_h2o)

#Agregamos mas columnas al frame que hemos creado
ale_h2o= h2o.H2OFrame(python_obj=ALEATORIO, column_names=['RANDOM'], column_types=["float"])
datos= datos.cbind(ale_h2o)


# In[ ]:


datos


# In[ ]:


# Convierte la variable objetivo a factor utilizando la función asfactor()
datos['SPECTYPE'] = datos['SPECTYPE'].asfactor()

# Define las columnas predictoras y la variable objetivo
predictores = datos.columns[1:]  # Todas las columnas excepto la primera (SPECTYPE)
objetivo='SPECTYPE'


# In[ ]:


# Dividir el conjunto de datos en entrenamiento y prueba de manera estratificada
#train, test = datos.split_frame(ratios=[0.6], seed=42, destination_frames=['train', 'test'], stratify='SPECTYPE')
train, test = datos.split_frame(ratios=[0.6], seed=42)


# In[ ]:


train['SPECTYPE'].table()


# In[ ]:


test['SPECTYPE'].table()


# ### Red neuronal

# In[ ]:


# Configura y entrena el modelo de red neuronal
modelo_nn = H2ODeepLearningEstimator(epochs=1000,
                                     hidden=[64, 128, 256, 256],
                                     distribution="multinomial",
                                     activation="RectifierWithDropout",
                                     variable_importances=True)
#modelo_nn = H2ODeepLearningEstimator(epochs=10, hidden=[10, 10], distribution="multinomial", activation="RectifierWithDropout", variable_importances=True)
modelo_nn.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprime métricas de rendimiento en el conjunto de prueba
print(modelo_nn.model_performance(test_data=test))

# Obtener las importancias de las variables
importancias_variables = modelo_nn.varimp()
# Imprimir las importancias de las variables
print(importancias_variables)


#Random forest, gradient boosting


# In[ ]:


# Graficar el Loss en cada época de entrenamiento y validación
training_metrics = modelo_nn.score_history()
plt.plot(training_metrics['epochs'], training_metrics['training_classification_error'], label='Training Error')
plt.plot(training_metrics['epochs'], training_metrics['validation_classification_error'], label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()


# ### Random forest

# In[ ]:


# Configura y entrena el modelo de Random Forest
modelo_rf = H2ORandomForestEstimator(ntrees=200, max_depth=20, seed=42)
modelo_rf.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprime métricas de rendimiento en el conjunto de prueba
print(modelo_rf.model_performance(test_data=test))

# Obtener las importancias de las variables
importancias_variables_rf = modelo_rf.varimp(True)
print("\nImportancias relativas de las características de entrenamiento:")
print(importancias_variables_rf)


# In[ ]:


# Graficar las métricas de rendimiento en cada árbol
rf_metrics = modelo_rf.score_history()
plt.plot(rf_metrics['number_of_trees'], rf_metrics['training_classification_error'], label='Training Error')
plt.plot(rf_metrics['number_of_trees'], rf_metrics['validation_classification_error'], label='Validation Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.legend()
plt.show()


# ### Mejor familia

# In[ ]:


# Configurar y ejecutar la búsqueda automática de modelos
exclude_algos = ["DeepLearning", "StackedEnsemble"]  # Excluir redes neuronales y ensambles
automl = H2OAutoML(max_models=10, seed=42, exclude_algos=exclude_algos)
automl.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprimir métricas de rendimiento en el conjunto de prueba para el mejor modelo
print(automl.leader.model_performance(test_data=test))


# In[ ]:


# Obtiene y muestra el modelo líder
best_model = automl.leader
print("Mejor modelo:")
print(best_model)

# Imprime métricas de rendimiento en el conjunto de prueba para el mejor modelo
print("Métricas del mejor modelo en el conjunto de prueba:")
print(best_model.model_performance(test_data=test))


# In[ ]:


# Obtener y mostrar el modelo líder
best_model = automl.leader
print("Mejor modelo:")
print(best_model)

# Imprimir métricas de rendimiento en el conjunto de prueba para el mejor modelo
print("Métricas del mejor modelo en el conjunto de prueba:")
print(best_model.model_performance(test_data=test))

# Obtener la tabla completa de importancias relativas de las características de entrenamiento
importancias_variables = best_model.varimp(True)

# Mostrar la tabla completa de importancias relativas de las características de entrenamiento
print("\nImportancias relativas de las características de entrenamiento:")
print(importancias_variables)


# ### GBM

# In[ ]:


# Importar la clase LightGBM de H2O
from h2o.estimators import H2OGradientBoostingEstimator

# Configurar y entrenar el modelo LightGBM
modelo_lgbm = H2OGradientBoostingEstimator(ntrees=300, max_depth=10, seed=42)
modelo_lgbm.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprimir métricas de rendimiento en el conjunto de prueba
print(modelo_lgbm.model_performance(test_data=test))

# Obtener las importancias de las variables
importancias_variables_lgbm = modelo_lgbm.varimp(True)
# Imprimir las importancias de las variables
print(importancias_variables_lgbm)


# In[ ]:


# Obtener las métricas de rendimiento del modelo
lgbm_metrics = modelo_lgbm.score_history()

# Graficar las métricas de rendimiento en cada iteración
plt.plot(lgbm_metrics['number_of_trees'], lgbm_metrics['training_classification_error'], label='Training Error')
plt.plot(lgbm_metrics['number_of_trees'], lgbm_metrics['validation_classification_error'], label='Validation Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.legend()
plt.title('LightGBM Performance Metrics')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Cierra el cluster de H2O
h2o.cluster().shutdown()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paso 1: Abrir el archivo CSV
df = pd.read_csv('matriz_correlacion.csv')

# Paso 2: Graficar la matriz de correlación
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Correlation matrix')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5,  fmt=".2f")
plt.title('Correlation matrix')
plt.savefig('Correlation matrix')
plt.show()



# In[ ]:




