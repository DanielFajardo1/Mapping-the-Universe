#!/usr/bin/env python
# coding: utf-8

# In[2]:


print('programa iniciado: Catalogo')


# In[1]:


import sklearn
import tensorflow
import astropy

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

from keras.layers import Dropout
from keras.layers import BatchNormalization

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import chisquare
from scipy.interpolate import interp1d

from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import class_weight

import pickle
import joblib

from tensorflow.keras.models import load_model


# In[4]:


print('modulos cargados')


# In[5]:


### We will attempt to do it automatically for all standardized spectra


# In[ ]:





# In[6]:


wave= fits.open('B_R_Z_wavelenght.fits')
Bwave = wave[1].data
Rwave = wave[2].data
Zwave = wave[3].data
wavelenght = np.hstack((Bwave, Rwave, Zwave)) #Contiene la cadena completa de longitudes de onda B+Z+R para cada espectro


# In[ ]:





# In[5]:


archivos= ['DataDESI_76.fits', 'DataDESI_152.fits', 'DataDESI_213.fits', 'DataDESI_284.fits', 'DataDESI_351.fits',
           'DataDESI_438.fits', 'DataDESI_530.fits', 'DataDESI_606.fits', 'DataDESI_690.fits', 'DataDESI_752.fits']
#archivos= ['DataDESI_752.fits']

#Generamos las listas con los datos:
spectra = None #Este tensor contiene los elementos de flujo completo R+Z+B
Spectra_set = None

spectype = np.array([])  # Esta lista contiene las etiquetas para el ejercicio de clasificación
z = np.array([]) #Esta matriz contiene los corrimientos z para el ejercicio de regresion

for h in range(len(archivos)):
    espc = fits.open(archivos[h]) #open file
    len_espc= len(espc[2].data)
    
    #leemos la informacion
    Bflux= espc[2].data
    Rflux= espc[3].data
    Zflux= espc[4].data
    
    spectrum = np.hstack((Bflux, Rflux, Zflux)) #Contiene la cadena completa de flujo B+Z+R para cada espectro
    spectrum_cnn = spectrum.reshape(spectrum.shape[0], spectrum.shape[1], 1)

    if spectra is None:
        spectra = spectrum
        Spectra_set = spectrum_cnn
    else:
        spectra = np.concatenate((spectra, spectrum), axis=0)
        Spectra_set = np.concatenate((Spectra_set, spectrum_cnn), axis=0)

    # Obtener la clase espectral y corrimiento para cada espectro
    clases_espectrales = Table.read(espc, hdu=1)['SPECTYPE'].data
    corrimiento = Table.read(espc, hdu=1)['Z'].data
    
    spectype = np.append(spectype,clases_espectrales)
    z = np.append(z, corrimiento)
    z = z.reshape(-1,1)

    
# Tenemos el tensor spectra que contiene todos los flujos de los .fits seleccionados
# spectype es una lista con las etiquetas de dichos espectros
# z una matriz con los valores de corrimiento de cada espectro.


# In[ ]:





# In[8]:


indices = [index for index, value in enumerate(spectype) if value == 'GALAXY']
spectra_galaxy = np.array([spectra[index] for index in indices])


# In[9]:


indices = [index for index, value in enumerate(spectype) if value == 'QSO']
spectra_qso = np.array([spectra[index] for index in indices])


# In[10]:


indices = [index for index, value in enumerate(spectype) if value == 'STAR']
spectra_star = np.array([spectra[index] for index in indices])


# In[11]:

#Spectra_set= Spectra_set[:1000]
#spectra= spectra[:1000]
#spectype= spectype[:1000]


# In[12]:


print('spectra y etiquetas creados')


# In[13]:


### We calculate the features


# In[14]:


#flujos dentro de las bandas ugriz y sinteticas
FLUX_U=np.array([])
FLUX_G=np.array([])
FLUX_R=np.array([])
FLUX_I=np.array([])
FLUX_Z=np.array([])
F0=np.array([])
F1=np.array([])
F2=np.array([])
F3=np.array([])

mascara_U = (wavelenght >= 3055.11) & (wavelenght <= 4030.64)
mascara_G = (wavelenght >= 3797.64) & (wavelenght <= 5553.04)
mascara_R = (wavelenght >= 5418.23) & (wavelenght <= 6994.42)
mascara_I = (wavelenght >= 6692.41) & (wavelenght <= 8400.32)
mascara_Z = (wavelenght >= 8385) & (wavelenght <= 9875)
mascara_F0 = (wavelenght >= 3055.11) & (wavelenght <= 3756)
mascara_F1 = (wavelenght >= 5578) & (wavelenght <= 5582)
mascara_F2 = (wavelenght >= 7601) & (wavelenght <= 7603)
mascara_F3 = (wavelenght >= 9310) & (wavelenght <= 9570)

wavelenght_filtrado_U= wavelenght[mascara_U]
wavelenght_filtrado_G= wavelenght[mascara_G]
wavelenght_filtrado_R= wavelenght[mascara_R]
wavelenght_filtrado_I= wavelenght[mascara_I]
wavelenght_filtrado_Z= wavelenght[mascara_Z]
wavelenght_filtrado_F0= wavelenght[mascara_F0]
wavelenght_filtrado_F1= wavelenght[mascara_F1]
wavelenght_filtrado_F2= wavelenght[mascara_F2]
wavelenght_filtrado_F3= wavelenght[mascara_F3]

#Curvatura
CURV=np.array([])

def func(x, a, b, c):
    return a * x**2 + b * x + c

#MAD
MAD=np.array([])

#Chi cuadrado
CHI_GALAXY=np.array([])
CHI_QSO=np.array([])
CHI_STAR=np.array([])

spectra_galaxy_mean = np.mean(spectra_galaxy, axis=0)
spectra_qso_mean = np.mean(spectra_qso, axis=0)
spectra_star_mean = np.mean(spectra_star, axis=0)

#ABBE
ABBE=np.array([])

#RANDOM
ALEATORIO=np.array([])
random.seed(42)

#Recorremos cada espectro y calculamos las features que iran dentro de las listas de arriba
for spectrum in spectra:
    #Flujos
    espectro_filtrado_U= spectrum[mascara_U]
    espectro_filtrado_G= spectrum[mascara_G]
    espectro_filtrado_R= spectrum[mascara_R]
    espectro_filtrado_I= spectrum[mascara_I]
    espectro_filtrado_Z= spectrum[mascara_Z]
    espectro_filtrado_F0= spectrum[mascara_F0]
    espectro_filtrado_F1= spectrum[mascara_F1]
    espectro_filtrado_F2= spectrum[mascara_F2]
    espectro_filtrado_F3= spectrum[mascara_F3]
    
    area_U= trapz(espectro_filtrado_U, wavelenght_filtrado_U)
    area_G= trapz(espectro_filtrado_G, wavelenght_filtrado_G)
    area_R= trapz(espectro_filtrado_R, wavelenght_filtrado_R)
    area_I= trapz(espectro_filtrado_I, wavelenght_filtrado_I)
    area_Z= trapz(espectro_filtrado_Z, wavelenght_filtrado_Z)
    area_F0= trapz(espectro_filtrado_F0, wavelenght_filtrado_F0)
    area_F1= trapz(espectro_filtrado_F1, wavelenght_filtrado_F1)
    area_F2= trapz(espectro_filtrado_F2, wavelenght_filtrado_F2)
    area_F3= trapz(espectro_filtrado_F3, wavelenght_filtrado_F3)
    
    FLUX_U= np.append(FLUX_U, area_U)
    FLUX_G= np.append(FLUX_G, area_G)
    FLUX_R= np.append(FLUX_R, area_R)
    FLUX_I= np.append(FLUX_I, area_I)
    FLUX_Z= np.append(FLUX_Z, area_Z)
    F0= np.append(F0, area_F0)
    F1= np.append(F1, area_F1)
    F2= np.append(F2, area_F2)
    F3= np.append(F3, area_F3)

    #Curvatura
    try:
        # Realiza el ajuste de curva a los datos suavizados
        popt, _ = curve_fit(func, wavelenght, spectrum)
        # Obtiene los parámetros ajustados
        a, b, c = popt
        # Calcula la curvatura (segunda derivada)
        curvature = 2 * a
        # Agrega la curvatura al listado
        CURV= np.append(CURV, curvature)
    except RuntimeError:
        # Si no se puede calcular la curvatura, agrega 0
        CURV= np.append(CURV, 0)

    #MAD
    mad = np.median(np.abs(spectrum - np.median(spectrum)))
    MAD= np.append(MAD, mad)

    #Chi cuadrado con plantillas
    chi_qso =  np.sum(((spectrum - spectra_qso_mean)**2)/spectra_qso_mean) 
    chi_galaxy = np.sum(((spectrum - spectra_galaxy_mean)**2)/spectra_galaxy_mean) 
    chi_star = np.sum(((spectrum - spectra_star_mean)**2)/spectra_star_mean)

    CHI_GALAXY= np.append(CHI_GALAXY, chi_galaxy)
    CHI_QSO= np.append(CHI_QSO, chi_qso)
    CHI_STAR= np.append(CHI_STAR, chi_star)

    #ABBE
    parte_1 = (len(spectrum)/(2*(len(spectrum)-1))) * (1/(np.sum((spectrum-np.mean(spectrum))**2)))
    parte_2 = 0
    for i in range(len(spectrum)-1):
        parte_2 += (spectrum[i+1]-spectrum[i])**2
    abbe = parte_1 * parte_2
    ABBE= np.append(ABBE, abbe)

    #RANDOM
    ALEATORIO= np.append(ALEATORIO, random.random())


#Calculamos los indices de color

U_R = FLUX_U-FLUX_R 
U_G = FLUX_U-FLUX_G
G_Z = FLUX_G-FLUX_Z
R_Z = FLUX_R-FLUX_Z
I_Z = FLUX_I-FLUX_Z

#Calculamos las razones entre los flujos ugriz y sinteticos

RU= FLUX_R/FLUX_U
GZ= FLUX_G/FLUX_Z
RZ= FLUX_R/FLUX_Z
IZ= FLUX_I/FLUX_Z
UG= FLUX_U/FLUX_G
UZ= FLUX_U/FLUX_Z

_0Z= F0/FLUX_Z
_0G= F0/FLUX_G
_0R= F0/FLUX_R
_0I= F0/FLUX_I
_01= F0/F1
_02= F0/F2
_03= F0/F3
_1U= F1/FLUX_U
_1Z= F1/FLUX_Z
_1R= F1/FLUX_R
_1I= F1/FLUX_I
_12= F1/F2
_2Z= F2/FLUX_Z
_2R= F2/FLUX_R
_2I= F2/FLUX_I
_23= F2/F3
_3U= F3/FLUX_U
_3Z= F3/FLUX_Z
_3R= F3/FLUX_R
_3I= F3/FLUX_I

G_Z_R_Z= (G_Z)/(R_Z)

#Calculamos la razon entre los valores de chi cuadrado
CHI_GALAXYQSO= CHI_GALAXY/CHI_QSO
CHI_GALAXYSTAR= CHI_GALAXY/CHI_STAR
CHI_QSOSTAR= CHI_QSO/CHI_STAR

#Temperatura de color
T_UR= -2.5*np.log10(np.abs(FLUX_U/FLUX_R))
T_UG= -2.5*np.log10(np.abs(FLUX_U/FLUX_G))
T_GZ= -2.5*np.log10(np.abs(FLUX_G/FLUX_Z))
T_RZ= -2.5*np.log10(np.abs(FLUX_R/FLUX_Z))
T_IZ= -2.5*np.log10(np.abs(FLUX_I/FLUX_Z))


# In[15]:


print('features calculadas')


# In[16]:


### Random Forest


# In[17]:


#Haremos que X sea una matriz que contiene todas las features. En donde cada fila corresponde a un objeto distinto,
#y cada columna contiene una #features distinta


# In[18]:


# Definir los datos de características (features) y etiquetas
X = np.array([FLUX_U, FLUX_G, FLUX_R, FLUX_I, FLUX_Z, F0, F1, F2, F3, CURV, MAD, CHI_GALAXY, CHI_QSO, CHI_STAR, ABBE, U_R, U_G, 
              G_Z, R_Z, I_Z, RU, GZ, RZ, IZ, UG, UZ, _0Z, _0G, _0R, _0I, _01, _02, _03, _1U, _1Z, _1R, _1I, _12, _2Z, _2R, _2I,
              _23, _3U, _3Z, _3R, _3I, G_Z_R_Z, CHI_GALAXYQSO, CHI_GALAXYSTAR, CHI_QSOSTAR, T_UR, T_UG, T_GZ, T_RZ, T_IZ, ALEATORIO])
X = X.T
y = spectype

# Ahora con la CNN ----------------------
label_encoder = LabelEncoder()
y_cnn = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_cnn)


print('listo para llamar modelos')

# Cargar los modelos desde el archivo .pkl y .h5
model_RF = joblib.load('best_random_forest_model.pkl')
model_lightGBM = joblib.load('best_lightGBM_model.pkl')
model_SAM = load_model("best_SAM_model.h5")

print('modelos importados')

# Realizar predicciones
predictions_RF = model_RF.predict(X)
predictions_lightGBM = model_lightGBM.predict(X)

print('prediciones de RF y GBM hechas')

predictions_SAM_0 = model_SAM.predict(Spectra_set)    #GALAXY=0, QSO=1, STAR=2
predictions_SAM_1 = np.argmax(predictions_SAM_0, axis=1)
predictions_SAM = label_encoder.inverse_transform(predictions_SAM_1)


print('termine predicciones')

# Convertir las predicciones a valores booleanos
predictions_RF_bool_star = (predictions_RF == 'STAR').astype(int)
predictions_lightGBM_bool_star = (predictions_lightGBM == 'STAR').astype(int)
predictions_SAM_bool_star = (predictions_SAM == 'STAR').astype(int)

predictions_RF_bool_gal = (predictions_RF == 'GALAXY').astype(int)
predictions_lightGBM_bool_gal = (predictions_lightGBM == 'GALAXY').astype(int)
predictions_SAM_bool_gal = (predictions_SAM == 'GALAXY').astype(int)

predictions_RF_bool_qso = (predictions_RF == 'QSO').astype(int)
predictions_lightGBM_bool_qso = (predictions_lightGBM == 'QSO').astype(int)
predictions_SAM_bool_qso = (predictions_SAM == 'QSO').astype(int)

# Sumar las predicciones booleanas

sum_predictions_star = predictions_RF_bool_star + predictions_lightGBM_bool_star + predictions_SAM_bool_star
star_indices_at_least_two = np.where(sum_predictions_star >= 2)[0]

sum_predictions_gal = predictions_RF_bool_gal + predictions_lightGBM_bool_gal + predictions_SAM_bool_gal
gal_indices_at_least_two = np.where(sum_predictions_gal >= 2)[0]

sum_predictions_qso = predictions_RF_bool_qso + predictions_lightGBM_bool_qso + predictions_SAM_bool_qso
qso_indices_at_least_two = np.where(sum_predictions_qso >= 2)[0]

# Crear conjuntos de índices
all_indices = set(range(len(predictions_RF)))
star_indices_set = set(star_indices_at_least_two)
gal_indices_set = set(gal_indices_at_least_two)
qso_indices_set = set(qso_indices_at_least_two)

# Encontrar los índices que faltan en todas las listas y convertirlos a una lista
indices_empty = np.array(list((all_indices - (star_indices_set | gal_indices_set | qso_indices_set))))



# Crear un arreglo de numpy lleno con 'REV'
new_catalogue = np.full(len(predictions_RF), 'EXAMINE')

# Asignar 'STAR' a los índices de star_indices_at_least_two
new_catalogue[star_indices_at_least_two] = 'STAR'

# Asignar 'GALAXY' a los índices de gal_indices_at_least_two
new_catalogue[gal_indices_at_least_two] = 'GALAXY'

# Asignar 'QSO' a los índices de qso_indices_at_least_two
new_catalogue[qso_indices_at_least_two] = 'QSO'


print(new_catalogue)



# Encontrar los índices donde spectype es 'GALAXY'
galaxy_indices_spectype = np.where(spectype == 'GALAXY')[0]
star_indices_spectype = np.where(spectype == 'STAR')[0]
qso_indices_spectype = np.where(spectype == 'QSO')[0]

# Contador para contar cuántas veces coinciden los valores en new_catalogue con spectype cuando spectype es 'GALAXY' 'STAR' 'QSO'
coincidencias_galaxy = 0
coincidencias_star = 0
coincidencias_qso = 0

# Contador para contar cuántas veces el valor en new_catalogue difiere de 'GALAXY' 'STAR' 'QSO' cuando spectype es 'GALAXY' 'STAR' 'QSO'
diferencias_galaxy = 0
diferencias_star = 0
diferencias_qso = 0

for index in galaxy_indices_spectype:
    if new_catalogue[index] == 'GALAXY':
        coincidencias_galaxy += 1
    else:
        diferencias_galaxy += 1

for index in star_indices_spectype:
    if new_catalogue[index] == 'STAR':
        coincidencias_star += 1
    else:
        diferencias_star += 1

for index in qso_indices_spectype:
    if new_catalogue[index] == 'QSO':
        coincidencias_qso += 1
    else:
        diferencias_qso += 1

print("galaxy coincidences: ", coincidencias_galaxy)
print("differences galaxies: ", diferencias_galaxy)

print("star coincidences: ", coincidencias_star)
print("differences star: ", diferencias_star)

print("qso coincidences: ", coincidencias_qso)
print("differences qso: ", diferencias_qso)




lens=[]
for h in range(len(archivos)):
    espc = fits.open(archivos[h]) #open file
    len_espc= len(espc[2].data)
    lens.append(len_espc)

lens = np.array(lens)


# Nombre del archivo FITS
nombre_archivo = 'SPECTYPE_SPASS.fits'
# Crear un nuevo archivo FITS
hdulist = fits.HDUList()

c=1
n=0
m=0
for k in lens:
    m += k
    classes = new_catalogue[n:m]
    n += k
    
    # Crear un DataFrame de pandas
    df = pd.DataFrame({'SPECTYPE': classes})
    # Convertir el DataFrame en una tabla FITS
    table = Table.from_pandas(df)
    # Asignar un nombre único a la tabla
    table.meta['EXTNAME'] = 'SPECTYPE_FILE_' + str(c)
    # Crear una nueva HDU con los datos de la tabla FITS
    hdu = fits.table_to_hdu(table)

    # Agregar la HDU a la lista de HDUs
    hdulist.append(hdu)
    
    c += 1
    print("Copiado")

# Guardar el archivo FITS
hdulist.writeto(nombre_archivo, overwrite=True)

print("Archivo FITS creado correctamente:", nombre_archivo)


