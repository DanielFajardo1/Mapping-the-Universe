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


# In[2]:


espc = fits.open('DataDESI_691_752.fits') #open file
wave= fits.open('B_R_Z_wavelenght.fits')
espc.info() #resume el contenido de la tabla
wave.info()


# In[ ]:


Table.read(espc, hdu=1)


# In[ ]:


#Leemos la info del archivo .fits
flux_b=espc[2].data
flux_r=espc[3].data
flux_z=espc[4].data

Bwave = wave[1].data
Rwave = wave[2].data
Zwave = wave[3].data

#Seleccionamos un espectro con indice i (QSO=-43, GALAXY=-3, STAR=-4)
i=-3035

espectro_b=flux_b[i]
espectro_r=flux_r[i]
espectro_z=flux_z[i]

targetid = espc[1].data['TARGETID'][i]
formatted_targetid = str(int(targetid))
print('TARGETID:', formatted_targetid)

spectype= espc[1].data['SPECTYPE'][i]
print('SPECTYPE:', spectype)

espectro=np.hstack((espectro_b, espectro_r, espectro_z))
wavelenght = np.hstack((Bwave, Rwave, Zwave)) #Contiene la cadena completa de longitudes de onda B+Z+R para cada espectro


# In[ ]:


plt.figure()
plt.plot(wavelenght, espectro)
plt.show()


# In[ ]:


#Leemos la info del archivo .fits
flux_b=espc[2].data
flux_r=espc[3].data
flux_z=espc[4].data

Bwave = wave[1].data
Rwave = wave[2].data
Zwave = wave[3].data

#Seleccionamos un espectro con indice i (QSO=-43, GALAXY=-3, STAR=-4)

j=-3035
k=-43

wavelenght = np.hstack((Bwave, Rwave, Zwave)) #Contiene la cadena completa de longitudes de onda B+Z+R para cada espectro

espectro_b=flux_b[j]
espectro_r=flux_r[j]
espectro_z=flux_z[j]
espectro2=np.hstack((espectro_b, espectro_r, espectro_z))

espectro_b=flux_b[k]
espectro_r=flux_r[k]
espectro_z=flux_z[k]
espectro3=np.hstack((espectro_b, espectro_r, espectro_z))

# Crea una figura y subtramas con tres filas
plt.figure(figsize=(8, 9))

# Subtrama 1
plt.subplot(3, 1, 1)
plt.plot(wavelenght, spectra_qso_z, color='black')
plt.title('QSO A ')
plt.xlabel('Wavelength')
plt.ylabel('Flux')

# Subtrama 2
plt.subplot(3, 1, 2)
plt.plot(wavelenght, espectro2, color='black')
plt.title('QSO B')
plt.xlabel('Wavelength')
plt.ylabel('Flux')

# Subtrama 3
plt.subplot(3, 1, 3)
plt.plot(wavelenght, espectro3, color='black')
plt.title('QSO C')
plt.xlabel('Wavelength')
plt.ylabel('Flux')

# Ajusta el espacio entre las subtramas
plt.tight_layout()

# Muestra la figura
plt.show()


# In[3]:


#Leemos la info del archivo .fits
flux_b=espc[2].data
flux_r=espc[3].data
flux_z=espc[4].data

Bwave = wave[1].data
Rwave = wave[2].data
Zwave = wave[3].data

#Seleccionamos un espectro con indice i (QSO=-43, GALAXY=-3, STAR=-4)
i=-3
j=-4
k=-43

espectro_b=flux_b[i]
espectro_r=flux_r[i]
espectro_z=flux_z[i]
espectro1=np.hstack((espectro_b, espectro_r, espectro_z))
wavelenght = np.hstack((Bwave, Rwave, Zwave)) #Contiene la cadena completa de longitudes de onda B+Z+R para cada espectro

espectro_b=flux_b[j]
espectro_r=flux_r[j]
espectro_z=flux_z[j]
espectro2=np.hstack((espectro_b, espectro_r, espectro_z))

espectro_b=flux_b[k]
espectro_r=flux_r[k]
espectro_z=flux_z[k]
espectro3=np.hstack((espectro_b, espectro_r, espectro_z))

# Crea una figura y subtramas con tres filas
plt.figure(figsize=(8, 9))

# Subtrama 1
plt.subplot(3, 1, 1)
plt.plot(wavelenght, espectro1, color='black')
plt.title('GALAXY 39633162703212632')
plt.xlabel('Wavelength')
plt.ylabel('Flux')

# Subtrama 2
plt.subplot(3, 1, 2)
plt.plot(wavelenght, espectro2, color='black')
plt.title('STAR 39633162699015128')
plt.xlabel('Wavelength')
plt.ylabel('Flux')

# Subtrama 3
plt.subplot(3, 1, 3)
plt.plot(wavelenght, espectro3, color='black')
plt.title('QSO 39633162703209832')
plt.xlabel('Wavelength')
plt.ylabel('Flux')

# Ajusta el espacio entre las subtramas
plt.tight_layout()

# Muestra la figura
plt.show()


# In[19]:


#Metodo 1 para calcular la curvatura de un espectro
# Define una función para ajustar (aquí se usa un polinomio de segundo grado como ejemplo)
def func(x, a, b, c):
    return a * x**2 + b * x + c

# Realiza el ajuste de curva a los datos suavizados
popt, _ = curve_fit(func, wavelenght, espectro1)
# Obtiene los parámetros ajustados
a, b, c = popt
# Calcula la curvatura (segunda derivada)
curvature = 2 * a
# Imprime la curvatura
print("Curvatura del espectro 1:", curvature)

# Plotea los datos y la función ajustada
plt.plot(wavelenght, espectro1, 'k-', label='Datos')
plt.plot(wavelenght, func(wavelenght, *popt), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro 1: Galaxia')
plt.show()


# Realiza el ajuste de curva a los datos suavizados
popt, _ = curve_fit(func, wavelenght, espectro2)
# Obtiene los parámetros ajustados
a, b, c = popt
# Calcula la curvatura (segunda derivada)
curvature = 2 * a
# Imprime la curvatura
print("Curvatura del espectro 2:", curvature)

# Plotea los datos y la función ajustada
plt.plot(wavelenght, espectro2, 'k-', label='Datos')
plt.plot(wavelenght, func(wavelenght, *popt), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro 2: estrella')
plt.show()


# Realiza el ajuste de curva a los datos suavizados
popt, _ = curve_fit(func, wavelenght, espectro3)
# Obtiene los parámetros ajustados
a, b, c = popt
# Calcula la curvatura (segunda derivada)
curvature = 2 * a
# Imprime la curvatura
print("Curvatura del espectro 3:", curvature)

# Plotea los datos y la función ajustada
plt.plot(wavelenght, espectro3, 'k-', label='Datos')
plt.plot(wavelenght, func(wavelenght, *popt), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro 3: qso')
plt.show()


# In[21]:


#Metodo 1 para calcular la curvatura de un espectro. Primero suavizar, y al suavizado le calculamos el ajuste
# Crea una instancia de la clase Spectrum
sp = pyspeckit.Spectrum(data=espectro1, xarr=wavelenght, xarrkwargs={'unit':'angstroms'})

# Realiza un suavizado utilizando el método smooth con un ancho de ventana específico
sp.smooth(15)  # Ajusta el valor de beam_radius según tus necesidades

# Grafica el espectro original y el suavizado
plt.figure(figsize=(10, 6))
plt.plot(wavelenght, espectro1, label='Espectro Original')
plt.plot(sp.xarr, sp.data, label='Espectro Suavizado')
plt.xlabel('Longitud de Onda (Angstroms)')
plt.ylabel('Intensidad')
plt.legend()
plt.show()

# Define una función para ajustar (aquí se usa un polinomio de segundo grado como ejemplo)
def func(x, a, b, c):
    return a * x**2 + b * x + c

# Realiza el ajuste de curva a los datos suavizados
popt, _ = curve_fit(func, sp.xarr, sp.data)
# Obtiene los parámetros ajustados
a, b, c = popt
# Calcula la curvatura (segunda derivada)
curvature = 2 * a
# Imprime la curvatura
print("Curvatura del espectro 1:", curvature)
# Plotea los datos y la función ajustada
plt.plot(sp.xarr, sp.data, 'k-', label='suavizado')
plt.plot(wavelenght, func(wavelenght, *popt), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro 1: galaxia')
plt.show()



# Crea una instancia de la clase Spectrum
sp = pyspeckit.Spectrum(data=espectro2, xarr=wavelenght, xarrkwargs={'unit':'angstroms'})

# Realiza un suavizado utilizando el método smooth con un ancho de ventana específico
sp.smooth(15)  # Ajusta el valor de beam_radius según tus necesidades

# Grafica el espectro original y el suavizado
plt.figure(figsize=(10, 6))
plt.plot(wavelenght, espectro2, label='Espectro Original')
plt.plot(sp.xarr, sp.data, label='Espectro Suavizado')
plt.xlabel('Longitud de Onda (Angstroms)')
plt.ylabel('Intensidad')
plt.legend()
plt.show()

# Define una función para ajustar (aquí se usa un polinomio de segundo grado como ejemplo)
def func(x, a, b, c):
    return a * x**2 + b * x + c

# Realiza el ajuste de curva a los datos suavizados
popt, _ = curve_fit(func, sp.xarr, sp.data)
# Obtiene los parámetros ajustados
a, b, c = popt
# Calcula la curvatura (segunda derivada)
curvature = 2 * a
# Imprime la curvatura
print("Curvatura del espectro 2:", curvature)
# Plotea los datos y la función ajustada
plt.plot(sp.xarr, sp.data, 'k-', label='suavizado')
plt.plot(wavelenght, func(wavelenght, *popt), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro 2: estrella')
plt.show()


# Crea una instancia de la clase Spectrum
sp = pyspeckit.Spectrum(data=espectro3, xarr=wavelenght, xarrkwargs={'unit':'angstroms'})

# Realiza un suavizado utilizando el método smooth con un ancho de ventana específico
sp.smooth(25)  # Ajusta el valor de beam_radius según tus necesidades

# Grafica el espectro original y el suavizado
plt.figure(figsize=(10, 6))
plt.plot(wavelenght, espectro3, label='Espectro Original')
plt.plot(sp.xarr, sp.data, label='Espectro Suavizado')
plt.xlabel('Longitud de Onda (Angstroms)')
plt.ylabel('Intensidad')
plt.legend()
plt.show()

# Define una función para ajustar (aquí se usa un polinomio de segundo grado como ejemplo)
def func(x, a, b, c):
    return a * x**2 + b * x + c

# Realiza el ajuste de curva a los datos suavizados
popt, _ = curve_fit(func, sp.xarr, sp.data)
# Obtiene los parámetros ajustados
a, b, c = popt
# Calcula la curvatura (segunda derivada)
curvature = 2 * a
# Imprime la curvatura
print("Curvatura del espectro 3:", curvature)
# Plotea los datos y la función ajustada
plt.plot(sp.xarr, sp.data, 'k-', label='suavizado')
plt.plot(wavelenght, func(wavelenght, *popt), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro 3: qso')
plt.show()


# ### Sumamos los valores de flujo comprendidos en la banda de interes

# In[8]:


# Definir una máscara booleana para las longitudes de onda entre 3055.11 y 4030.64 (Banda u con lambda_eff=3608.04)
mascara = (wavelenght >= 3055.11) & (wavelenght <= 4030.64)

espectro_filtrado= espectro[mascara]
wavelenght_filtrado= wavelenght[mascara]

area= trapz(espectro_filtrado, wavelenght_filtrado)
print("flux_u:", area)


# In[9]:


# Definir una máscara booleana para las longitudes de onda entre 3797.64 y 5553.04 (Banda g con lambda_eff=4671.82)
mascara = (wavelenght >= 3797.64) & (wavelenght <= 5553.04)

espectro_filtrado= espectro[mascara]
wavelenght_filtrado= wavelenght[mascara]

area= trapz(espectro_filtrado, wavelenght_filtrado)
print("flux_g:", area)


# In[10]:


# Definir una máscara booleana para las longitudes de onda entre 5418.23 y 6994.42 (Banda r con lambda_eff=6141.12)
mascara = (wavelenght >= 5418.23) & (wavelenght <= 6994.42)

espectro_filtrado= espectro[mascara]
wavelenght_filtrado= wavelenght[mascara]

area= trapz(espectro_filtrado, wavelenght_filtrado)
print("flux_r:", area)


# In[11]:


# Definir una máscara booleana para las longitudes de onda entre 6692.41 y 8400.32 (Banda i con lambda_eff=7457.89)
mascara = (wavelenght >= 6692.41) & (wavelenght <= 8400.32)

espectro_filtrado= espectro[mascara]
wavelenght_filtrado= wavelenght[mascara]

area= trapz(espectro_filtrado, wavelenght_filtrado)
print("flux_i:", area)


# In[12]:


# Definir una máscara booleana para las longitudes de onda entre 7964.70 y 10873.33 (Banda z con lambda_eff=8922.78)
mascara = (wavelenght >= 8385) & (wavelenght <= 9875)

espectro_filtrado= espectro[mascara]
wavelenght_filtrado= wavelenght[mascara]

area= trapz(espectro_filtrado, wavelenght_filtrado)
print("flux_z:", area)


# ### Intentaremos hacerlo de manera automatica para todos los espectros, estandarizados

# In[5]:


B_mean=[]# en estas listas se guardaran todos los espectros
Z_mean=[]
R_mean=[]
B_var = []  # Lista para guardar las varianzas
Z_var = []
R_var = []
N_espectros= 0 #Este se utiliza para calcular la media de cada pixel
N_espectros_menos_1= 0 #Este se utiliza para calcular la varianza agrupada 

n=0
#archivos= ['DataDESI_1_76.fits', 'DataDESI_77_152.fits', 'DataDESI_153_213.fits', 'DataDESI_214_284.fits', 'DataDESI_285_351.fits', 'DataDESI_352_438.fits'
#             , 'DataDESI_439_530.fits', 'DataDESI_531_606.fits', 'DataDESI_607_690.fits', 'DataDESI_691_752.fits']
archivos= ['DataDESI_285_351.fits']

for h in range(len(archivos)):
    espc = fits.open(archivos[h]) #open file
    len_espc= len(espc[2].data)
    
    #leemos la informacion
    Bflux=espc[2].data
    Rflux=espc[3].data
    Zflux=espc[4].data

    promedios_b = np.mean(Bflux, axis=0)
    promedios_r = np.mean(Rflux, axis=0)
    promedios_z = np.mean(Zflux, axis=0)

    promedios_b_promedio = [arr * (len_espc) for arr in promedios_b] #Estas guardan NhPh
    promedios_r_promedio = [arr * (len_espc) for arr in promedios_r]
    promedios_z_promedio = [arr * (len_espc) for arr in promedios_z]

    N_espectros+= len_espc #Va sumando los espectros para tener en cuenta la muestra total

    B_mean.append(promedios_b_promedio) #Estas guardan N1P1,N2P2,...,N11P11
    Z_mean.append(promedios_z_promedio)
    R_mean.append(promedios_r_promedio)


    varianza_b = np.var(Bflux, axis=0)
    varianza_r = np.var(Rflux, axis=0)
    varianza_z = np.var(Zflux, axis=0)

    varianza_b_varianza = [arr * (len_espc-1) for arr in varianza_b] #Estas guardan (Nh-1)sigmah
    varianza_r_varianza = [arr * (len_espc-1) for arr in varianza_r]
    varianza_z_varianza = [arr * (len_espc-1) for arr in varianza_z]

    N_espectros_menos_1+= len_espc-1

    B_var.append(varianza_b_varianza) #Estas guardan (N1-1)sigma1,(N2-1)sigma2,...,(N11-1)sigma11
    Z_var.append(varianza_z_varianza)
    R_var.append(varianza_r_varianza)
    
    n+=1
    print(n)


ponderacion_B = [sum(valores) for valores in zip(*B_mean)] #Suma N1P1+N2P2+...+N11P11 solo falta dividir entre la cantidad total
ponderacion_Z = [sum(valores) for valores in zip(*Z_mean)]
ponderacion_R = [sum(valores) for valores in zip(*R_mean)]

media_pixel_B = np.array([elemento / N_espectros for elemento in ponderacion_B])#Estas son las listas con los valores promedios
media_pixel_Z = np.array([elemento / N_espectros for elemento in ponderacion_Z])# en cada pixel
media_pixel_R = np.array([elemento / N_espectros for elemento in ponderacion_R])


var_agrupada_B = [sum(valores) for valores in zip(*B_var)] #Suma (N1-1)sigma1+(N2-1)sigma2+...+(N11-1)sigma11
var_agrupada_Z = [sum(valores) for valores in zip(*Z_var)]
var_agrupada_R = [sum(valores) for valores in zip(*R_var)]

var_B = np.array([elemento / N_espectros_menos_1 for elemento in var_agrupada_B])#Estas son las listas con las varianzas
var_Z = np.array([elemento / N_espectros_menos_1 for elemento in var_agrupada_Z])# en cada pixel
var_R = np.array([elemento / N_espectros_menos_1 for elemento in var_agrupada_R])


desv_B = np.sqrt(var_B) #Estas son las listas con las desviaciones en cada pixel
desv_Z = np.sqrt(var_Z)
desv_R = np.sqrt(var_R)


# In[6]:


#archivos= ['DataDESI_1_36.fits', 'DataDESI_37_72.fits', 'DataDESI_73_108.fits', 'DataDESI_109_144.fits', 'DataDESI_145_180.fits', 'DataDESI_181_216.fits'
# , 'DataDESI_217_252.fits', 'DataDESI_253_288.fits', 'DataDESI_289_324.fits', 'DataDESI_325_360.fits', 'DataDESI_361_379.fits']

archivos= ['DataDESI_285_351.fits']

for h in range(len(archivos)):
    espc = fits.open(archivos[h])
    
    B_FLUX_STAN=[]
    Z_FLUX_STAN=[]
    R_FLUX_STAN=[]
    
    for i in range(len(espc[4].data)):
        espc_b=((espc[2].data[i])-media_pixel_B)/desv_B
        espc_z=((espc[4].data[i])-media_pixel_Z)/desv_Z
        espc_r=((espc[3].data[i])-media_pixel_R)/desv_R
    
        B_FLUX_STAN.append(espc_b)
        Z_FLUX_STAN.append(espc_z)
        R_FLUX_STAN.append(espc_r)
        
    
    B_FLUX_STAN=np.array(B_FLUX_STAN, dtype=np.float32)
    Z_FLUX_STAN=np.array(Z_FLUX_STAN, dtype=np.float32)
    R_FLUX_STAN=np.array(R_FLUX_STAN, dtype=np.float32)


# In[7]:


espc = fits.open('DataDESI_285_351.fits') #open file
wave= fits.open('B_R_Z_wavelenght.fits')
espc.info() #resume el contenido de la tabla
wave.info()


# In[8]:


Table= Table.read(espc, hdu=1)

flux_b= B_FLUX_STAN
flux_r= R_FLUX_STAN
flux_z= Z_FLUX_STAN

Bwave = wave[1].data
Rwave = wave[2].data
Zwave = wave[3].data


spectra= np.hstack((flux_b, flux_r, flux_z))
wavelenght = np.hstack((Bwave, Rwave, Zwave)) #Contiene la cadena completa de longitudes de onda B+Z+R para cada espectro


# In[ ]:





# In[9]:


spectra


# In[10]:


mascara_gal = Table['SPECTYPE'] == 'GALAXY'
just_galaxy = Table[mascara_gal]
spectra_galaxy = spectra[mascara_gal]

mascara_qso = Table['SPECTYPE'] == 'QSO'
just_qso = Table[mascara_qso]
spectra_qso = spectra[mascara_qso]

mascara_star = Table['SPECTYPE'] == 'STAR'
just_star = Table[mascara_star]
spectra_star = spectra[mascara_star]


# In[ ]:





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


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[13]:


# Crear una figura con tres subgráficos en una fila
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Primer subgráfico para FLUX_U_GALAXY
axs[0].scatter(FLUX_U_GALAXY, FLUX_U_GALAXY, color='purple')
axs[0].set_xlabel('FLUX_U_GALAXY')
axs[0].set_ylabel('FLUX_U_GALAXY')
axs[0].set_title('Valores de area bajo la curva para galaxias en U')
# Segundo subgráfico para FLUX_U_QSO
axs[1].scatter(FLUX_U_QSO, FLUX_U_QSO, color='purple')
axs[1].set_xlabel('FLUX_U_QSO')
axs[1].set_ylabel('FLUX_U_QSO')
axs[1].set_title('Valores de area bajo la curva para cuásares en U')
# Tercer subgráfico para FLUX_U_STAR
axs[2].scatter(FLUX_U_STAR, FLUX_U_STAR, color='purple')
axs[2].set_xlabel('FLUX_U_STAR')
axs[2].set_ylabel('FLUX_U_STAR')
axs[2].set_title('Valores de area bajo la curva para estrellas en U')
# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar la gráfica
plt.show()

# Crear una figura con tres subgráficos en una fila
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Primer subgráfico para FLUX_V_G_GALAXY
axs[0].scatter(FLUX_V_G_GALAXY, FLUX_V_G_GALAXY, color='green')
axs[0].set_xlabel('FLUX_V_G_GALAXY')
axs[0].set_ylabel('FLUX_V_G_GALAXY')
axs[0].set_title('Valores de area bajo la curva para galaxias en V_G')
# Segundo subgráfico para FLUX_V_G_QSO
axs[1].scatter(FLUX_V_G_QSO, FLUX_V_G_QSO, color='green')
axs[1].set_xlabel('FLUX_V_G_QSO')
axs[1].set_ylabel('FLUX_V_G_QSO')
axs[1].set_title('Valores de area bajo la curva para cuásares en V_G')
# Tercer subgráfico para FLUX_V_G_STAR
axs[2].scatter(FLUX_V_G_STAR, FLUX_V_G_STAR, color='green')
axs[2].set_xlabel('FLUX_V_G_STAR')
axs[2].set_ylabel('FLUX_V_G_STAR')
axs[2].set_title('Valores de area bajo la curva para estrellas en V_G')
# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar la gráfica
plt.show()


# Crear una figura con tres subgráficos en una fila
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Primer subgráfico para FLUX_R_GALAXY
axs[0].scatter(FLUX_R_GALAXY, FLUX_R_GALAXY, color='violet')
axs[0].set_xlabel('FLUX_R_GALAXY')
axs[0].set_ylabel('FLUX_R_GALAXY')
axs[0].set_title('Valores de area bajo la curva para galaxias en R')
# Segundo subgráfico para FLUX_R_QSO
axs[1].scatter(FLUX_R_QSO, FLUX_R_QSO, color='violet')
axs[1].set_xlabel('FLUX_R_QSO')
axs[1].set_ylabel('FLUX_R_QSO')
axs[1].set_title('Valores de area bajo la curva para cuásares en R')
# Tercer subgráfico para FLUX_R_STAR
axs[2].scatter(FLUX_R_STAR, FLUX_R_STAR, color='violet')
axs[2].set_xlabel('FLUX_R_STAR')
axs[2].set_ylabel('FLUX_R_STAR')
axs[2].set_title('Valores de area bajo la curva para estrellas en R')
# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar la gráfica
plt.show()


# Crear una figura con tres subgráficos en una fila
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Primer subgráfico para FLUX_I_GALAXY
axs[0].scatter(FLUX_I_GALAXY, FLUX_I_GALAXY, color='red')
axs[0].set_xlabel('FLUX_I_GALAXY')
axs[0].set_ylabel('FLUX_I_GALAXY')
axs[0].set_title('Valores de area bajo la curva para galaxias en I')
# Segundo subgráfico para FLUX_I_QSO
axs[1].scatter(FLUX_I_QSO, FLUX_I_QSO, color='red')
axs[1].set_xlabel('FLUX_I_QSO')
axs[1].set_ylabel('FLUX_I_QSO')
axs[1].set_title('Valores de area bajo la curva para cuásares en I')
# Tercer subgráfico para FLUX_I_STAR
axs[2].scatter(FLUX_I_STAR, FLUX_I_STAR, color='red')
axs[2].set_xlabel('FLUX_I_STAR')
axs[2].set_ylabel('FLUX_I_STAR')
axs[2].set_title('Valores de area bajo la curva para estrellas en I')
# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar la gráfica
plt.show()


# Crear una figura con tres subgráficos en una fila
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Primer subgráfico para FLUX_z_GALAXY
axs[0].scatter(FLUX_z_GALAXY, FLUX_z_GALAXY, color='darkred')
axs[0].set_xlabel('FLUX_z_GALAXY')
axs[0].set_ylabel('FLUX_z_GALAXY')
axs[0].set_title('Valores de area bajo la curva para galaxias en z')
# Segundo subgráfico para FLUX_z_QSO
axs[1].scatter(FLUX_z_QSO, FLUX_z_QSO, color='darkred')
axs[1].set_xlabel('FLUX_z_QSO')
axs[1].set_ylabel('FLUX_z_QSO')
axs[1].set_title('Valores de area bajo la curva para cuásares en z')
# Tercer subgráfico para FLUX_z_STAR
axs[2].scatter(FLUX_z_STAR, FLUX_z_STAR, color='darkred')
axs[2].set_xlabel('FLUX_z_STAR')
axs[2].set_ylabel('FLUX_z_STAR')
axs[2].set_title('Valores de area bajo la curva para estrellas en z')
# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar la gráfica
plt.show()


# In[14]:


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

    ax.set_xlabel('Clases')
    ax.set_ylabel(f'Valores de FLUX_{["U", "G", "R", "I", "z"][i]}')
    ax.set_title(f'Distribución de valores de FLUX_{["U", "G", "R", "I", "z"][i]}')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# ### Analisis ANOVA

# In[15]:


from scipy.stats import f_oneway

# Lista de etiquetas para las clases
clases = ['GALAXY', 'STAR', 'QSO']

# Iterar sobre cada filtro y realizar ANOVA
for i in range(5):  # 5 filtros: U, V_G, R, I, z
    # Organizar los datos en un formato adecuado para ANOVA
    data_for_anova = []
    for j in range(3):  # 3 clases: GALAXY, STAR, QSO
        data_for_anova.append(flux_values[i][j])

    # Realizar ANOVA
    anova_result = f_oneway(*data_for_anova)

    # Imprimir el resultado del ANOVA para el filtro actual
    print(f"Resultado del ANOVA para FLUX_{['U', 'V_G', 'R', 'I', 'z'][i]}:")
    print(anova_result)

    # Verificar la significancia estadística
    if anova_result.pvalue < 0.05:
        print("La diferencia entre al menos dos grupos es estadísticamente significativa.")
    else:
        print("No hay evidencia suficiente para rechazar la hipótesis nula de igualdad de medias.")

    print("-" * 50)  # Separador entre resultados de diferentes filtros


# In[16]:


from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Lista de etiquetas para las clases
clases = ['GALAXY', 'STAR', 'QSO']

# Iterar sobre cada filtro y realizar ANOVA
for i in range(5):  # 5 filtros: U, V_G, R, I, z
    # Organizar los datos en un formato adecuado para ANOVA
    data_for_anova = []
    for j in range(3):  # 3 clases: GALAXY, STAR, QSO
        data_for_anova.append(flux_values[i][j])

    # Realizar ANOVA
    anova_result = f_oneway(*data_for_anova)

    # Imprimir el resultado del ANOVA para el filtro actual
    print(f"Resultado del ANOVA para FLUX_{['U', 'V_G', 'R', 'I', 'z'][i]}:")
    print(anova_result)

    # Obtener la longitud mínima de los grupos para evitar el error
    min_group_length = min(len(group) for group in data_for_anova)

    # Realizar prueba de Tukey para comparaciones múltiples
    tukey_result = pairwise_tukeyhsd(np.concatenate([group[:min_group_length] for group in data_for_anova]),
                                      np.repeat(clases, min_group_length))

    # Imprimir los resultados de la prueba de Tukey
    print("Resultados de la prueba de Tukey:")
    print(tukey_result)

    # Verificar si los grupos son estadísticamente distintos
    if any(tukey_result.reject):
        print("Al menos un par de grupos es estadísticamente distinto.")
    else:
        print("No hay evidencia suficiente para rechazar la hipótesis nula de igualdad de medias.")

    print("-" * 50)  # Separador entre resultados de diferentes filtros


# ### Analisis de Kruskall-Wallis

# In[17]:


from scipy.stats import kruskal
from scipy.stats import chi2

# Lista de etiquetas para las clases
clases = ['GALAXY', 'STAR', 'QSO']

# Grados de libertad para Kruskal-Wallis
df = len(clases) - 1

# Nivel de significancia
alpha = 0.05

# Valor crítico de la tabla chi cuadrado
crit_value = chi2.ppf(1 - alpha, df)

# Iterar sobre cada filtro y realizar Kruskal-Wallis
for i in range(5):  # 5 filtros: U, V_G, R, I, z
    # Organizar los datos en un formato adecuado para Kruskal-Wallis
    data_for_kruskal = []
    for j in range(3):  # 3 clases: GALAXY, STAR, QSO
        data_for_kruskal.append(flux_values[i][j])

    # Realizar Kruskal-Wallis
    kruskal_result = kruskal(*data_for_kruskal)

    # Imprimir el resultado de Kruskal-Wallis para el filtro actual
    print(f"Resultado de Kruskal-Wallis para FLUX_{['U', 'V_G', 'R', 'I', 'z'][i]}:")
    print(kruskal_result)

    # Imprimir el valor crítico
    print(f"Valor crítico de la tabla chi cuadrado para alpha = {alpha} y df = {df}: {crit_value}")

    # Verificar la significancia estadística
    if kruskal_result.statistic > crit_value:
        print("El valor del estadístico H supera el valor crítico. La diferencia entre al menos dos grupos es estadísticamente significativa.")
    else:
        print("El valor del estadístico H no supera el valor crítico. No hay evidencia suficiente para rechazar la hipótesis nula de igualdad de medias.")

    print("-" * 50)  # Separador entre resultados de diferentes filtros


# ### Analisis de Kruskall-Wallis con valores de la integral de flujo normalizados por filtro

# In[18]:


# Lista para almacenar los datos estandarizados
flux_values_stand = []

for i in flux_values: #abrimos los filtros U,V_G,R,I,z
    Flux= np.concatenate([i[0],i[1],i[2]])
    med= np.mean(Flux)
    desv= np.sqrt(np.var(Flux))
    
    flux_standardized = [(i[0] - med) / desv, (i[1] - med) / desv, (i[2] - med) / desv]
    flux_values_stand.append(flux_standardized)
        


# In[19]:


flux_values_stand


# In[20]:


from scipy.stats import kruskal
from scipy.stats import chi2

# Lista de etiquetas para las clases
clases = ['GALAXY', 'STAR', 'QSO']

# Grados de libertad para Kruskal-Wallis
df = len(clases) - 1

# Nivel de significancia
alpha = 0.05

# Valor crítico de la tabla chi cuadrado
crit_value = chi2.ppf(1 - alpha, df)

# Iterar sobre cada filtro y realizar Kruskal-Wallis
for i in range(5):  # 5 filtros: U, V_G, R, I, z
    # Organizar los datos en un formato adecuado para Kruskal-Wallis
    data_for_kruskal = []
    for j in range(3):  # 3 clases: GALAXY, STAR, QSO
        data_for_kruskal.append(flux_values_stand[i][j])

    # Realizar Kruskal-Wallis
    kruskal_result = kruskal(*data_for_kruskal)

    # Imprimir el resultado de Kruskal-Wallis para el filtro actual
    print(f"Resultado de Kruskal-Wallis para FLUX_{['U', 'V_G', 'R', 'I', 'z'][i]}:")
    print(kruskal_result)

    # Imprimir el valor crítico
    print(f"Valor crítico de la tabla chi cuadrado para alpha = {alpha} y df = {df}: {crit_value}")

    # Verificar la significancia estadística
    if kruskal_result.statistic > crit_value:
        print("El valor del estadístico H supera el valor crítico. La diferencia entre al menos dos grupos es estadísticamente significativa.")
    else:
        print("El valor del estadístico H no supera el valor crítico. No hay evidencia suficiente para rechazar la hipótesis nula de igualdad de medias.")

    print("-" * 50)  # Separador entre resultados de diferentes filtros


# ### Graficaremos el logaritmo natural de la cantidad de elementos en el histograma

# In[14]:


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

# Iterar sobre cada subgráfico y crear un histograma para cada filtro
for i in range(5):  # Cambiando el rango a 5
    ax = axs[i]
    flux_values_current = flux_values[i]
    counts, bins, _ = ax.hist(flux_values_current, bins=10, label=clases, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_xlabel(f'Valores de FLUX_{["U", "G", "R", "I", "z"][i]}')
    ax.set_ylabel('Logaritmo Natural de la Cantidad de objetos')
    ax.set_title(f'Histograma de FLUX_{["U", "G", "R", "I", "z"][i]}')
    ax.legend(loc='upper right', title='Clases')  # Mover la leyenda a cada subgráfico

    # Aplicar logaritmo natural a los valores del eje y
    ax.set_yscale('log')
    ax.set_yticks([1, 10, 100, 1000, 10000])  # Puedes personalizar los ticks según tus necesidades

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# ### Ahora, repetiremos esto, pero excluyendo los valores atipicos (outliers)

# In[76]:


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
    
    ax.set_xlabel('Clases')
    ax.set_ylabel(f'Valores de FLUX_{["U", "G", "R", "I", "z"][i]}')
    ax.set_title(f'Distribución de valores de FLUX_{["U", "G", "R", "I", "z"][i]}')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# ### Ahora, Graficamos las correlaciones entre los valores de Flujo que entraremos como parametros

# In[16]:


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


# In[17]:


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


# In[21]:


# Crear una figura con subgráficos para cada filtro
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
# Scatter Plot 1
axs[0].scatter(FLUX_V_G_GALAXY, FLUX_I_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[0].scatter(FLUX_V_G_STAR, FLUX_I_STAR, label='STAR', color='green', alpha=0.7)
axs[0].scatter(FLUX_V_G_QSO, FLUX_I_QSO, label='QSO', color='red', alpha=0.7)
axs[0].set_xlabel('FLUX_G')
axs[0].set_ylabel('FLUX_I')
axs[0].set_title('FLUX_G vs FLUX_I')
axs[0].legend()
# Scatter Plot 2
axs[1].scatter(FLUX_V_G_GALAXY, FLUX_z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[1].scatter(FLUX_V_G_STAR, FLUX_z_STAR, label='STAR', color='green', alpha=0.7)
axs[1].scatter(FLUX_V_G_QSO, FLUX_z_QSO, label='QSO', color='red', alpha=0.7)
axs[1].set_xlabel('FLUX_G')
axs[1].set_ylabel('FLUX_Z')
axs[1].set_title('FLUX_G vs FLUX_Z')
axs[1].legend()
# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[19]:


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


# ### Extraemos las features: u, g, r, i, z, (u-r), (u-g), (g-z), (r-z), (i-z)

# In[19]:


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


# In[38]:


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
        ax.set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
        ax.set_title(labels[i * 3 + j])
        ax.set_xlabel('Tipo de Objeto')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[39]:


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
        ax.set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
        ax.set_title(labels[i * 3 + j])
        ax.set_xlabel('Tipo de Objeto')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[40]:


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
        ax.set_ylabel('Frecuencia')
        ax.set_title(labels[i * 3 + j])
        ax.legend()

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[41]:


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
        ax.set_ylabel('Frecuencia')
        ax.set_title(labels[i * 3 + j])
        ax.legend()
        
        # Establecer la escala logarítmica en el eje y
        ax.set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[42]:


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
labels = ['RU', 'UG', 'GZ', 'RZ', 'IZ', 'IZ']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j])
        ax.set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
        ax.set_title(labels[i * 3 + j])
        ax.set_xlabel('Tipo de Objeto')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[43]:


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
labels = ['RU', 'UG', 'GZ', 'RZ', 'IZ', 'IZ']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j], showfliers=False)
        ax.set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
        ax.set_title(labels[i * 3 + j])
        ax.set_xlabel('Tipo de Objeto')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[44]:


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
labels = ['RU', 'UG', 'GZ', 'RZ', 'IZ', 'UZ']
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
        ax.set_ylabel('Frecuencia')
        ax.set_title(labels[i * 3 + j])
        ax.legend()
        
        # Establecer la escala logarítmica en el eje y
        ax.set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[ ]:





# In[56]:


R_G


# In[57]:


RG


# In[24]:


# Crear una figura con subgráficos para cada combinación de índices de color
fig, axs = plt.subplots(5, 2, figsize=(12, 18))

# Scatter Plot 1: U_R vs G_Z
axs[0, 0].scatter(U_R_GALAXY, G_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[0, 0].scatter(U_R_STAR, G_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[0, 0].scatter(U_R_QSO, G_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[0, 0].set_xlabel('U_R')
axs[0, 0].set_ylabel('G_Z')
axs[0, 0].set_title('U_R vs G_Z')
axs[0, 0].legend()

# Scatter Plot 2: U_R vs R_Z
axs[0, 1].scatter(U_R_GALAXY, R_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[0, 1].scatter(U_R_STAR, R_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[0, 1].scatter(U_R_QSO, R_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[0, 1].set_xlabel('U_R')
axs[0, 1].set_ylabel('R_Z')
axs[0, 1].set_title('U_R vs R_Z')
axs[0, 1].legend()

# Scatter Plot 3: U_R vs I_Z
axs[1, 0].scatter(U_R_GALAXY, I_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[1, 0].scatter(U_R_STAR, I_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[1, 0].scatter(U_R_QSO, I_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[1, 0].set_xlabel('U_R')
axs[1, 0].set_ylabel('I_Z')
axs[1, 0].set_title('U_R vs I_Z')
axs[1, 0].legend()

# Scatter Plot 4: U_R vs U_G
axs[1, 1].scatter(U_R_GALAXY, U_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[1, 1].scatter(U_R_STAR, U_G_STAR, label='STAR', color='green', alpha=0.7)
axs[1, 1].scatter(U_R_QSO, U_G_QSO, label='QSO', color='red', alpha=0.7)
axs[1, 1].set_xlabel('U_R')
axs[1, 1].set_ylabel('U_G')
axs[1, 1].set_title('U_R vs U_G')
axs[1, 1].legend()

# Scatter Plot 5: G_Z vs R_Z
axs[2, 0].scatter(G_Z_GALAXY, R_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[2, 0].scatter(G_Z_STAR, R_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[2, 0].scatter(G_Z_QSO, R_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[2, 0].set_xlabel('G_Z')
axs[2, 0].set_ylabel('R_Z')
axs[2, 0].set_title('G_Z vs R_Z')
axs[2, 0].legend()

# Scatter Plot 6: G_Z vs I_Z
axs[2, 1].scatter(G_Z_GALAXY, I_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[2, 1].scatter(G_Z_STAR, I_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[2, 1].scatter(G_Z_QSO, I_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[2, 1].set_xlabel('G_Z')
axs[2, 1].set_ylabel('I_Z')
axs[2, 1].set_title('G_Z vs I_Z')
axs[2, 1].legend()

# Scatter Plot 7: G_Z vs U_G
axs[3, 0].scatter(G_Z_GALAXY, U_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[3, 0].scatter(G_Z_STAR, U_G_STAR, label='STAR', color='green', alpha=0.7)
axs[3, 0].scatter(G_Z_QSO, U_G_QSO, label='QSO', color='red', alpha=0.7)
axs[3, 0].set_xlabel('G_Z')
axs[3, 0].set_ylabel('U_G')
axs[3, 0].set_title('G_Z vs U_G')
axs[3, 0].legend()

# Scatter Plot 8: R_Z vs I_Z
axs[3, 1].scatter(R_Z_GALAXY, I_Z_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[3, 1].scatter(R_Z_STAR, I_Z_STAR, label='STAR', color='green', alpha=0.7)
axs[3, 1].scatter(R_Z_QSO, I_Z_QSO, label='QSO', color='red', alpha=0.7)
axs[3, 1].set_xlabel('R_Z')
axs[3, 1].set_ylabel('I_Z')
axs[3, 1].set_title('R_Z vs I_Z')
axs[3, 1].legend()

# Scatter Plot 9: R_Z vs U_G
axs[4, 0].scatter(R_Z_GALAXY, U_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[4, 0].scatter(R_Z_STAR, U_G_STAR, label='STAR', color='green', alpha=0.7)
axs[4, 0].scatter(R_Z_QSO, U_G_QSO, label='QSO', color='red', alpha=0.7)
axs[4, 0].set_xlabel('R_Z')
axs[4, 0].set_ylabel('U_G')
axs[4, 0].set_title('R_Z vs U_G')
axs[4, 0].legend()

# Scatter Plot 10: I_Z vs U_G
axs[4, 1].scatter(I_Z_GALAXY, U_G_GALAXY, label='GALAXY', color='blue', alpha=0.7)
axs[4, 1].scatter(I_Z_STAR, U_G_STAR, label='STAR', color='green', alpha=0.7)
axs[4, 1].scatter(I_Z_QSO, U_G_QSO, label='QSO', color='red', alpha=0.7)
axs[4, 1].set_xlabel('I_Z')
axs[4, 1].set_ylabel('U_G')
axs[4, 1].set_title('I_Z vs U_G')
axs[4, 1].legend()

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# In[26]:


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
    axs[i].scatter(eval(filtro + '_GALAXY'), eval(indice_color + '_GALAXY'), label='GALAXY', color='blue', alpha=0.7)
    axs[i].scatter(eval(filtro + '_STAR'), eval(indice_color + '_STAR'), label='STAR', color='green', alpha=0.7)
    axs[i].scatter(eval(filtro + '_QSO'), eval(indice_color + '_QSO'), label='QSO', color='red', alpha=0.7)
    axs[i].set_xlabel(filtro)
    axs[i].set_ylabel(indice_color)
    axs[i].set_title(f'{indice_color} vs {filtro}')
    axs[i].legend()

# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[27]:


# Lista de todas las razones entre flujos
razones_flujos = ['RU', 'GZ', 'RZ', 'IZ', 'UG', 'UZ']

# Generar todas las combinaciones posibles entre razones de flujos
combinaciones_razones = list(itertools.combinations(razones_flujos, 2))

# Crear una figura con subgráficos para cada combinación de razones de flujos
fig, axs = plt.subplots(len(combinaciones_razones), 1, figsize=(10, 5*len(combinaciones_razones)))

# Iterar sobre todas las combinaciones y crear los gráficos correspondientes
for i, (razon1, razon2) in enumerate(combinaciones_razones):
    # Scatter Plot para la combinación actual
    axs[i].scatter(eval(razon1 + '_GALAXY'), eval(razon2 + '_GALAXY'), label='GALAXY', color='blue', alpha=0.7)
    axs[i].scatter(eval(razon1 + '_STAR'), eval(razon2 + '_STAR'), label='STAR', color='green', alpha=0.7)
    axs[i].scatter(eval(razon1 + '_QSO'), eval(razon2 + '_QSO'), label='QSO', color='red', alpha=0.7)
    axs[i].set_xlabel(razon1)
    axs[i].set_ylabel(razon2)
    axs[i].set_title(f'{razon1} vs {razon2}')
    axs[i].legend()

# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[30]:


import itertools

# Lista de todas las razones entre flujos
razones_flujos = ['RU', 'GZ', 'RZ', 'IZ', 'UG', 'UZ']

# Lista de todos los índices de color
indices_color = ['U_R', 'U_G', 'G_Z', 'R_Z', 'I_Z']

# Lista de todos los filtros
filtros = ['FLUX_U', 'FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z']

# Generar todas las combinaciones posibles entre razones de flujos, índices de color y filtros
combinaciones = list(itertools.product(razones_flujos, filtros))
combinaciones += list(itertools.product(razones_flujos, indices_color))

# Crear una figura para cada combinación
for razon_flujo, parametro in combinaciones:
    fig, ax = plt.subplots(figsize=(8, 6))
    # Scatter Plot para la combinación actual
    ax.scatter(eval(parametro + '_GALAXY'), eval(razon_flujo + '_GALAXY'), label='GALAXY', color='blue', alpha=0.7)
    ax.scatter(eval(parametro + '_STAR'), eval(razon_flujo + '_STAR'), label='STAR', color='green', alpha=0.7)
    ax.scatter(eval(parametro + '_QSO'), eval(razon_flujo + '_QSO'), label='QSO', color='red', alpha=0.7)
    ax.set_xlabel(parametro)
    ax.set_ylabel(razon_flujo)
    ax.set_title(f'{razon_flujo} vs {parametro}')
    ax.legend()
    # Mostrar el gráfico
    plt.show()



# In[ ]:





# ### Parametro de curvatura del espectro

# spectra_galaxy
# spectra_qso
# spectra_star

# In[8]:


wavelenght


# In[20]:


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


# In[21]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [curv_gal, curv_star, curv_qso],
    [b_gal, b_star, b_qso],
    [c_gal, c_star, c_qso]
]

# Etiquetas para los boxplots
labels = ['Curvaturas', 'Parámetro b', 'Parámetro c']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i, ax in enumerate(axs):
    ax.boxplot(data[i])
    ax.set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
    ax.set_title(labels[i])
    ax.set_xlabel('Tipo de Objeto')
    ax.set_ylabel(labels[i])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[22]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [
    [curv_gal, curv_star, curv_qso],
    [b_gal, b_star, b_qso],
    [c_gal, c_star, c_qso]
]

# Etiquetas para los boxplots
labels = ['Curvaturas', 'Parámetro b', 'Parámetro c']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i, ax in enumerate(axs):
    ax.boxplot(data[i], showfliers=False)
    ax.set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
    ax.set_title(labels[i])
    ax.set_xlabel('Tipo de Objeto')
    ax.set_ylabel(labels[i])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[26]:


# Datos para los histogramas (sustituye estos datos por los tuyos)
data_curv = [curv_gal, curv_star, curv_qso]
data_b = [b_gal, b_star, b_qso]
data_c = [c_gal, c_star, c_qso]

# Etiquetas para los histogramas
labels = ['Curvaturas', 'Parámetro b', 'Parámetro c']
colors = ['blue', 'green', 'red']

# Crear una figura con subgráficos para cada clase
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase
for i, data in enumerate([data_curv, data_b, data_c]):
    for j, class_data in enumerate(data):
        axs[i].hist(class_data, bins=10, color=colors[j], alpha=0.7, label=['Galaxia', 'Estrella', 'QSO'][j])

    # Agregar leyenda y título
    axs[i].legend()
    axs[i].set_title(labels[i])
    axs[i].set_xlabel(labels[i])
    axs[i].set_ylabel('Frecuencia')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[27]:


# Datos para los histogramas (sustituye estos datos por los tuyos)
data_curv = [curv_gal, curv_star, curv_qso]
data_b = [b_gal, b_star, b_qso]
data_c = [c_gal, c_star, c_qso]

# Etiquetas para los histogramas
labels = ['Curvaturas', 'Parámetro b', 'Parámetro c']
colors = ['blue', 'green', 'red']

# Crear una figura con subgráficos para cada clase
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase
for i, data in enumerate([data_curv, data_b, data_c]):
    for j, class_data in enumerate(data):
        axs[i].hist(class_data, bins=10, color=colors[j], alpha=0.7, label=['Galaxia', 'Estrella', 'QSO'][j])

    # Agregar leyenda y título
    axs[i].legend()
    axs[i].set_title(labels[i])
    axs[i].set_xlabel(labels[i])
    axs[i].set_ylabel('Frecuencia')
    
    # Establecer la escala logarítmica en el eje y
    axs[i].set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[50]:


# Datos de ejemplo para las variables
UG = [UG_GALAXY, UG_STAR, UG_QSO]
b = [b_gal, b_star, b_qso]
c = [c_gal, c_star, c_qso]
R_Z = [R_Z_GALAXY, R_Z_STAR, R_Z_QSO]
Curvatura = [curv_gal, curv_star, curv_qso]

# Colores para las clases
colors = ['blue', 'green', 'red']
labels = ['Galaxia', 'Estrella', 'QSO']

# Crear la figura
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Etiquetas para los ejes
x_labels = ['UG', 'b', 'c', 'R_Z', 'UG', 'R_Z', 'UG', 'R_Z', 'UG']
y_labels = ['Curvatura', 'Curvatura', 'Curvatura', 'b', 'b', 'c', 'c', 'Curvatura', 'R_Z']

# Iterar sobre los datos y colorear según la clase
for i in range(len(UG)):
    axs[0, 0].scatter(UG[i], Curvatura[i], color=colors[i], label=labels[i], alpha=0.7)
    axs[0, 1].scatter(b[i], Curvatura[i], color=colors[i], label=labels[i], alpha=0.7)
    axs[0, 2].scatter(c[i], Curvatura[i], color=colors[i], label=labels[i], alpha=0.7)
    axs[1, 0].scatter(R_Z[i], b[i], color=colors[i], label=labels[i], alpha=0.7)
    axs[1, 1].scatter(R_Z[i], UG[i], color=colors[i], label=labels[i], alpha=0.7)
    axs[1, 2].scatter(UG[i], c[i], color=colors[i], label=labels[i], alpha=0.7)
    axs[2, 0].scatter(R_Z[i], c[i], color=colors[i], label=labels[i], alpha=0.7)
    axs[2, 1].scatter(UG[i], b[i], color=colors[i], label=labels[i], alpha=0.7)
    axs[2, 2].scatter(UG[i], R_Z[i], color=colors[i], label=labels[i], alpha=0.7)

# Añadir etiquetas y leyenda
for i in range(3):
    for j in range(3):
        if i * 3 + j < len(x_labels):
            axs[i, j].set_xlabel(x_labels[i * 3 + j])
            axs[i, j].set_ylabel(y_labels[i * 3 + j])
            axs[i, j].legend()

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# ### Parametro MAD (median absolute deviation)

# In[21]:


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


# In[62]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [mad_gal, mad_star, mad_qso]

# Etiquetas para los boxplots
labels = ['MAD GALAXY', 'MAD STAR', 'MAD QSO']

# Crear la figura y los ejes de la figura
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Crear el boxplot con outliers
axs[0].boxplot(data)
axs[0].set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
axs[0].set_title('Boxplot de MAD con Outliers')
axs[0].set_xlabel('Tipo de Objeto')
axs[0].set_ylabel('MAD')

# Crear el boxplot sin outliers
axs[1].boxplot(data, showfliers=False)
axs[1].set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
axs[1].set_title('Boxplot de MAD sin Outliers')
axs[1].set_xlabel('Tipo de Objeto')
axs[1].set_ylabel('MAD')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[65]:


# Datos para los histogramas
data = [mad_gal, mad_star, mad_qso]

# Etiquetas para los histogramas
labels = ['MAD GALAXY', 'MAD STAR', 'MAD QSO']

# Colores para los histogramas
colors = ['blue', 'green', 'red']

# Crear una figura con subgráficos para cada clase
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase en la primera columna
for i, class_data in enumerate(data):
    axs[0].hist(class_data, bins=10, color=colors[i], alpha=0.7, label=labels[i])

# Agregar leyenda y título a la primera columna
axs[0].legend()
axs[0].set_title('Histogramas de MAD por Clase')
axs[0].set_xlabel('MAD')
axs[0].set_ylabel('Frecuencia')

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase en la segunda columna
for i, class_data in enumerate(data):
    axs[1].hist(class_data, bins=10, color=colors[i], alpha=0.7, label=labels[i])

# Agregar leyenda y título a la segunda columna
axs[1].legend()
axs[1].set_title('Histogramas de MAD por Clase (Escala Logarítmica)')
axs[1].set_xlabel('MAD')
axs[1].set_ylabel('Frecuencia')

# Establecer escala logarítmica en el eje y para la segunda columna
axs[1].set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[67]:


# Datos de ejemplo para las variables
UG = [UG_GALAXY, UG_STAR, UG_QSO]
R_Z = [R_Z_GALAXY, R_Z_STAR, R_Z_QSO]
MAD = [mad_gal, mad_star, mad_qso]

# Colores para las clases
colors = ['blue', 'green', 'red']
labels = ['Galaxia', 'Estrella', 'QSO']

# Crear la figura
fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# Etiquetas para los ejes
x_labels = ['UG', 'R_Z']
y_labels = ['MAD', 'MAD']

# Iterar sobre los datos y colorear según la clase
for i in range(len(UG)):
    axs[0].scatter(UG[i], MAD[i], color=colors[i], label=labels[i], alpha=0.7)
    axs[1].scatter(R_Z[i], MAD[i], color=colors[i], label=labels[i], alpha=0.7)

# Añadir etiquetas y leyenda
for i in range(2):
    axs[i].set_xlabel(x_labels[i])
    axs[i].set_ylabel(y_labels[i])
    axs[i].legend()

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[ ]:





# ### Filtros sinteticos

# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[53]:


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

    ax.set_xlabel('Clases')
    ax.set_ylabel(f'Valores de FLUX_{i}')
    ax.set_title(f'Distribución de valores de FLUX_{i}')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[58]:


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

    ax.set_xlabel('Clases')
    ax.set_ylabel(f'Valores de FLUX_{i}')
    ax.set_title(f'Distribución de valores de FLUX_{i}')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[60]:


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

# Iterar sobre cada subgráfico y crear un histograma para cada filtro
for i in range(4):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    flux_values_current = flux_values[i]

    # Crear un histograma
    ax.hist(flux_values_current, bins=20, label=clases, alpha=0.7)
    
    ax.set_xlabel(f'Valores de FLUX_{i}')
    ax.set_ylabel('Frecuencia (log)')
    ax.set_title(f'Histograma de valores de FLUX_{i}')
    ax.legend()
    
    # Establecer escala logarítmica en el eje y
    ax.set_yscale('log')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[72]:


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


# In[74]:


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


# In[26]:


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


# In[95]:


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
    titulo = f'Boxplot de Ratios'
    ax.set_title(titulo)
    ax.set_xlabel('Clases')
    ax.set_ylabel('Valores de Ratios')

    # Mostrar el boxplot
    plt.show()


# In[91]:


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
    titulo = f'Histograma de Ratios (Escala Logarítmica)'
    ax.set_title(titulo)
    ax.set_xlabel('Valores de Ratios')
    ax.set_ylabel('Frecuencia (log)')
    ax.legend()

    # Mostrar el histograma
    plt.show()


# In[99]:


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
    axs[i].set_title(f'Histograma para {clases[i]}')
    axs[i].set_xlabel('Valor de _3I')
    axs[i].set_ylabel('Frecuencia')

# Ajustar el espacio entre los subgráficos
plt.tight_layout()

# Mostrar los histogramas
plt.show()


# In[ ]:





# In[ ]:





# ### Transformacion con logaritmos
# λ2

# mλ1 − mλ2 = −2.5 lg(Fλ1/Fλ2)
#           = a + b/Tc
# El logaritmo no acepta valores negativos, asi que lo haremos para el fljo total recibido, es decir, el absoluto

# In[27]:


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


# In[21]:


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
labels = ['U-R', 'U-G', 'G-Z', 'R-Z', 'I-Z', 'I-Z']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j])
        ax.set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
        ax.set_title(labels[i * 3 + j])
        ax.set_xlabel('Tipo de Objeto')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[23]:


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
labels = ['U-R T', 'U-G T', 'G-Z T', 'R-Z T', 'I-Z T', 'I-Z T']

# Crear la figura con subgráficos de 1 fila y 3 columnas
fig, axs = plt.subplots(2, 3, figsize=(15, 5))

# Iterar sobre cada subgráfico y crear el boxplot correspondiente
for i in range(2):
    for j in range(3):
        ax = axs[i, j]
        ax.boxplot(data[i * 3 + j], showfliers=False)
        ax.set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
        ax.set_title(labels[i * 3 + j])
        ax.set_xlabel('Tipo de Objeto')
        ax.set_ylabel(labels[i * 3 + j])

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[24]:


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
labels = ['U-R T', 'U-G T', 'G-Z T', 'R-Z T', 'I-Z T', 'I-Z T']
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
        ax.set_ylabel('Frecuencia')
        ax.set_title(labels[i * 3 + j])
        ax.legend()

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[25]:


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
labels = ['U-R T', 'U-G T', 'G-Z T', 'R-Z T', 'I-Z T', 'I-Z T']
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
        ax.set_ylabel('Frecuencia')
        ax.set_title(labels[i * 3 + j])
        ax.legend()
        
        # Establecer la escala logarítmica en el eje y
        ax.set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[26]:


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


# In[28]:


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





# ### Creamos modelos con H2O

# In[ ]:





# In[32]:


# Inicializa el cluster de H2O y mostrar info del cluster en uso
h2o.init()


# #Creamos el H2O frame con una columna
# datos = h2o.H2OFrame(python_obj=STYPE, column_names=['SPECTYPE'], column_types=["string"])
# 
# #Agregamos mas columnas al frame que hemos creado
# flux_u_h2o= h2o.H2OFrame(python_obj=FLUX_U, column_names=['FLUX_U'], column_types=["float"])
# datos= datos.cbind(flux_u_h2o)
# 
# #Agregamos mas columnas al frame que hemos creado
# flux_g_h2o= h2o.H2OFrame(python_obj=FLUX_G, column_names=['FLUX_G'], column_types=["float"])
# datos= datos.cbind(flux_g_h2o)
# 
# #Agregamos mas columnas al frame que hemos creado
# flux_r_h2o= h2o.H2OFrame(python_obj=FLUX_R, column_names=['FLUX_R'], column_types=["float"])
# datos= datos.cbind(flux_r_h2o)
# 
# #Agregamos mas columnas al frame que hemos creado
# flux_i_h2o= h2o.H2OFrame(python_obj=FLUX_I, column_names=['FLUX_I'], column_types=["float"])
# datos= datos.cbind(flux_i_h2o)
# 
# #Agregamos mas columnas al frame que hemos creado
# flux_z_h2o= h2o.H2OFrame(python_obj=FLUX_Z, column_names=['FLUX_Z'], column_types=["float"])
# datos= datos.cbind(flux_z_h2o)
# 
# #Agregamos mas columnas al frame que hemos creado
# flux_r_z_h2o= h2o.H2OFrame(python_obj=R_Z, column_names=['R-Z'], column_types=["float"])
# datos= datos.cbind(flux_r_z_h2o)
# 
# #Agregamos mas columnas al frame que hemos creado
# flux_r_g_h2o= h2o.H2OFrame(python_obj=R_G, column_names=['R-G'], column_types=["float"])
# datos= datos.cbind(flux_r_g_h2o)

# In[34]:


#Creamos el H2O frame con una columna
datos = h2o.H2OFrame(python_obj=STYPE, column_names=['SPECTYPE'], column_types=["string"])

#Agregamos mas columnas al frame que hemos creado
flux_u_h2o= h2o.H2OFrame(python_obj=FLUX_U, column_names=['FLUX_U'], column_types=["float"])
datos= datos.cbind(flux_u_h2o)

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


# In[35]:


datos


# In[36]:


datos[134028,0]


# In[37]:


# Convierte la variable objetivo a factor utilizando la función asfactor()
datos['SPECTYPE'] = datos['SPECTYPE'].asfactor()

# Define las columnas predictoras y la variable objetivo
predictores = datos.columns[1:]  # Todas las columnas excepto la primera (SPECTYPE)
objetivo='SPECTYPE'


# In[38]:


# Dividir el conjunto de datos en entrenamiento y prueba de manera estratificada
#train, test = datos.split_frame(ratios=[0.6], seed=42, destination_frames=['train', 'test'], stratify='SPECTYPE')
train, test = datos.split_frame(ratios=[0.6], seed=42)


# In[39]:


train['SPECTYPE'].table()


# In[40]:


test['SPECTYPE'].table()


# ### Red neuronal con H2O

# In[41]:


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


# In[42]:


# Graficar el Loss en cada época de entrenamiento y validación
training_metrics = modelo_nn.score_history()
plt.plot(training_metrics['epochs'], training_metrics['training_classification_error'], label='Training Error')
plt.plot(training_metrics['epochs'], training_metrics['validation_classification_error'], label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()


# ### Random Forest con H2O

# In[43]:


# Configura y entrena el modelo de Random Forest
modelo_rf = H2ORandomForestEstimator(ntrees=200, max_depth=20, seed=42)
modelo_rf.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprime métricas de rendimiento en el conjunto de prueba
print(modelo_rf.model_performance(test_data=test))

# Obtener las importancias de las variables
importancias_variables_rf = modelo_rf.varimp()
# Imprimir las importancias de las variables
print(importancias_variables_rf)


# In[44]:


# Graficar las métricas de rendimiento en cada árbol
rf_metrics = modelo_rf.score_history()
plt.plot(rf_metrics['number_of_trees'], rf_metrics['training_classification_error'], label='Training Error')
plt.plot(rf_metrics['number_of_trees'], rf_metrics['validation_classification_error'], label='Validation Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.legend()
plt.show()


# ### Seleccion de la mejor familia con H2O

# In[45]:


# Configura y ejecuta la búsqueda automática de modelos
automl = H2OAutoML(max_models=10, seed=42)
automl.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprime métricas de rendimiento en el conjunto de prueba para el mejor modelo
print(automl.leader.model_performance(test_data=test))


# In[46]:


# Obtiene y muestra el modelo líder
best_model = automl.leader
print("Mejor modelo:")
print(best_model)

# Imprime métricas de rendimiento en el conjunto de prueba para el mejor modelo
print("Métricas del mejor modelo en el conjunto de prueba:")
print(best_model.model_performance(test_data=test))


# In[106]:


# Cierra el cluster de H2O
h2o.cluster().shutdown()


# ### Miramos el espectro total de cada tipo, para saber si el flujo se satura en ciertas longitudes de onda

# In[9]:


archivos= ['DataDESI_1_76.fits', 'DataDESI_77_152.fits', 'DataDESI_153_213.fits', 'DataDESI_214_284.fits', 'DataDESI_285_351.fits', 'DataDESI_352_438.fits'
 , 'DataDESI_439_530.fits', 'DataDESI_531_606.fits', 'DataDESI_607_690.fits', 'DataDESI_691_752.fits']

# Inicializar listas para guardar las sumas de flujos B, R y Z para cada clase de objeto celeste
B_total_star = []
R_total_star = []
Z_total_star = []

B_total_qso = []
R_total_qso = []
Z_total_qso = []

B_total_gal = []
R_total_gal = []
Z_total_gal = []

# Recorrido de los archivos FITS
for archivo in archivos:
    espc = fits.open(archivo)
    
    # Leer flujos B, R y Z del archivo FITS
    Bflux = espc[2].data
    Zflux = espc[4].data
    Rflux = espc[3].data
    
    # Obtener la clase espectral para cada espectro
    clases_espectrales = Table.read(espc, hdu=1)['SPECTYPE'].data
    
    # Seleccionar solo los espectros de la clase espectral objetivo
    indices_objetivo_star = np.where(clases_espectrales == 'STAR')[0]
    indices_objetivo_qso = np.where(clases_espectrales == 'QSO')[0]
    indices_objetivo_gal = np.where(clases_espectrales == 'GALAXY')[0]
    
    # Filtrar los flujos para cada clase de objeto celeste
    Bflux_objetivo_star = Bflux[indices_objetivo_star]
    Zflux_objetivo_star = Zflux[indices_objetivo_star]
    Rflux_objetivo_star = Rflux[indices_objetivo_star]
    
    Bflux_objetivo_qso = Bflux[indices_objetivo_qso]
    Zflux_objetivo_qso = Zflux[indices_objetivo_qso]
    Rflux_objetivo_qso = Rflux[indices_objetivo_qso]
    
    Bflux_objetivo_gal = Bflux[indices_objetivo_gal]
    Zflux_objetivo_gal = Zflux[indices_objetivo_gal]
    Rflux_objetivo_gal = Rflux[indices_objetivo_gal]
    
    # Sumar los flujos para cada clase de objeto celeste
    B_total_star.append(np.sum(Bflux_objetivo_star, axis=0))
    R_total_star.append(np.sum(Rflux_objetivo_star, axis=0))
    Z_total_star.append(np.sum(Zflux_objetivo_star, axis=0))
    
    B_total_qso.append(np.sum(Bflux_objetivo_qso, axis=0))
    R_total_qso.append(np.sum(Rflux_objetivo_qso, axis=0))
    Z_total_qso.append(np.sum(Zflux_objetivo_qso, axis=0))
    
    B_total_gal.append(np.sum(Bflux_objetivo_gal, axis=0))
    R_total_gal.append(np.sum(Rflux_objetivo_gal, axis=0))
    Z_total_gal.append(np.sum(Zflux_objetivo_gal, axis=0))

# Sumar los flujos para cada clase de objeto celeste
B_total_star = np.sum(B_total_star, axis=0)
R_total_star = np.sum(R_total_star, axis=0)
Z_total_star = np.sum(Z_total_star, axis=0)
    
B_total_qso = np.sum(B_total_qso, axis=0)
R_total_qso = np.sum(R_total_qso, axis=0)
Z_total_qso = np.sum(Z_total_qso, axis=0)
    
B_total_gal = np.sum(B_total_gal, axis=0)
R_total_gal = np.sum(R_total_gal, axis=0)
Z_total_gal = np.sum(Z_total_gal, axis=0)


# In[10]:


espectro_star=np.hstack((B_total_star, R_total_star, Z_total_star))
espectro_qso=np.hstack((B_total_qso, R_total_qso, Z_total_qso))
espectro_gal=np.hstack((B_total_gal, R_total_gal, Z_total_gal))


# In[68]:


# Normalizar los espectros
espectro_gal_normalizado = espectro_gal / np.max(espectro_gal)
espectro_qso_normalizado = espectro_qso / np.max(espectro_qso)
espectro_star_normalizado = espectro_star / np.max(espectro_star)

plt.figure()
plt.plot(wavelenght, espectro_gal_normalizado, color='blue', label='Galaxia')
plt.plot(wavelenght, espectro_qso_normalizado, color='red', label='QSO')
#plt.plot(wavelenght, espectro_star_normalizado, color='green', label='Estrella')
plt.yscale('log')  # Establecer escala logarítmica en el eje y
plt.xlabel('Longitud de onda')
plt.ylabel('Espectro (Normalizado)')
plt.legend()
plt.show()


# In[54]:


plt.figure()
plt.plot(wavelenght, espectro_gal, color='black')
plt.show()


# In[8]:


plt.figure()
plt.plot(wavelenght, espectro_qso, color='black')
plt.show()


# In[13]:


plt.figure()
plt.plot(wavelenght, espectro_star, color='black')
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


# ### Templates de chi^2
# Espectros netos (reemplaza estos datos con tus espectros)
espectro_gal
espectro_star
espectro_qso
wavelenght
# In[13]:


def planck(x, T):
    h = 6.62607015e-34  # Constante de Planck en Julios por segundo
    c = 3.0e8  # Velocidad de la luz en metros por segundo
    k = 1.380649e-23  # Constante de Boltzmann en Julios por Kelvin
    return (2 * h * c**2 / x**5) * (1 / (np.exp(h * c / (x * k * T)) - 1))

# Realiza el ajuste de curva a los datos de la galaxia utilizando la función de cuerpo negro de Planck
popt_gal, _ = curve_fit(planck, wavelenght, espectro_gal)

# Plotea los datos y la función ajustada para la galaxia
plt.plot(wavelenght, espectro_gal, 'k-', label='Datos')
plt.plot(wavelenght, planck(wavelenght, *popt_gal), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Longitud de Onda')
plt.ylabel('Flujo')
plt.title('Ajuste de curva y datos del espectro neto de galaxia (cuerpo negro de Planck)')
plt.show()


# In[28]:


# Define una función polinómica para ajustar
def polynomial(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Realizar el ajuste de curva a los datos de la galaxia
popt_gal, _ = curve_fit(polynomial, wavelenght, espectro_gal)

# Plotea los datos y la función ajustada para la galaxia
plt.plot(wavelenght, espectro_gal, 'k-', label='Datos')
plt.plot(wavelenght, polynomial(wavelenght, *popt_gal), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de galaxia (polinómico)')
plt.show()

# Almacena el template de la galaxia
template_gal = polynomial(wavelenght, *popt_gal)


# In[107]:


# Define una función polinómica de décimo grado para ajustar
def polynomial(x, a, b, c, d, e, f, g, h, i, j, k):
    return   g * x**4 + h * x**3 + i * x**2 + j * x + k

# Realizar el ajuste de curva a los datos de la galaxia
popt_gal, _ = curve_fit(polynomial, wavelenght, espectro_gal)

# Plotea los datos y la función ajustada para la galaxia
plt.plot(wavelenght, espectro_gal, 'k-', label='Datos')
plt.plot(wavelenght, polynomial(wavelenght, *popt_gal), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de galaxia (polinómico de grado 4)')
plt.show()


# In[34]:


# Define una función polinómica de quinto grado para ajustar
def polynomial(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

# Realizar el ajuste de curva a los datos de la galaxia
popt_gal, _ = curve_fit(polynomial, wavelenght, espectro_gal)

# Plotea los datos y la función ajustada para la galaxia
plt.plot(wavelenght, espectro_gal, 'k-', label='Datos')
plt.plot(wavelenght, polynomial(wavelenght, *popt_gal), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de galaxia (polinómico de grado 5)')
plt.show()


# In[ ]:





# In[ ]:





# In[29]:


# Define una función polinómica de décimo grado para ajustar
def polynomial(x, a, b, c, d, e, f, g, h, i, j, k):
    return  h * x**3 + i * x**2 + j * x + k

# Realizar el ajuste de curva a los datos del espectro de estrella
popt_star, _ = curve_fit(polynomial, wavelenght, espectro_star)

# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, espectro_star, 'k-', label='Datos')
plt.plot(wavelenght, polynomial(wavelenght, *popt_star), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de estrella (polinómico de grado 3)')
plt.show()

# Almacena el template del espectro de estrella
template_star = polynomial(wavelenght, *popt_star)


# In[103]:


# Define una función polinómica de cuarto grado para ajustar
def polynomial(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Realizar el ajuste de curva a los datos del espectro de estrella
popt_star, _ = curve_fit(polynomial, wavelenght, espectro_star)

# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, espectro_star, 'k-', label='Datos')
plt.plot(wavelenght, polynomial(wavelenght, *popt_star), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de estrella (polinómico de grado 4)')
plt.show()


# In[104]:


# Define una función polinómica de quinto grado para ajustar
def polynomial(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

# Realizar el ajuste de curva a los datos del espectro de estrella
popt_star, _ = curve_fit(polynomial, wavelenght, espectro_star)

# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, espectro_star, 'k-', label='Datos')
plt.plot(wavelenght, polynomial(wavelenght, *popt_star), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de estrella (polinómico de grado 5)')
plt.show()


# In[105]:


# Lista de tipos de ajuste de modelos
fit_types = ['gaussian', 'lorentzian']

# Crear una instancia de Spectrum
spec = pyspeckit.Spectrum(data= espectro_star, xarr=wavelenght)

# Iterar sobre los tipos de ajuste de modelos y realizar el ajuste
for fit_type in fit_types:
    spec.specfit(fittype=fit_type)
    
    # Graficar el espectro y el ajuste
    spec.plotter()
    spec.specfit.plot_fit()
    
    # Mostrar el gráfico
    plt.show()    
    # Limpiar la figura para el siguiente ajuste
    plt.clf()


# In[ ]:





# In[ ]:





# In[16]:


# Define una función polinómica de décimo grado para ajustar
def polynomial(x, a, b, c, d, e, f, g, h, i, j, k):
    return  h * x**3 + i * x**2 + j * x + k

# Realizar el ajuste de curva a los datos del espectro de estrella
popt_qso, _ = curve_fit(polynomial, wavelenght, espectro_qso)

# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, espectro_qso, 'k-', label='Datos')
plt.plot(wavelenght, polynomial(wavelenght, *popt_qso), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de estrella (polinómico de grado 3)')
plt.show()

# Almacena el template del espectro de estrella
template_qso = polynomial(wavelenght, *popt_qso)


# In[112]:


# Define una función polinómica de décimo grado para ajustar
def polynomial(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Realizar el ajuste de curva a los datos del espectro de estrella
popt_qso, _ = curve_fit(polynomial, wavelenght, espectro_qso)

# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, espectro_qso, 'k-', label='Datos')
plt.plot(wavelenght, polynomial(wavelenght, *popt_qso), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de estrella (polinómico de grado 4)')
plt.show()


# In[115]:


# Define una función polinómica de décimo grado para ajustar
def polynomial(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
    
# Realizar el ajuste de curva a los datos del espectro de estrella
popt_qso, _ = curve_fit(polynomial, wavelenght, espectro_qso)

# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, espectro_qso, 'k-', label='Datos')
plt.plot(wavelenght, polynomial(wavelenght, *popt_qso), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de estrella (polinómico de grado 5)')
plt.show()


# In[119]:


# Define una función para ajustar una ley de potencias
def power_law(x, a, b):
    return a * x**b

# Realizar el ajuste de curva a los datos del espectro de estrella
popt_qso, _ = curve_fit(power_law, wavelenght, espectro_qso)

# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, espectro_qso, 'k-', label='Datos')
plt.plot(wavelenght, power_law(wavelenght, *popt_qso), 'r-', label='Ajuste')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Ajuste de curva y datos del espectro neto de estrella (ley de potencias)')
plt.show()


# In[121]:


# Lista de tipos de ajuste de modelos
fit_types = ['gaussian', 'lorentzian']

# Crear una instancia de Spectrum
spec = pyspeckit.Spectrum(data= espectro_qso, xarr=wavelenght)

# Iterar sobre los tipos de ajuste de modelos y realizar el ajuste
for fit_type in fit_types:
    spec.specfit(fittype=fit_type)
    
    # Graficar el espectro y el ajuste
    spec.plotter()
    spec.specfit.plot_fit()
    
    # Mostrar el gráfico
    plt.show()    
    # Limpiar la figura para el siguiente ajuste
    plt.clf()


# ### Ahora haremos un ajusto polinomico de grado 3 a todos los espectros y calcularmeos chi^2 respecto a las Templates
spectra_galaxy
spectra_qso
spectra_star
wavelenght
# In[30]:


def polynomial(x, a, b, c, d, e, f, g, h, i, j, k):
    return  h * x**3 + i * x**2 + j * x + k

galaxy_chi_as_qso = []
galaxy_chi_as_galaxy = []
galaxy_chi_as_star = []

# Iterar sobre los espectros de galaxia
for spectrum_galaxy in spectra_galaxy:
    # Realizar el ajuste de curva polinómica de grado 3 al espectro de galaxia
    popt_galaxy, _ = curve_fit(polynomial, wavelenght, spectrum_galaxy)
    spec = polynomial(wavelenght, *popt_galaxy)
    
    # Calcular el chi-cuadrado para cada modelo utilizando la función chisquare
    chi_qso =  np.sum(((spec - template_qso)**2)/template_qso) 
    chi_galaxy = np.sum(((spec - template_gal)**2)/template_gal) 
    chi_star = np.sum(((spec - template_star)**2)/template_star) 
    
    galaxy_chi_as_qso.append(chi_qso)
    galaxy_chi_as_galaxy.append(chi_galaxy)
    galaxy_chi_as_star.append(chi_star)


# In[40]:


def polynomial(x, a, b, c, d, e, f, g, h, i, j, k):
    return  h * x**3 + i * x**2 + j * x + k

qso_chi_as_qso = []
qso_chi_as_galaxy = []
qso_chi_as_star = []

for spectrum_qso in spectra_qso:
    popt_qso, _ = curve_fit(polynomial, wavelenght, spectrum_qso)
    spec = polynomial(wavelenght, *popt_qso)
    
    # Calcular el chi-cuadrado para cada modelo utilizando la función chisquare
    chi_qso =  np.sum(((spec - template_qso)**2)/template_qso) 
    chi_galaxy = np.sum(((spec - template_gal)**2)/template_gal) 
    chi_star = np.sum(((spec - template_star)**2)/template_star) 
    
    qso_chi_as_qso.append(chi_qso)
    qso_chi_as_galaxy.append(chi_galaxy)
    qso_chi_as_star.append(chi_star)


# In[41]:


def polynomial(x, a, b, c, d, e, f, g, h, i, j, k):
    return  h * x**3 + i * x**2 + j * x + k

star_chi_as_qso = []
star_chi_as_galaxy = []
star_chi_as_star = []

for spectrum_star in spectra_star:
    popt_star, _ = curve_fit(polynomial, wavelenght, spectrum_star)
    spec = polynomial(wavelenght, *popt_star)
    
    # Calcular el chi-cuadrado para cada modelo utilizando la función chisquare
    chi_qso =  np.sum(((spec - template_qso)**2)/template_qso) 
    chi_galaxy = np.sum(((spec - template_gal)**2)/template_gal) 
    chi_star = np.sum(((spec - template_star)**2)/template_star) 
    
    star_chi_as_qso.append(chi_qso)
    star_chi_as_galaxy.append(chi_galaxy)
    star_chi_as_star.append(chi_star)


# In[49]:


# Crear subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para QSO
axs[0].hist(galaxy_chi_as_qso, bins=20, alpha=0.5, color='b')
axs[0].set_title('Histograma para QSO')
axs[0].set_xlabel('Valor de Chi-cuadrado')
axs[0].set_ylabel('Frecuencia')
axs[0].set_yscale('log')

# Histograma para Galaxia
axs[1].hist(galaxy_chi_as_galaxy, bins=20, alpha=0.5, color='g')
axs[1].set_title('Histograma para Galaxia')
axs[1].set_xlabel('Valor de Chi-cuadrado')
axs[1].set_ylabel('Frecuencia')
axs[1].set_yscale('log')

# Histograma para Estrella
axs[2].hist(galaxy_chi_as_star, bins=20, alpha=0.5, color='r')
axs[2].set_title('Histograma para Estrella')
axs[2].set_xlabel('Valor de Chi-cuadrado')
axs[2].set_ylabel('Frecuencia')
axs[2].set_yscale('log')

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Mostrar los subgráficos
plt.show()


# In[50]:


# Crear subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para QSO
axs[0].hist(qso_chi_as_qso, bins=20, alpha=0.5, color='b')
axs[0].set_title('Histograma para QSO')
axs[0].set_xlabel('Valor de Chi-cuadrado')
axs[0].set_ylabel('Frecuencia')
axs[0].set_yscale('log')

# Histograma para Galaxia
axs[1].hist(qso_chi_as_galaxy, bins=20, alpha=0.5, color='g')
axs[1].set_title('Histograma para Galaxia')
axs[1].set_xlabel('Valor de Chi-cuadrado')
axs[1].set_ylabel('Frecuencia')
axs[1].set_yscale('log')

# Histograma para Estrella
axs[2].hist(qso_chi_as_star, bins=20, alpha=0.5, color='r')
axs[2].set_title('Histograma para Estrella')
axs[2].set_xlabel('Valor de Chi-cuadrado')
axs[2].set_ylabel('Frecuencia')
axs[2].set_yscale('log')

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Mostrar los subgráficos
plt.show()


# In[51]:


# Crear subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para QSO
axs[0].hist(star_chi_as_qso, bins=20, alpha=0.5, color='b')
axs[0].set_title('Histograma para QSO')
axs[0].set_xlabel('Valor de Chi-cuadrado')
axs[0].set_ylabel('Frecuencia')
axs[0].set_yscale('log')

# Histograma para Galaxia
axs[1].hist(star_chi_as_galaxy, bins=20, alpha=0.5, color='g')
axs[1].set_title('Histograma para Galaxia')
axs[1].set_xlabel('Valor de Chi-cuadrado')
axs[1].set_ylabel('Frecuencia')
axs[1].set_yscale('log')

# Histograma para Estrella
axs[2].hist(star_chi_as_star, bins=20, alpha=0.5, color='r')
axs[2].set_title('Histograma para Estrella')
axs[2].set_xlabel('Valor de Chi-cuadrado')
axs[2].set_ylabel('Frecuencia')
axs[2].set_yscale('log')

# Ajustar el espaciado entre subgráficos
plt.tight_layout()

# Mostrar los subgráficos
plt.show()


# ### Ahora intentamos chi^2, pero con el valor de flujo medio de los espectros estandarizados
spectra_galaxy
spectra_qso
spectra_star
# In[31]:


spectra_galaxy_mean = np.mean(spectra_galaxy, axis=0)
spectra_qso_mean = np.mean(spectra_qso, axis=0)
spectra_star_mean = np.mean(spectra_star, axis=0)


# In[23]:


# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, spectra_star_mean, 'k-', label='Datos')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Espectro flujo medio de estrellas')
plt.show()


# In[24]:


# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, spectra_galaxy_mean, 'k-', label='Datos')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Espectro flujo medio de galaxias')
plt.show()


# In[25]:


# Plotea los datos y la función ajustada para el espectro de estrella
plt.plot(wavelenght, spectra_qso_mean, 'k-', label='Datos')
plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Espectro flujo medio de qso')
plt.show()


# In[34]:


# Normalizar los espectros
spectra_galaxy_mean_norm = spectra_galaxy_mean / np.max(spectra_galaxy_mean)
spectra_qso_mean_norm = spectra_qso_mean / np.max(spectra_qso_mean)
spectra_star_mean_norm = spectra_star_mean / np.max(spectra_star_mean)

plt.figure()
plt.plot(wavelenght, spectra_qso_mean_norm, color='blue', label='QSO')
plt.plot(wavelenght, spectra_galaxy_mean_norm, color='red', label='GALAXIA')

#plt.plot(wavelenght, espectro_star_normalizado, color='green', label='Estrella')
plt.yscale('log')  # Establecer escala logarítmica en el eje y
plt.xlabel('Longitud de onda')
plt.ylabel('Espectro (Normalizado)')
plt.legend()
plt.show()


# In[32]:


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


# In[33]:


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


# In[34]:


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


# In[48]:


# Crear subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para QSO
axs[0].hist(galaxy_chi_qso, bins=20, alpha=0.5, color='b')
axs[0].set_title('Histograma GALAXY como QSO')
axs[0].set_xlabel('Valor de Chi-cuadrado')
axs[0].set_ylabel('Frecuencia')
axs[0].set_yscale('log')

# Histograma para Galaxia
axs[1].hist(galaxy_chi_galaxy, bins=20, alpha=0.5, color='g')
axs[1].set_title('Histograma GALAXY como GALAXY')
axs[1].set_xlabel('Valor de Chi-cuadrado')
axs[1].set_ylabel('Frecuencia')
axs[1].set_yscale('log')

# Histograma para Estrella
axs[2].hist(galaxy_chi_star, bins=20, alpha=0.5, color='r')
axs[2].set_title('Histograma GALAXY como STAR')
axs[2].set_xlabel('Valor de Chi-cuadrado')
axs[2].set_ylabel('Frecuencia')
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
ax.set_xticklabels(['GALAXY como QSO', 'GALAXY como GALAXY', 'GALAXY como STAR'])
ax.set_ylabel('Valor de Chi-cuadrado')
ax.set_title('Boxplots para GALAXY como QSO, GALAXY y STAR')

# Mostrar la gráfica
plt.show()


# In[49]:


# Crear subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para QSO
axs[0].hist(qso_chi_qso, bins=20, alpha=0.5, color='b')
axs[0].set_title('Histograma QSO como QSO')
axs[0].set_xlabel('Valor de Chi-cuadrado')
axs[0].set_ylabel('Frecuencia')
axs[0].set_yscale('log')

# Histograma para Galaxia
axs[1].hist(qso_chi_galaxy, bins=20, alpha=0.5, color='g')
axs[1].set_title('Histograma QSO como GALAXY')
axs[1].set_xlabel('Valor de Chi-cuadrado')
axs[1].set_ylabel('Frecuencia')
axs[1].set_yscale('log')

# Histograma para Estrella
axs[2].hist(qso_chi_star, bins=20, alpha=0.5, color='r')
axs[2].set_title('Histograma QSO como STAR')
axs[2].set_xlabel('Valor de Chi-cuadrado')
axs[2].set_ylabel('Frecuencia')
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
ax.set_xticklabels(['QSO como QSO', 'QSO como GALAXY', 'QSO como STAR'])
ax.set_ylabel('Valor de Chi-cuadrado')
ax.set_title('Boxplots para QSO como QSO, GALAXY y STAR')

# Mostrar la gráfica
plt.show()


# In[50]:


# Crear subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para QSO
axs[0].hist(star_chi_qso, bins=20, alpha=0.5, color='b')
axs[0].set_title('Histograma STAR como QSO')
axs[0].set_xlabel('Valor de Chi-cuadrado')
axs[0].set_ylabel('Frecuencia')
axs[0].set_yscale('log')

# Histograma para Galaxia
axs[1].hist(star_chi_galaxy, bins=20, alpha=0.5, color='g')
axs[1].set_title('Histograma STAR como GALAXY')
axs[1].set_xlabel('Valor de Chi-cuadrado')
axs[1].set_ylabel('Frecuencia')
axs[1].set_yscale('log')

# Histograma para Estrella
axs[2].hist(qso_chi_star, bins=20, alpha=0.5, color='r')
axs[2].set_title('Histograma STAR como STAR')
axs[2].set_xlabel('Valor de Chi-cuadrado')
axs[2].set_ylabel('Frecuencia')
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
ax.set_xticklabels(['STAR como QSO', 'STAR como GALAXY', 'STAR como STAR'])
ax.set_ylabel('Valor de Chi-cuadrado')
ax.set_title('Boxplots para STAR como QSO, GALAXY y STAR')

# Mostrar la gráfica
plt.show()


# Intentemos ver como se comparan los azules, los verdes y los rojos

# In[63]:


# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma para GALAXY como QSO
ax.hist(galaxy_chi_qso, bins=20, alpha=0.5, color='b', label='GALAXY como QSO')
# Histograma para QSO como QSO
ax.hist(qso_chi_qso, bins=20, alpha=0.5, color='r', label='QSO como QSO')
ax.hist(star_chi_qso, bins=20, alpha=0.5, color='g', label='STAR como QSO')

# Configuraciones adicionales
ax.set_xlabel('Valor de Chi-cuadrado')
ax.set_ylabel('Frecuencia')
ax.set_title('Histograma para GALAXY, STAR y QSO como QSO')
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
ax.set_xticklabels(['GALAXY como QSO', 'QSO como QSO', 'STAR como QSO'])
ax.set_ylabel('Valor de Chi-cuadrado')
ax.set_title('Boxplots para GALAXY, STAR, QSO como QSO')

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
ax.set_xticklabels(['GALAXY como QSO', 'QSO como QSO'])
ax.set_ylabel('Valor de Chi-cuadrado')
ax.set_title('Boxplots para GALAXY, QSO como QSO')

# Mostrar la gráfica
plt.show()


# In[67]:


# Crear la figura y el subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma para GALAXY como QSO
ax.hist(galaxy_chi_galaxy, bins=20, alpha=0.5, color='b', label='GALAXY como GALAXY')
# Histograma para QSO como QSO
ax.hist(qso_chi_galaxy, bins=20, alpha=0.5, color='r', label='QSO como GALAXY')
ax.hist(star_chi_galaxy, bins=20, alpha=0.5, color='g', label='STAR como GALAXY')

# Configuraciones adicionales
ax.set_xlabel('Valor de Chi-cuadrado')
ax.set_ylabel('Frecuencia')
ax.set_title('Histograma para GALAXY, STAR y QSO como GALAXY')
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
ax.set_xticklabels(['GALAXY como GALAXY', 'QSO como GALAXY', 'STAR como GALAXY'])
ax.set_ylabel('Valor de Chi-cuadrado')
ax.set_title('Boxplots para GALAXY, QSO y STAR como GALAXY')

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
ax.set_xticklabels(['GALAXY como GALAXY', 'QSO como GALAXY'])
ax.set_ylabel('Valor de Chi-cuadrado')
ax.set_title('Boxplots para GALAXY, QSO como GALAXY')

# Mostrar la gráfica
plt.show()


# In[76]:


# Crear la figura y el subplot para el histograma
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma para GALAXY como STAR
ax.hist(galaxy_chi_star, bins=20, alpha=0.5, color='b', label='GALAXY como STAR')
# Histograma para QSO como STAR
ax.hist(qso_chi_star, bins=20, alpha=0.5, color='r', label='QSO como STAR')
# Histograma para STAR como STAR
ax.hist(star_chi_star, bins=20, alpha=0.5, color='g', label='STAR como STAR')

# Configuraciones adicionales
ax.set_xlabel('Valor de Chi-cuadrado')
ax.set_ylabel('Frecuencia')
ax.set_title('Histograma para GALAXY, QSO y STAR como STAR')
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
ax.set_xticklabels(['GALAXY como STAR', 'QSO como STAR', 'STAR como STAR'])
ax.set_ylabel('Valor de Chi-cuadrado')
ax.set_title('Boxplots para GALAXY, QSO y STAR como STAR')

# Mostrar la gráfica de los boxplots
plt.show()

Ahora hagamos un grafico 3d Donde cada las coordenadas son el chi_star, chi_galaxy, chi_qso
# In[114]:


from mpl_toolkits.mplot3d import Axes3D

# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, galaxy_chi_star, c='b', marker='o', label='Galaxy')
ax.scatter(qso_chi_qso, qso_chi_galaxy, qso_chi_star, c='r', marker='o', label='QSO')
ax.scatter(star_chi_qso, star_chi_galaxy, star_chi_star, c='g', marker='o', label='Star')
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi Galaxy')
ax.set_zlabel('Chi Star')
ax.set_title('Gráfico 3D de valores de chi^2')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()


# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, galaxy_chi_star, c='b', marker='o', label='Galaxy')
ax.scatter(qso_chi_qso, qso_chi_galaxy, qso_chi_star, c='r', marker='o', label='QSO')
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi Galaxy')
ax.set_zlabel('Chi Star')
ax.set_title('Gráfico 3D de valores de chi^2')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()

# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(qso_chi_qso, qso_chi_galaxy, qso_chi_star, c='r', marker='o', label='QSO')
ax.scatter(star_chi_qso, star_chi_galaxy, star_chi_star, c='g', marker='o', label='Star')
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi Galaxy')
ax.set_zlabel('Chi Star')
ax.set_title('Gráfico 3D de valores de chi^2')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()

# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, galaxy_chi_star, c='b', marker='o', label='Galaxy')
ax.scatter(star_chi_qso, star_chi_galaxy, star_chi_star, c='g', marker='o', label='Star')
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi Galaxy')
ax.set_zlabel('Chi Star')
ax.set_title('Gráfico 3D de valores de chi^2')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()


# In[105]:


# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, galaxy_chi_star, c='b', marker='o', label='Galaxy')
ax.scatter(galaxy_chi_qso, galaxy_chi_galaxy, np.zeros_like(galaxy_chi_star), color='gray', alpha=0.3)  # Agregar sombra
ax.scatter(galaxy_chi_qso, np.zeros_like(galaxy_chi_galaxy), galaxy_chi_star, color='gray', alpha=0.3)  # Agregar sombra
ax.scatter(np.zeros_like(galaxy_chi_qso), galaxy_chi_galaxy, galaxy_chi_star, color='gray', alpha=0.3)  # Agregar sombra
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi Galaxy')
ax.set_zlabel('Chi Star')
ax.set_title('Gráfico 3D de valores de chi^2')
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
ax.set_ylabel('Chi Galaxy')
ax.set_zlabel('Chi Star')
ax.set_title('Gráfico 3D de valores de chi^2')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()


# Crear la figura y el subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Graficar los puntos en 3D con sombra
ax.scatter(star_chi_qso, star_chi_galaxy, star_chi_star, c='g', marker='o', label='Star')
ax.scatter(star_chi_qso, star_chi_galaxy, np.zeros_like(star_chi_star), color='gray', alpha=0.1)  # Agregar sombra
ax.scatter(star_chi_qso, np.zeros_like(star_chi_galaxy), star_chi_star, color='gray', alpha=0.1)  # Agregar sombra
ax.scatter(np.zeros_like(star_chi_qso), star_chi_galaxy, star_chi_star, color='gray', alpha=0.1)  # Agregar sombra
# Configuraciones adicionales
ax.set_xlabel('Chi QSO')
ax.set_ylabel('Chi Galaxy')
ax.set_zlabel('Chi Star')
ax.set_title('Gráfico 3D de valores de chi^2')
ax.legend()  # Mostrar la leyenda
# Mostrar el gráfico 3D
plt.show()


# Ahora, veremos si las pendientes pueden contener informacion util

# In[35]:


gal_galqso= np.array(galaxy_chi_galaxy)/np.array(galaxy_chi_qso)
gal_galstar= np.array(galaxy_chi_galaxy)/np.array(galaxy_chi_star)
gal_qsostar= np.array(galaxy_chi_qso)/np.array(galaxy_chi_star)


# In[36]:


qso_galqso= np.array(qso_chi_galaxy)/np.array(qso_chi_qso)
qso_galstar= np.array(qso_chi_galaxy)/np.array(qso_chi_star)
qso_qsostar= np.array(qso_chi_qso)/np.array(qso_chi_star)


# In[37]:


star_galqso= np.array(star_chi_galaxy)/np.array(star_chi_qso)
star_galstar= np.array(star_chi_galaxy)/np.array(star_chi_star)
star_qsostar= np.array(star_chi_qso)/np.array(star_chi_star)


# In[124]:


# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_galqso
plt.hist(gal_galqso, bins=20, color='b', alpha=0.5, label='GAL: GALAXY / QSO')
# Histograma para qso_galqso
plt.hist(qso_galqso, bins=20, color='r', alpha=0.5, label='QSO: GALAXY / QSO')
# Histograma para star_galqso
plt.hist(star_galqso, bins=20, color='g', alpha=0.5, label='STAR: GALAXY / QSO')
# Configuraciones adicionales
plt.title('Histogramas de GALAXY/ QSO')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()


# In[126]:


# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_galstar
plt.hist(gal_galstar, bins=20, color='b', alpha=0.5, label='GAL: GALAXY / STAR')
# Histograma para qso_galstar
plt.hist(qso_galstar, bins=20, color='r', alpha=0.5, label='QSO: GALAXY / STAR')
# Histograma para star_galstar
plt.hist(star_galstar, bins=20, color='g', alpha=0.5, label='STAR: GALAXY / STAR')
# Configuraciones adicionales
plt.title('Histogramas de GALAXY/ STAR')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()

# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_galstar
plt.hist(gal_galstar, bins=20, color='b', alpha=0.5, label='GAL: GALAXY / STAR')
# Histograma para qso_galstar
plt.hist(qso_galstar, bins=20, color='r', alpha=0.5, label='QSO: GALAXY / STAR')
plt.title('Histogramas de GALAXY/ STAR')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()


# In[128]:


# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_qsostar
plt.hist(gal_qsostar, bins=20, color='b', alpha=0.5, label='GAL: QSO / STAR')
# Histograma para qso_qsostar
plt.hist(qso_qsostar, bins=20, color='r', alpha=0.5, label='QSO: QSO / STAR')
# Histograma para star_qsostar
plt.hist(star_qsostar, bins=20, color='g', alpha=0.5, label='STAR: QSO / STAR')
# Configuraciones adicionales
plt.title('Histogramas de QSO / STAR')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()

# Crear la figura
plt.figure(figsize=(10, 6))
# Histograma para gal_qsostar
plt.hist(gal_qsostar, bins=20, color='b', alpha=0.5, label='GAL: QSO / STAR')
# Histograma para qso_qsostar
plt.hist(qso_qsostar, bins=20, color='r', alpha=0.5, label='QSO: QSO / STAR')
# Configuraciones adicionales
plt.title('Histogramas de QSO / STAR')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.yscale('log')  # Escala logarítmica en el eje y
# Mostrar el histograma
plt.show()


# ### Parametro Abbe
spectra_galaxy
spectra_qso
spectra_star
# In[38]:


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


# In[118]:


# Datos para los boxplots (sustituye estos datos por los tuyos)
data = [abbe_gal, abbe_star, abbe_qso]

# Etiquetas para los boxplots
labels = ['ABBE GALAXY', 'ABBE STAR', 'ABBE QSO']

# Crear la figura y los ejes de la figura
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Crear el boxplot con outliers
axs[0].boxplot(data)
axs[0].set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
axs[0].set_title('Boxplot de ABBE con Outliers')
axs[0].set_xlabel('Tipo de Objeto')
axs[0].set_ylabel('ABBE')

# Crear el boxplot sin outliers
axs[1].boxplot(data, showfliers=False)
axs[1].set_xticklabels(['Galaxia', 'Estrella', 'QSO'])
axs[1].set_title('Boxplot de ABBE sin Outliers')
axs[1].set_xlabel('Tipo de Objeto')
axs[1].set_ylabel('ABBE')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[119]:


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
axs[0].set_title('Histogramas de ABBE por Clase')
axs[0].set_xlabel('ABBE')
axs[0].set_ylabel('Frecuencia')

# Iterar sobre cada conjunto de datos y crear un histograma para cada clase en la segunda columna
for i, class_data in enumerate(data):
    axs[1].hist(class_data, bins=10, color=colors[i], alpha=0.7, label=labels[i])

# Agregar leyenda y título a la segunda columna
axs[1].legend()
axs[1].set_title('Histogramas de ABBE por Clase (Escala Logarítmica)')
axs[1].set_xlabel('ABBE')
axs[1].set_ylabel('Frecuencia')

# Establecer escala logarítmica en el eje y para la segunda columna
axs[1].set_yscale('log')

# Ajustar el diseño
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[ ]:





# ### Ahora, Intentaremos con la FFT

# In[7]:


espc = fits.open('DataDESI_691_752.fits') #open file
wave= fits.open('B_R_Z_wavelenght.fits')
espc.info() #resume el contenido de la tabla
wave.info()


# In[8]:


#Leemos la info del archivo .fits
flux_b=espc[2].data
flux_r=espc[3].data
flux_z=espc[4].data

Bwave = wave[1].data
Rwave = wave[2].data
Zwave = wave[3].data

#Seleccionamos un espectro con indice i (QSO=-43, GALAXY=-3, STAR=-4)
i=-4

espectro_b=flux_b[i]
espectro_r=flux_r[i]
espectro_z=flux_z[i]

targetid = espc[1].data['TARGETID'][i]
formatted_targetid = str(int(targetid))
print('TARGETID:', formatted_targetid)

spectype= espc[1].data['SPECTYPE'][i]
print('SPECTYPE:', spectype)

espectro=np.hstack((espectro_b, espectro_r, espectro_z))
wavelenght = np.hstack((Bwave, Rwave, Zwave)) #Contiene la cadena completa de longitudes de onda B+Z+R para cada espectro


# In[9]:


plt.figure()
plt.plot(wavelenght, espectro)
plt.show()


# In[10]:


f_fund=[]
stype=[]

# Realiza la FFT
fft_result = np.fft.fft(espectro)
# Obtén la longitud del espectro
n = len(espectro)
# Obtén las frecuencias correspondientes solo para la mitad del espectro
frequencies = np.fft.fftfreq(n, d=(wavelenght[1] - wavelenght[0]))
# Obtén solo la mitad de la FFT (ignorando el reflejo conjugado)
fft_result = fft_result[:n//2]
frequencies = frequencies[:n//2]

# Obtenemos un FFT filtrado
threshold = 0  # Ajusta el umbral según tus necesidades, positivo
fft_result_filtered = fft_result * (np.abs(fft_result) > threshold)

# Encuentra la posición de la frecuencia más grande
indice_frecuencia_maxima = np.argmax(np.abs(fft_result_filtered))
# Obtén la frecuencia correspondiente
frecuencia_maxima = fft_result_filtered[indice_frecuencia_maxima]

# Guardar la frecuencia en la lista f_fund
f_fund.append(frecuencia_maxima)

# Aplicar la FFT inversa a todas las frecuencias
filtered_spectrum = np.fft.ifft(np.concatenate((fft_result_filtered, np.conj(fft_result_filtered[::-1]))))

# Construimos un nuevo arreglo de frecuencias, donde todas son cero, menos la mas grande
fft_inversa = np.zeros_like(fft_result)
fft_inversa[indice_frecuencia_maxima] = frecuencia_maxima
# Aplicar la FFT inversa completa
filtered_spectrum_fund = np.fft.ifft(np.concatenate((fft_inversa, np.conj(fft_inversa[::-1]))))


# In[11]:


# Crear una figura con subgráficos de tres filas y dos columnas
fig, axs = plt.subplots(4, 2, figsize=(15, 15))
# Primer conjunto de subgráficos
# Subgráfico 1
axs[0, 0].plot(frequencies, fft_result, label='FFT del Espectro', color='green')
axs[0, 0].set_xlabel('Frecuencia (1/longitud de onda)')
axs[0, 0].set_ylabel('Amplitud')
axs[0, 0].set_title('FFT del Espectro')
axs[0, 0].legend()
axs[0, 0].grid(True)
# Subgráfico 2
axs[0, 1].plot(frequencies, np.abs(fft_result), label='FFT del Espectro con abs', color='purple')
axs[0, 1].set_xlabel('Frecuencia (1/longitud de onda)')
axs[0, 1].set_ylabel('Amplitud')
axs[0, 1].set_title('FFT del Espectro con abs')
axs[0, 1].legend()
axs[0, 1].grid(True)
# Segundo conjunto de subgráficos
# Subgráfico 1
axs[1, 0].plot(frequencies, fft_result_filtered, label='FFT filtrado', color='green')
axs[1, 0].set_xlabel('Frecuencia (1/longitud de onda)')
axs[1, 0].set_ylabel('Amplitud')
axs[1, 0].set_title('FFT filtrado')
axs[1, 0].legend()
axs[1, 0].grid(True)
# Subgráfico 2
axs[1, 1].plot(frequencies, np.abs(fft_result_filtered), label='FFT filtrado con abs', color='purple')
axs[1, 1].set_xlabel('Frecuencia (1/longitud de onda)')
axs[1, 1].set_ylabel('Amplitud')
axs[1, 1].set_title('FFT filtrado con abs')
axs[1, 1].legend()
axs[1, 1].grid(True)
# Tercer conjunto de subgráficos
# Subgráfico 1
axs[2, 0].plot(wavelenght, espectro, label='Espectro original', color='black')
axs[2, 0].set_xlabel('Wavelength')
axs[2, 0].set_ylabel('Flujo')
axs[2, 0].set_title('Espectro original')
axs[2, 0].legend()
axs[2, 0].grid(True)
# Subgráfico 2
axs[2, 1].plot(wavelenght, filtered_spectrum, label='Espectro filtrado', color='black')
axs[2, 1].set_xlabel('Wavelength')
axs[2, 1].set_ylabel('Flujo')
axs[2, 1].set_title('IFFT del espectro filtrado')
axs[2, 1].legend()
axs[2, 1].grid(True)
# Cuarto conjunto de subgráficos
# Subgráfico 1
axs[3, 0].plot(filtered_spectrum_fund, label='IFFT fundamental', color='green')
axs[3, 0].set_xlabel('Wavelength')
axs[3, 0].set_ylabel('Flujo')
axs[3, 0].set_title('IFFT de la frecuencia fundamental')
axs[3, 0].legend()
axs[3, 0].grid(True)
# Subgráfico 2
axs[3, 1].plot(wavelenght, filtered_spectrum_fund, label='IFFT fundamental', color='purple')
axs[3, 1].set_xlabel('Wavelength')
axs[3, 1].set_ylabel('Flujo')
axs[3, 1].set_title('IFFT de la frecuencia fundamental')
axs[3, 1].legend()
axs[3, 1].grid(True)
# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar la gráfica
plt.show()


# ### Encontraremos la frecuencia fundamental para muchos mas espectros

# Utilizaremos los espectros estandarizados, ejecutando el codigo de antes.

# In[7]:


stype= Table['SPECTYPE'].data


# In[8]:


wavelenght


# In[9]:


spectra


# In[10]:


f_fund=[]

for i in spectra:
    espectro=i
    # Realiza la FFT
    fft_result = np.fft.fft(espectro)
    # Obtén la longitud del espectro
    n = len(espectro)
    # Obtén las frecuencias correspondientes solo para la mitad del espectro
    frequencies = np.fft.fftfreq(n, d=(wavelenght[1] - wavelenght[0]))
    # Obtén solo la mitad de la FFT (ignorando el reflejo conjugado)
    fft_result = fft_result[:n//2]
    frequencies = frequencies[:n//2]
    
    # Obtenemos un FFT filtrado
    threshold = 0  # Ajusta el umbral según tus necesidades, positivo
    fft_result_filtered = fft_result * (np.abs(fft_result) > threshold)
    
    # Encuentra la posición de la frecuencia más grande
    indice_frecuencia_maxima = np.argmax(np.abs(fft_result_filtered))
    # Obtén la frecuencia correspondiente
    frecuencia_maxima = np.abs(fft_result_filtered[indice_frecuencia_maxima])
    
    # Guardar la frecuencia en la lista f_fund
    f_fund.append(frecuencia_maxima)


# In[11]:


f_fund= np.array(np.real(f_fund))


# In[12]:


f_fund


# In[13]:


stype


# In[14]:


indices_galaxy = np.where(stype == 'GALAXY')[0]
indices_star = np.where(stype == 'STAR')[0]
indices_qso = np.where(stype == 'QSO')[0]

# Filtrar las frecuencias correspondientes a 'GALAXY'
f_fund_galaxy = f_fund[indices_galaxy]
f_fund_star = f_fund[indices_star]
f_fund_qso = f_fund[indices_qso]


# In[15]:


len(f_fund_galaxy)+len(f_fund_star)+len(f_fund_qso)


# In[16]:


# Crear una lista con todas las frecuencias
all_frequencies = [f_fund_galaxy, f_fund_star, f_fund_qso]

# Crear una figura con una fila y dos columnas
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Boxplot con outliers
axs[0].boxplot(all_frequencies, labels=['GALAXY', 'STAR', 'QSO'])
axs[0].set_xlabel('Tipo Espectral')
axs[0].set_ylabel('Frecuencias')
axs[0].set_title('Boxplot de Frecuencias por Tipo Espectral (Con Outliers)')

# Boxplot sin outliers
axs[1].boxplot(all_frequencies, labels=['GALAXY', 'STAR', 'QSO'], showfliers=False)
axs[1].set_xlabel('Tipo Espectral')
axs[1].set_ylabel('Frecuencias')
axs[1].set_title('Boxplot de Frecuencias por Tipo Espectral (Sin Outliers)')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# In[17]:


# Crear una lista con todas las frecuencias
all_frequencies = [f_fund_galaxy, f_fund_star, f_fund_qso]

# Crear una figura con una fila y tres columnas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para GALAXY
axs[0].hist(f_fund_galaxy, bins=20, color='blue', alpha=0.7)
axs[0].set_xlabel('Frecuencias')
axs[0].set_ylabel('Frecuencia Relativa')
axs[0].set_title('Histograma de Frecuencias - GALAXY')

# Histograma para STAR
axs[1].hist(f_fund_star, bins=20, color='green', alpha=0.7)
axs[1].set_xlabel('Frecuencias')
axs[1].set_ylabel('Frecuencia Relativa')
axs[1].set_title('Histograma de Frecuencias - STAR')

# Histograma para QSO
axs[2].hist(f_fund_qso, bins=20, color='red', alpha=0.7)
axs[2].set_xlabel('Frecuencias')
axs[2].set_ylabel('Frecuencia Relativa')
axs[2].set_title('Histograma de Frecuencias - QSO')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# In[18]:


# Crear una lista con todas las frecuencias
all_frequencies = [f_fund_galaxy, f_fund_star, f_fund_qso]

# Crear una figura con una fila y tres columnas
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histograma para GALAXY
axs[0].hist(f_fund_galaxy, bins=20, color='blue', alpha=0.7)
axs[0].set_yscale('log')  # Escala logarítmica en el eje y
axs[0].set_xlabel('Frecuencias')
axs[0].set_ylabel('Frecuencia (log)')
axs[0].set_title('Histograma de Frecuencias - GALAXY')

# Histograma para STAR
axs[1].hist(f_fund_star, bins=20, color='green', alpha=0.7)
axs[1].set_yscale('log')  # Escala logarítmica en el eje y
axs[1].set_xlabel('Frecuencias')
axs[1].set_ylabel('Frecuencia (log)')
axs[1].set_title('Histograma de Frecuencias - STAR')

# Histograma para QSO
axs[2].hist(f_fund_qso, bins=20, color='red', alpha=0.7)
axs[2].set_yscale('log')  # Escala logarítmica en el eje y
axs[2].set_xlabel('Frecuencias')
axs[2].set_ylabel('Frecuencia (log)')
axs[2].set_title('Histograma de Frecuencias - QSO')

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show()


# In[22]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import Normalize

# Crear una figura tridimensional
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Definir los cortes para cada eje
x_bins = np.linspace(min(f_fund_star), max(f_fund_star), 10)  # Cambia según tus necesidades
y_bins = np.linspace(min(f_fund_galaxy), max(f_fund_galaxy), 15)  # Cambia según tus necesidades

# Contar cuántas estrellas y galaxias caen en cada región
hist, x_edges, y_edges = np.histogram2d(f_fund_star, f_fund_galaxy, bins=[x_bins, y_bins])
hist = hist.astype(int)  # Convertir a enteros

# Obtener las coordenadas para las barras 3D
xpos, ypos = np.meshgrid(x_edges[:-1] + np.diff(x_edges)/2, y_edges[:-1] + np.diff(y_edges)/2, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

# Tamaños de las barras en x y y
dx = np.diff(x_edges)[0]
dy = np.diff(y_edges)[0]
dz = hist.ravel()

# Normalizar la escala de color entre los valores mínimos y máximos de dz
norm = Normalize(vmin=min(dz), vmax=max(dz))

# Crear el gráfico de barras 3D con gradiente de color
bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, cmap='viridis', norm=norm)

# Agregar una barra de color (colorbar) con leyenda
cbar = fig.colorbar(bars, ax=ax, pad=0.1)
cbar.set_label('Densidad de Objetos')  # Agregar leyenda

# Configuración de etiquetas y título
ax.set_xlabel('Frecuencias - STAR')
ax.set_ylabel('Frecuencias - GALAXY')
ax.set_zlabel('Cantidad de Objetos')
ax.set_title('Histograma 3D con Grillas Irregulares y Gradiente de Color Ajustado')

# Mostrar el gráfico
plt.show()


# ### Analisis ANOVA

# In[26]:


# Listas de frecuencias para cada clase
data_for_anova = [f_fund_galaxy, f_fund_star, f_fund_qso]

# Realizar ANOVA para cada clase
for i, clase in enumerate(['GALAXY', 'STAR', 'QSO']):
    # Realizar ANOVA
    anova_result = f_oneway(*data_for_anova)

    # Imprimir el resultado del ANOVA para la clase actual
    print(f"Resultado del ANOVA para la clase {clase}:")
    print(anova_result)

    # Verificar la significancia estadística
    if anova_result.pvalue < 0.05:
        print("La diferencia entre al menos dos grupos es estadísticamente significativa.")
    else:
        print("No hay evidencia suficiente para rechazar la hipótesis nula de igualdad de medias.")

    print("-" * 50)  # Separador entre resultados de diferentes clases


# In[27]:


import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Listas de frecuencias para cada clase
data_for_anova = [f_fund_galaxy, f_fund_star, f_fund_qso]

# Obtener la longitud mínima de los grupos
min_group_length = min(len(group) for group in data_for_anova)

# Concatenar los datos para la prueba de Tukey
all_data = np.concatenate([group[:min_group_length] for group in data_for_anova])

# Crear las etiquetas de grupo para la prueba de Tukey
group_labels = np.repeat(['GALAXY', 'STAR', 'QSO'], min_group_length)

# Realizar ANOVA para cada clase
for i, clase in enumerate(['GALAXY', 'STAR', 'QSO']):
    # Realizar ANOVA
    anova_result = f_oneway(*data_for_anova)

    # Imprimir el resultado del ANOVA para la clase actual
    print(f"Resultado del ANOVA para la clase {clase}:")
    print(anova_result)

    # Realizar prueba de Tukey para comparaciones múltiples
    tukey_result = pairwise_tukeyhsd(all_data, group_labels)

    # Imprimir los resultados de la prueba de Tukey
    print("Resultados de la prueba de Tukey:")
    print(tukey_result)

    # Verificar si los grupos son estadísticamente distintos
    if any(tukey_result.reject):
        print("Al menos un par de grupos es estadísticamente distinto.")
    else:
        print("No hay evidencia suficiente para rechazar la hipótesis nula de igualdad de medias.")

    print("-" * 50)  # Separador entre resultados de diferentes clases


# ### Kruskal-Wallis

# In[28]:


from scipy.stats import kruskal

# Listas de frecuencias para cada clase
data_for_kruskal = [f_fund_galaxy, f_fund_star, f_fund_qso]

# Realizar Kruskal-Wallis para cada clase
for i, clase in enumerate(['GALAXY', 'STAR', 'QSO']):
    # Realizar Kruskal-Wallis
    kruskal_result = kruskal(*data_for_kruskal)

    # Imprimir el resultado del Kruskal-Wallis para la clase actual
    print(f"Resultado de Kruskal-Wallis para la clase {clase}:")
    print(kruskal_result)

    # Verificar la significancia estadística
    if kruskal_result.pvalue < 0.05:
        print("La diferencia entre al menos dos grupos es estadísticamente significativa.")
    else:
        print("No hay evidencia suficiente para rechazar la hipótesis nula de igualdad de medias.")

    print("-" * 50)  # Separador entre resultados de diferentes clases


# In[ ]:





# ### H2O con FFT 

# In[21]:


# Inicializa el cluster de H2O y mostrar info del cluster en uso
h2o.init()


# In[22]:


STYPE=[]
for i in range(len(f_fund_galaxy)):
    STYPE.append('GALAXY')
for j in range(len(f_fund_star)):
    STYPE.append('STAR')
for k in range(len(f_fund_qso)):
    STYPE.append('QSO')
STYPE= np.array(STYPE)

FREQ= np.concatenate([f_fund_galaxy,f_fund_star,f_fund_qso])

#Creamos el H2O frame con una columna
datos_f = h2o.H2OFrame(python_obj=STYPE, column_names=['SPECTYPE'], column_types=["string"])

#Agregamos mas columnas al frame que hemos creado
f_h2o= h2o.H2OFrame(python_obj=FREQ, column_names=['FUNDAMENTAL_f'], column_types=["float"])
datos_f= datos_f.cbind(f_h2o)


# In[23]:


datos_f


# In[24]:


# Convierte la variable objetivo a factor utilizando la función asfactor()
datos_f['SPECTYPE'] = datos_f['SPECTYPE'].asfactor()

# Define las columnas predictoras y la variable objetivo
predictores = datos_f.columns[1:]  # Todas las columnas excepto la primera (SPECTYPE)
objetivo='SPECTYPE'


# In[25]:


train, test = datos_f.split_frame(ratios=[0.6], seed=42)


# In[26]:


train['SPECTYPE'].table()


# In[27]:


test['SPECTYPE'].table()


# ### Red neuronal con H20

# In[28]:


# Configura y entrena el modelo de red neuronal
modelo_nn = H2ODeepLearningEstimator(epochs=1000, hidden=[64, 128, 256, 256], distribution="multinomial", activation="RectifierWithDropout", variable_importances=True)
#modelo_nn = H2ODeepLearningEstimator(epochs=10, hidden=[10, 10], distribution="multinomial", activation="RectifierWithDropout", variable_importances=True)
modelo_nn.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprime métricas de rendimiento en el conjunto de prueba
print(modelo_nn.model_performance(test_data=test))

# Obtener las importancias de las variables
importancias_variables = modelo_nn.varimp()
# Imprimir las importancias de las variables
print(importancias_variables)


# In[29]:


# Graficar el Loss en cada época de entrenamiento y validación
training_metrics = modelo_nn.score_history()
plt.plot(training_metrics['epochs'], training_metrics['training_classification_error'], label='Training Error')
plt.plot(training_metrics['epochs'], training_metrics['validation_classification_error'], label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()


# ### Random forest con H2O

# In[30]:


# Configura y entrena el modelo de Random Forest
modelo_rf = H2ORandomForestEstimator(ntrees=200, max_depth=20, seed=42)
modelo_rf.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprime métricas de rendimiento en el conjunto de prueba
print(modelo_rf.model_performance(test_data=test))

# Obtener las importancias de las variables
importancias_variables_rf = modelo_rf.varimp()
# Imprimir las importancias de las variables
print(importancias_variables_rf)


# In[31]:


# Graficar las métricas de rendimiento en cada árbol
rf_metrics = modelo_rf.score_history()
plt.plot(rf_metrics['number_of_trees'], rf_metrics['training_classification_error'], label='Training Error')
plt.plot(rf_metrics['number_of_trees'], rf_metrics['validation_classification_error'], label='Validation Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.legend()
plt.show()


# ### Seleccion de la mejor familia con H2O

# In[32]:


# Configura y ejecuta la búsqueda automática de modelos
automl = H2OAutoML(max_models=10, seed=42)
automl.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprime métricas de rendimiento en el conjunto de prueba para el mejor modelo
print(automl.leader.model_performance(test_data=test))


# In[33]:


# Obtiene y muestra el modelo líder
best_model = automl.leader
print("Mejor modelo:")
print(best_model)

# Imprime métricas de rendimiento en el conjunto de prueba para el mejor modelo
print("Métricas del mejor modelo en el conjunto de prueba:")
print(best_model.model_performance(test_data=test))


# In[34]:


# Cierra el cluster de H2O
h2o.cluster().shutdown()


# ### Graficaremos espectros de las tres clases principales en distintos valores de corrimiento al rojo

# ### Galaxias

# In[9]:


Z=Table['Z'].data

mascara_gal = Table['SPECTYPE'] == 'GALAXY'
just_galaxy = Table[mascara_gal]
spectra_galaxy = spectra[mascara_gal]
z_galaxy = Z[mascara_gal]

mascara_qso = Table['SPECTYPE'] == 'QSO'
just_qso = Table[mascara_qso]
spectra_qso = spectra[mascara_qso]
z_qso = Z[mascara_qso]

mascara_star = Table['SPECTYPE'] == 'STAR'
just_star = Table[mascara_star]
spectra_star = spectra[mascara_star]
z_star = Z[mascara_star]


# In[136]:


# Obtener el rango de redshifts
z_min = np.min(z_galaxy)
z_max = np.max(z_galaxy)

# Seleccionar 10 valores distintos dentro del rango
step = (z_max - z_min) / 10
redshifts = np.arange(z_min, z_max, step)

# Crear la figura
plt.figure(figsize=(10, 6))

# Graficar spectra_galaxy vs wavelength para cada valor de redshift
for z in redshifts:
    # Encontrar el índice correspondiente en z_galaxy
    idx = np.where((z_galaxy >= z) & (z_galaxy < z + step))[0][0]
    
    # Obtener el espectro correspondiente
    spectra_gal_z = spectra_galaxy[idx]
    
    # Graficar
    plt.plot(wavelenght, spectra_gal_z, label=f'Redshift: {z:.2f}')

# Configuraciones adicionales
plt.xlabel('Wavelength')
plt.ylabel('Spectra Galaxy')
plt.title('Spectra Galaxy vs Wavelength para distintos redshifts')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()


# In[137]:


# Obtener el rango de redshifts
z_min = np.min(z_galaxy)
z_max = np.max(z_galaxy)

# Seleccionar 10 valores distintos dentro del rango
step = (z_max - z_min) / 10
redshifts = np.arange(z_min, z_max, step)

# Crear la figura con subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 10))

# Iterar sobre los valores de redshifts y graficar en los subplots
for i, z in enumerate(redshifts):
    # Encontrar el índice correspondiente en z_galaxy
    idx = np.where((z_galaxy >= z) & (z_galaxy < z + step))[0][0]
    
    # Obtener el espectro correspondiente
    spectra_gal_z = spectra_galaxy[idx]
    
    # Obtener las coordenadas del subplot
    row = i // 5
    col = i % 5
    
    # Graficar en el subplot correspondiente
    axs[row, col].plot(wavelenght, spectra_gal_z)
    axs[row, col].set_title(f'Redshift: {z:.2f}')
    axs[row, col].set_xlabel('Wavelength')
    axs[row, col].set_ylabel('Spectra Galaxy')
    axs[row, col].grid(True)

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# ### QSO

# In[138]:


# Obtener el rango de redshifts
z_min = np.min(z_qso)
z_max = np.max(z_qso)

# Seleccionar 10 valores distintos dentro del rango
step = (z_max - z_min) / 10
redshifts = np.arange(z_min, z_max, step)

# Crear la figura
plt.figure(figsize=(10, 6))

# Graficar spectra_galaxy vs wavelength para cada valor de redshift
for z in redshifts:
    # Encontrar el índice correspondiente en z_galaxy
    idx = np.where((z_qso >= z) & (z_qso < z + step))[0][0]
    
    # Obtener el espectro correspondiente
    spectra_qso_z = spectra_qso[idx]
    
    # Graficar
    plt.plot(wavelenght, spectra_qso_z, label=f'Redshift: {z:.2f}')

# Configuraciones adicionales
plt.xlabel('Wavelength')
plt.ylabel('Spectra qso')
plt.title('Spectra qso vs Wavelength para distintos redshifts')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()


# In[10]:


# Obtener el rango de redshifts
z_min = np.min(z_qso)
z_max = np.max(z_qso)

# Seleccionar 10 valores distintos dentro del rango
step = (z_max - z_min) / 10
redshifts = np.arange(z_min, z_max, step)

# Crear la figura con subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 10))

# Iterar sobre los valores de redshifts y graficar en los subplots
for i, z in enumerate(redshifts):
    # Encontrar el índice correspondiente en z_galaxy
    idx = np.where((z_qso >= z) & (z_qso < z + step))[0][0]
    
    # Obtener el espectro correspondiente
    spectra_qso_z = spectra_qso[idx]
    
    # Obtener las coordenadas del subplot
    row = i // 5
    col = i % 5
    
    # Graficar en el subplot correspondiente
    axs[row, col].plot(wavelenght, spectra_qso_z)
    axs[row, col].set_title(f'Redshift: {z:.2f}')
    axs[row, col].set_xlabel('Wavelength')
    axs[row, col].set_ylabel('Spectra qso')
    axs[row, col].grid(True)
    print(idx)

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# In[12]:


spectra_qso_z = spectra_qso[8]


# ### STAR

# In[141]:


# Obtener el rango de redshifts
z_min = np.min(z_star)
z_max = np.max(z_star)

# Seleccionar 10 valores distintos dentro del rango
step = (z_max - z_min) / 10
redshifts = np.arange(z_min, z_max, step)

# Crear la figura
plt.figure(figsize=(10, 6))

# Graficar spectra_galaxy vs wavelength para cada valor de redshift
for z in redshifts:
    # Encontrar el índice correspondiente en z_galaxy
    idx = np.where((z_star >= z) & (z_star < z + step))[0][0]
    
    # Obtener el espectro correspondiente
    spectra_star_z = spectra_star[idx]
    
    # Graficar
    plt.plot(wavelenght, spectra_star_z, label=f'Redshift: {z:.2f}')

# Configuraciones adicionales
plt.xlabel('Wavelength')
plt.ylabel('Spectra star')
plt.title('Spectra star vs Wavelength para distintos redshifts')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()


# In[142]:


# Obtener el rango de redshifts
z_min = np.min(z_star)
z_max = np.max(z_star)

# Seleccionar 10 valores distintos dentro del rango
step = (z_max - z_min) / 10
redshifts = np.arange(z_min, z_max, step)

# Crear la figura con subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 10))

# Iterar sobre los valores de redshifts y graficar en los subplots
for i, z in enumerate(redshifts):
    # Encontrar el índice correspondiente en z_galaxy
    idx = np.where((z_star >= z) & (z_star < z + step))[0][0]
    
    # Obtener el espectro correspondiente
    spectra_star_z = spectra_star[idx]
    
    # Obtener las coordenadas del subplot
    row = i // 5
    col = i % 5
    
    # Graficar en el subplot correspondiente
    axs[row, col].plot(wavelenght, spectra_star_z)
    axs[row, col].set_title(f'Redshift: {z:.2f}')
    axs[row, col].set_xlabel('Wavelength')
    axs[row, col].set_ylabel('Spectra qso')
    axs[row, col].grid(True)

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# ### Diagrama de magnitud color

# In[65]:


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


# In[69]:


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


# In[74]:


#Lista de todos los índices de color
indices_color = ['U_COLOR', 'G_COLOR', 'R_COLOR', 'I_COLOR', 'Z_COLOR']

# Lista de todos los filtros
colores = ['U_R_COLOR', 'U_G_COLOR', 'G_Z_COLOR', 'R_Z_COLOR', 'I_Z_COLOR']

# Generar todas las combinaciones posibles entre índices de color y filtros
combinaciones = list(itertools.product(indices_color, colores))

# Crear una figura con subgráficos para cada combinación de índices de color y filtros
fig, axs = plt.subplots(len(combinaciones), 1, figsize=(10, 5*len(combinaciones)))

# Iterar sobre todas las combinaciones y crear los gráficos correspondientes
for i, (indice_color, filtro) in enumerate(combinaciones):
    # Scatter Plot para la combinación actual
    axs[i].scatter(eval(filtro + '_GALAXY'), eval(indice_color + '_GALAXY'), label='GALAXY', color='blue', alpha=0.6)
    axs[i].scatter(eval(filtro + '_STAR'), eval(indice_color + '_STAR'), label='STAR', color='green', alpha=0.6)
    axs[i].scatter(eval(filtro + '_QSO'), eval(indice_color + '_QSO'), label='QSO', color='red', alpha=0.6)
    axs[i].set_xlabel(filtro)
    axs[i].set_ylabel(indice_color)
    axs[i].set_title(f'{indice_color} vs {filtro}')
    axs[i].legend()
    # Invertir el eje y
    axs[i].invert_yaxis()

# Ajustar el diseño de los subgráficos
plt.tight_layout()
# Mostrar el gráfico
plt.show()


# In[80]:


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
# In[8]:


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

# In[39]:


# Establecer la semilla
random.seed(42)  # Puedes cambiar este número por cualquier otro valor entero

# Generar números aleatorios
aleatorio = [random.random() for _ in spectra]
aleatorio = np.array(aleatorio)


# In[13]:


np.unique(aleatorio)


# ### Construimos el modelo con las 55 features+ la aleatoria

# In[40]:


# Inicializa el cluster de H2O y mostrar info del cluster en uso
h2o.init()


# In[41]:


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


# In[42]:


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


# In[43]:


datos


# In[44]:


# Convierte la variable objetivo a factor utilizando la función asfactor()
datos['SPECTYPE'] = datos['SPECTYPE'].asfactor()

# Define las columnas predictoras y la variable objetivo
predictores = datos.columns[1:]  # Todas las columnas excepto la primera (SPECTYPE)
objetivo='SPECTYPE'


# In[45]:


# Dividir el conjunto de datos en entrenamiento y prueba de manera estratificada
#train, test = datos.split_frame(ratios=[0.6], seed=42, destination_frames=['train', 'test'], stratify='SPECTYPE')
train, test = datos.split_frame(ratios=[0.6], seed=42)


# In[46]:


train['SPECTYPE'].table()


# In[47]:


test['SPECTYPE'].table()


# ### Red neuronal

# In[84]:


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


# In[85]:


# Graficar el Loss en cada época de entrenamiento y validación
training_metrics = modelo_nn.score_history()
plt.plot(training_metrics['epochs'], training_metrics['training_classification_error'], label='Training Error')
plt.plot(training_metrics['epochs'], training_metrics['validation_classification_error'], label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()


# ### Random forest

# In[53]:


# Configura y entrena el modelo de Random Forest
modelo_rf = H2ORandomForestEstimator(ntrees=200, max_depth=20, seed=42)
modelo_rf.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprime métricas de rendimiento en el conjunto de prueba
print(modelo_rf.model_performance(test_data=test))

# Obtener las importancias de las variables
importancias_variables_rf = modelo_rf.varimp(True)
print("\nImportancias relativas de las características de entrenamiento:")
print(importancias_variables_rf)


# In[54]:


# Graficar las métricas de rendimiento en cada árbol
rf_metrics = modelo_rf.score_history()
plt.plot(rf_metrics['number_of_trees'], rf_metrics['training_classification_error'], label='Training Error')
plt.plot(rf_metrics['number_of_trees'], rf_metrics['validation_classification_error'], label='Validation Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.legend()
plt.show()


# ### Mejor familia

# In[55]:


# Configurar y ejecutar la búsqueda automática de modelos
exclude_algos = ["DeepLearning", "StackedEnsemble"]  # Excluir redes neuronales y ensambles
automl = H2OAutoML(max_models=10, seed=42, exclude_algos=exclude_algos)
automl.train(x=predictores, y=objetivo, training_frame=train, validation_frame=test)

# Imprimir métricas de rendimiento en el conjunto de prueba para el mejor modelo
print(automl.leader.model_performance(test_data=test))


# In[58]:


# Obtiene y muestra el modelo líder
best_model = automl.leader
print("Mejor modelo:")
print(best_model)

# Imprime métricas de rendimiento en el conjunto de prueba para el mejor modelo
print("Métricas del mejor modelo en el conjunto de prueba:")
print(best_model.model_performance(test_data=test))


# In[59]:


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

# In[62]:


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


# In[63]:


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




