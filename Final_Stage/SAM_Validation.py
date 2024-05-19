#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('programa iniciado: SAM')


# In[1]:


print('1')
import numpy as np
print('2')
import sklearn
print('3')
import tensorflow
print('4')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import BatchNormalization
print('5')
from astropy.io import fits
print('5.6')
import pandas as pd
print('5.7')
import matplotlib.pyplot as plt
print('5.8')
import astropy

print('6')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import class_weight
import seaborn as sns
from sklearn.model_selection import GridSearchCV
print('7')
from astropy.io import fits
#from astropy.table import Table
print('8')
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.models import load_model


# In[2]:


wave= fits.open('B_R_Z_wavelenght.fits')
Bwave = wave[1].data
Rwave = wave[2].data
Zwave = wave[3].data
wavelenght = np.hstack((Bwave, Rwave, Zwave)) #Contiene la cadena completa de longitudes de onda B+Z+R para cada espectro

# Definir una máscara booleana para las longitudes de onda del flujo en U
#mascara = (wavelenght >= 3055.11) & (wavelenght <= 4030.64)
#wavelenght_filtrado= wavelenght[mascara]

archivos= ['DataDESI_76.fits', 'DataDESI_152.fits', 'DataDESI_213.fits', 'DataDESI_284.fits', 'DataDESI_351.fits',
          'DataDESI_438.fits' , 'DataDESI_530.fits', 'DataDESI_606.fits', 'DataDESI_690.fits', 'DataDESI_752.fits']
#archivos= ['DataDESI_752.fits']

#Generamos las listas con los datos:
Spectra_set = None #Este tensor contiene los elementos de flujo completo R+Z+B
spectype = np.array([])  # Esta lista contiene las etiquetas para el ejercicio de clasificación
z = np.array([]) #Esta matriz contiene los corrimientos z para el ejercicio de regresion

for h in range(len(archivos)):
    espc = fits.open(archivos[h]) #open file
    len_espc= len(espc[2].data)
    
    #leemos la informacion
    Bflux= espc[2].data
    Rflux= espc[3].data
    Zflux= espc[4].data
    
    spectra=np.hstack((Bflux, Rflux, Zflux)) #Contiene la cadena completa de flujo B+Z+R para cada espectro
    # Aplicar la máscara a la matriz spectra
#    spectra = spectra[:, mascara] 
    spectra = spectra.reshape(spectra.shape[0], spectra.shape[1], 1)
    
    if Spectra_set is None:
        Spectra_set = spectra
    else:
        Spectra_set = np.concatenate((Spectra_set, spectra), axis=0)

    # Obtener la clase espectral y corrimiento para cada espectro
#    clases_espectrales = Table.read(espc, hdu=1)['SPECTYPE'].data
#    corrimiento = Table.read(espc, hdu=1)['Z'].data
    # Read the FITS file and access the SPECTYPE and Z extensions

    clases_espectrales = espc[1].data['SPECTYPE']  # Get class data
    corrimiento = espc[1].data['Z']  # Get redshift data
    
    
    spectype = np.append(spectype,clases_espectrales)
    z = np.append(z, corrimiento)
    z=z.reshape(-1,1)

# Tenemos el tensor Spectra_set que contiene todos los flujos de los .fits seleccionados
# spectype es una lista con las etiquetas de dichos espectros
# z una matriz con los valores de corrimiento de cada espectro.


# In[3]:


#Spectra_set= Spectra_set[:35000]
#spectype= spectype[:35000]


# ### Cargamos el modelo de clasifiacion y comprobamos

# In[6]:


y = spectype
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y_categorical = to_categorical(y)

# Cargar el modelo desde el archivo HDF5
loaded_model = load_model("best_SAM_model.h5")

print('cargo el modelo .h5')

# In[ ]:


# Hacer predicciones sobre la nueva muestra de datos
predictions_new = loaded_model.predict(Spectra_set)    #GALAXY=0, QSO=1, STAR=2




galaxy_probability = predictions_new[:, 0]
# Probabilidades de cuásares
quasar_probability = predictions_new[:, 1]
# Probabilidades de estrellas
star_probability = predictions_new[:, 2]

print('he calculado las probas por clase')


predictions_new = np.argmax(predictions_new, axis=1)



print('se hicieron las predicciones')
accuracy = accuracy_score(y, predictions_new)
print("Overall accuracy:", accuracy)



# Crear histograma para las probabilidades de galaxia
plt.figure(figsize=(8, 6))
plt.hist(galaxy_probability, bins=10, color='blue', alpha=0.7)
plt.yscale('log')  # Establecer el eje Y en escala logarítmica
plt.title('Histogram of Probabilities for Galaxies')
plt.xlabel('Probability')
plt.ylabel('Ln(Frequency)')
plt.grid(True)
plt.savefig('his_proba_SAM_gal')
plt.show()
plt.close()


# Crear histograma para las probabilidades de quasar
plt.figure(figsize=(8, 6))
plt.hist(quasar_probability, bins=10, color='green', alpha=0.7)
plt.yscale('log')  # Establecer el eje Y en escala logarítmica
plt.title('Histogram of Probabilities for Quasars')
plt.xlabel('Probability')
plt.ylabel('Ln(Frequency)')
plt.grid(True)
plt.savefig('his_proba_SAM_qso')
plt.show()
plt.close()

# Crear histograma para las probabilidades de estrella
plt.figure(figsize=(8, 6))
plt.hist(star_probability, bins=10, color='red', alpha=0.7)
plt.yscale('log')  # Establecer el eje Y en escala logarítmica
plt.title('Histogram of Probabilities for Stars')
plt.xlabel('Probability')
plt.ylabel('Ln(Frequency)')
plt.grid(True)
plt.savefig('his_proba_SAM_star')
plt.show()
plt.close()


# In[7]:


# Obtener las etiquetas reales y predichas
true_labels = np.argmax(y_categorical, axis=1)
predicted_labels = predictions_new

# Crear la matriz de confusión
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Visualizar la matriz de confusión con seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['GALAXY', 'QSO', 'STAR'], yticklabels=['GALAXY', 'QSO', 'STAR'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('conf_matrix_SAM_validation')
plt.show()
plt.close()


# In[17]:


# Calcular precision, recall, y F1 Score
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')


# In[ ]:


# Obtener las etiquetas verdaderas y las predicciones del modelo
true_labels = y
predicted_labels = predictions_new

# Inicializar contadores para clasificaciones correctas e incorrectas por clase
correct_counts = {'GALAXY': 0, 'QSO': 0, 'STAR': 0}
incorrect_counts = {'GALAXY': 0, 'QSO': 0, 'STAR': 0}

# Calcular clasificaciones correctas e incorrectas para cada clase
for true_label, predicted_label in zip(true_labels, predicted_labels):
    if true_label == predicted_label:
        correct_counts[true_label] += 1
    else:
        incorrect_counts[true_label] += 1

# Clases
classes = ['GALAXY', 'QSO', 'STAR']
x_pos = np.arange(len(classes))

# Obtener las alturas de las barras
correct_heights = [correct_counts[c] for c in classes]
incorrect_heights = [incorrect_counts[c] for c in classes]

# Configurar la anchura de las barras
bar_width = 0.35

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(x_pos, correct_heights, bar_width, label='Correctly Classified', color='g')
plt.bar(x_pos + bar_width, incorrect_heights, bar_width, label='Incorrectly Classified', color='r')

# Etiquetas y título
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Correct vs Incorrect Classification Count by Class')
plt.xticks(x_pos + bar_width / 2, classes)
plt.legend()
# Mostrar el gráfico
plt.tight_layout()
plt.savefig('hist_accuracy_SAM_validation')
plt.show()
plt.close()

