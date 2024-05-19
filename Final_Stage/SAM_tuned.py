#!/usr/bin/env python
# coding: utf-8

# In[2]:


print('programa iniciado: SAM')


# In[12]:

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

# In[4]:


print('modulos cargados')


# In[5]:


### Leemos datos del archivo .fits pero unicamente de la clase de interes 'GALAXY', 'STAR', 'QSO'


# In[2]:


wave= fits.open('B_R_Z_wavelenght.fits')
Bwave = wave[1].data
Rwave = wave[2].data
Zwave = wave[3].data
wavelenght = np.hstack((Bwave, Rwave, Zwave)) #Contiene la cadena completa de longitudes de onda B+Z+R para cada espectro


# In[7]:


# Definir una máscara booleana para las longitudes de onda del flujo en U
#mascara = (wavelenght >= 3055.11) & (wavelenght <= 4030.64)
#wavelenght_filtrado= wavelenght[mascara]


# In[3]:


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


# In[9]:


print('archivos leidos y data creado')


# In[4]:


#Spectra_set= Spectra_set[:1000]
#spectype= spectype[:1000]


# In[11]:


### Ejercicio de clasifiación. Aqui predecimos la clase o tipo espectral de cada espectro de entrada


# In[12]:


num_classes = 3  # Número de clases
X=Spectra_set

#Aigna un numero entero a cada clase
y = spectype
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Divide los datos en conjuntos de entrenamiento y prueba de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

# Convierte las etiquetas a codificación one-hot
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)      #Galazxy=[0,0,1]

# Calcular los pesos de clase balanceados
class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
# Convertir los pesos a un diccionario
class_weights_dict = dict(enumerate(class_weights))

# Crea el modelo de CNN
model = Sequential()

#model.add(Conv1D(filters=64, kernel_size=20, activation='relu', strides=2!!, input_shape=(len(X[0]),1)))
model.add(Conv1D(filters=64, kernel_size=15, activation='relu', strides=2, input_shape=(len(X[0]),1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=15, activation='relu', strides=2))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=15, activation='relu', strides=2))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=15, activation='relu', strides=2))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compila el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrena el modelo
history = model.fit(X_train, y_train, epochs=75, batch_size=32, validation_split=0.4, sample_weight=class_weights)

# Guardar el modelo y sus pesos en un solo archivo HDF5
model.save("best_SAM_model.h5")

#Visualización del Rendimiento
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Loss_curve_SAM')
plt.close()
plt.show()


# In[13]:


print('modelo entrenado')


# In[14]:


# Hacer predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Convierte las predicciones one-hot a etiquetas
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Crear la matriz de confusión
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Visualizar la matriz de confusión con seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['GALAXY', 'QSO', 'STAR'], yticklabels=['GALAXY', 'QSO', 'STAR'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('conf_matrix_SAM')
plt.close()
plt.show()


# In[15]:


# Calcular precision, recall, y F1 Score
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')


# In[16]:


# Mapeo de índices de clase a nombres
class_names = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}

# Calcular precision, recall, y F1 Score por clase
precision_per_class, recall_per_class, f1_score_per_class, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)

# Mostrar las métricas por clase
for i in range(num_classes):
    class_name = class_names[i]
    print(f'Class: {class_name}')
    print(f'Precision: {precision_per_class[i]:.4f}')
    print(f'Recall: {recall_per_class[i]:.4f}')
    print(f'F1 Score: {f1_score_per_class[i]:.4f}')
    print('---')


# In[17]:


# Contar las instancias correctamente clasificadas y las incorrectamente clasificadas
correctly_classified = np.where(predicted_labels == true_labels)[0]
incorrectly_classified = np.where(predicted_labels != true_labels)[0]

# Obtener las etiquetas verdaderas de las instancias mal clasificadas
true_labels_incorrect = true_labels[incorrectly_classified]

# Contar las clases correctamente clasificadas
correct_counts = np.bincount(true_labels[correctly_classified], minlength=num_classes)

# Contar las clases incorrectamente clasificadas
incorrect_counts = np.bincount(true_labels_incorrect, minlength=num_classes)

# Crear el histograma
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(num_classes)
plt.bar(index, correct_counts, bar_width, label='Correctly Classified', color='g')
plt.bar(index + bar_width, incorrect_counts, bar_width, label='Incorrectly Classified', color='r')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Correct vs Incorrect Classification Count by Class')
plt.xticks(index + bar_width / 2, ['GALAXY', 'QSO', 'STAR'])
plt.legend()
plt.savefig('hist_accuracy_SAM')
plt.close()
plt.show()


# In[18]:


print('metricas calculadas, proceso finalizado')

