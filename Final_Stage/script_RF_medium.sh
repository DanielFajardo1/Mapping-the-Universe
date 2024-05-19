#!/bin/bash

# ###### Zona de ParÃ¡metros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=Model_RF_long_half             #Nombre del job
#SBATCH -p long                        #Cola a usar, Default=short (Ver colas y lÃ­mites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1                            #Nodos requeridos, Default=1
#SBATCH -n 1                            #Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=11               #Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=262000              #Memoria en Mb por CPU, Default=2048
#SBATCH --time=10-00:00:00                 #Tiempo mÃ¡ximo de corrida, Default=2 horas
#SBATCH --mail-user=d.fajardo@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH -o Model_RF_long_half.o%j                 #Nombre de archivo de salida
#
########################################################################################

# ############################### Zona Carga de Modulos ################################
module load anaconda/python3.9
source activate Cluster2
#pip install tensorflow
########################################################################################

# ###### Zona de Ejecucion de codigo y comandos a ejecutar secuencialmente #############
host=`/bin/hostname`
date=`/bin/hostname`
echo "Soy un JOB de prueba"
echo "Corri en la maquina: "$host
echo "Corri el: "$date

echo -e "Ejecutando Script de python \n"
python RF_tuned.py
echo -e "Finalice la ejecucion del script \n"
########################################################################################

