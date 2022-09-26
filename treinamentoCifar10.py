# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 09:31:52 2021

@author: André

https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/

Base de dados do cifar10

"""

#importando a base dados do cifar10 para o projeto
from keras.datasets import cifar10

#dividindo os dados entre treinamento e teste
(X_treino, y_treino), (X_teste, y_teste) = cifar10.load_data();

print("X_treino quantidade: ", X_treino.shape)
print("y_treino quantidade: ", y_treino.shape)
print("X_teste quantidade: ", X_teste.shape)
print("y_teste quantidade: ", y_teste.shape)

#preparação para aplicar a CNN

# 1° importar o classificador sequencial
from keras.models import Sequential

# 2° importar as classes que representam as camadas convolucionais
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

# 3° importar a classe utils do keras para operações numéricas
from keras.utils import np_utils

# aplicar a operação de Flattening nas imagens
# As imagens serão transformadas em vetores colunas
# sabendo que, as imagens originais da base mnist possuem dimensão
# 28 pixels de largura por 28 pixels de altura
# Após o Flattening, teremos um vetor coluna com tamanho de 784
# e valores normalizados entre 0 e 1

# 1° passo do flattening - transformação da imagem em vetor coluna
# 784 = 28 x 28
X_treino = X_treino.reshape(X_treino.shape[0], 32, 32, 3)
X_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)

# 2° passo: transformar os dados em float32 para facilicar operações numéricas
X_treino = X_treino.astype('float32')
X_teste = X_teste.astype('float32')

# 3° passo: Normalizar os dados para valores entre 0 e 1
# Objetivo será melhorar o treinamento
X_treino = X_treino/255
X_teste = X_teste/255

# declarar a quantidade de classes e obrigar a variável Y
# adquirir valores entre a quantidade de classes informada
n_classes = 10
print("Classes antes das labels serem estabelecidas: ", y_treino.shape)
Y_treino = np_utils.to_categorical(y_treino, 10)
Y_teste = np_utils.to_categorical(y_teste, 10)
print("Classes depois das labels serem estabelecidas: ", Y_treino.shape)

# 4° passo: construir a nossa rede neural sequencial
modelo = Sequential()

# convolutional layer
modelo.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
modelo.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
modelo.add(MaxPool2D(pool_size=(2,2)))
modelo.add(Dropout(0.25))

modelo.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
modelo.add(MaxPool2D(pool_size=(2,2)))
modelo.add(Dropout(0.25))

# flatten output of conv
modelo.add(Flatten())

# hidden layer
modelo.add(Dense(500, activation='relu'))
modelo.add(Dropout(0.4))
modelo.add(Dense(250, activation='relu'))
modelo.add(Dropout(0.3))

# output layer
modelo.add(Dense(10, activation='softmax'))

# compiling the sequential model
modelo.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
modelo.fit(X_treino, Y_treino, batch_size=128, epochs=10, validation_data=(X_teste, Y_teste))
