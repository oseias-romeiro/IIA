# -*- coding: utf-8 -*-
"""Trabalho2.ipynb

# Trabalho 2
Este trabalho aborda os conteúdos de Self-Organizing Map (SOM) e Convolutional Neural Network (CNN), permitindo que os alunos explorem conceitos fundamentais de redes neurais e aprendizado de máquina por meio de implementações práticas.

## Dempendências
"""

import numpy as np

"""## Questão 1
Você deve implementar um Self-Organizing Map (SOM) para mapear dados em um espaço 2D. O SOM é uma rede neural não supervisionada que organiza dados de alta dimensão em um mapa de baixa dimensão, preservando a topologia dos dados.

O código base já define a classe SOM com os métodos find_best_matching_unit, update_weights e train. Sua tarefa é completar o método train para que ele treine a rede SOM e retorne a posição da Best Matching Unit (BMU) para o primeiro ponto de dados após o treinamento.
"""

class SOM:
    def __init__(self, input_dim, map_size, learning_rate=0.1, sigma=1.0):
        self.input_dim = input_dim
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = np.random.rand(map_size[0], map_size[1], input_dim)

    def find_best_matching_unit(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index

    def update_weights(self, input_vector, bmu_index, epoch, max_epochs):
        lr = self.learning_rate * (1 - epoch / max_epochs)
        radius = self.sigma * (1 - epoch / max_epochs)

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))
                influence = np.exp(-distance_to_bmu**2 / (2 * radius**2))
                self.weights[i, j] += lr * influence * (input_vector - self.weights[i, j])

    def train(self, data, epochs):
        # retornar a posição da BMU para o primeiro ponto de dados.
        for epoch in range(epochs):
            for input_vector in data:
                bmu_index = self.find_best_matching_unit(input_vector)
                self.update_weights(input_vector, bmu_index, epoch, epochs)
        return self.find_best_matching_unit(data[0])

"""### Exemplo
Ao executar o código abaixo, o resultado deve ser:
```
Posição da BMU: (0, 2)
```
"""

# Entrada fixa
data = np.array([[0.1, 0.1], [0.9, 0.9]])  # Dois pontos de dados fixos
som = SOM(input_dim=2, map_size=(5, 5), learning_rate=0.1, sigma=1.0)

# Fixando a semente aleatória para garantir resultados determinísticos
np.random.seed(42)
som.weights = np.random.rand(5, 5, 2)  # Pesos fixos

# Treinamento e saída
bmu_position = som.train(data, epochs=10)
print("Posição da BMU:", bmu_position)

"""## Questão 2

Você deve implementar uma Convolutional Neural Network (CNN) simplificada para classificação de imagens. A CNN consiste em uma camada convolucional, uma camada de pooling e uma camada totalmente conectada.

O código base já define a classe CNN com os métodos convolution, pooling e forward. Sua tarefa é completar os métodos convolution e pooling  para que ele realize a passagem direta (forward pass) da rede e retorne a classe prevista para uma imagem de entrada.
"""

class CNN:
    def __init__(self, input_shape, num_filters, filter_size, num_classes):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_classes = num_classes
        self.display=True

        # Inicializa os filtros da camada convolucional
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size**2)
        self.num_filters = num_filters

        # Calcula as dimensões da saída da camada de pooling
        self.pool_output_size = (input_shape[0] - filter_size + 1) // 2  # Assumindo pool_size=2 e stride=2
        self.flattened_size = num_filters * (self.pool_output_size ** 2)

        # Inicializa os pesos da camada totalmente conectada
        self.weights = np.random.randn(self.flattened_size, num_classes) / 100
        self.bias = np.zeros(num_classes)


    def convolution(self, input_image):
        """
        Realiza a operação de convolução na entrada usando os filtros.

        Parâmetros:
        - input_image (ndarray): Imagem de entrada como matriz bidimensional.

        Retorno:
        - output (ndarray): Mapa de características resultante da convolução com dimensões reduzidas.
        """
        height, width = input_image.shape

        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1

        conv_output = np.zeros((self.num_filters, output_height, output_width))

        for f in range(0, self.num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    submatrix = input_image[i:i+self.filter_size, j:j+self.filter_size]
                    conv_output[f, i, j] = np.sum(submatrix * self.filters[f])

        return conv_output


    def pooling(self, input_feature_map, pool_size=2, stride=2):
        """
        Realiza a operação de pooling max na entrada.

        Parâmetros:
        - input_feature_map (ndarray): Mapa de características gerado pela camada convolucional.
        - pool_size (int): Tamanho da janela de pooling (default=2).
        - stride (int): Passo do pooling (default=2).

        Retorno:
        - output (ndarray): Mapa de características reduzido após a operação de pooling.
        """
        num_maps = len(input_feature_map)

        input_height, input_width = input_feature_map[0].shape # assumo que todos os mapas têm o mesmo tamanho

        output_height = (input_height - pool_size) // stride + 1
        output_width = (input_width - pool_size) // stride + 1

        pooled_output = np.zeros((num_maps, output_height, output_width))

        for m, ifm in enumerate(input_feature_map):
            for i in range(output_height):
                for j in range(output_width):
                    submatrix = ifm[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                    pooled_output[m, i, j] = np.max(submatrix)

        return pooled_output

    def forward(self, input_image):
        # Passagem pela camada convolucional
        conv_output = self.convolution(input_image)
        if(self.display): print("conv_output:\n", conv_output)

        # Passagem pela camada de pooling
        pooled_output = self.pooling(conv_output)
        if(self.display): print("pooled_output:\n", pooled_output)

        # Achata a saída para um vetor
        flattened_output = pooled_output.flatten()
        if(self.display): print("flattened_output:\n", flattened_output)

        if(self.display): print("wheights:\n", self.weights)

        # Passagem pela camada totalmente conectada
        logits = np.dot(flattened_output, self.weights) + self.bias
        if(self.display): print("logits:\n", logits)

        return np.argmax(logits)  # Retorna a classe prevista

"""### Testes

#### Teste 1
Resultado esperado: `Classe prevista: 5`
"""

np.random.seed(42)  # Garante resultados consistentes
input_image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])  # Imagem 3x3 fixa

cnn = CNN(input_shape=(3, 3), num_filters=1, filter_size=2, num_classes=10)

# Configurando filtros e pesos fixos para garantir resultado determinístico
cnn.filters = np.random.randn(1, 2, 2)
cnn.weights = np.random.randn(1 * 1, 10) / 100  # Ajuste do tamanho dos pesos

# Classe prevista
predicted_class = cnn.forward(input_image)
print("Classe prevista:", predicted_class)

"""#### Teste 2

Resultado esperado: `Classe prevista: 0`
"""

np.random.seed(40) # Garante resultados consistentes
input_image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])  # Imagem 4x4

cnn = CNN(input_shape=(4, 4), num_filters=2, filter_size=2, num_classes=3)

# Configurando filtros e pesos fixos para garantir resultado determinístico
cnn.filters = np.random.randn(2, 2, 2)
cnn.weights = np.random.randn(2 * 1, 3) / 100  # Ajuste do tamanho dos pesos

# Classe prevista
predicted_class = cnn.forward(input_image)
print("Classe prevista:", predicted_class)

"""#### Teste 3

Resultado esperado `Classe prevista: 2`
"""

np.random.seed(42) # Garante resultados consistentes

input_image = np.array([
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    ])  # Imagem 6x6

cnn = CNN(input_shape=(6, 6), num_filters=1, filter_size=3, num_classes=5)

# Configurando filtros e pesos fixos para garantir resultado determinístico
cnn.filters = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
])

cnn.weights = np.random.randn(1 * 4, 5) / 100  # Ajuste do tamanho dos pesos

# Classe prevista
predicted_class = cnn.forward(input_image)
print("Classe prevista:", predicted_class)