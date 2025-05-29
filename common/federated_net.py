# common/federated_net.py

import torch
import torch.nn.functional as F
import torch.nn as nn

# Define a classe da rede neural federada.
# Ela herda de nn.Module, a classe base para todos os módulos de redes neurais no PyTorch.
class FederatedNet(nn.Module):
    # O construtor da classe, onde as camadas da rede são definidas.
    def __init__(self):
        # Chama o construtor da classe pai (nn.Module).
        super().__init__()
        # Define a primeira camada convolucional:
        # 3 canais de entrada (para imagens RGB), 20 canais de saída (filtros),
        # kernel de tamanho 7x7.
        self.conv1 = nn.Conv2d(3, 20, 7)
        # Define a segunda camada convolucional:
        # 20 canais de entrada (saída da conv1), 40 canais de saída,
        # kernel de tamanho 7x7.
        self.conv2 = nn.Conv2d(20, 40, 7)
        # Define uma camada de max-pooling:
        # Janela de 2x2, passo de 2 (reduz a dimensão espacial pela metade).
        self.maxpool = nn.MaxPool2d(2, 2)
        # Define uma camada de "achatamento" (flatten):
        # Converte as dimensões espaciais e de canal em uma única dimensão
        # antes de passar para a camada linear.
        self.flatten = nn.Flatten()
        # Define a camada linear (totalmente conectada):
        # A entrada é 4000 (40 canais * 10x10 dimensão espacial após pooling para 32x32 input).
        # A saída é 10 (para as 10 classes do CIFAR-10).
        self.linear = nn.Linear(4000, 10) 

        # Define a função de ativação não-linear como ReLU (Rectified Linear Unit).
        self.non_linearity = F.relu
        # Dicionário que armazena as camadas cujos parâmetros serão rastreados e transferidos.
        # Isso é importante para o aprendizado federado, onde apenas os parâmetros dessas camadas
        # são trocados entre clientes e servidor.
        self.track_layers = {'conv1': self.conv1, 'conv2': self.conv2, 'linear': self.linear}

    # Define o passe forward da rede, que descreve como os dados fluem através das camadas.
    def forward(self, x):
        # Aplica a primeira convolução seguida pela função de ativação.
        x = self.non_linearity(self.conv1(x))
        # Aplica a segunda convolução seguida pela função de ativação.
        x = self.non_linearity(self.conv2(x))
        # Aplica a camada de max-pooling.
        x = self.maxpool(x)
        # Achata a saída para uma dimensão única.
        x = self.flatten(x)
        # Aplica a camada linear final.
        x = self.linear(x)
        # Retorna a saída (logits) da rede.
        return x

    # Método para obter os parâmetros (pesos e bias) das camadas rastreadas.
    def get_parameters(self):
        # Retorna um dicionário onde a chave é o nome da camada e o valor é um dicionário
        # contendo os tensores de peso e bias (copiados para evitar referências).
        return {name: {'weight': layer.weight.data.clone(), 'bias': layer.bias.data.clone()} for name, layer in self.track_layers.items()}

    # Método para aplicar um novo conjunto de parâmetros à rede.
    def apply_parameters(self, parameters):
        # Desabilita o cálculo de gradientes durante a aplicação dos parâmetros,
        # pois não é uma operação de treinamento.
        with torch.no_grad():
            # Itera sobre os parâmetros fornecidos.
            for name in parameters:
                # Copia os novos pesos para a camada correspondente na rede.
                self.track_layers[name].weight.data.copy_(parameters[name]['weight'])
                # Copia os novos bias para a camada correspondente na rede.
                self.track_layers[name].bias.data.copy_(parameters[name]['bias'])