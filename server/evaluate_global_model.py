# server/evaluate_global_model.py

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
import os
import sys

# Importar o modelo da pasta comum
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
from federated_net import FederatedNet

# Define a função para avaliar o modelo.
def evaluate_model(model_path=os.path.join(os.path.dirname(__file__), 'global_parameters.pkl')):
    print("\n--- Avaliando o Modelo Global Final ---")

    # Verifica se o arquivo de parâmetros do modelo global existe no caminho especificado.
    if not os.path.exists(model_path):
        print(f"Erro: Arquivo de parâmetros do modelo global não encontrado em {model_path}.")
        print("Certifique-se de que o servidor concluiu o treinamento e salvou 'global_parameters.pkl'.")
        sys.exit(1) # Sai do script se o arquivo não for encontrado.

    # Carrega os parâmetros do modelo global do arquivo.
    with open(model_path, 'rb') as f:
        global_parameters = pickle.load(f)

    # Inicializa uma nova instância da rede neural.
    net = FederatedNet()
    # Aplica os parâmetros carregados à rede.
    net.apply_parameters(global_parameters)
    # Coloca o modelo em modo de avaliação.
    # Isso desabilita camadas como Dropout e ajusta o comportamento de BatchNorm (se existissem).
    net.eval() 

    # Carrega o dataset de teste CIFAR-10.
    # train=False indica que é o conjunto de teste.
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CIFAR10(root='./data_temp', train=False, download=True, transform=transform)
    # Cria um DataLoader para iterar sobre o dataset de teste em batches.
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    correct = 0 # Contador para o número de previsões corretas.
    total = 0 # Contador para o número total de amostras.
    # Desabilita o cálculo de gradientes durante a avaliação para economizar memória e tempo,
    # pois não estamos treinando.
    with torch.no_grad():
        # Itera sobre os batches do DataLoader de teste.
        for inputs, labels in test_dataloader:
            outputs = net(inputs) # Realiza o passe forward.
            _, predicted = torch.max(outputs.data, 1) # Obtém a classe prevista.
            total += labels.size(0) # Acumula o número de amostras.
            correct += (predicted == labels).sum().item() # Acumula as previsões corretas.

    # Calcula a acurácia em porcentagem.
    accuracy = 100 * correct / total
    print(f"Acurácia do Modelo Global no Dataset de Teste: {accuracy:.2f}%")
    print("--- Avaliação Concluída ---")

# Bloco executado apenas se o script for rodado diretamente.
if __name__ == "__main__":
    evaluate_model()