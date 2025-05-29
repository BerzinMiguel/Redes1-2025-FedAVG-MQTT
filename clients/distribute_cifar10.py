# clients/distribute_cifar10.py

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import pickle

# Define uma classe CustomSubset que permite criar um subconjunto de um Dataset PyTorch
# usando uma lista específica de índices.
class CustomSubset(Dataset):
    # O construtor recebe o dataset original e uma lista de índices.
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    # O método __getitem__ retorna a amostra no índice especificado na lista de índices.
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    # O método __len__ retorna o número total de amostras neste subconjunto.
    def __len__(self):
        return len(self.indices)

# Função para distribuir o dataset CIFAR-10 de forma IID (independentemente e identicamente distribuída).
def distribute_cifar10_iid(num_clients, output_base_dir='./clients'):
    # Define as transformações a serem aplicadas às imagens:
    # Apenas converte a imagem para um tensor PyTorch.
    transform = transforms.Compose([transforms.ToTensor()])
    # Carrega o dataset CIFAR-10 de treinamento.
    # Se não existir, ele baixa automaticamente para './data_temp'.
    full_dataset = CIFAR10(root='./data_temp', train=True, download=True, transform=transform)

    # Loop para criar diretórios de dados para cada cliente.
    for i in range(num_clients):
        # Constrói o caminho para a pasta de dados de cada cliente (e.g., './clients/client_0/data').
        client_data_dir = os.path.join(output_base_dir, f'client_{i}', 'data')
        # Cria o diretório se ele não existir (exist_ok=True evita erro se já existir).
        os.makedirs(client_data_dir, exist_ok=True)

    # INÍCIO DA DISTRIBUIÇÃO IID:
    # Obtém uma lista de todos os índices do dataset completo.
    all_indices = list(range(len(full_dataset)))
    # Embaralha aleatoriamente a lista de índices.
    np.random.shuffle(all_indices)

    # Calcula o tamanho que cada subconjunto de dados de cliente terá.
    split_size = len(full_dataset) // num_clients
    # Inicializa um dicionário para armazenar os índices de dados para cada cliente.
    client_data_indices = {i: [] for i in range(num_clients)}

    # Loop para dividir os índices embaralhados igualmente entre os clientes.
    for i in range(num_clients):
        # Calcula o índice de início e fim para a fatia de dados do cliente atual.
        start_idx = i * split_size
        end_idx = start_idx + split_size
        # Se for o último cliente, ele recebe todas as amostras restantes
        # para garantir que nenhuma amostra seja perdida devido à divisão inteira.
        if i == num_clients - 1:
            client_data_indices[i] = all_indices[start_idx:]
        else:
            # Caso contrário, o cliente recebe sua fatia designada.
            client_data_indices[i] = all_indices[start_idx:end_idx]

    # Loop para criar e salvar o dataset específico para cada cliente.
    for i in range(num_clients):
        # Cria um CustomSubset usando os índices atribuídos ao cliente.
        client_dataset = CustomSubset(full_dataset, client_data_indices[i])
        # Abre um arquivo em modo binário de escrita para salvar o dataset do cliente.
        with open(os.path.join(output_base_dir, f'client_{i}', 'data', f'cifar10_client_{i}.pkl'), 'wb') as f:
            # Serializa o objeto CustomSubset e o salva no arquivo.
            pickle.dump(client_dataset, f)
        
        # Opcional: Verifica e imprime a distribuição de classes para cada cliente
        # para confirmar que a distribuição IID funciona (classes bem misturadas).
        client_labels = [full_dataset[idx][1] for idx in client_data_indices[i]]
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))
        
        print(f"Client {i}: {len(client_dataset)} samples assigned. Label distribution (counts): {label_distribution}")

    print("Distribuição do dataset CIFAR-10 (IID) concluída.")

# Bloco executado apenas se o script for rodado diretamente (não importado como módulo).
if __name__ == "__main__":
    # Define o número de clientes. Pode ser ajustado aqui ou passado como argumento.
    num_clients = 3
    # Chama a função para distribuir os dados.
    distribute_cifar10_iid(num_clients)