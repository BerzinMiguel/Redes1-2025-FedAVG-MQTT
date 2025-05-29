# clients/client_X/client.py

import torch
from torch.utils.data import DataLoader
import sys
import os
import pickle
import paho.mqtt.client as mqtt # Importa a biblioteca Paho MQTT.
import time
from datetime import datetime
import numpy as np

# --- INÍCIO: LINHAS DE DEPURACÃO PARA O CAMINHO 'common' (Remova após verificar o funcionamento) ---
# Obtém o diretório do arquivo Python atual.
current_dir = os.path.dirname(__file__)
# Calcula o caminho absoluto para a pasta 'common'.
# '..' sobe um nível (de client_X/ para clients/), outro '..' sobe mais um (para federated_learning/).
calculated_common_path_client = os.path.abspath(os.path.join(current_dir, '..', '..', 'common'))
# Imprime o diretório atual e o caminho calculado para fins de depuração.
print(f"DEBUG (Client): Diretório atual do client.py: {current_dir}")
print(f"DEBUG (Client): Caminho calculado para 'common' no cliente: {calculated_common_path_client}")
# --- FIM: LINHAS DE DEPURACÃO ---

# Adiciona o caminho calculado para 'common' ao sys.path, permitindo a importação de módulos de lá.
sys.path.append(calculated_common_path_client)
# Importa a classe FederatedNet do módulo federated_net (que está em 'common').
from federated_net import FederatedNet 

# NOVO: Adiciona o diretório 'clients' (que contém 'distribute_cifar10.py') ao sys.path.
# Isso é necessário para que o pickle.load() consiga encontrar a definição da classe CustomSubset.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Importa a classe CustomSubset do módulo distribute_cifar10.
from distribute_cifar10 import CustomSubset

# Define a classe Cliente.
class Client:
    # Construtor da classe Cliente.
    def __init__(self, client_id, broker_address="localhost", broker_port=1883, epochs=3):
        # Armazena o ID único do cliente.
        self.client_id = client_id
        # Define o número de épocas para o treinamento local em cada rodada.
        self.epochs = epochs
        # Carrega o dataset CIFAR-10 específico para este cliente.
        self.dataset = self.load_data() 
        # Inicializa uma instância da rede neural.
        self.net = FederatedNet()
        
        # Inicializa o cliente MQTT.
        # mqtt.CallbackAPIVersion.VERSION2 especifica o uso da API de callbacks da versão 2.0.
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, f"client_{client_id}")
        
        # Atribui os métodos de callback para eventos MQTT.
        self.client.on_connect = self.on_connect # Chamado quando o cliente se conecta ao broker.
        # Usa um wrapper para rotear mensagens para manipuladores específicos de tópicos.
        self.client.on_message = self._on_message_handler_wrapper 
        # Endereço do broker MQTT.
        self.broker_address = broker_address
        # Porta do broker MQTT.
        self.broker_port = broker_port
        # Flag para controlar se os parâmetros iniciais foram recebidos.
        self.received_initial_parameters = False
        # Armazena os parâmetros globais recebidos do servidor.
        self.global_parameters = None
        # O número da rodada atual.
        self.round_num = 0
        # NOVO: Flag para sinalizar que o treinamento federado terminou.
        self.training_finished = False 

    # Método para carregar o dataset CIFAR-10 específico do cliente.
    def load_data(self):
        # Constrói o caminho para o arquivo .pkl do dataset do cliente.
        data_path = os.path.join(os.path.dirname(__file__), 'data', f'cifar10_client_{self.client_id}.pkl')
        # Verifica se o arquivo de dados existe.
        if not os.path.exists(data_path):
            print(f"Erro: Arquivo de dados para o cliente {self.client_id} não encontrado em {data_path}.")
            print("Execute 'python clients/distribute_cifar10.py' primeiro.")
            sys.exit(1) # Sai do programa se o arquivo não for encontrado.
        
        # Abre o arquivo em modo binário de leitura.
        with open(data_path, 'rb') as f:
            # Carrega (des-serializa) o dataset do arquivo.
            dataset = pickle.load(f) 
        return dataset

    # Método de callback chamado quando o cliente se conecta ao broker MQTT.
    # 'properties' é um novo argumento na API v2.0 do paho-mqtt.
    def on_connect(self, client, userdata, flags, rc, properties): 
        # Verifica se a conexão foi bem-sucedida (rc=0).
        if rc == 0:
            print(f"Client {self.client_id}: Conectado ao broker MQTT com sucesso!")
            # Inscreve-se no tópico para receber parâmetros iniciais do servidor.
            self.client.subscribe(f"server/initial_parameters/{self.client_id}")
            # Inscreve-se no tópico para receber parâmetros globais atualizados do servidor.
            self.client.subscribe(f"server/global_parameters/{self.client_id}")
            # NOVO: Inscreve-se no tópico para receber o sinal de término do servidor.
            self.client.subscribe("client/terminate") 
            print(f"Client {self.client_id}: Inscrito nos tópicos 'server/initial_parameters/{self.client_id}', 'server/global_parameters/{self.client_id}' e 'client/terminate'.")
            
            # NOVO: Cliente envia um sinal de "pronto" para o servidor.
            # O payload é o ID do cliente, e o QoS (Quality of Service) 1 garante entrega.
            self.client.publish("client/ready", str(self.client_id), qos=1)
            print(f"Client {self.client_id}: Enviou sinal de 'pronto' para o servidor.")

        else:
            print(f"Client {self.client_id}: Falha na conexão, código de retorno: {rc}")

    # NOVO: Manipulador de mensagens para o tópico de término.
    def on_terminate_message(self, client, userdata, msg):
        print(f"\n{'-'*50}")
        print(f"Client {self.client_id}: Recebido sinal de término do servidor. Desconectando...")
        print(f"{'-'*50}\n")
        # Define a flag de término para True, fazendo o loop principal do start() parar.
        self.training_finished = True 
        
    # Método de callback chamado quando uma mensagem é recebida nos tópicos de parâmetros.
    def on_parameters_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload
        
        # Determina o número da rodada. A rodada 0 é para parâmetros iniciais, as subsequentes incrementam.
        if "initial_parameters" in topic:
            self.round_num = 0
        else: # Mensagem de global_parameters
            self.round_num += 1 

        # Des-serializa os parâmetros recebidos (de bytes para objeto Python).
        parameters_bytes = payload
        parameters = pickle.loads(parameters_bytes)
        
        # Atualiza os parâmetros globais do cliente com os recebidos.
        self.global_parameters = parameters
        # Indica que os parâmetros iniciais foram recebidos (para sair da espera inicial no start()).
        self.received_initial_parameters = True
        
        print(f"\n{'-'*50}")
        print(f"Client {self.client_id}: Iniciando Rodada {self.round_num}")
        print(f"Client {self.client_id}: Parâmetros do servidor recebidos.")
        
        # Registra o tempo de início do treinamento local.
        start_time = time.time()
        # Executa o treinamento local e obtém os parâmetros atualizados, perda e acurácia.
        updated_parameters, train_loss, accuracy = self.train(self.global_parameters)
        # Registra o tempo de fim do treinamento local.
        end_time = time.time()
        
        # Calcula a duração do treinamento local.
        training_time = end_time - start_time
        # Estima o tamanho dos parâmetros atualizados que serão transferidos (upload).
        transferred_data_size_bytes = sys.getsizeof(pickle.dumps(updated_parameters)) 
        
        print(f"Client {self.client_id}: Treinamento local concluído para Rodada {self.round_num}.")
        print(f"  Tempo de treinamento: {training_time:.2f} segundos")
        print(f"  Tamanho dos dados transferidos (upload): {transferred_data_size_bytes / 1024:.2f} KB")
        print(f"  Perda de Treinamento: {train_loss:.4f}")
        print(f"  Acurácia Local: {accuracy:.2f}%")
        print(f"{'-'*50}\n")
        
        # Publica os parâmetros atualizados no tópico específico do cliente para o servidor.
        self.client.publish(f"client/updated_parameters/{self.client_id}", pickle.dumps(updated_parameters), qos=1)
        print(f"Client {self.client_id}: Parâmetros atualizados para Rodada {self.round_num} enviados para o servidor.")

    # NOVO: Wrapper para on_message para direcionar mensagens para os manipuladores corretos.
    # O paho-mqtt chama apenas um on_message, então este método decide qual manipulador chamar
    # com base no tópico da mensagem recebida.
    def _on_message_handler_wrapper(self, client, userdata, msg):
        # Se a mensagem for para terminar o treinamento.
        if msg.topic == "client/terminate":
            self.on_terminate_message(client, userdata, msg)
        # Se a mensagem for de parâmetros iniciais ou globais.
        elif msg.topic.startswith("server/initial_parameters/") or msg.topic.startswith("server/global_parameters/"):
            self.on_parameters_message(client, userdata, msg)
        else:
            print(f"Client {self.client_id}: Mensagem recebida em tópico não esperado: {msg.topic}")


    # Método para realizar o treinamento local do modelo.
    def train(self, parameters):
        # Aplica os parâmetros globais recebidos à rede local do cliente.
        self.net.apply_parameters(parameters)
        # Define o otimizador SGD (Stochastic Gradient Descent) com uma taxa de aprendizado.
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        # Cria um DataLoader para iterar sobre o dataset do cliente em batches.
        dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)

        total_loss = 0.0 # Acumulador para a perda total.
        correct_predictions = 0 # Acumulador para o número de previsões corretas.
        total_samples = 0 # Acumulador para o número total de amostras processadas.

        self.net.train() # Coloca a rede em modo de treinamento (habilita dropout/batchnorm, se houver).
        # Loop sobre o número de épocas.
        for epoch in range(self.epochs):
            # Loop sobre os batches de dados do DataLoader.
            for inputs, labels in dataloader:
                optimizer.zero_grad() # Zera os gradientes acumulados de passes anteriores.
                outputs = self.net(inputs) # Realiza o passe forward.
                loss = torch.nn.functional.cross_entropy(outputs, labels) # Calcula a perda de entropia cruzada.
                loss.backward() # Realiza o passe backward para calcular os gradientes.
                optimizer.step() # Atualiza os pesos do modelo usando o otimizador.
                
                total_loss += loss.item() * inputs.size(0) # Acumula a perda do batch.
                _, predicted = torch.max(outputs.data, 1) # Obtém a classe prevista (índice com maior probabilidade).
                total_samples += labels.size(0) # Acumula o número de amostras no batch.
                correct_predictions += (predicted == labels).sum().item() # Conta as previsões corretas.
        
        avg_loss = total_loss / total_samples # Calcula a perda média.
        accuracy = (correct_predictions / total_samples) * 100 # Calcula a acurácia em porcentagem.
        
        # Retorna os parâmetros atualizados da rede local, a perda média e a acurácia.
        return self.net.get_parameters(), avg_loss, accuracy

    # Método para iniciar o cliente MQTT e seu loop de execução.
    def start(self):
        try:
            # Tenta conectar ao broker MQTT.
            self.client.connect(self.broker_address, self.broker_port, 60)
            # Inicia o loop de rede em um thread separado para processar mensagens em segundo plano.
            self.client.loop_start() 
            
            print(f"Client {self.client_id}: Aguardando comandos do servidor...")
            # NOVO: Loop principal do cliente espera até que a flag training_finished seja True.
            while not self.training_finished:
                time.sleep(1) # Aguarda 1 segundo para evitar uso excessivo da CPU.
        # Captura qualquer exceção que ocorra durante a execução do cliente.
        except Exception as e:
            print(f"Client {self.client_id}: Erro durante a execução: {e}")
        # O bloco finally é sempre executado, independentemente de exceções.
        finally:
            # Para o loop de rede do cliente MQTT.
            self.client.loop_stop()
            # Desconecta o cliente do broker MQTT.
            self.client.disconnect()
            print(f"Client {self.client_id}: Encerrado.") # NOVO: Mensagem de encerramento.

# Bloco executado apenas se o script for rodado diretamente.
if __name__ == "__main__":
    # Verifica se os argumentos de linha de comando foram fornecidos corretamente.
    if len(sys.argv) != 3:
        print("Uso: python client.py <client_id> <num_epochs>")
        sys.exit(1) # Sai com erro se os argumentos estiverem incorretos.

    # Obtém o client_id e o número de épocas dos argumentos de linha de comando.
    client_id = int(sys.argv[1])
    num_epochs = int(sys.argv[2])

    # Cria uma instância do Cliente e a inicia.
    client_instance = Client(client_id=client_id, epochs=num_epochs)
    client_instance.start()