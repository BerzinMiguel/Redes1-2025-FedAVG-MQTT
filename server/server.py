# server/server.py

import torch
import pickle
import os
import paho.mqtt.client as mqtt
import time
from collections import defaultdict
import sys
from datetime import datetime
import numpy as np # Importado para np.mean nas métricas.

# --- LINHAS DE DEPURACÃO PARA O CAMINHO 'common'  ---
# Obtém o diretório do arquivo Python atual.
current_dir = os.path.dirname(__file__)
# Calcula o caminho absoluto para a pasta 'common'.
# '..' sobe um nível (de server/ para federated_learning/).
calculated_common_path_server = os.path.abspath(os.path.join(current_dir, '..', 'common'))
# Imprime o diretório atual e o caminho calculado para fins de depuração.
print(f"DEBUG (Server): Diretório atual do server.py: {current_dir}")
print(f"DEBUG (Server): Caminho calculado para 'common' no servidor: {calculated_common_path_server}")
# --- FIM: LINHAS DE DEPURACÃO ---

# Adiciona o caminho calculado para 'common' ao sys.path, permitindo a importação de módulos de lá.
sys.path.append(calculated_common_path_server)
# Importa a classe FederatedNet do módulo federated_net (que está em 'common').
from federated_net import FederatedNet

# Define a classe Server.
class Server:
    # Construtor da classe Server.
    def __init__(self, num_rounds=50, num_clients=2, broker_address="localhost", broker_port=1883):
        # Define o número total de rodadas de aprendizado federado.
        self.num_rounds = num_rounds
        # Define o número esperado de clientes.
        self.num_clients = num_clients
        # Endereço do broker MQTT.
        self.broker_address = broker_address
        # Porta do broker MQTT.
        self.broker_port = broker_port
        # Inicializa uma instância da rede neural global.
        self.global_net = FederatedNet()
        # Obtém os parâmetros iniciais da rede global.
        self.global_parameters = self.global_net.get_parameters()
        # O número da rodada atual.
        self.current_round = 0

        # Inicializa o cliente MQTT do servidor.
        # mqtt.CallbackAPIVersion.VERSION2 especifica o uso da API de callbacks da versão 2.0.
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "server")
        
        # Atribui os métodos de callback para eventos MQTT.
        self.client.on_connect = self.on_connect
        # Usa um wrapper para rotear mensagens para manipuladores específicos de tópicos.
        self.client.on_message = self._on_message_handler_wrapper 
        
        # Dicionário para armazenar os pesos recebidos de cada cliente por rodada.
        self.round_client_parameters = defaultdict(dict)
        # Conjunto para rastrear quais clientes já enviaram seus pesos na rodada atual.
        self.received_clients_in_round = set()
        # Conjunto para rastrear quais clientes já sinalizaram que estão prontos.
        self.connected_clients = set() 
        
        # Registra o tempo de início da rodada para calcular a duração.
        self.round_start_time = time.time()
        # NOVO: Lista para armazenar métricas detalhadas de cada rodada.
        self.round_metrics = [] 

        # NOVO: Define o tópico para enviar o sinal de término aos clientes.
        self.terminate_clients_topic = "client/terminate" 

    # Método de callback chamado quando o servidor se conecta ao broker MQTT.
    # 'properties' é um novo argumento na API v2.0 do paho-mqtt.
    def on_connect(self, client, userdata, flags, rc, properties):
        # Verifica se a conexão foi bem-sucedida (rc=0).
        if rc == 0:
            print("Servidor: Conectado ao broker MQTT com sucesso!")
            # Inscreve-se no tópico genérico para receber atualizações de parâmetros de todos os clientes.
            self.client.subscribe("client/updated_parameters/+") 
            # Inscreve-se no tópico para receber sinais de "pronto" dos clientes.
            self.client.subscribe("client/ready") 
            print("Servidor: Inscrito nos tópicos 'client/updated_parameters/+' e 'client/ready'.")
        else:
            print(f"Servidor: Falha na conexão, código de retorno: {rc}")

    # NOVO: Manipulador de mensagens para o tópico 'client/ready'.
    def on_client_ready_message(self, client, userdata, msg):
        try:
            # Decodifica o ID do cliente da payload da mensagem.
            client_id = int(msg.payload.decode('utf-8'))
            # Se o cliente ainda não estiver na lista de clientes conectados, o adiciona.
            if client_id not in self.connected_clients:
                self.connected_clients.add(client_id)
                print(f"Servidor: Cliente {client_id} sinalizou estar pronto. Total de clientes prontos: {len(self.connected_clients)}/{self.num_clients}")
        # Captura erro se a payload não puder ser convertida para int.
        except ValueError:
            print(f"Servidor: Mensagem 'ready' mal formatada de {msg.topic}: {msg.payload}")

    # Manipulador de mensagens para o tópico 'client/updated_parameters/+'.
    def on_updated_parameters_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload
        try:
            # Extrai o ID do cliente do tópico da mensagem.
            client_id = int(topic.split('/')[-1])
        except ValueError:
            print(f"Servidor: Tópico inesperado ou mal formatado: {topic}")
            return

        # Des-serializa os parâmetros atualizados recebidos do cliente.
        parameters = pickle.loads(payload)
        
        # Garante que haja um dicionário para a rodada atual nos parâmetros do cliente.
        if self.current_round not in self.round_client_parameters:
            self.round_client_parameters[self.current_round] = {}

        # Armazena os parâmetros recebidos do cliente específico para a rodada atual.
        self.round_client_parameters[self.current_round][client_id] = parameters
        # Adiciona o ID do cliente ao conjunto de clientes que já enviaram pesos nesta rodada.
        self.received_clients_in_round.add(client_id)
        
        print(f"Servidor: Recebido parâmetros do cliente {client_id} para a rodada {self.current_round}.")

        # Verifica se todos os clientes esperados já enviaram seus pesos para a rodada atual.
        if len(self.received_clients_in_round) == self.num_clients:
            # Calcula a duração total da rodada.
            end_round_time = time.time()
            round_duration = end_round_time - self.round_start_time

            print(f"\n{'-'*50}")
            print(f"Servidor: Rodada {self.current_round} concluída pelos clientes.")
            print(f"Servidor: Tempo total da Rodada {self.current_round}: {round_duration:.2f} segundos.")
            
            aggregation_start_time = time.time() # NOVO: Registra tempo de início da agregação.
            self.aggregate_parameters() # Chama o método para agregar os pesos.
            aggregation_end_time = time.time() # NOVO: Registra tempo de fim da agregação.
            aggregation_time = aggregation_end_time - aggregation_start_time # NOVO: Calcula duração da agregação.

            # NOVO: Coleta de métricas da rodada para registro.
            # Tamanho dos parâmetros globais enviados pelo servidor (download para clientes).
            current_round_data_sent_per_client = sys.getsizeof(pickle.dumps(self.global_parameters)) 
            # Média do tamanho dos parâmetros atualizados recebidos dos clientes (upload de clientes).
            current_round_data_received_per_client = np.mean([sys.getsizeof(pickle.dumps(p)) for p in self.round_client_parameters[self.current_round].values()])
            
            # NOVO: Adiciona as métricas da rodada atual à lista de histórico.
            self.round_metrics.append({
                'round_num': self.current_round,
                'round_duration': round_duration,
                'aggregation_time': aggregation_time,
                'data_sent_per_client_kb': current_round_data_sent_per_client / 1024,
                'data_received_per_client_kb': current_round_data_received_per_client / 1024
            })
            
            # NOVO: Exibe métricas detalhadas da rodada no terminal.
            print(f"Servidor: Agregação concluída em {aggregation_time:.4f} segundos.")
            print(f"Servidor: Média de dados enviados aos clientes nesta rodada: {current_round_data_sent_per_client / 1024:.2f} KB/cliente")
            print(f"Servidor: Média de dados recebidos dos clientes nesta rodada: {current_round_data_received_per_client / 1024:.2f} KB/cliente")
            # Estimativa do total de dados transferidos (enviado + recebido) para todos os clientes nesta rodada.
            print(f"Servidor: Total de dados transferidos (aprox.) nesta rodada: {((current_round_data_sent_per_client * self.num_clients) + (current_round_data_received_per_client * self.num_clients)) / 1024:.2f} KB")


            self.current_round += 1 # Incrementa o contador da rodada.
            self.received_clients_in_round.clear() # Limpa o conjunto de clientes recebidos para a próxima rodada.
            
            # Verifica se ainda há rodadas a serem executadas.
            if self.current_round < self.num_rounds:
                print(f"Servidor: Iniciando Rodada {self.current_round + 1} de {self.num_rounds}.")
                self.distribute_global_parameters() # Distribui os novos parâmetros globais.
                self.round_start_time = time.time() # Reinicia o timer para a nova rodada.
            else:
                # Se todas as rodadas foram concluídas.
                print(f"\n{'='*50}")
                print("Servidor: Treinamento federado concluído!")
                print(f"Servidor: Total de Rodadas Executadas: {self.num_rounds}")
                print(f"{'='*50}\n")
                
                # NOVO: Exibe o resumo das métricas de todas as rodadas no terminal.
                print("\n--- RESUMO DAS MÉTRICAS POR RODADA ---")
                for r_metrics in self.round_metrics:
                    print(f"Rodada {r_metrics['round_num']}: Tempo Rodada {r_metrics['round_duration']:.2f}s | Agregação {r_metrics['aggregation_time']:.4f}s | Dados enviados {r_metrics['data_sent_per_client_kb']:.2f}KB/c | Dados recebidos {r_metrics['data_received_per_client_kb']:.2f}KB/c")
                print("--------------------------------------\n")

                self.save_global_parameters() # Salva o modelo global final.
                # NOVO: Publica uma mensagem no tópico de término para que os clientes encerrem.
                self.client.publish(self.terminate_clients_topic, "TERMINATE", qos=1) 
                print(f"Servidor: Sinal de término enviado aos clientes em '{self.terminate_clients_topic}'.")
                self.client.disconnect() # Desconecta o cliente MQTT do servidor.
                sys.exit(0) # Termina o script do servidor.
            print(f"{'-'*50}\n")

    # NOVO: Wrapper para on_message para direcionar mensagens para os manipuladores corretos.
    # O paho-mqtt chama apenas um on_message, então este método decide qual manipulador chamar
    # com base no tópico da mensagem recebida.
    def _on_message_handler_wrapper(self, client, userdata, msg):
        # Se a mensagem for um sinal de cliente pronto.
        if msg.topic == "client/ready":
            self.on_client_ready_message(client, userdata, msg)
        # Se a mensagem for de parâmetros atualizados de um cliente.
        elif msg.topic.startswith("client/updated_parameters/"):
            self.on_updated_parameters_message(client, userdata, msg)
        else:
            print(f"Servidor: Mensagem recebida em tópico não esperado: {msg.topic}")


    # Método para agregar os parâmetros (pesos) recebidos dos clientes.
    def aggregate_parameters(self):
        # Inicializa um novo dicionário de parâmetros com zeros, com a mesma estrutura do modelo global.
        new_parameters = {name: {'weight': torch.zeros_like(param['weight']), 'bias': torch.zeros_like(param['bias'])} 
                          for name, param in self.global_parameters.items()}

        # Agregação ponderada (média simples neste caso, assumindo datasets de tamanhos similares).
        # Itera sobre os IDs dos clientes que enviaram parâmetros na rodada atual.
        for client_id in self.round_client_parameters[self.current_round]:
            client_params = self.round_client_parameters[self.current_round][client_id]
            # Itera sobre as camadas (nome) e seus parâmetros.
            for name in client_params:
                # Soma os pesos de cada cliente e divide pelo número total de clientes (média).
                new_parameters[name]['weight'] += client_params[name]['weight'] / self.num_clients
                # Soma os bias de cada cliente e divide pelo número total de clientes (média).
                new_parameters[name]['bias'] += client_params[name]['bias'] / self.num_clients
        
        # Aplica os parâmetros agregados à rede global do servidor.
        self.global_net.apply_parameters(new_parameters)
        # Atualiza a referência dos parâmetros globais do servidor.
        self.global_parameters = new_parameters
        print(f"Servidor: Parâmetros globais atualizados para a rodada {self.current_round}.")

    # Método para distribuir os parâmetros iniciais aos clientes no começo do treinamento.
    def distribute_initial_parameters(self):
        # Serializa os parâmetros globais para bytes para envio via MQTT.
        parameters_bytes = pickle.dumps(self.global_parameters)
        # Loop para publicar os parâmetros para cada cliente.
        for client_id in range(self.num_clients):
            # Publica no tópico exclusivo de cada cliente.
            self.client.publish(f"server/initial_parameters/{client_id}", parameters_bytes, qos=1)
            print(f"Servidor: Parâmetros iniciais enviados para o cliente {client_id}")
        # Reinicia o timer da rodada.
        self.round_start_time = time.time()

    # Método para distribuir os parâmetros globais atualizados aos clientes em cada nova rodada.
    def distribute_global_parameters(self):
        # Serializa os parâmetros globais para bytes.
        parameters_bytes = pickle.dumps(self.global_parameters)
        # Loop para publicar os parâmetros para cada cliente.
        for client_id in range(self.num_clients):
            # Publica no tópico exclusivo de cada cliente.
            self.client.publish(f"server/global_parameters/{client_id}", parameters_bytes, qos=1)

    # Método para salvar o modelo global em um arquivo.
    def save_global_parameters(self):
        # NOVO: Define o caminho de saída do arquivo na mesma pasta do script do servidor.
        output_path = os.path.join(os.path.dirname(__file__), "global_parameters.pkl")
        # Abre o arquivo em modo binário de escrita.
        with open(output_path, 'wb') as f:
            # Serializa e salva os parâmetros globais.
            pickle.dump(self.global_parameters, f)
        print(f"Servidor: Parâmetros do modelo global final salvos em {output_path}")

    # Método para iniciar o servidor MQTT e seu loop de execução.
    def start(self):
        try:
            # Tenta conectar ao broker MQTT.
            self.client.connect(self.broker_address, self.broker_port, 60)
            # Inicia o loop de rede em um thread separado para processar mensagens em segundo plano.
            self.client.loop_start() 
            
            print(f"\n{'='*50}")
            print(f"Servidor: Aguardando {self.num_clients} clientes se conectarem e sinalizarem que estão prontos...")
            print(f"{'='*50}\n")
            
            # NOVO: Espera ativa pelos clientes sinalizarem que estão prontos.
            while len(self.connected_clients) < self.num_clients:
                time.sleep(1) # Aguarda 1 segundo para evitar uso excessivo da CPU.
            
            print(f"\n{'='*50}")
            print(f"Servidor: Todos os {self.num_clients} clientes estão prontos! Iniciando Rodada 0 de {self.num_rounds}.")
            print(f"{'='*50}\n")
            # Distribui os parâmetros iniciais para começar o treinamento.
            self.distribute_initial_parameters()
            
            # Loop principal do servidor que continua até que todas as rodadas sejam concluídas.
            while self.current_round < self.num_rounds:
                time.sleep(1) # Mantém o servidor ativo aguardando as atualizações dos clientes.
                
        # Captura qualquer exceção (incluindo KeyboardInterrupt) durante a execução do servidor.
        except Exception as e:
            print(f"Servidor: Erro durante a execução: {e}")
        # O bloco finally é sempre executado, independentemente de exceções ou término manual.
        finally:
            # Para o loop de rede do cliente MQTT.
            self.client.loop_stop()
            # Desconecta o cliente do broker MQTT.
            self.client.disconnect()
            # Condicional para salvar o modelo em caso de interrupção.
            if self.current_round > 0 and self.current_round < self.num_rounds:
                # Se o treinamento foi interrompido, mas já havia começado.
                print("\nServidor: Interrupção detectada. Salvando o estado atual do modelo global...")
                self.save_global_parameters()
            elif self.current_round == 0:
                # Se o treinamento foi interrompido antes mesmo de iniciar a primeira rodada.
                print("\nServidor: Interrupção detectada antes do início do treinamento. Modelo não salvo.")

# Bloco executado apenas se o script for rodado diretamente.
if __name__ == "__main__":
    # Define o número total de rodadas. Ajuste conforme a necessidade de acurácia.
    num_rounds_server = 2 
    # Define o número de clientes que se conectarão ao servidor.
    num_clients_server = 3 

    # Cria uma instância do Servidor e a inicia.
    server_instance = Server(num_rounds=num_rounds_server, num_clients=num_clients_server)
    server_instance.start()
