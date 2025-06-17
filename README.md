# Projeto-Redes 1-2025.1

# Aprendizado Federado com CIFAR-10 usando FedAVG e MQTT

Buscamos, no projeto, implementar um sistema de Aprendizado Federado para treinar um modelo de classificação de imagens no dataset CIFAR-10. O sistema utiliza o algoritmo Federated Averaging (FedAVG) para agregar modelos treinados de modo local pelos clientes e o protocolo MQTT para a comunicação entre um servidor central e múltiplos clientes. Em nossos testes, O Servidor Central era Miguel Berzin, e os clientes Sérgio Henrique, Paulo Massa e Hugo Andrade.

## Objetivos da Segunda Fase do Projeto
O principal objetivo desta fase é implementar um sistema de aprendizado federado, onde:
* Cada cliente treina um modelo de aprendizado de máquina localmente em um subconjunto exclusivo do dataset CIFAR-10.
* Os parâmetros (pesos) dos modelos treinados localmente são enviados pelos clientes para um servidor central.
* O servidor central agrega os parâmetros recebidos usando o algoritmo FedAVG.
* O modelo global atualizado é redistribuído para os clientes para novas rodadas de treinamento.
* São realizadas medições e análises da quantidade de dados transmitidos, tempo de comunicação/processamento e requisitos computacionais.
---

## Arquitetura do Sistema🏛️

O sistema é composto por três componentes principais:

1.  **Servidor Central (`server.py`)**:
    * Inicializa um modelo global de rede neural.
    * Gerencia o ciclo de treinamento federado, coordenando as rodadas.
    * Aguarda que os clientes se conectem e sinalizem que estão prontos.
    * Distribui os parâmetros do modelo global (inicial e atualizado) para os clientes.
    * Recebe os parâmetros dos modelos treinados localmente de cada cliente.
    * Agrega os parâmetros recebidos usando o algoritmo FedAVG para atualizar o modelo global.
    * Coleta e exibe métricas sobre o processo de treinamento (duração da rodada, tempo de agregação, volume de dados transferidos).
    * Salva o modelo global treinado ao final do processo.
    * Envia um sinal de término para os clientes ao concluir todas as rodadas.

2.  **Clientes (`client.py`)**:
    * Cada cliente opera com um subconjunto exclusivo do dataset CIFAR-10.
    * Recebe os parâmetros do modelo global do servidor.
    * Treina o modelo localmente em seus dados por um número definido de épocas.
    * Envia os parâmetros atualizados do seu modelo local de volta para o servidor.
    * Calcula e exibe métricas locais (tempo de treinamento, perda, acurácia, volume de dados enviados).
    * Repete o processo por várias rodadas até receber um sinal de término do servidor.

3.  **Broker MQTT**:
    * Atua como intermediário para a troca de mensagens (parâmetros do modelo, sinais de controle) entre o servidor e os clientes. Este projeto foi testado com o Mosquitto.

### Fluxo do Aprendizado Federado:

1.  **Inicialização**: O servidor inicializa o modelo global.
2.  **Distribuição Inicial**: O servidor envia os parâmetros do modelo global inicial para todos os clientes conectados e prontos.
3.  **Treinamento Local**: Cada cliente treina o modelo recebido usando seu conjunto de dados local.
4.  **Envio de Atualizações**: Os clientes enviam os parâmetros (pesos) de seus modelos treinados de volta para o servidor. Os dados brutos dos clientes nunca saem de seus dispositivos.
5.  **Agregação (FedAVG)**: O servidor agrega as atualizações recebidas (calculando a média dos parâmetros) para criar um novo modelo global aprimorado.
6.  **Redistribuição e Nova Rodada**: O servidor envia o modelo global atualizado para os clientes, iniciando uma nova rodada de treinamento.
7.  **Repetição**: Os passos 3-6 são repetidos por um número pré-definido de rodadas.
8.  **Término**: Ao final das rodadas, o servidor salva o modelo global final e envia um sinal para os clientes encerrarem.

### Dataset

* **CIFAR-10**: O projeto utiliza o dataset CIFAR-10, que consiste em 60.000 imagens coloridas de 32x32 pixels, divididas em 10 classes. O script `distribute_cifar10.py` divide o conjunto de treinamento deste dataset de forma Independente e Identicamente Distribuída (IID) entre os clientes.

---
```plaintext
## 📁 Estrutura de Arquivos
raiz_do_projeto/
├── README.md                   # Este arquivo
├── requirements.txt            # Dependências do projeto
├── CLIENTS/                    # Contém arquivos relacionados aos clientes
│   ├── client_0/               # Pasta específica para o cliente 0
│   │   ├── client.py           # Script principal do cliente (replicado para client_1, client_2)
│   │   └── data/               # Criada por distribute_cifar10.py para armazenar dados do cliente
│   │       └── cifar10_client_0.pkl # Dataset específico do cliente 0
│   ├── client_1/
│   │   ├── client.py
│   │   └── data/
│   │       └── cifar10_client_1.pkl
│   ├── client_2/               # Exemplo para 3 clientes
│   │   ├── client.py
│   │   └── data/
│   │       └── cifar10_client_2.pkl
│   └── distribute_cifar10.py   # Script para distribuir o dataset CIFAR-10 entre os clientes
├── COMMON/                     # Contém código comum ao servidor e clientes
│   └── federated_net.py        # Define a arquitetura da rede neural convolucional
└── SERVER/                     # Contém arquivos relacionados ao servidor
├── evaluate_global_model.py # Script para avaliar o modelo global treinado
├── server.py                # Script principal do servidor
└── global_parameters.pkl    # Criado pelo server.py ao salvar o modelo global treinado
```

**Atenção!**: A pasta `data_temp/` será criada na raiz do projeto pelos scripts `distribute_cifar10.py` e `evaluate_global_model.py` para baixar o dataset CIFAR-10, caso ainda não exista localmente.

---

## Pré-requisitos🛠️ 

* Python 3.8+
* Bibliotecas Python listadas em `requirements.txt`:
    * `torch`: Biblioteca de Deep-Learning.
    * `torchvision`: Utilities para datasets e modelos de visão computacional.
    * `paho-mqtt`: Cliente MQTT Python.
    * `numpy`
* Broker MQTT em execução (ex: Mosquitto).O arquivo `requirements.txt` menciona `mosquitto -c mosquitto.conf`.

---

## Como Executar🚀

Inicie o Broker MQTT:
   Caso você seja o host, Abra um novo terminal e inicie o broker Mosquitto. no arquivo `mosquitto.conf`, coloque as instruções abaixo:
   ```plaintext
   allow_anonymous true
   listener 1883 0.0.0.0
   listener 9001
   protocol websockets
   ```
    
---
  **Distribua o Dataset CIFAR-10 para os Clientes:**
    Este script irá baixar o CIFAR-10 (se necessário) e dividi-lo entre o número de clientes especificado. As pastas `clients/client_X/data/` serão criadas.
    Abra um terminal na raiz do projeto e execute:
    ```bash
    python CLIENTS/distribute_cifar10.py
    ```
    Como foi desenvolvido por um grupo de 4 pessoas, este script está configurado para 3 clientes. Mas podemos alterar a variável `num_clients` dentro do script `distribute_cifar10.py` caso precise de um número diferente de clientes.

  **Iniciação do Servidor:**
    Abra um novo terminal na raiz do projeto e execute:
    ```bash
    python server/server.py
    ```
    O servidor irá aguardar que o número esperado de clientes (configurado dentro de `server.py`) se conecte.

  **Inicie os Clientes:**
    Para cada cliente, abra um novo terminal na raiz do projeto e execute o script `client.py`, fornecendo o `client_id` (começando em 0) e o número de `epochs` para treinamento local.
    Você verá logs nos terminais do servidor e dos clientes mostrando o progresso do treinamento, envio e recebimento de parâmetros, agregação e métricas por rodada.
    Por exemplo, para 3 clientes e 5 épocas de treinamento local por rodada: 
    
    * **Terminal Cliente 0:**
        ```bash
        python clients/client_0/client.py 0 5
        ```
    * **Terminal Cliente 1:**
        ```bash
        python clients/client_1/client.py 1 5
        ```
    * **Terminal Cliente 2:**
        ```bash
        python clients/client_2/client.py 2 5
        ```
    **Importante**: O número de clientes iniciado deve corresponder ao `num_clients` configurado no `server.py`. As pastas `clients/client_X/` devem existir para cada cliente que você iniciar.

  **Avaliação do Modelo Global (Após o término do treinamento):**
    Após o servidor completar todas as rodadas de treinamento, ele salvará o modelo global final em `server/global_parameters.pkl`. Você pode avaliar a performance deste modelo no conjunto de teste do CIFAR-10 executando:
    ```bash
    python SERVER/evaluate_global_model.py
    ```

---

## Medições e Análises

Seguindo as orientações do projeto, o sistema foi desenvolvido justamente para permitir a coleta e análise de:

* **Quantidade de Dados Transmitidos:**
    * **Clientes**: O script `client.py` calcula e exibe o tamanho (em KB) dos parâmetros do modelo que são enviados ao servidor a cada rodada.
    * **Servidor**: O script `server.py` calcula e exibe:
        * O tamanho (em KB) dos parâmetros globais enviados a cada cliente por rodada.
        * A média do tamanho (em KB) dos parâmetros recebidos de cada cliente por rodada.
        * Uma estimativa do total de dados transferidos (enviados + recebidos agregados) por rodada.

* **Tempo Necessário para Comunicação e Processamento:**
    * **Clientes**: O script `client.py` calcula e exibe o tempo de treinamento local por rodada.
    * **Servidor**: O script `server.py` calcula e exibe:
        * O tempo total para cada rodada de aprendizado federado (desde a distribuição dos parâmetros até o recebimento de todas as atualizações dos clientes).
        * O tempo específico para a etapa de agregação dos parâmetros (FedAVG).

* **Requisitos Computacionais:**
    * Embora não medido explicitamente em termos de CPU/GPU ou uso de memória de forma contínua, o design do sistema (troca de parâmetros) é inerentemente mais eficiente em comparação com o treinamento centralizado tradicional.
    * O tempo de treinamento local nos clientes e o tempo de agregação no servidor fornecem proxies para a carga computacional em cada componente.
    * O tamanho dos modelos e a frequência das rodadas influenciam diretamente os requisitos.

Feito com ♥️ na POLI!
