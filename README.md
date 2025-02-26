DESCRIÇÃO DO PROJETO
    Este projeto visa desenvolver um sistema de monitoramento de desempenho acadêmico para prever e reduzir a evasão escolar. A solução se baseia em dados de notas e faltas, processados utilizando Apache Spark, R e Apache Airflow, permitindo a identificação precoce de alunos em risco de reprovação.
    A infraestrutura será automatizada com Vagrant, Ansible, Kubernetes e Docker, garantindo escalabilidade e eficiência. Jenkins será usado para CI/CD, enquanto Prometheus, Grafana e Loki fornecerão monitoramento e logs do sistema.

PROBLEMA A SER SOLUCIONADO
    A reprovação acadêmica pode levar à evasão escolar, especialmente quando os alunos acumulam notas baixas e faltas excessivas sem receber suporte adequado. Atualmente, muitos sistemas acadêmicos apenas registram os dados sem oferecer análises preditivas ou alertas preventivos.
    Sem um sistema de monitoramento proativo, professores e coordenadores só percebem o risco quando já é tarde demais para intervenção.

SOLUÇÃO PROPOSTA
    Desenvolver um sistema automatizado para monitoramento de desempenho acadêmico, que:
        - Coleta e processa dados de faltas e notas de alunos utilizando Apache Airflow e Apache Spark.
        - - Utiliza Machine Learning (R) para prever quais alunos estão em risco de reprovação.
        - Gera alertas automáticos para alunos e professores quando há risco de reprovação.
        - Disponibiliza um Dashboard Interativo para acompanhamento em tempo real.
        - Automatiza a infraestrutura com Vagrant e Ansible.
        - Utiliza Kubernetes e Docker para escalabilidade e orquestração.
        - CI/CD com Jenkins para deploy contínuo.
        - Monitoramento com Prometheus, Grafana e Loki.

TECNOLOGIAS UTILIZADAS
    - Apache Airflow – Para automação do fluxo de coleta e processamento dos dados.
    - Apache Spark – Para manipulação eficiente de grandes volumes de dados.
    - R (rpart, randomForest, ggplot2, dplyr) – Para análise estatística e modelagem preditiva.
    - Power BI/Shiny – Para visualização dos resultados.
    - SQL/NoSQL – Para armazenamento de dados.
    - API REST/Webhooks – Para integração e envio de notificações.
    - Docker & Kubernetes – Para containerização e orquestração de serviços.
    - Vagrant & Ansible – Para provisionamento e configuração automatizada da infraestrutura.
    - Jenkins – Para integração e entrega contínua (CI/CD).
    - Prometheus & Grafana – Para monitoramento de métricas.
    - Loki – Para centralização e análise de logs.

DIVISÃO DAS TAREFAS
    A equipe será dividida em 3 grupos principais, cada um responsável por tarefas específicas de uma matéria.
    Todos os membros trabalharão com Análise e Estatística para garantir que os modelos preditivos tenham fundamentos estatísticos adequados.

    Grupo 1: Análise Exploratória de Dados
        - Responsáveis: Luciano Aimon e Vinycius Oblonczyk
        - Objetivo: Trabalhar na geração, limpeza, organização e visualização dos dados antes da modelagem preditiva.

        - Tarefas
            Coletar e explorar os dados de notas e faltas.
            Realizar limpeza e tratamento dos dados.
            Aplicar estatísticas descritivas para entender padrões e tendências.
            Criar gráficos e relatórios iniciais no Power BI ou R (ggplot2).
            Validar quais variáveis influenciam mais na reprovação.
            Trabalhar com o cientista de dados para preparar os dados para modelagem.

    Grupo 2: Big Data
        - Responsáveis: Matheus Pinto e Bruno Cardoso
        - Objetivo: Criar um pipeline robusto para processamento de grandes volumes de dados.

        - Tarefas
            Implementar Apache Spark para manipulação de dados escaláveis.
            Desenvolver consultas eficientes para extrair e processar os dados.
            Criar rotinas para otimizar o tempo de processamento das análises.
            Integrar Apache Spark com Apache Airflow para automação do ETL.
            Trabalhar junto ao grupo de Machine Learning para fornecer os dados limpos

    Grupo 3: DevOps
        - Responsáveis: Igor Guilherme e Igor Gaspar
        - Objetivo: Configurar a infraestrutura automatizada, garantindo escalabilidade e monitoramento do sistema.

        - Tarefas
            Configurar Docker e Kubernetes para orquestração de containers.
            Criar ambientes de desenvolvimento automatizados com Vagrant e Ansible.
            Implementar Jenkins para CI/CD e deploy contínuo do sistema.
            Configurar Prometheus e Grafana para monitoramento de métricas de desempenho.
            Configurar Loki para gerenciamento e análise de logs.
            Trabalhar junto ao backend para garantir que as APIs funcionem corretamente.

    Análise e Estatística (Todos os Membros)
        - Objetivo: Aplicar estatísticas e Machine Learning para prever quais alunos têm maior risco de reprovação.

        - Tarefas
            Criar modelos preditivos utilizando R (rpart, randomForest, glm).
            Calcular acurácia, recall e precisão dos modelos.
            Ajustar os hiperparâmetros para melhorar a performance do modelo.
            Desenvolver um sistema de alertas para alunos em risco de reprovação.
            Avaliar se os modelos realmente ajudam a reduzir a evasão escolar.

PAPEL DOS LIDERES DO PROJETO
    Luciano Aimon – Scrum Master:
        Responsável por facilitar as cerimônias ágeis e remover impedimentos da equipe.
    Igor Gaspar – Product Owner: 
        Responsável por garantir que as entregas do projeto atendam às necessidades dos usuários e stakeholders.