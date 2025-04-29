import mysql.connector
import pandas as pd

# Conectar ao banco de dados MySQL
conexao = mysql.connector.connect(
    host="localhost",      # Altere se necessário
    user="root",    # Seu usuário do MySQL
    password="",  # Sua senha do MySQL
    database="faculdade"   # Nome do banco de dados
)

# Criar um cursor e executar a consulta
consulta = "SELECT * FROM alunos_detalhes;"
df = pd.read_sql(consulta, conexao)

consulta_frequencia = "SELECT * FROM frequencia;"
df_frequencia = pd.read_sql(consulta_frequencia, conexao)

consulta_notas = "SELECT * FROM notas;"
df_notas = pd.read_sql(consulta_notas, conexao)

# Verificar as primeiras linhas para garantir que os dados estão corretos
print(df.head())
print(df_frequencia.head())
print(df_notas.head())

# Exportar para CSV com separação de colunas
df.to_csv("alunos_detalhes.csv", index=False, encoding="utf-8", sep=";")
df_frequencia.to_csv("frequencia_detalhes.csv", index=False, encoding="utf-8", sep=";")
df_notas.to_csv("notas_detalhes.csv", index=False, encoding="utf-8", sep=";")

# Fechar conexão
conexao.close()

print("Dados exportados para 'alunos_detalhes.csv' com sucesso!")
