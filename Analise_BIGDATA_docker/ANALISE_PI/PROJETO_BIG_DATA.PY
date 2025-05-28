# ===============================
# 📦 Importação das Dependências
# ===============================

# 🔢 Manipulação de Dados
import pandas as pd
import numpy as np
import time

# 📊 Visualização de Dados
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pivottablejs import pivot_ui

# 🤖 Machine Learning - Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    silhouette_score,
    r2_score,
    mean_squared_error
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 🧠 Aprendizado de Máquina - Balanceamento de Dados
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 📚 Processamento de Linguagem Natural (PLN)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# ⚡ Big Data com PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix

# ================================
# 📊 Visualização Inicial com mysql

# Configuração da conexão
usuario = 'root'
senha = ''
host = 'localhost'
porta = 3306
nome_banco = 'faculdade'

# String de conexão
engine = create_engine(f'mysql+pymysql://{usuario}:{senha}@{host}:{porta}/{nome_banco}')

#dw 
# String de conexão geral
engine = create_engine(f'mysql+pymysql://{usuario}:{senha}@{host}:{porta}/{nome_banco}')

# Função para atualizar o DW (ETL)
def atualizar_dw():
    print("Atualizando o Data Warehouse...")

    # Conexões
    engine_operacional = create_engine("mysql+pymysql://root:@localhost:3306/faculdade")
    engine_dw = create_engine("mysql+pymysql://root:@localhost:3306/faculdade_dw")

    # Extrair dados da base operacional
    df_alunos = pd.read_sql("SELECT * FROM alunos", con=engine_operacional)

    # Dimensões
    dim_aluno = df_alunos[['aluno_id', 'nome_aluno', 'sexo', 'idade']].drop_duplicates()
    
    dim_tempo = df_alunos[['semestre', 'bimestre']].drop_duplicates().reset_index(drop=True)
    dim_tempo['id_tempo'] = dim_tempo.index + 1

    dim_curso = df_alunos[['curso']].drop_duplicates().reset_index(drop=True)
    dim_curso['id_curso'] = dim_curso.index + 1

    dim_risco = pd.DataFrame({
        'id_risco': [0, 1, 2],
        'descricao': ['Baixo', 'Médio', 'Alto']
    })

    # Tabela fato
    fato = df_alunos[['aluno_id', 'semestre', 'bimestre', 'curso', 'risco_evasao']].copy()
    fato = fato.merge(dim_tempo, on=['semestre', 'bimestre'], how='left')
    fato = fato.merge(dim_curso, on='curso', how='left')
    fato['id_risco'] = fato['risco_evasao'].fillna(0).astype(int)

    # Carregar no DW
    dim_aluno.to_sql('dim_aluno', con=engine_dw, if_exists='replace', index=False)
    dim_tempo.to_sql('dim_tempo', con=engine_dw, if_exists='replace', index=False)
    dim_curso.to_sql('dim_curso', con=engine_dw, if_exists='replace', index=False)
    dim_risco.to_sql('dim_risco', con=engine_dw, if_exists='replace', index=False)
    fato[['aluno_id', 'id_tempo', 'id_curso', 'id_risco']].to_sql('fato_evasao', con=engine_dw, if_exists='replace', index=False)

    print("Data Warehouse atualizado com sucesso!")

# Executar ETL
atualizar_dw()

# Conectar ao DW
engine_dw = create_engine("mysql+pymysql://root:@localhost:3306/faculdade_dw")

dim_aluno = pd.read_sql("SELECT * FROM dim_aluno", con=engine_dw)
dim_tempo = pd.read_sql("SELECT * FROM dim_tempo", con=engine_dw)
dim_curso = pd.read_sql("SELECT * FROM dim_curso", con=engine_dw)
dim_risco = pd.read_sql("SELECT * FROM dim_risco", con=engine_dw)
fato = pd.read_sql("SELECT * FROM fato_evasao", con=engine_dw)

# Montar df_olap
df_olap = fato.merge(dim_aluno, on='aluno_id') \
              .merge(dim_tempo, on='id_tempo') \
              .merge(dim_curso, on='id_curso') \
              .merge(dim_risco, on='id_risco')

# Preparar OLAP visual
# Remover colunas técnicas
df_olap_vis = df_olap.drop(columns=['aluno_id', 'id_tempo', 'id_curso', 'id_risco'], errors='ignore')

# Renomear colunas
df_olap_vis.rename(columns={
    'nome_aluno': 'Nome do Aluno',
    'sexo': 'Sexo',
    'idade': 'Idade',
    'semestre': 'Semestre',
    'bimestre': 'Bimestre',
    'curso': 'Curso',
    'descricao': 'Nível de Risco'
}, inplace=True)

# Selecionar colunas úteis
colunas_uteis = [
    'Nome do Aluno', 'Sexo', 'Idade', 'Curso',
    'Semestre', 'Bimestre', 'Nível de Risco'
]
df_olap_vis = df_olap_vis[[col for col in colunas_uteis if col in df_olap_vis.columns]]

# Criar OLAP interativo (padrão)
saida_olap = 'olap_interativo.html'
pivot_ui(df_olap_vis, outfile_path=saida_olap)

print(f"OLAP gerado com sucesso em: {saida_olap}") #da pra colocar automático se precisar

# ===============================

# Consulta os dados da tabela alunos
query = "SELECT * FROM alunos"

df = pd.read_sql(query, con=engine)

# Conferir os dados
print(df.head())
print(df.info())
print(df.isnull().sum())
# Verificar os tipos de dados
print(df.dtypes)
# Verificar os valores únicos de cada coluna
for col in df.columns:
    print(f"Valores únicos em {col}: {df[col].unique()}")
    

# =================================

# ===============================
# 📌 Pré-processamento (Pandas)
# ===============================

# Remover registros com valores nulos e criar cópia
df_model = df.dropna().copy()

# Label Encoding para colunas categóricas (exceto target)
label_encoders = {}
for c in df_model.select_dtypes(include='object').columns:
    if c != 'risco_evasao':
        le = LabelEncoder()
        df_model.loc[:, c] = le.fit_transform(df_model[c].astype(str))
        label_encoders[c] = le

# Separar features e target
X = df_model.drop(columns=['risco_evasao'])
y = df_model['risco_evasao']

# Codificar target
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# Normalizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 📊 Modelo de Classificação (Random Forest)
# ===============================

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=le_y.classes_.astype(str)))

# ===============================
# 📉 Regressão Linear (Ridge) para Desempenho e Risco
# ===============================

#Explicação com base nos nossos resultados 

# =============================================================
# Explicação do bloco de Regressão Linear Ridge aplicada no projeto.
# =============================================================

# Neste projeto, foi utilizada a técnica de Regressão Linear com penalização Ridge 
# para prever as variáveis de desempenho acadêmico dos alunos e o risco de evasão.

# A escolha deste modelo se justifica pela sua capacidade de lidar com multicolinearidade 
# entre variáveis preditoras — um cenário comum em bases educacionais com muitos atributos 
# inter-relacionados.

# Para cada variável-alvo (desempenho_1, desempenho_2, ..., risco_evasao), 
# o modelo foi treinado e avaliado utilizando duas métricas principais:

# R² (Coeficiente de Determinação): 
# - Indica a proporção da variância da variável-alvo explicada pelas variáveis preditoras.
# - Valores próximos de 1 indicam bom desempenho.

# MSE (Mean Squared Error): 
# - Quantifica o erro médio quadrático entre os valores reais e os preditos.
# - Quanto menor, melhor.

# Os resultados médios obtidos pelos modelos foram:

# - Média dos R²: 0.78 
# - Média dos MSE: 1.42

# Isso indica que, em média, os modelos foram capazes de explicar 78% da variação 
# das variáveis de interesse, com um erro médio quadrado de 1.42.

# Essa abordagem permite identificar quais fatores (socioeconômicos, demográficos e acadêmicos) 
# possuem maior influência sobre o desempenho dos alunos e sobre o risco de evasão, 
# permitindo à instituição planejar intervenções mais eficazes e direcionadas.
# FIM DA EXPLICAÇÃO

# Importar bibliotecas necessárias
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# ✅ Conectar ao banco e carregar dados no pandas
engine = create_engine("mysql+pymysql://root:@localhost:3306/faculdade")
df_model = pd.read_sql("SELECT * FROM alunos", con=engine)

# ✅ Lista de targets a treinar
targets_reg = ['desempenho_1', 'desempenho_2', 'desempenho_3', 'desempenho_4', 'desempenho_5', 'risco_evasao']

# ✅ Armazenar resultados
resultados_r2 = []
resultados_mse = []

# ✅ Loop para treinar regressão para cada target
for target in targets_reg:
    if target in df_model.columns:
        # ✅ Remover colunas não usadas e separar target
        X_reg = df_model.drop(columns=targets_reg + ['nome_aluno', 'email', 'professor_1', 'professor_2',
                                                     'professor_3', 'professor_4', 'professor_5'], errors='ignore')
        y_reg = df_model[target]

        # ✅ Verificar variação no target
        if y_reg.nunique() < 2:
            print(f"⚠️ Coluna {target} não tem variação suficiente para regressão.")
            continue

        # ✅ Se target for categórico, aplicar LabelEncoder
        if y_reg.dtype == 'object':
            le = LabelEncoder()
            y_reg = le.fit_transform(y_reg)

        # ✅ Transformar variáveis categóricas em dummies
        X_reg_encoded = pd.get_dummies(X_reg, drop_first=True)

        # ✅ Dividir dados treino/teste
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_reg_encoded, y_reg, test_size=0.2, random_state=42
        )

        # ✅ Pipeline com imputação, normalização e regressão Ridge
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('regressor', Ridge())
        ])

        # ✅ Treinar pipeline
        pipeline.fit(X_train_r, y_train_r)

        # ✅ Fazer predições
        y_pred_r = pipeline.predict(X_test_r)

        # ✅ Avaliar o modelo
        r2 = r2_score(y_test_r, y_pred_r)
        mse = mean_squared_error(y_test_r, y_pred_r)

        resultados_r2.append(r2)
        resultados_mse.append(mse)

        print(f"📊 {target} - R²: {r2:.4f}, MSE: {mse:.4f}")

    else:
        print(f"❌ Coluna {target} não encontrada.")

# ✅ Calcular médias das métricas
if resultados_r2:
    media_r2 = sum(resultados_r2) / len(resultados_r2)
    media_mse = sum(resultados_mse) / len(resultados_mse)

    print(f"\n📈 Média dos R²: {media_r2:.4f}")
    print(f"📉 Média dos MSE: {media_mse:.4f}")

    # 📊 Gráfico de barras das médias
    plt.figure(figsize=(10,5))
    plt.bar(["R² Médio", "MSE Médio"], [media_r2, media_mse], color=['#2ecc71', '#e74c3c'])
    plt.title("📊 Média das Métricas das Regressões (5 Desempenhos + Risco)")
    plt.ylabel("Valor")
    plt.ylim(0, max(media_r2, media_mse) * 1.3)
    for i, v in enumerate([media_r2, media_mse]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

else:
    print("❌ Nenhuma regressão válida foi executada.")
# ===============================
# 📉 Fim Regressão Linear
# ===============================


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix

# ===============================
# 🔧 Iniciar sessão Spark
# ===============================

spark = SparkSession.builder \
    .appName("PrevisaoRiscoEvasao") \
    .config("spark.jars", "file:///C:/pyspark_drivers/mysql-connector-j-9.3.0.jar") \
    .getOrCreate()

# ===============================
# 📥 Leitura dos dados via JDBC
# ===============================
df_spark = spark.read.format("jdbc").options(
    url="jdbc:mysql://localhost:3306/faculdade",
    driver="com.mysql.cj.jdbc.Driver",
    dbtable="alunos",
    user="root",
    password=""
).load()

df_spark.printSchema()
df_spark.show(5)

# ===============================
# 🧼 Limpeza e conversão
# ===============================
colunas_numericas = ["idade", "renda_familiar", "faltas_1", "notas_1", "faltas_2", "notas_2",
                     "faltas_3", "notas_3", "faltas_4", "notas_4", "faltas_5", "notas_5"]
colunas_categoricas = ["sexo", "trabalha", "acompanhamento_medico", "tem_filho",
                       "estado_civil", "desempenho_1", "desempenho_2",
                       "desempenho_3", "desempenho_4", "desempenho_5", "risco_evasao"]

for coln in colunas_numericas:
    df_spark = df_spark.withColumn(coln, col(coln).cast("double"))
    df_spark = df_spark.na.fill({coln: -1})

for colc in colunas_categoricas:
    df_spark = df_spark.na.fill({colc: "desconhecido"})
    indexer = StringIndexer(inputCol=colc, outputCol=colc + "_idx", handleInvalid="keep")
    df_spark = indexer.fit(df_spark).transform(df_spark)

# ===============================
# 🔄 Vetorização
# ===============================
feature_cols = colunas_numericas + [col + "_idx" for col in colunas_categoricas[:-1]]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
df_spark = assembler.transform(df_spark)

# ===============================
# 🌳 Random Forest Classifier
# ===============================
rf = RandomForestClassifier(labelCol="risco_evasao_idx", featuresCol="features_vec")
model = rf.fit(df_spark)
predictions = model.transform(df_spark)

# ===============================
# 📊 Avaliação
# ===============================
evaluator = MulticlassClassificationEvaluator(
    labelCol="risco_evasao_idx", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Acurácia do modelo: {accuracy:.4f}")

# ===============================
# 📈 Visualizações
# ===============================
# Coleta os dados manualmente
data = predictions.select("risco_evasao_idx", "prediction").collect()

# Converte para pandas
preds_pd = pd.DataFrame(data, columns=["risco_evasao_idx", "prediction"])

# ===============================
# 📈 Visualizações com Pandas, Seaborn e Matplotlib
# ===============================
# Coleta os dados manualmente
data = predictions.select("risco_evasao_idx", "prediction").collect()

# Converte para pandas
preds_pd = pd.DataFrame(data, columns=["risco_evasao_idx", "prediction"])

# ===============================
# 📊 Converter para Pandas e gerar gráfico
# ===============================

# Converter para Pandas
data = df_spark.collect()
df_pandas = pd.DataFrame([row.asDict() for row in data])

# Tratar valores nulos
df_pandas['risco_evasao'] = df_pandas['risco_evasao'].fillna('Desconhecido')

# ===============================
# 📋 Previsões Spark para análise detalhada
# ===============================

# Mapear as classes numéricas para texto (exemplo)
map_labels = {0: "Baixo", 1: "Médio", 2: "Alto"}

# Converter previsões Spark para Pandas
predictions_pandas = predictions.select("aluno_id", "prediction").toPandas()
predictions_pandas['risco_predito'] = predictions_pandas['prediction'].map(map_labels)

# Ajustar tipos para merge
df_model['aluno_id'] = df_model['aluno_id'].astype(str)
predictions_pandas['aluno_id'] = predictions_pandas['aluno_id'].astype(str)

# Remover coluna risco_predito anterior para evitar conflito
if 'risco_predito' in df_model.columns:
    df_model = df_model.drop(columns=['risco_predito'])

# Merge das previsões Spark no df_model
df_model = df_model.merge(predictions_pandas[['aluno_id', 'risco_predito']], on='aluno_id', how='inner')

print("Colunas após merge:", df_model.columns)
print(df_model.head())

# Filtrar alunos de alto risco
alunos_alto_risco = df_model[df_model['risco_predito'] == 'Alto']
print(alunos_alto_risco[['nome_aluno', 'desempenho_1', 'faltas_1', 'notas_1']].head())

# Gráfico percentual dos riscos preditos
# Calcular os percentuais
risco_counts = df_model['risco_predito'].value_counts(normalize=True) * 100

# Converter para DataFrame para plotar
df_risco_counts = risco_counts.reset_index()
df_risco_counts.columns = ['risco_predito', 'percentual']

# Gráfico interativo
fig = px.bar(
    df_risco_counts,
    x='risco_predito',
    y='percentual',
    color='risco_predito',
    text='percentual',
    color_discrete_sequence=px.colors.qualitative.Set1,
    title='Distribuição percentual dos riscos preditos'
)

fig.update_layout(
    yaxis_title="Porcentagem (%)",
    xaxis_title="Risco Predito",
    showlegend=False
)

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

fig.show()

# Variáveis (features) que vai usar para treinar (sem identificadores e target)
features_all = [
    "curso", "status", "turma", "sexo", "idade", "trabalha", "renda_familiar", "acompanhamento_medico",
    "tem_filho", "estado_civil", "semestre", "bimestre",
    "aula_1", "professor_1", "notas_1", "faltas_1", "desempenho_1",
    "aula_2", "professor_2", "notas_2", "faltas_2", "desempenho_2",
    "aula_3", "professor_3", "notas_3", "faltas_3", "desempenho_3",
    "aula_4", "professor_4", "notas_4", "faltas_4", "desempenho_4",
    "aula_5", "professor_5", "notas_5", "faltas_5", "desempenho_5"
]

# Preprocessar seu dataframe (exemplo rápido, ajuste conforme seu caso):
X = df_model[features_all].copy()
y = df_model['risco_evasao']

# Se tem variáveis categóricas, transforme em numérico (LabelEncoder ou get_dummies)
X_encoded = pd.get_dummies(X, drop_first=True)  # codifica as categorias

# Treina modelo RandomForest
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_spark = assembler.transform(df_spark)

rf = RandomForestClassifier(labelCol="risco_evasao_idx", featuresCol="features", numTrees=100)
model = rf.fit(df_spark)

# Extrai importância das features
importances = model.featureImportances.toArray()

# Nomes das features após o get_dummies
feature_names = assembler.getInputCols()

# 4. Organizar para visualização
df_imp = pd.DataFrame({
    'Variável': feature_names,
    'Importância': importances
}).sort_values(by='Importância', ascending=False)

# 5. Gráfico interativo
fig = px.bar(
    df_imp.head(10),
    x='Importância',
    y='Variável',
    orientation='h',
    color='Importância',
    color_continuous_scale='Viridis',
    title='Importância das Variáveis para Predição do Risco de Evasão (PySpark)',
    labels={'Importância': 'Importância da Feature', 'Variável': 'Feature'}
)
fig.update_layout(yaxis=dict(categoryorder='total ascending'))
fig.show()

# Copiar e preparar dados, garantir que coluna de evasão está presente e limpa
df_plot = df.copy().dropna(subset=['renda_familiar', 'risco_evasao'])

# Converter para numérico, forçando erros para NaN
df_plot['renda_familiar'] = pd.to_numeric(df_plot['renda_familiar'], errors='coerce')

# Remover valores nulos resultantes da conversão
df_plot = df_plot.dropna(subset=['renda_familiar'])

# Criar categorias econômicas baseadas na renda familiar
df_plot['classe_economica'] = pd.cut(
    df_plot['renda_familiar'],
    bins=[-np.inf, 4000, 8000, np.inf],
    labels=['Baixa', 'Média', 'Alta']
)

# Assumindo que 'risco_evasao' é 1 para evadiu e 0 para não evadiu
# Filtrar apenas alunos que evadiram
df_evadidos = df_plot[df_plot['risco_evasao'] == 1]

# Contar número de evadidos por classe econômica
df_contagem = df_evadidos.groupby('classe_economica').size().reset_index(name='quantidade_evadidos')

# Se quiser mostrar barras mesmo para classes sem evadidos, garantimos todas as categorias
todas_classes = ['Baixa', 'Média', 'Alta']
df_contagem = df_contagem.set_index('classe_economica').reindex(todas_classes, fill_value=0).reset_index()

# Gráfico interativo
fig = px.bar(
    df_contagem,
    x='classe_economica',
    y='quantidade_evadidos',
    labels={'classe_economica': 'Classe Econômica', 'quantidade_evadidos': 'Quantidade de Alunos Evadidos'},
    title='Quantidade de Alunos Evadidos por Classe Econômica (Renda Familiar)',
    color='classe_economica',
    color_discrete_map={'Baixa': 'red', 'Média': 'orange', 'Alta': 'green'},
    width=700,
    height=500
)

fig.update_layout(
    yaxis=dict(title='Quantidade de Alunos Evadidos', dtick=1),
    xaxis=dict(title='Classe Econômica'),
    showlegend=False,
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
)

fig.show()

# Preparar dados: filtrar faltas e evasão
df_faltas = df.copy().dropna(subset=['faltas_1', 'risco_evasao'])

# Filtrar apenas alunos que evadiram
df_evadidos = df_faltas[df_faltas['risco_evasao'] == 1]

# Contar evadidos por número de faltas
df_faltas_count = df_evadidos.groupby('faltas_1').size().reset_index(name='quantidade_evadidos')

# Ordenar por número de faltas para visualização mais clara
df_faltas_count = df_faltas_count.sort_values(by='faltas_1')

# Gráfico interativo
fig = px.bar(
    df_faltas_count,
    x='faltas_1',
    y='quantidade_evadidos',
    labels={'faltas_1': 'Número de Faltas', 'quantidade_evadidos': 'Quantidade de Alunos Evadidos'},
    title='Quantidade de Alunos Evadidos por Número de Faltas',
    width=800,
    height=500
)

fig.update_layout(
    xaxis=dict(dtick=1),  # mostrar cada número de falta no eixo x
    yaxis=dict(dtick=1),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
)

fig.show()

# 1. Preparar os dados
df_evasao = df.copy().dropna(subset=['semestre', 'risco_evasao'])

# Filtrar apenas alunos com risco de evasão
df_evasao = df_evasao[df_evasao['risco_evasao'] == 1]

# 2. Contar evasões por semestre
evasao_por_semestre = df_evasao['semestre'].value_counts().reset_index()
evasao_por_semestre.columns = ['semestre', 'total_evasoes']

# 3. Calcular porcentagem do total
total_alunos_por_semestre = df['semestre'].value_counts().reset_index()
total_alunos_por_semestre.columns = ['semestre', 'total_alunos']

dados_plot = evasao_por_semestre.merge(total_alunos_por_semestre, on='semestre')
dados_plot['percentual_evasao'] = (dados_plot['total_evasoes'] / dados_plot['total_alunos'] * 100).round(1)

# 4. Criar gráfico interativo
fig = px.bar(
    dados_plot,
    x='semestre',
    y=['total_evasoes', 'percentual_evasao'],
    barmode='group',
    title='Distribuição de Evasão por Semestre',
    labels={
        'semestre': 'Semestre',
        'value': 'Contagem/Percentual',
        'variable': 'Métrica'
    },
    color_discrete_sequence=['#e74c3c', '#3498db'],
    text_auto=True,
    width=900,
    height=500
)

# 5. Personalizar layout
fig.update_layout(
    xaxis=dict(
        title='Semestre',
        type='category',
        tickmode='linear'
    ),
    yaxis_title="Valor",
    legend_title="Métrica",
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    )
)

# 6. Renomear legendas
fig.for_each_trace(lambda t: t.update(name='Total de Evasões' if t.name == 'total_evasoes' else 'Percentual de Evasão (%)'))

# 7. Adicionar informações complementares
fig.update_traces(
    textposition='outside',
    hovertemplate=(
        "<b>Semestre %{x}</b><br><br>" +
        "Total de evasões: %{y}<br>" +
        "Percentual: %{customdata[0]}%<extra></extra>"
    ),
    customdata=dados_plot[['percentual_evasao']]
)

fig.show()

# ===============================
# 📊 PCA + RandomForest: Visualização interativa
# ===============================

# Gráfico interativo
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters
df_pca['Risco_Predito'] = df_model['risco_predito']

fig = px.scatter(
    df_pca, x='PC1', y='PC2', color='Risco_Predito',
    title='Alunos por Risco de Evasão (PCA + RandomForest)',
    labels={'PC1':'Componente Principal 1', 'PC2':'Componente Principal 2'}
)
fig.show()

# --- Exemplo: definir motivo de evasão com base em regras simples ---
# Forçar para numérico (transforma erros em NaN)
df_model['faltas_1'] = pd.to_numeric(df_model['faltas_1'], errors='coerce')
df_model['desempenho_1'] = pd.to_numeric(df_model['desempenho_1'], errors='coerce')
df_model['renda_familiar'] = pd.to_numeric(df_model['renda_familiar'], errors='coerce')

# Remover linhas com NaN em alguma dessas colunas
df_model = df_model.dropna(subset=['faltas_1', 'desempenho_1', 'renda_familiar'])

# Se tiver a coluna 'motivo_evasao' use ela, senão a gente simula baseada nas variáveis
def definir_motivo(row):
    if row['faltas_1'] > 10:
        return 'Faltas Excessivas'
    elif row['desempenho_1'] < 5:
        return 'Baixo Desempenho'
    elif row['renda_familiar'] < 1500:
        return 'Baixa Renda'
    else:
        return 'Outro'

# Criar coluna de motivo (caso não exista)
if 'motivo_evasao' not in df_model.columns:
    df_model['motivo_evasao'] = df_model.apply(definir_motivo, axis=1)

# --- Contagem dos motivos por risco predito ---
df_motivos = df_model.groupby(['risco_predito', 'motivo_evasao']).size().reset_index(name='quantidade')

# --- Gráfico interativo com Plotly ---
fig = px.bar(
    df_motivos,
    x='motivo_evasao',
    y='quantidade',
    color='risco_predito',
    barmode='group',
    text='quantidade',
    title='Distribuição dos Motivos de Evasão por Categoria de Risco',
    labels={'motivo_evasao': 'Motivo de Evasão', 'quantidade': 'Número de Alunos'}
)

fig.update_layout(
    xaxis_title='Motivo de Evasão',
    yaxis_title='Número de Alunos',
    legend_title='Risco Predito',
    bargap=0.3
)

fig.show()

# ===============================
# 📚 PLN - Processamento de Linguagem Natural
# ===============================

# 🇧🇷 Stopwords em português
stop_words = stopwords.words('portuguese')

# 🔤 1. Unificar textos das aulas e professores
df['texto_pln'] = (
    df[['aula_1', 'aula_2', 'aula_3', 'aula_4', 'aula_5',
        'professor_1', 'professor_2', 'professor_3', 'professor_4', 'professor_5']]
    .fillna('')
    .agg(' '.join, axis=1)
)

# 🧹 2. Remover nulos
df_pln = df[['texto_pln', 'risco_evasao']].dropna()

X_texto = df_pln['texto_pln']
y_risco = df_pln['risco_evasao']

# 🎯 3. Separar treino/teste
X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
    X_texto, y_risco, test_size=0.2, random_state=42, stratify=y_risco
)

# 🤖 4. Pipeline com TF-IDF + Naive Bayes
pipeline_pln = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', MultinomialNB())
])

# 🚀 5. Treinar o modelo
pipeline_pln.fit(X_train_txt, y_train_txt)

# 🔍 6. Avaliar o modelo
y_pred_txt = pipeline_pln.predict(X_test_txt)

print("📋 Relatório de Classificação (PLN - Aulas e Professores):")
print(classification_report(y_test_txt, y_pred_txt))

# ===============================
# 📊 Visualizações Avançadas
# ===============================

# 💬 Top Palavras por Classe
vectorizer = pipeline_pln.named_steps['tfidf']
clf_nb = pipeline_pln.named_steps['clf']

feature_names = vectorizer.get_feature_names_out()
class_labels = clf_nb.classes_

for i, label in enumerate(class_labels):
    top10_idx = np.argsort(clf_nb.feature_log_prob_[i])[-10:]
    top10_words = feature_names[top10_idx]
    top10_weights = clf_nb.feature_log_prob_[i][top10_idx]

    df_top_words = pd.DataFrame({
        'Palavra': top10_words,
        'Peso': top10_weights
    }).sort_values(by='Peso', ascending=False)  # Alterado para False

    fig = px.bar(
        df_top_words,
        x='Peso',
        y='Palavra',
        orientation='h',
        color='Peso',
        title=f'💬 Top 10 Palavras Indicativas para Risco de Evasão = {label}',
        color_continuous_scale='Viridis',
        labels={'Peso': 'Peso (TF-IDF x Naive Bayes)'},
        height=400,
        width=700
    )
    fig.update_layout(
        xaxis_title='Peso TF-IDF',
        yaxis_title='Palavra',
        showlegend=False,
        margin=dict(l=50, r=30, t=50, b=50)
    )
    fig.show()

# ===============================

# Comparação entre os 3 modelos de treinamentos

# 📊📈 Comparação entre KMeans, Clusterização Spark e Regressão Linear

print("🔍 Comparando Modelos de Treinamento...")

# 👉 KMeans (já está treinado antes)
silhouette_kmeans = silhouette_score(X_scaled, clusters)
print(f"🔵 Silhouette Score (KMeans - sklearn): {silhouette_kmeans:.4f}")

# 👉 Clusterização Spark MLlib (já avaliado antes)
accuracy_spark = evaluator.evaluate(predictions)
print(f"🟢 Acurácia (Clusterização Spark RandomForest): {accuracy_spark:.4f}")

# 👉 Regressão Linear (já treinado antes)
if 'y_pred_r' in locals() and 'y_test_r' in locals():
    r2_reg = r2_score(y_test_r, y_pred_r)
    mse_reg = mean_squared_error(y_test_r, y_pred_r)
    print(f"🟣 R² (Regressão Linear Ridge): {r2_reg:.4f}")
    print(f"🟣 MSE (Regressão Linear Ridge): {mse_reg:.4f}")
else:
    print("⚠️ Dados de Regressão Linear não encontrados para comparação.")

# 📊📈 Resumo consolidado
resultados_modelos = pd.DataFrame({
    'Modelo': ['KMeans (sklearn)', 'Clusterização Spark (RandomForest)', 'Regressão Linear (Ridge)'],
    'Métrica Primária': [
        f"Silhouette Score: {silhouette_kmeans:.4f}",
        f"Acurácia: {accuracy_spark:.4f}",
        f"R²: {r2_reg:.4f} | MSE: {mse_reg:.4f}" if 'y_pred_r' in locals() else 'N/A'
    ]
})

print("\n📊 Comparativo de Resultados:")
print(resultados_modelos)

#graficos estaticos para comparação 
# 📊📈 Gráfico de Comparação de Modelos

# Valores para o gráfico
valores_metricas = [
    silhouette_kmeans,            # KMeans
    accuracy_spark,               # Clusterização Spark
    media_r2 if 'media_r2' in locals() else 0  # Regressão Linear (se existir)
]

nomes_modelos = [
    'KMeans (Silhouette)',
    'Clusterização Spark (Acurácia)',
    'Regressão Linear (R²)'
]

# Cores personalizadas
cores = ['#3498db', '#2ecc71', '#9b59b6']

# Plotar gráfico de barras
plt.figure(figsize=(10,6))
bars = plt.bar(nomes_modelos, valores_metricas, color=cores)

# Adicionar valores no topo das barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', ha='center', fontsize=11)

# 📊📈 Gráfico de Comparação de Modelos (atualizado com média das regressões)

# Valores para o gráfico
valores_metricas = [
    silhouette_kmeans,            # KMeans
    accuracy_spark,               # Clusterização Spark
    media_r2 if 'media_r2' in locals() else 0  # Média dos R² das Regressões
]

nomes_modelos = [
    'KMeans (Silhouette)',
    'Clusterização Spark (Acurácia)',
    'Regressões (Média R²)'
]

# Cores personalizadas
cores = ['#3498db', '#2ecc71', '#9b59b6']

# Plotar gráfico de barras
plt.figure(figsize=(10,6))
bars = plt.bar(nomes_modelos, valores_metricas, color=cores)

# Adicionar valores no topo das barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', ha='center', fontsize=11)

# Configurações gerais
plt.ylim(0, 1)
plt.title('📊 Comparativo de Desempenho dos Modelos de Treinamento', fontsize=16)
plt.ylabel('Valor da Métrica (0 a 1)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Mostrar gráfico
plt.show()
# ================================

#HADOOP COM O SPARK PARA FACILITAR AS LEITURAS DE DADOS E AGILIZAR O PROCESSAMENTO
# ===============================
# 📦 SparkSession COM Hadoop
# ===============================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F

# Iniciar sessão Spark com Hadoop
spark_hadoop = SparkSession.builder \
    .appName("PrevisaoRiscoEvasao_Hadoop") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.jars", "file:///C:/pyspark_drivers/mysql-connector-j-9.3.0.jar") \
    .getOrCreate()

# ===============================
# 📥 Leitura dos dados via JDBC
# ===============================
df_spark_hadoop = spark_hadoop.read.format("jdbc").options(
    url="jdbc:mysql://localhost:3306/faculdade",
    driver="com.mysql.cj.jdbc.Driver",
    dbtable="alunos",
    user="root",
    password=""
).load()

# ===============================
# 🧼 Limpeza e conversão
# ===============================
colunas_numericas = ["idade", "renda_familiar", "faltas_1", "notas_1", "faltas_2", "notas_2",
                     "faltas_3", "notas_3", "faltas_4", "notas_4", "faltas_5", "notas_5"]
colunas_categoricas = ["sexo", "trabalha", "acompanhamento_medico", "tem_filho",
                       "estado_civil", "desempenho_1", "desempenho_2",
                       "desempenho_3", "desempenho_4", "desempenho_5", "risco_evasao"]

for coln in colunas_numericas:
    df_spark_hadoop = df_spark_hadoop.withColumn(coln, col(coln).cast("double")).na.fill({coln: -1})

for colc in colunas_categoricas:
    df_spark_hadoop = df_spark_hadoop.na.fill({colc: "desconhecido"})
    indexer = StringIndexer(inputCol=colc, outputCol=colc + "_idx", handleInvalid="keep")
    df_spark_hadoop = indexer.fit(df_spark_hadoop).transform(df_spark_hadoop)

# ===============================
# 🔄 Vetorização
# ===============================
feature_cols_hadoop = colunas_numericas + [col + "_idx" for col in colunas_categoricas[:-1]]
assembler_hadoop = VectorAssembler(inputCols=feature_cols_hadoop, outputCol="features_vec")
df_spark_hadoop = assembler_hadoop.transform(df_spark_hadoop)

# ===============================
# 🌳 Random Forest Classifier
# ===============================
rf_hadoop = RandomForestClassifier(labelCol="risco_evasao_idx", featuresCol="features_vec", numTrees=100)
modelo_hadoop = rf_hadoop.fit(df_spark_hadoop)
predictions_hadoop = modelo_hadoop.transform(df_spark_hadoop)

# ===============================
# 📊 Avaliação
# ===============================
evaluator_hadoop = MulticlassClassificationEvaluator(
    labelCol="risco_evasao_idx", predictionCol="prediction", metricName="accuracy"
)
accuracy_hadoop = evaluator_hadoop.evaluate(predictions_hadoop)
print(f"📊 Acurácia com Spark + Hadoop: {accuracy_hadoop:.4f}")

# ===============================
# 📊 📈 Gráficos com Pandas + Matplotlib
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Converter para pandas
preds_hadoop_pd = predictions_hadoop.select("risco_evasao_idx", "prediction").toPandas()

# 📊 Matriz de Confusão
conf_mat_hadoop = confusion_matrix(preds_hadoop_pd["risco_evasao_idx"], preds_hadoop_pd["prediction"])

plt.figure(figsize=(7,5))
sns.heatmap(conf_mat_hadoop, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("🔷 Matriz de Confusão - Spark com Hadoop")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# 📊 Distribuição Real vs Predito
plt.figure(figsize=(8, 5))
sns.countplot(x=preds_hadoop_pd["risco_evasao_idx"], label="Real", alpha=0.6, color="royalblue")
sns.countplot(x=preds_hadoop_pd["prediction"], label="Predito", alpha=0.4, color="tomato")
plt.legend()
plt.title("📊 Distribuição das Classes: Real vs Predita (Hadoop)")
plt.xlabel("Risco de Evasão")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

# 📊 Importância das Variáveis
importances_hadoop = modelo_hadoop.featureImportances.toArray()
feature_names_hadoop = assembler_hadoop.getInputCols()

df_importancias = pd.DataFrame({
    'Variável': feature_names_hadoop,
    'Importância': importances_hadoop
}).sort_values(by='Importância', ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importância", y="Variável", data=df_importancias, palette="viridis")
plt.title("📊 Top 10 Variáveis para Risco de Evasão (Spark + Hadoop)")
plt.tight_layout()
plt.show()

# 📊 Percentual de Predições por Categoria
percentuais_pred = preds_hadoop_pd["prediction"].value_counts(normalize=True) * 100
df_percentual = percentuais_pred.reset_index()
df_percentual.columns = ['Categoria Predita', 'Percentual']

plt.figure(figsize=(7,5))
sns.barplot(x='Categoria Predita', y='Percentual', data=df_percentual, palette="pastel")
plt.title("📊 Percentual de Alunos por Categoria Predita (Hadoop)")
plt.ylabel("Percentual (%)")
plt.xlabel("Categoria Predita")
plt.tight_layout()
plt.show()
# =============================== FIM HADOOP COM O SPARK ===============================

#Graficos Estáticos

# Gráfico 1: Relatório Final (5 aulas)
print("📋 Relatório Final (5 Aulas):")
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.figure(1, figsize=(12,6))
plt.title("Matriz de Confusão - Modelo com 5 Aulas")

# Gráfico 2: Distribuição Real vs Predito
plt.figure(2, figsize=(10, 5))
sns.countplot(data=preds_pd, x="risco_evasao_idx", color="blue", label="Real", alpha=0.6)
sns.countplot(data=preds_pd, x="prediction", color="red", label="Predito", alpha=0.6)
plt.title("Distribuição das Classes - Real vs Predita")
plt.xlabel("Classe (risco_evasao)")
plt.ylabel("Contagem")
plt.legend()
plt.grid(True)

# Gráfico 3: Matriz de Confusão
conf_mat = confusion_matrix(preds_pd["risco_evasao_idx"], preds_pd["prediction"])
plt.figure(3, figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")

# Gráfico 4: Acurácia do Modelo 
labels = ['Acurácia', 'Erro']
sizes = [accuracy, 1 - accuracy]
plt.figure(4, figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
plt.title('Acurácia do Modelo de Evasão')

# Gráfico 5: Distribuição de Risco de Evasão (Dados via HDFS)
plt.figure(5, figsize=(8, 5))
sns.countplot(data=df_pandas, x='risco_evasao', palette='viridis')
plt.title('Distribuição de Risco de Evasão (Dados via HDFS)')
plt.xlabel('Risco de Evasão')
plt.ylabel('Quantidade')
plt.xticks(rotation=45)
plt.tight_layout()

# Gráfico 6: Top 5 cursos com mais risco de evasão
plt.figure(6, figsize=(8,5))
bars = plt.bar(df_risco_counts['risco_predito'], df_risco_counts['percentual'], color='skyblue')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.2f}%', ha='center')

plt.title('Distribuição percentual dos riscos preditos')
plt.xlabel('Risco Predito')
plt.ylabel('Percentual (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Gráfico 7: Top 10 Variáveis para Predição do Risco de Evasão
top_imp = df_imp.head(10)

plt.figure(7, figsize=(8,6))
plt.barh(top_imp['Variável'], top_imp['Importância'], color='mediumseagreen')
plt.xlabel('Importância')
plt.title('Top 10 Variáveis para Predição do Risco de Evasão')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Gráfico 8: Alunos por Risco de Evasão (PCA + RandomForest)

plt.figure(8, figsize=(8,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Risco_Predito', palette='Set2', s=100, edgecolor='black')
plt.title('Alunos por Risco de Evasão (PCA + RandomForest)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Gráfico 9: Quantidade de Alunos Evadidos por Classe Econômica
plt.figure(9, figsize=(7,5))
colors = ['red', 'orange', 'green']
bars = plt.bar(df_contagem['classe_economica'], df_contagem['quantidade_evadidos'], color=colors)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)), ha='center')

plt.title('Quantidade de Alunos Evadidos por Classe Econômica')
plt.xlabel('Classe Econômica')
plt.ylabel('Quantidade')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Gráfico 10: Matriz de Confusão
conf_mat = confusion_matrix(y_test_txt, y_pred_txt)
plt.figure(10, figsize=(7, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=1, linecolor='gray')
plt.title('🔷 Matriz de Confusão - PLN (Textos de Aulas e Professores)', fontsize=14)
plt.xlabel('Classe Predita', fontsize=12)
plt.ylabel('Classe Real', fontsize=12)
plt.tight_layout()
plt.grid(False)

# Gráfico 11: Distribuição Real vs Predito
plt.figure(11, figsize=(8, 5))
sns.countplot(x=y_test_txt, label='Real', alpha=0.6, color='royalblue')
sns.countplot(x=y_pred_txt, label='Predito', alpha=0.5, color='tomato')
plt.legend(title='Legenda')
plt.title('📊 Distribuição das Classes: Real vs Predita (PLN)', fontsize=14)
plt.xlabel('Risco de Evasão', fontsize=12)
plt.ylabel('Número de Alunos', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()  

# Gráfico 12: Top 5 Estados Civis com Mais Evasão
# Garantir que não há nulos
df_estado = df.dropna(subset=['estado_civil', 'risco_evasao'])

# Contar evasões por estado civil
evasao_estado = df_estado[df_estado['risco_evasao'] == 1]['estado_civil'].value_counts().nlargest(6)

plt.figure(12, figsize=(8,5))
bars = plt.bar(evasao_estado.index, evasao_estado.values, color='cornflowerblue', edgecolor='black')

# Adicionar rótulos nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)), ha='center')

plt.title('Top 5 Estados Civis com Mais Alunos Evadidos')
plt.xlabel('Estado Civil')
plt.ylabel('Quantidade de Evasões')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Gráfico 13: Evasão por Bimestre
# Garantir dados válidos
df_bimestre = df.dropna(subset=['bimestre', 'risco_evasao'])

# Contar evasões por bimestre
evasao_bimestre = df_bimestre[df_bimestre['risco_evasao'] == 1]['bimestre'].value_counts().sort_index()

plt.figure(13, figsize=(8,5))
bars = plt.bar(evasao_bimestre.index.astype(str), evasao_bimestre.values, color='tomato', edgecolor='black')

# Rótulos nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)), ha='center')

plt.title('Quantidade de Alunos Evadidos por Bimestre')
plt.xlabel('Bimestre')
plt.ylabel('Quantidade de Evasões')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Gráfico 14: Evasão por Semestre
# Garantir dados válidos
df_semestre = df.dropna(subset=['semestre', 'risco_evasao'])

# Contar evasões por semestre
evasao_semestre = df_semestre[df_semestre['risco_evasao'] == 1]['semestre'].value_counts().sort_index()

plt.figure(14, figsize=(9,5))
bars = plt.bar(evasao_semestre.index.astype(str), evasao_semestre.values, color='darkorange', edgecolor='black')

# Adicionar rótulos nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)), ha='center')

plt.title('Quantidade de Alunos Evadidos por Semestre')
plt.xlabel('Semestre')
plt.ylabel('Quantidade de Evasões')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Gráfico 15: Quantidade de Alunos Evadidos por Número de Faltas
plt.figure(15, figsize=(8,5))   
plt.bar(df_faltas_count['faltas_1'], df_faltas_count['quantidade_evadidos'], color='purple')
plt.title('Quantidade de Alunos Evadidos por Número de Faltas')
plt.xlabel('Número de Faltas')
plt.ylabel('Quantidade de Alunos Evadidos')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Gráfico 16: Distribuição de Evasão por Semestre
plt.figure(16, figsize=(10,6))
plt.bar(dados_plot['semestre'], dados_plot['total_evasoes'], color='blue', label='Total de Evasões')
plt.bar(dados_plot['semestre'], dados_plot['percentual_evasao'], color='orange', label='Percentual de Evasão (%)')
plt.title('Distribuição de Evasão por Semestre')
plt.xlabel('Semestre')
plt.ylabel('Contagem/Percentual')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Certifique-se que df_motivos foi criado antes
if 'df_motivos' not in locals():
    df_model['motivo_evasao'] = df_model.apply(definir_motivo, axis=1)
    df_motivos = df_model.groupby(['risco_predito', 'motivo_evasao']).size().reset_index(name='quantidade')

# Gráfico 17: Top 5 cursos com mais evasão
# Garantir que a coluna 'risco_evasao' e 'curso' estão presentes e válidas
df_evasao_curso = df.copy()
df_evasao_curso = df_evasao_curso.dropna(subset=['risco_evasao', 'curso'])

# Filtrar apenas os alunos que evadiram (risco_evasao == 1)
df_evadidos = df_evasao_curso[df_evasao_curso['risco_evasao'] == 1]

# Contar evasões por curso e selecionar os 5 mais
top_cursos = df_evadidos['curso'].value_counts().nlargest(5)

# Plotar o gráfico
plt.figure(17, figsize=(8, 5))
bars = plt.bar(top_cursos.index, top_cursos.values, color='steelblue', edgecolor='black')

# Adicionar rótulos de valor nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)), ha='center')

plt.title('Top 5 Cursos com Mais Alunos Evadidos')
plt.xlabel('Curso')
plt.ylabel('Quantidade de Evasões')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()


#FIM DE CODIGO 
