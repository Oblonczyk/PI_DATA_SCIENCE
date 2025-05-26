import subprocess
import sys

# Lista de pacotes a serem instalados
packages = [
    # Manipulação de Dados
    "pandas",
    "numpy",

    # Visualização de Dados
    "matplotlib",
    "seaborn",
    "plotly",

    # Machine Learning
    "scikit-learn",
    "imbalanced-learn",

    # Processamento de Linguagem Natural
    "nltk",

    # Big Data com PySpark
    "pyspark",

    # Conexão com banco de dados
    "sqlalchemy"
]

# Função para instalar pacotes
def install(package):
    try:
        print(f"Instalando: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"Erro ao instalar o pacote: {package}")

# Instalando todos os pacotes
for pkg in packages:
    install(pkg)

# Downloads adicionais para nltk
try:
    import nltk
    nltk.download("stopwords")
except ImportError:
    print("nltk não instalado corretamente para baixar os recursos.")


#Caso não funcione, tente instalar manualmente
# Instalação manual de pacotes
#pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn nltk pyspark sqlalchemy
#python -c "import nltk; nltk.download('stopwords')"
