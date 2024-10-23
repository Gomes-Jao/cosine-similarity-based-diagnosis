import pandas as pd
import dados
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 1. Criação da base de casos

base_de_casos = pd.DataFrame(dados.dados)

# 2. Função para transformar sintomas em vetores para cálculo de similaridade
def vetorizar_sintomas(casos, sintomas_paciente):
    vectorizer = CountVectorizer().fit(casos)
    casos_vetorizados = vectorizer.transform(casos)
    sintomas_vetorizados = vectorizer.transform([sintomas_paciente])
    return casos_vetorizados, sintomas_vetorizados

# 3. Função de recuperação de casos com base na similaridade
def diagnosticar(sintomas_paciente):
    casos_vetorizados, sintomas_vetorizados = vetorizar_sintomas(
        base_de_casos["sintomas"], sintomas_paciente
    )
    similaridades = cosine_similarity(sintomas_vetorizados, casos_vetorizados)
    indice_caso_mais_proximo = similaridades.argmax()
    doenca = base_de_casos.iloc[indice_caso_mais_proximo]["doenca"]
    return doenca

# 4. Entrada dos sintomas do paciente
sintomas_paciente = "Dor abdominal intensa Febre Diarreia Febre Dor de cabeça Dor muscular"
diagnostico = diagnosticar(sintomas_paciente)

# 5. Apresentação do diagnóstico
print(f"Diagnóstico: {diagnostico}")