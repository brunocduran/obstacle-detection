#Arquivo responsável por pegar as imagens recebidas do APP (já conferidas) e gerar as métricas

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_PATH, 'images-teste')
RESULT_PATH = os.path.join(BASE_PATH, 'details-results', 'results_predict', '')

def load_data():
    # Definir extensões de arquivos válidos (imagens)
    valid_extensions = ('.jpg', '.jpeg', '.png')

    filenames = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid_extensions)]

    print(filenames)

    aLabel = []
    aPredict = []

    for filename in filenames:
        #Carregando os labels
        label = filename.split('.')[2]
        if label == 'clear':
            aLabel.append(1)
        else:
            aLabel.append(0)


        predict = filename.split('.')[0]
        if predict == 'clear':
            aPredict.append(1)
        else:
            aPredict.append(0)

    df = pd.DataFrame({
        'label': aLabel,
        'predict': aPredict
    })

    return df

def metricsView():
    #Carregando as imagens em um dataframe
    df = load_data()

    hidden_labels = df["label"].to_numpy()
    hidden_pred = df["predict"].to_numpy()

    # Calcular a matriz de confusão
    cm = confusion_matrix(hidden_labels, hidden_pred)

    # Obter Verdadeiro Negativo (VN), Falso Positivo (FP), Falso Negativo (FN) e Verdadeiro Positivo (VP)
    tn, fp, fn, tp = cm.ravel()

    # Plotar e salvar a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - Testes")
    plt.savefig(f"{RESULT_PATH}matriz_confusao_testes.png")
    plt.close()

    # Calcular métricas detalhadas
    accuracy = accuracy_score(hidden_labels, hidden_pred)
    f1 = f1_score(hidden_labels, hidden_pred)
    roc_score = roc_auc_score(hidden_labels, hidden_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    self = []

    # Armazenar as informações para exportação
    self.append({
        "Época": "Testes",
        "Acurácia": round(accuracy, 4),
        "Pontuação F1": round(f1, 4),
        "Pontuação ROC": round(roc_score, 4),
        "Precisão": round(precision, 4),
        "Sensibilidade (Recall)": round(recall, 4),
        "Especificidade": round(specificity, 4),
        "Taxa de Falsos Positivos (FPR)": round(fpr, 4),
        "Taxa de Falsos Negativos (FNR)": round(fnr, 4),
        "Verdadeiro Negativo (VN)": tn,
        "Falso Positivo (FP)": fp,
        "Falso Negativo (FN)": fn,
        "Verdadeiro Positivo (VP)": tp,
        "Total de Amostras": tn + fp + fn + tp
    })

    # Salvar os dados detalhados em um arquivo Excel ao final do treinamento
    df_epoch_data = pd.DataFrame(self)

    # Criar um arquivo Excel com duas folhas
    with pd.ExcelWriter(f"{RESULT_PATH}dados_testes_detalhado.xlsx") as writer:
        # Escrever os dados dos resultados na primeira aba
        df_epoch_data.to_excel(writer, sheet_name="Dados dos testes", index=False)

        # Criar uma segunda aba com as explicações
        # Criar uma segunda aba com as explicações
        explicacoes = {
            "Métrica": [
                "Acurácia", "Pontuação F1", "Pontuação ROC", "Precisão", "Sensibilidade (Recall)",
                "Especificidade", "Taxa de Falsos Positivos (FPR)", "Taxa de Falsos Negativos (FNR)",
                "Classe Positiva", "Classe Negativa"
            ],
            "Descrição": [
                "Proporção de predições corretas sobre o total de amostras.",
                "Média harmônica entre Precisão e Sensibilidade, usada para avaliar equilíbrio.",
                "Área sob a curva ROC, indicando a capacidade de separação entre classes.",
                "Proporção de predições corretas da classe positiva.",
                "Proporção de exemplos positivos corretamente identificados.",
                "Proporção de exemplos negativos corretamente identificados.",
                "Proporção de exemplos negativos incorretamente classificados como positivos.",
                "Proporção de exemplos positivos incorretamente classificados como negativos.",
                "Representa a classe 'clear' (sem obstáculo).",
                "Representa a classe 'non-clear' (com obstáculo)."
            ]
        }
        df_explicacoes = pd.DataFrame(explicacoes)
        df_explicacoes.to_excel(writer, sheet_name="Explicações", index=False)

if __name__ == "__main__":
    metricsView()