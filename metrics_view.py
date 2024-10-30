#Arquivo responsável por pegar as imagens recebidas do APP (já conferidas) e gerar as métricas

#base libraries
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

#----------------------- FUNCTION ------------------------------------------------

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

    #Gerando as informações
    with open(RESULT_PATH + "data_detailed.csv", "a+") as f_data:
        f_data.write("MobileNetV2,")
        f_data.write("MLP,") #MLP - Multilayer Perceptron
        f_data.write(str(0)+", ")
        f_data.write(str("{0:.4f}".format(accuracy_score(hidden_labels,hidden_pred)))+",") #Acc Score
        f_data.write(str("{0:.4f}".format(f1_score(hidden_labels,hidden_pred)))+",") #F1 Score
        f_data.write(str("{0:.4f}".format(roc_auc_score(hidden_labels,hidden_pred)))+",") #ROC Score

    #Gerando a Matrix de Confusão
    cm = confusion_matrix(hidden_labels, hidden_pred)
    sns.heatmap(cm, annot=True)

    #Plotando a figura, salvando e limpando
    plt.plot(cm)
    plt.savefig(RESULT_PATH + "matriz_confusao.png", dpi=100)
    plt.clf()

if __name__ == "__main__":
    metricsView()