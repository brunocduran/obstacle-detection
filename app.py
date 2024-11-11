import numpy as np
import pandas as pd
import tensorflow as tf
import random
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import FirebaseHelper, extract_feature, modeloCombinado, metrics_view
from FirebaseHelper import *
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

app = Flask(__name__)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

EXTENSAO_PERMITIDA = set(['png', 'jpg', 'jpeg'])

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

#PREDICT_PATH = os.path.join(BASE_PATH, 'images-teste')
#DATASET_PATH = os.path.join(BASE_PATH, 'images-treino')
FEATURE_PATH = os.path.join(BASE_PATH, 'features', 'features.csv')
RESULT_PATH = os.path.join(BASE_PATH, 'details-results', '')
MODEL_PATH = os.path.join(BASE_PATH, 'modelo', 'model')
TFLITE_MODEL_PATH = os.path.join(BASE_PATH, 'modelo', 'model.tflite')

data_filename = RESULT_PATH + "data_detailed.csv"

image_size = (224, 224)


import matplotlib
matplotlib.use('Agg')


def remove_all_png_files(directory):
    png_files = glob.glob(os.path.join(directory, "*.png"))
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Removido: {file_path}")
        except Exception as e:
            print(f"Erro ao remover {file_path}: {e}")


class ConfusionMatrixCallback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.epoch_data = []  # Armazenar informações para o Excel

    def on_epoch_end(self, epoch, logs=None):
        # Prever os valores de validação
        y_pred = (self.model.predict(self.X_val) > 0.5).astype("int32").flatten()

        # Calcular a matriz de confusão
        cm = confusion_matrix(self.y_val, y_pred)

        # Obter Verdadeiro Negativo (VN), Falso Positivo (FP), Falso Negativo (FN) e Verdadeiro Positivo (VP)
        tn, fp, fn, tp = cm.ravel()

        # Plotar e salvar a matriz de confusão
        #plt.figure(figsize=(8, 6))
        #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        #plt.xlabel("Predito")
        #plt.ylabel("Real")
        #plt.title(f"Matriz de Confusão - Época {epoch + 1}")
        #plt.savefig(f"{RESULT_PATH}matriz_confusao_epoca_{epoch + 1}.png")
        #plt.close()


        # Configurações para rótulos personalizados e tamanho das fontes
        labels = ['VP', 'FP', 'FN', 'VN']
        cm_display = [[cm[1, 1], cm[0, 1]], [cm[1, 0], cm[0, 0]]]

        # Plotar e salvar a matriz de confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_display, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['Positivo', 'Negativo'], yticklabels=['Positivo', 'Negativo'],
                    annot_kws={"size": 16})  # Aumenta o tamanho dos números dentro dos quadrados
        plt.xlabel("Predito", fontsize=16)
        plt.ylabel("Real", fontsize=16)
        plt.title(f"Matriz de Confusão - Época {epoch + 1}", fontsize=18)

        # Adicionar os rótulos VN, VP, FP, FN no canto superior esquerdo
        for i in range(2):
            for j in range(2):
                plt.text(j, i, labels[i * 2 + j], ha='left', va='top', color="red", fontsize=12, weight="bold")

        plt.savefig(f"{RESULT_PATH}matriz_confusao_epoca_{epoch + 1}.png")
        plt.close()

        # Calcular métricas detalhadas
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        roc_score = roc_auc_score(self.y_val, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Armazenar as informações para exportação
        self.epoch_data.append({
            "Época": epoch + 1,
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

    def on_train_end(self, logs=None):
        # Salvar os dados detalhados em um arquivo Excel ao final do treinamento
        df_epoch_data = pd.DataFrame(self.epoch_data)

        # Criar um arquivo Excel com duas folhas
        with pd.ExcelWriter(f"{RESULT_PATH}dados_epoca_detalhado.xlsx") as writer:
            # Escrever os dados das épocas na primeira aba
            df_epoch_data.to_excel(writer, sheet_name="Dados por Época", index=False)

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

def load_feature():
    # Carregar o CSV sem a restrição de colunas
    df = pd.read_csv(FEATURE_PATH, sep=',')

    # Contar o número de colunas
    num_cols = df.shape[1]
    #print(f"O número de colunas no arquivo CSV é: {num_cols}")

    # Carregar novamente o CSV, agora com o número correto de colunas
    df = pd.read_csv(FEATURE_PATH, sep=',', usecols=range(1, num_cols))

    return df


@app.route('/extrair_feature', methods=['GET'])
def extrair_features():
    try:
        extract_feature.main_extract_feature()
        return jsonify({'message': 'Features extraídas e salvas com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Função para treinar o modelo e salvar em formato TensorFlow Lite
@app.route('/salvar_modelo', methods=['GET'])
def salvar_modelo():
    try:
        # remover todas as matrizes de confusao da pasta do resultado detalhado
        remove_all_png_files(RESULT_PATH)

        # Carregar os dados
        df = extract_feature.load_data()

        # Convertendo os valores da coluna 'category' para strings adequadas para 'binary' mode
        df['category'] = df['category'].replace({1: 'clear', 0: 'non-clear'})  # Converte para strings apenas para ImageDataGenerator

        # Carregar as features já extraídas do arquivo CSV
        features_df = load_feature()
        features = features_df.to_numpy()

        # Após a divisão, converta os rótulos de volta para 0 e 1
        df['category'] = df['category'].replace({'clear': 1, 'non-clear': 0})  # Converta de volta para números

        # Dividir os dados em treino e validação (80% treino, 20% validação)
        X_train, X_val, y_train, y_val = train_test_split(features, df['category'], test_size=0.2, random_state=SEED)

        # Garantir que o número de amostras e rótulos seja o mesmo
        assert X_train.shape[0] == len(y_train), f"Mismatch: Features train shape {X_train.shape[0]} vs Labels {len(y_train)}"
        assert X_val.shape[0] == len(y_val), f"Mismatch: Features val shape {X_val.shape[0]} vs Labels {len(y_val)}"

        # Criar o modelo com base nas features carregadas
        model = Sequential()
        #model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))  # Ajuste de entrada
        model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Saída binária

        # Compilar o modelo
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        # Treinar o modelo com as features extraídas
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Adicionar o callback para matriz de confusão
        confusion_matrix_callback = ConfusionMatrixCallback(X_val, y_val)

        model.fit(X_train, y_train, epochs=35, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, confusion_matrix_callback])

        # Salvar o modelo Keras
        keras_model_path = os.path.join(MODEL_PATH, 'model.h5')
        model.save(keras_model_path)

        # Converter o modelo Keras para TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Salvar o modelo TensorFlow Lite
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)

        return jsonify({'message': 'Modelo treinado e salvo com sucesso', 'tflite_model_path': TFLITE_MODEL_PATH})

    except Exception as e:
        return jsonify({'error': str(e)})

# Callback personalizado para gerar matriz de confusão ao final de cada epoch

@app.route('/salvar_modelo_combinado', methods=['GET'])
def salvar_modelo_combinado():
    try:
        modeloCombinado.gerarModeloCombinado()
        return jsonify({'message': 'Modelo combinado gerado e salvo com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/enviar_modelo', methods=['GET'])
def enviar_modelo():
    try:
        FirebaseHelper.upload_model_to_storage()
        return jsonify({'message': 'Modelo enviado para o storage do Firebase'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_imagens', methods=['GET'])
def download_imagens():
    try:
        FirebaseHelper.download_images_from_storage()
        return jsonify({'message': 'Foi realizado o download das imagens do storage para a pasta local'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/run', methods=['GET'])
def run():
    try:
        extrair_features()
        salvar_modelo()
        salvar_modelo_combinado()
        enviar_modelo()

        return jsonify({'message': 'Features extraídas, modelo gerado e enviado para o Firebase'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        metrics_view.metricsView()
        return jsonify({'message': 'Métricas geradas'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# Callback personalizado para gerar matriz de confusão ao final de cada epoch e salvar métricas
