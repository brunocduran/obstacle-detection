import numpy as np
import pandas as pd
import tensorflow as tf
import random
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import FirebaseHelper, extract_feature, modeloCombinado
from FirebaseHelper import *
import os

app = Flask(__name__)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

EXTENSAO_PERMITIDA = set(['png', 'jpg', 'jpeg'])

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

PREDICT_PATH = os.path.join(BASE_PATH, 'images-teste')
DATASET_PATH = os.path.join(BASE_PATH, 'images-treino')
FEATURE_PATH = os.path.join(BASE_PATH, 'features', 'features.csv')
RESULT_PATH = os.path.join(BASE_PATH, 'details-results', '')
MODEL_PATH = os.path.join(BASE_PATH, 'modelo', 'model')
TFLITE_MODEL_PATH = os.path.join(BASE_PATH, 'modelo', 'model.tflite')

data_filename = RESULT_PATH + "data_detailed.csv"

image_size = (224, 224)


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
        #model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=35, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)