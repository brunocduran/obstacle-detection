# region LIBRARY

# Bibliotecas base FLASK
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Bibliotecas base Python
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import time
import os

# Importing Image module from PIL package
from PIL import Image
import PIL

# Transformação
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import RepeatedKFold
from PIL import Image

# Classificador
from sklearn.svm import SVC

# Métricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import FirebaseHelper
from FirebaseHelper import *

# endregion

app = Flask(__name__)

# region VARIABLE GLOBAL

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

EXTENSAO_PERMITIDA = set(['png', 'jpg', 'jpeg'])

#PREDICT_PATH = 'C:/TCC/obstacle-detection/images-teste'
#DATASET_PATH = "C:/TCC/obstacle-detection/images-treino"
#FEATURE_PATH = "C:/TCC/obstacle-detection/features/features.csv"
#RESULT_PATH = "C:/TCC/obstacle-detection/details-results/"

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

PREDICT_PATH = os.path.join(BASE_PATH, 'images-teste')
DATASET_PATH = os.path.join(BASE_PATH, 'images-treino')
FEATURE_PATH = os.path.join(BASE_PATH, 'features', 'features.csv')
RESULT_PATH = os.path.join(BASE_PATH, 'details-results', '')
MODEL_PATH = os.path.join(BASE_PATH, 'modelo', 'model')
TFLITE_MODEL_PATH = os.path.join(BASE_PATH, 'modelo', 'model.tflite')

data_filename = RESULT_PATH + "data_detailed.csv"

# Criando folds para cross-validation - 10fold
kfold_n_splits = 10
kfold_n_repeats = 1
kf = RepeatedKFold(n_splits=kfold_n_splits, n_repeats=kfold_n_repeats, random_state=SEED)

image_size = (224, 224)

bGerarInformacoes = True


# endregion

# region FUNCTION

def fileValid(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSAO_PERMITIDA


def load_data():
    # Definir extensões de arquivos válidos (imagens)
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Listar apenas arquivos que possuem extensões de imagem válidas
    filenames = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid_extensions)]

    categories = []

    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'clear':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df


def load_feature():
    return pd.read_csv(FEATURE_PATH, sep=',', usecols=range(1, 50177))


def gen_dataset(features, labels, train, test):
    max_index = len(features) - 1
    train = [idx for idx in train if idx <= max_index]
    test = [idx for idx in test if idx <= max_index]

    dataset_train = np.array(features[train])
    dataset_train_label = np.array(labels[train])
    dataset_test = np.array(features[test])
    dataset_test_label = np.array(labels[test])

    return dataset_train, dataset_train_label, dataset_test, dataset_test_label



def training(train_data, train_label, test_data, clf):
    # Treinando o modelo
    start = time.time()
    clf = clf.fit(train_data, train_label)
    end = time.time()

    time_trainning = end - start

    # Testando o modelo
    start = time.time()
    classification_result = clf.predict(test_data)
    end = time.time()

    time_prediction = end - start

    return time_trainning, time_prediction, classification_result


def feature_model_extract(df):
    time_start = time.time()

    features_VGG16 = extract_features(df, modelVGG16, preprocessing_function_VGG16)

    features_VGG19 = extract_features(df, modelVGG19, preprocessing_function_VGG19)

    # concatenate array features VGG16+VGG19
    features = np.hstack((features_VGG16, features_VGG19))

    time_end = time.time()

    time_feature_extration = time_end - time_start

    return features, time_feature_extration


def create_models_VGG():
    IMAGE_CHANNELS = 3
    POOLING = None

    from keras.api.applications.vgg16 import VGG16, preprocess_input
    global modelVGG16
    modelVGG16 = VGG16(weights='imagenet', include_top=False, pooling=POOLING,
                       input_shape=(224, 224) + (IMAGE_CHANNELS,))

    global preprocessing_function_VGG16
    preprocessing_function_VGG16 = preprocess_input

    from keras.api.applications.vgg19 import VGG19, preprocess_input
    global modelVGG19
    modelVGG19 = VGG19(weights='imagenet', include_top=False, pooling=POOLING,
                       input_shape=(224, 224) + (IMAGE_CHANNELS,))

    global preprocessing_function_VGG19
    preprocessing_function_VGG19 = preprocess_input

    from keras.api.layers import Flatten
    from keras.api.models import Model

    # VGG16
    output = Flatten()(modelVGG16.layers[-1].output)
    modelVGG16 = Model(inputs=modelVGG16.inputs, outputs=output)

    # VGG19
    output = Flatten()(modelVGG19.layers[-1].output)
    modelVGG19 = Model(inputs=modelVGG19.inputs, outputs=output)

    return True


def extract_features(df, model, preprocessing_function):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )

    total = df.shape[0]
    batch_size = 4

    generator = datagen.flow_from_dataframe(
        df,
        PREDICT_PATH,
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=False
    )

    features = model.predict(generator, steps=int(np.ceil(total / batch_size)))

    return features


def classification(ds_features):
    return clf.predict(ds_features)


# endregion

# region ROUTE

@app.route('/')
def main():

    df_feature = load_feature()

    df_data = load_data()

    # Carregando Labels
    labels = df_data["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)

    kf.split(df_data)

    all_feature = df_feature.to_numpy()

    # Instanciando classificador
    global clf
    clf = SVC(kernel="linear", C=0.025)

    # kfold loop
    for index, [train, test] in enumerate(kf.split(df_data)):
        # Gerando os datasets
        dataset_train, dataset_train_label, dataset_test, dataset_test_label = gen_dataset(all_feature, labels, train,
                                                                                           test)

        # Treinando o modelo
        time_trainning, time_prediction, pred = training(dataset_train, dataset_train_label, dataset_test, clf)

        hidden_labels = dataset_test_label.copy()
        hidden_pred = pred.copy()

        if bGerarInformacoes:
            # Gerando as informações
            with open(data_filename, "a+") as f_data:
                f_data.write("VGG16+VGG19,")
                f_data.write("LinearSVM,")
                f_data.write(str(index + 1) + ",")  # Kfold index
                f_data.write(str(np.shape(all_feature)[1]) + ",")  # CNN_features
                f_data.write(str(0) + ", ")
                f_data.write(str("{0:.4f}".format(accuracy_score(hidden_labels, hidden_pred))) + ",")  # Acc Score
                f_data.write(str("{0:.4f}".format(f1_score(hidden_labels, hidden_pred))) + ",")  # F1 Score
                f_data.write(str("{0:.4f}".format(roc_auc_score(hidden_labels, hidden_pred))) + ",")  # ROC Score
                f_data.write(str("{0:.4f}".format(time_trainning)) + ",")  # Time Classifier Trainning
                f_data.write(str("{0:.4f}".format(time_prediction)) + ",\n")  # Time Classifier Predict

            # Gerando a Matrix de Confusão
            cm = confusion_matrix(hidden_labels, hidden_pred)
            sns.heatmap(cm, annot=True)

            # Plotando a figura, salvando e limpando
            plt.plot(cm)
            plt.savefig(RESULT_PATH + str(index) + ".png", dpi=100)
            plt.clf()

            # Gerando o grafico com hiperplano
            # plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');

    # Criando e Treinando as CNN para extrair as features
    create_models_VGG()

    return "Classificador Treinado"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'Imagem não encontrada na solicitação'})
        resp.status_code = 400
        return resp

    try:
        file = request.files['file']

        if file and fileValid(file.filename):
            sTime = str(time.time())
            sExtensao = file.filename.rsplit('.', 1)[1].lower()

            newFilename = sTime + "." + sExtensao

            # Guardando a imagem em um dataframe
            df = pd.DataFrame({'filename': newFilename, "category": ["clear"]})

            filename = secure_filename(newFilename)

            # criando um objeto da imagem
            image = Image.open(file)

            # Girando a imagem 270 graus no sentido anti-horário
            image = image.rotate(270, PIL.Image.NEAREST, expand=1)

            # Redimensionando a imagem
            image = image.resize((750, 1000), Image.LANCZOS)

            # Salva a imagem
            image.save(os.path.join(PREDICT_PATH, filename))

            # Extraindo as caracteristicas da imagem
            features, time_feature_extration = feature_model_extract(df)

            result = classification(features)

            # Renomeia o arquivo de acordo com o resultado obtido (SOMENTE PARA VALIDAÇÃO)
            if 1 == 1:
                if result == 1:
                    sNomeArquivo = "clear." + sTime
                else:
                    sNomeArquivo = "noclear." + sTime

                sNomeAntigo = os.path.join(PREDICT_PATH, filename)
                sNomeRenomeado = os.path.join(PREDICT_PATH, sNomeArquivo + "." + sExtensao)

                os.rename(sNomeAntigo, sNomeRenomeado)

            resp = jsonify({'result': str(result[0])})
            resp.status_code = 201
            return resp

    #except:
    #    resp = jsonify({"result": 0})
    #    resp.status_code = 500
    #    return resp
    except Exception as e:
        print(e)
        resp = jsonify({"result": 0})
        resp.status_code = 500
        return resp

# Função para criar o modelo com Keras
def create_model(input_size):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_size,)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Saída binária

    # Compilar o modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Função para treinar o modelo e salvar em formato TensorFlow Lite
@app.route('/salvar_modelo', methods=['GET'])
def salvar_modelo():
    try:
        # Carregar os dados
        df = load_data()

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
        model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))  # Ajuste de entrada
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Saída binária

        # Compilar o modelo
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Treinar o modelo com as features extraídas
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

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



@app.route('/enviar_modelo', methods=['GET'])
def enviar_modelo():
    try:
        FirebaseHelper.upload_model_to_storage()
        return jsonify({'message': 'Modelo enviado para o storage do firebase'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_imagens', methods=['GET'])
def download_imagens():
    try:
        FirebaseHelper.download_images_from_storage()
        return jsonify({'message': 'Foi realizado o download das imagens do storage para a pasta local'})

    except Exception as e:
        return jsonify({'error': str(e)})


# endregion

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)