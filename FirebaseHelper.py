import firebase_admin
from firebase_admin import credentials, storage
import os
from datetime import datetime

# Inicializa o Firebase Admin SDK
cred = credentials.Certificate(
    "credenciais\\credencial.json")  # Substitua com o caminho para o arquivo JSON das credenciais do Firebase
firebase_admin.initialize_app(cred, {
    'storageBucket': 'tccobstacledetection-7bfe8.appspot.com'  # Substitua com o bucket do Firebase Storage
})

bucket = storage.bucket()

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'modelo')


# Função para fazer o download das imagens e salvá-las localmente
def download_images_from_storage():
    # Referência para a pasta 'images' no Firebase Storage
    images_folder = bucket.list_blobs(prefix='imagens/')

    # Caminho para salvar as imagens localmente
    local_path = os.path.join(BASE_PATH, 'images-teste')

    # Certifique-se de que a pasta existe
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    for blob in images_folder:
        if not blob.name.endswith('/'):  # Evita subpastas vazias
            # Define o caminho completo para o arquivo local
            local_file_path = os.path.join(local_path, os.path.basename(blob.name))

            # Faz o download do arquivo
            blob.download_to_filename(local_file_path)
            print(f'Arquivo {blob.name} baixado para {local_file_path}')

            # Após o download, exclui o arquivo do Firebase Storage
            blob.delete()
            print(f'Arquivo {blob.name} excluído do Firebase Storage.')


# Função para fazer o upload do modelo treinado para o Storage
def upload_model_to_storage():
    print(f'Iniciando o envio do modelo')
    # Nome do arquivo modelo
    model_local_path = os.path.join(MODEL_PATH, 'combined_model.tflite')

    # Cria o nome do arquivo com base na data e hora atuais
    current_datetime = datetime.now().strftime("%m%d%Y%H%M%S")
    model_storage_name = f'models/modelo_{current_datetime}.tflite'

    # Faz o upload para o Storage
    blob = bucket.blob(model_storage_name)
    blob.upload_from_filename(model_local_path)
    print(f'Modelo {model_local_path} enviado para {model_storage_name} no Firebase Storage.')

    # Agora vamos listar e excluir os outros arquivos da pasta models/ no Storage
    models_folder = bucket.list_blobs(prefix='models/')

    # Iterar sobre todos os blobs (arquivos) na pasta 'models'
    for blob in models_folder:
        if blob.name != model_storage_name:  # Exclui todos os arquivos, exceto o que foi acabado de enviar
            blob.delete()
            print(f'Arquivo {blob.name} excluído do Firebase Storage.')

    print(f'Todos os arquivos antigos foram excluídos, mantendo apenas {model_storage_name}.')


# Exemplo de execução
if __name__ == "__main__":
    download_images_from_storage()
    upload_model_to_storage()
