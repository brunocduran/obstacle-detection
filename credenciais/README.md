# Configuração do Firebase e Credenciais

- Acesse o **Firebase Console** com sua conta: https://console.firebase.google.com/

- Inicie um novo projeto:

<p align="center">
  <img width="80%" src="img/1-Inicio.png" />
</p>

- Crie o projeto com qualquer nome, aceite os termos do Firebase e clique em **Continuar**:

<p align="center">
  <img width="80%" src="img/2-Criar um projeto.png" />
</p>

- Ativar o **Google Analytics** é opcional. Clique em **Continuar**:

<p align="center">
  <img width="80%" src="img/3-Google Analytics.png" />
</p>

- Caso tenha ativado o **Google Analytics**, configure-o e clique em **Criar Projeto**

<p align="center">
  <img width="80%" src="img/4-Configurar o Google Analytics.png" />
</p>

- O projeto foi criado. Clique em **Continuar**:

<p align="center">
  <img width="80%" src="img/5-Projeto pronto.png" />
</p>

- Vá para **Configurações do Projeto**:

<p align="center">
  <img width="80%" src="img/6-Configuracao do Projeto.png" />
</p>

- Em seguida, acesse a aba **Contas de serviço**, selecione a opção **Python** e clique em **Gerar nova chave privada**.

<p align="center">
  <img width="80%" src="img/7-Contas de Servico.png" />
</p>

- Após o passo anterior, será baixado um arquivo **JSON**. Esse arquivo deve ser renomeado para **credencial.json** e copiado para a pasta **credenciais**, localizada na raiz do projeto, no diretório: *obstacle-detection\credenciais*.

<p align="center">
  <img width="80%" src="img/8-Pasta Credenciais.png" />
</p>

- Volte ao menu principal do Firebase, acesse o menu lateral **Criação** e selecione **Storage**.

<p align="center">
  <img width="80%" src="img/9-Criar Storage.png" />
</p>

- Nesse momento, para utilizar o **Storage**, será exigido um upgrade da conta para o plano **Blaze**, uma medida adotada pelo Firebase desde 30/10/2024. Mais detalhes podem ser encontrados nos seguintes portais: <br>
https://firebase.google.com/docs/storage/web/start?hl=pt#before-you-begin <br>
https://firebase.google.com/pricing?hl=pt-br <br>
Porém, ainda é possível utilizar o **Storage** gratuitamente, desde que não ultrapasse o limite de uso. Então, clique em **Fazer upgrade do projeto** e, em seguida, em **Criar uma conta do Cloud Billing**. Será necessário informar os dados de um cartão de crédito para possíveis cobranças adicionais:

<p align="center">
  <img width="80%" src="img/10-Storage.png" />
</p>

<p align="center">
  <img width="80%" src="img/11-Criar Conta Cloud Billing.png" />
</p>

- Após concluir o processo anterior, é possível prosseguir com a criação do **Storage**. Clique em **Começar**:

<p align="center">
  <img width="80%" src="img/12-Storage Comecar.png" />
</p>

- Na configuração do **Bucket**, mantenha todas as opções padrão e clique em **Continuar**:

<p align="center">
  <img width="80%" src="img/13-Storage Opcoes de Bucket.png" />
</p>

- Na configuração de **Regras de segurança**, selecione a opção **Iniciar no modo de teste** e clique em **Criar**:

<p align="center">
  <img width="80%" src="img/14-Storage Regras de Seguranca.png" />
</p>

- Após a criação do **Storage**, acesse a aba **Regras**, altere o código para permitir acesso até uma data mais posterior e clique em **Publicar**:

<p align="center">
  <img width="80%" src="img/15-Storage Editar Regras de Seguranca.png" />
</p>

- Volte para a aba **Arquivos** e crie duas pastas com os nomes **imagens** e **models**:

<p align="center">
  <img width="80%" src="img/16-Storage Criar Pastas.png" />
</p>

- Ficando da seguinte forma:

<p align="center">
  <img width="80%" src="img/17-Storage Pastas Criadas.png" />
</p>

- Para finalizar, ainda no **Storage**, clique em **Copiar caminho do arquivo**:

<p align="center">
  <img width="80%" src="img/18-Copiar Caminho Storage.png" />
</p>

- Dentro do projeto, no arquivo **FirebaseHelper.py**, cole o caminho copiado no valor de **storageBucket** e remova a parte inicial do caminho, **gs://**, para que fique da seguinte forma:

<p align="center">
  <img width="80%" src="img/19-Alterar storageBucket.png" />
</p>

<p align="center">
  <img width="80%" src="img/20-Alterar storageBucket finalizado.png" />
</p>