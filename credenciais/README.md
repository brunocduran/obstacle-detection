# Configuração do Firebase e Credenciais

- Acesse o **Firebase Console** com sua conta: https://console.firebase.google.com/

- Inicie um novo projeto:

![1-Inicio](img/1-Inicio.png)

- Crie o projeto com qualquer nome, aceite os termos do Firebase e clique em **Continuar**:

![2-Criar-um-projeto](img/2-Criar-um-projeto.png)

- Ativar o **Google Analytics** é opcional. Clique em **Continuar**:

![3-Google-Analytics](img/3-Google-Analytics.png)

- Caso tenha ativado o **Google Analytics**, configure-o e clique em **Criar Projeto**

![4-Configurar-o-Google-Analytics](img/4-Configurar-o-Google-Analytics.png)

- O projeto foi criado. Clique em **Continuar**:

![5-Projeto-pronto](img/5-Projeto-pronto.png)

- Vá para **Configurações do Projeto**:

![6-Configuracao-do-Projeto](img/6-Configuracao-do-Projeto.png)

- Em seguida, acesse a aba **Contas de serviço**, selecione a opção **Python** e clique em **Gerar nova chave privada**.

![7-Contas-de-Servico](img/7-Contas-de-Servico.png)

- Após o passo anterior, será baixado um arquivo **JSON**. Esse arquivo deve ser renomeado para **credencial.json** e copiado para a pasta **credenciais**, localizada na raiz do projeto, no diretório: *obstacle-detection\credenciais*.

![8-Pasta-Credenciais](img/8-Pasta-Credenciais.png)

- Volte ao menu principal do Firebase, acesse o menu lateral **Criação** e selecione **Storage**.

![9-Criar-Storage](img/9-Criar-Storage.png)

- Nesse momento, para utilizar o **Storage**, será exigido um upgrade da conta para o plano **Blaze**, uma medida adotada pelo Firebase desde 30/10/2024. Mais detalhes podem ser encontrados nos seguintes portais:  
https://firebase.google.com/docs/storage/web/start?hl=pt#before-you-begin  
https://firebase.google.com/pricing?hl=pt-br  
Porém, ainda é possível utilizar o **Storage** gratuitamente, desde que não ultrapasse o limite de uso. Então, clique em **Fazer upgrade do projeto** e, em seguida, em **Criar uma conta do Cloud Billing**. Será necessário informar os dados de um cartão de crédito para possíveis cobranças adicionais:

![10-Storage](img/10-Storage.png)

![11-Criar-Conta-Cloud-Billing](img/11-Criar-Conta-Cloud-Billing.png)

- Após concluir o processo anterior, é possível prosseguir com a criação do **Storage**. Clique em **Começar**:

![12-Storage-Comecar](img/12-Storage-Comecar.png)

- Na configuração do **Bucket**, mantenha todas as opções padrão e clique em **Continuar**:

![13-Storage-Opcoes-de-Bucket](img/13-Storage-Opcoes-de-Bucket.png)

- Na configuração de **Regras de segurança**, selecione a opção **Iniciar no modo de teste** e clique em **Criar**:

![14-Storage-Regras-de-Seguranca](img/14-Storage-Regras-de-Seguranca.png)

- Após a criação do **Storage**, acesse a aba **Regras**, altere o código para permitir acesso até uma data mais posterior e clique em **Publicar**:

![15-Storage-Editar-Regras-de-Seguranca](img/15-Storage-Editar-Regras-de-Seguranca.png)

- Volte para a aba **Arquivos** e crie duas pastas com os nomes **imagens** e **models**:

![16-Storage-Criar-Pastas](img/16-Storage-Criar-Pastas.png)

- Ficando da seguinte forma:

![17-Storage-Pastas-Criadas](img/17-Storage-Pastas-Criadas.png)

- Para finalizar, ainda no **Storage**, clique em **Copiar caminho do arquivo**:

![18-Copiar-Caminho-Storage](img/18-Copiar-Caminho-Storage.png)

- Dentro do projeto, no arquivo **FirebaseHelper.py**, cole o caminho copiado no valor de **storageBucket** e remova a parte inicial do caminho, **gs://**, para que fique da seguinte forma:

![19-Alterar-storageBucket](img/19-Alterar-storageBucket.png)

![20-Alterar-storageBucket-finalizado](img/20-Alterar-storageBucket-finalizado.png)