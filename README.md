<h2><center>Trabalho de conclusão do curso de Especialização em Ciência de Dados da Facens</center></h2>
<h3><center>Faculdade de Engenharia de Sorocaba</center></h3>

**Alunos :** 
<p>
<p>Dani Shizue Sakamoto - https://www.linkedin.com/in/dani-sakamoto-b8250593/
<p>Mariana de Castro Pires Tavares Albuquerque  - https://www.linkedin.com/in/mariana-de-c-pires-tavares-albuquerque-9b4b7689/
<p>Tan Kin Wah - https://www.linkedin.com/in/tan-kin-wah/

**Orientador:** 
Fernando Vieira da Silva

# Google Drive:
 https://drive.google.com/drive/folders/10_XLLffHITGkOY7zdC8mD_heSm2GgMCx?usp=sharing

**Introdução**

Buscamos um dataset que contivesse imagens de rostos, para que conseguíssemos efetuar a classificação dos sujeitos em diversas faixas etárias.
Pensamos que referida classificação pode ser útil em inúmeras situações, tais como a identificação de menores desacompanhados em locais públicos, como por exemplo aeroportos e rodoviárias. Desse modo, nos propusemos a dividir os dados em duas (adultos e crianças) ou três (adultos, crianças e adolescentes) categorias.




Análise efetuada a partir do dataset "Adience", disponível em "Unfiltered Faces for Age and Gender Classification" https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender.


### Aprendizagem Profunda

<p style='text-align: justify;'>
   O conceito de deep learning é que tenhamos um único modelo treinado de ponta a ponta, ou seja, um único modelo, única rede, extremamente profunda que recebe uma imagem como entrada, submete por todas suas camadas, faz toda sua atribuição de pesos e no final a gente tem uma saída. </p>
   
<p style='text-align: justify;'>
   Fundamental em deep learning é reusabilidade de modelos, o TRANFER LEARNING, que consiste na habilidade de utilizar modelos pré-treinados normalmente utilizando milhões de amostras de dados com centenas de hyperparametros como ponto de entrada do processo de aprendizado.</p>

<p style='text-align: justify;'>
   Uma rede convolucional profunda pode ser utilizada pra aprender a detectar um objeto a partir de um volume gigantesco de dados e logo em seguida podemos utilizar essa rede como entrada para outra tarefa, onde faremos apenas o ajuste fim da última camada para tarefa mais específica.</p>
 
   Deep Learning promete resolver alguns problemas sérios que enfrentamos com método tradicionais de machine learning.
   As principais são:
   * Extração automática de características (processo caro, requer conhecimento do negócio e muitos experimentos);
   * Reusabilidade de modelos;
   * Modelos ponta a ponta;
   * Performance superior;
    
<p style='text-align: justify;'>
   Quando trabalhamos com problemas mais complexos como visão computacional é comum a utilização de sistemas modulares, onde cada um se especializa numa determinada tarefa como:
   <ul>
<li>Na extração de features;</li>
<li>No alinhamento de imagens;</li>
<li>Na identificação de objetos;</li>
<li>E diversas outras etapas.</li>
</ul>
<p style='text-align: justify;'>
   Para conseguirmos um bom resultado quando se trata de classificação, detecção de objetos em imagem é comum fazer a junção de diversos modelos do início ao fim. Com deep learning todos os modelos já são enfileirados, onde a saída é apenas uma, ou seja, a classificação de determinado evento se torna a entrada do próximo. Então a partir de uma camada o resultado dessa camada se torna a entrada da próxima camada, formando assim um pipeline extremamente grande e complexo.</p>
   
### Tarefas da visão computacional
   * Classificação de Objetos (Qual categoria de objeto contém na imagem?);
   * Identificação de Objetos (Qual o tipo de determinado objeto contém na imagem?);
   * Verificação de Objetos (Na imagem contém determinado objeto? - Check);
   * Detecção de Objetos;
   * Detecção de pontos chaves do Objeto (Onde estão os pontos?);
   * Segmentação de Objetos (Quais os pixels que pertencem ao objeto na imagem?);
   * Reconhecimento de Objetos (Quais os objetos na imagem e onde eles estão?);
   * Recuperação de informação (Tarefa de encontrar imagens similares a partir de uma determinada consulta);
   
https://minerandodados.com.br/

## Reconhecimento Facial - problema do campo da visão Computacional

- **O processo de reconhecimento passa por diversas etapas:**
    
     - Detecção de uma ou mais faces e localizar fronteiras:
     - Extração de features: Significa identificar características da face através de pontos. Distâncias que podem ser utilizadas na tarefa de reconhecimento.
     
     Em uma solução de reconhecimento facial, podemos ter camadas que se especializam em detectar features específicas para identificar uma pessoa como:
        * Olhos;
        * Boca;
        * Nariz;
        * Formato do rosto, etc.
        
    Começam a ser identificada através de padrões de constraste local nas camadas iniciais como:
        * Bordas;
        * Texturas.
        
    Nas camadas escondidas começa a se especializar ao dado, começa a ter features de uma determinada face como:
        * Bocas, olhos, etc;
        * Começa a ter camadas que se especializam cada vez mais.

## Detecção de Face
    
Para a detecção de faces utilizamos o detector facial MTCNN.

### MTCNN - Multi-task Cascaded Convolutional Neural Networks
https://pypi.org/project/mtcnn/

### Melhor accuracy e performance do que opencv

<p style='text-align: justify;'>
MTCNN é uma biblioteca python (pip) escrita pelo usuário do Github ipacz , que implementa o artigo Zhang, Kaipeng et al. “Detecção e alinhamento conjunto de faces usando redes convolucionais em cascata multitarefa.” Cartas de processamento de sinais IEEE 23.10 (2016): 1499–1503. Crossref. Web . Faz o reconhecimento da face e alinhamento baseado em processamento mais denso de uma rede neural.</p>

### OpenCV - Open Source Computer Vision Library
https://opencv.org/

   Utillizado no Python, C, C++.
   
   <p style='text-align: justify;'>
   É a biblioteca mais completa para trabalhar com visão computacional.
   No OpenCV há modelos de machine learning pré-treinado para classificadores.(Na documentação do OpenCV tem o endereço do github que tem o acesso de todos os cascades/classificadores pré-treinados para detectar olho, detectar boca e outros atributos da face).</p>
  

### Transfer Learning
https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/ 
https://keras.io/guides/transfer_learning/

<p style='text-align: justify;'>
   O aprendizado por transferência é um método de aprendizado de máquina em que um modelo desenvolvido/pré-treinado(pesos e parâmetros) para uma tarefa, é reutilizado como ponto de partida para um modelo em uma segunda tarefa.</p>

<p style='text-align: justify;'>
   Tem o benefício de diminuir o tempo de treinamento para um modelo de rede neural e pode resultar em um erro de generalização menor. O objetivo é aproveitar os dados da primeira configuração para extrair informações que podem ser úteis ao aprender ou mesmo, ao fazer previsões diretamente na segunda configuração. </p>
 
<p style='text-align: justify;'>
    A aprendizagem por transferência é a reusabilidade de modelos, otimização e um atalho para economizar tempo ou obter melhor desempenho.</p>

<p style='text-align: justify;'>
    Utiliza modelos pré-treinados, normalmente utilizando milhões de amostras de dados com centenas de hyperparametros como ponto de entrada do processo de aprendizado. Conjunto de dados mais utilizado é o ImageNet.</p>
   
   
### ImageNet
http://www.image-net.org/

Explore Categorias:
http://www.image-net.org/explore

<p style='text-align: justify;'>
   A ImageNet é uma base de dados extremamente grande, cerca de 14.197.122 imagens de alta resolução e 21.841 indexadas(que tem informações de classes). Está organizado de acordo com a hierarquia WordNet (atualmente apenas os substantivos), em que cada nó da hierarquia é representado por centenas e milhares de imagens. Ela é usada em competições pra trabalhar com Visão Computacional como na ImageNet Large-Scale Visual Recognition Challenge (ILSVRC).
</p>
<p style='text-align: justify;'>
   Em uma Rede Neural Convolucional Pré-treinada como InceptionResnetV2 carregamos os pesos pré-treinados no ImageNet como no exemplo a seguir:</p>
   InceptionResnetV2(weights='imagenet')
    
<p style='text-align: justify;'>   
   Se a base analisada é muito diferente da ImageNet pode-se usar menos camadas pré-treinadas e mais para o problema específico.</p>


### Adience Benchmark 

### Artigo: Age and Gender Classification using Convolutional Neural Networks

    Gil Levi and Tal Hassner
    Department of Mathematics and Computer Science
    The Open University of Israel
    
https://talhassner.github.io/home/publication/2015_CVPR;
https://github.com/GilLevi/AgeGenderDeepLearning

**Classificação de idade e gênero usando redes neurais convolucionais**

Publicado no Workshop IEEE sobre Análise e Modelagem de Faces e Gestos (AMFG), no IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston , 2015.

<p style='text-align: justify;'>
"As fontes das imagens incluídas em nosso conjunto são os álbuns do Flickr, montados por upload automático a partir de dispositivos de smartphones iPhone5 (ou posteriores), e lançados por seus autores ao público em geral sob a licença Creative Commons (CC)".</p>



## Redes Utilizadas no Transfer Learning
### ResNet 50
 COMPARATIVO DOS RESULTADOS E TODO ANALISE EM -> https://drive.google.com/file/d/1wNkqbWvPC9ov-ZQu9EJ5h-KkDiBs83iw/view?usp=sharing

![resnet](https://miro.medium.com/max/1000/1*zbDxCB-0QDAc4oUGVtg3xw.png)

<p style='text-align: justify;'>
  Com o aumento da profundidade das Redes Neurais, a acurácia começa a ficar saturada, devido ao problema desaparecimento do gradiente. Desse modo, pesquisadores da Microsoft resolveram esse problema com a ResNet, "pulando" conexões, ou seja, com uma rede residual. </p>
   
<p style='text-align: justify;'>
   Além disso, a ResNet foi uma das primeiras a adotar o "batch normalisation", ou seja, para uma camada interna da rede, é normalizada a saída da camada anterior, subtraindo-se a média e dividindo-se pelo desvio padrão.</p>

<p style='text-align: justify;'>
   Desse modo, a ResNet 50 pode conter até 152 camadas com 26 milhões de parâmetros, sem comprometer o poder de generalização do modelo.</p>
<br> 
  
### Inception ResNet-V2

![inception resnet v2](https://miro.medium.com/max/1000/1*xpb6QFQ4IknSmxmgai8w-Q.png)
   
<p style='text-align: justify;'>
As Inception-ResNets foram introduzidas, por pesquisadores do Google, em 2016, no mesmo artigo que as Inception-V4, e possuem 56 milhões de parâmetros.</p>

<p style='text-align: justify;'>
   Em suma, os módulos de "Inception" já presentes nas redes do tipo "Inception", foram convertidos para "Residual Inception Blocks", ou seja, há módulos que "pulam" conexões, resultando em memórial residual. Além disso, foram adicionados mais módulos de "Inception", bem como um novo tipo de módulo (Inception-A).</p>
 
# Resultados e Comparativos
### 1 - Comparativo entre Rede Residual RESNET-50 e a Inception ResNet - V2
ResNet-50
------------------------------------------------------------------------------------------------------------------------
#### Performance do treinamento


<!DOCTYPE html> 
<html>
<body>
<table>
<tr>
<th><img src="https://drive.google.com/uc?id=1Qu0qPB_EG2LYL8qvayYDbavmNP8CyjUO"/></th>

<th><img src="https://drive.google.com/uc?id=145oV7WD6eB3cuHGrcT_dXtlwLDgH6Hmd"/></th>
</tr>
</table>
</body>
<html>

InceptionResnetV2
------------------------------------------------------------------------------------------------------------------------

#### Performance do treinamento
    
<!DOCTYPE html> 
<html>
<body>
<table>
<tr>
<th><img src="https://drive.google.com/uc?id=1y2rh4KReHdMtmxmus17dYML8x04yzZGk"/></th>
<th><img src="https://drive.google.com/uc?id=16PDFiZ1ge8Tb8uklM4s2wsNt73PLpHRO"/></th>
</tr>
</table>
</body>
<html>
 
 
 ResNet-50
--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
    Performance do treinamento                            |    Performance do teste

                  precision    recall  f1-score   support |                  precision    recall  f1-score   support
           Adult       0.65      0.99      0.78      7027 |           Adult       0.65      1.00      0.79      3012
           Child       0.96      0.20      0.32      4737 |           Child       0.96      0.20      0.33      2032
                                                                                                             
        accuracy                           0.67     11764 |        accuracy                           0.67      5044
       macro avg       0.80      0.59      0.55     11764 |       macro avg       0.81      0.60      0.56      5044
    weighted avg       0.77      0.67      0.60     11764 |    weighted avg       0.97      0.97      0.97      5044
--------------------------------------------------------------------------------------------------------------------

InceptionResnetV2
---------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------
    Performance do treinamento                             |    Performance do teste
    
                  precision    recall  f1-score   support  |                  precision    recall  f1-score   support
           Adult       1.00      1.00      1.00      7027  |           Adult       0.97      0.99      0.98      3012  
           Child       1.00      1.00      1.00      4737  |           Child       0.98      0.96      0.97      2032

        accuracy                           1.00     11764  |        accuracy                           0.97      5044
       macro avg       1.00      1.00      1.00     11764  |       macro avg       0.97      0.97      0.97      5044  
    weighted avg       1.00      1.00      1.00     11764  |    weighted avg       0.97      0.97      0.97      5044
---------------------------------------------------------------------------------------------------------------------


#### Matriz de Confusão

    |  ResNet-50   |     | InceptionResnetV2 |
    |  2997 |  15  |     |   2968  |   44    |
    |  1625 | 407  |     |    86   |  1946   |


<table>
<tr>
<th><img src="https://drive.google.com/uc?id=1u11y7cWJilyToHz10dlpSFdiBitkuHs2"/></th>
<th><img src="https://drive.google.com/uc?id=18ol54wm2uMj--0ghgQmjcp1LJvPJ0nfg"/></th>
</tr></table>

<p style='text-align: justify;'>
Conclusão: O modelo InceptioResnetV2 teve uma performance melhor do que o da ResNet-50.  O modelo Resnet foi desenvolvido para resolver o problema do gradiente descendente. Com os modelos Resnet, os modelos CNNs podem ir cada vez mais fundo. ResNets têm gradientes significativamente mais baixos e, portanto, podem contornar o problema do gradiente explosivo, permitindo o treinamento eficaz de redes muito mais profundas.
InceptionResnet é uma melhoria adicional no Resnet, combinado a técnica chamada Inception que tem como objetivo principal atuar como um extrator de características em vários níveis da rede. </p>

## 2 - Comparativo dos resultados utilizando o detector de faces MTCNN e sem o detector de faces :

ResNet-50 - Resultado do treino e teste com as imagens sem passar pelo detector de faces
----------------------------------------------------------------------------------------------------------------------------------
![](https://drive.google.com/uc?export=view&H)

https://drive.google.com/file/d/19KdKhQbgnx5qtD7MMAQVGOOTMbNWpt9s/view?usp=sharing
#### Performance do treinamento
<table>
<tr>
<th><img src="https://drive.google.com/uc?id=19KdKhQbgnx5qtD7MMAQVGOOTMbNWpt9s"/></th>
<th><img src="https://drive.google.com/uc?id=1c1L_NfGmbllCwQRP-0oC4lzjReVqRMx5"/></th>
</tr></table>


ResNet-50 - Resultado do treino e teste utilizando o detector de faces
----------------------------------------------------------------------------------------------------------------------------------

#### Performance do treinamento
<table>
<tr>
<th><img src="https://drive.google.com/uc?id=1n6p_iva_fqsf-xplV5KEkPRakhfyVDLY"/></th>
<th><img src="https://drive.google.com/uc?id=1wj_5FOevqepojnmUGKGELL1xO2zgLcJB"/></th>
</tr></table>

ResNet-50
-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
    Performance do treinamento - Sem detector de face       |     Performance do teste - Sem detector de face
    
                  precision    recall  f1-score   support   |                   precision    recall  f1-score   support
           Adult       0.79      0.87      0.83      7027   |            Adult       0.78      0.87      0.82      3012
           Child       0.77      0.84      0.80      4737   |            Child       0.76      0.83      0.79      2032
            Teen       0.40      0.01      0.03      1267   |             Teen       0.32      0.01      0.02       542

        accuracy                           0.78     13031   |         accuracy                           0.77      5586
       macro avg       0.65      0.58      0.55     13031   |        macro avg       0.62      0.57      0.55      5586
    weighted avg       0.74      0.78      0.74     13031   |     weighted avg       0.73      0.77      0.73      5586
-----------------------------------------------------------------------------------------------------------------------    
     Performance do treinamento - Com detector de face      |     Performance do teste - Com detector de face
    
                  precision    recall  f1-score   support   |                   precision    recall  f1-score   support
           Adult       0.99      0.97      0.98      5759   |            Adult       0.93      0.95      0.94      2472
           Child       0.99      0.99      0.99      3801   |            Child       0.95      0.93      0.94      1645
            Teen       0.89      0.97      0.93       997   |             Teen       0.73      0.68      0.70       437

        accuracy                           0.98     10557   |         accuracy                           0.92      4554
       macro avg       0.96      0.98      0.97     10557   |        macro avg       0.87      0.85      0.86      4554
    weighted avg       0.98      0.98      0.98     10557   |     weighted avg       0.91      0.92      0.91      4554
-----------------------------------------------------------------------------------------------------------------------


### Matriz de Confusão - ResNet-50

####  1 - Sem detector de face

    | Adulto  | Criança | Adolescente | 
    |   2613  |   390   |      9      |
    |    349  |  1677   |      6      |
    |    390  |   145   |      7      |
        
####  2 - Com detector de face

    | Adulto  | Criança | Adolescente | 
    |   2348  |   44    |      80     |
    |    88   |  1525   |      32     |
    |   100   |   40    |     297     |
                                
<table>
<tr>
<th><img src="https://drive.google.com/uc?id=1Bi6AxbWO6Z4DDhEiGyHPy76eTiEbEfAA"/></th>
<th><img src="https://drive.google.com/uc?id=1Ve1Hn7fuJsSOMGItBa4HN08ammwGtQJw"/></th>
</tr></table>

Conclusão: Utilizando o detector de faces obtivemos uma melhor performance em todos os testes realizados.

## 3 - Comparativo dos resultados dos treinos com o Modelo InceptionResnetV2
   - **1 - Adulto-Criança** 
   - **2 - Adulto-Adolescente-Criança**
   
#### Com detecção de faces utilizando MTCNN

InceptionResnetV2 - Adulto - Criança
------------------------------------------------------------------------------------------------------------------------

#### Performance do treinamento
<table>
<tr>
<th><img src="https://drive.google.com/uc?id=15D7B3_XuwUfoMdpEgusG_gGTHtI_8KO0"/></th>
<th><img src="https://drive.google.com/uc?id=1ZGwcasNhO080rlFycQSEdQIW0oDtQFX3"/></th>
</tr></table>

InceptionResnetV2 - Adulto - Adolescente - Criança
------------------------------------------------------------------------------------------------------------------------

#### Performance do treinamento
<table>
<tr>
<th><img src="https://drive.google.com/uc?id=17dGQNmRRJaqK7mVY-F7G6GVWXVfBS3RS"/></th>
<th><img src="https://drive.google.com/uc?id=16ZXyURLzDscOLsPEVG4T2fae6K9UM5yD"/></th>
</tr></table>


InceptionResnetV2 - Adulto - Criança
-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------    
     Performance do treinamento                             |     Performance do teste
    
                  precision    recall  f1-score   support   |                   precision    recall  f1-score   support
           Adult       0.99      1.00      0.99      5759   |            Adult       0.92      0.99      0.96      2472
           Child       1.00      0.99      0.99      3801   |            Child       0.99      0.87      0.93      1645

        accuracy                           0.99      9560   |         accuracy                           0.95      4117
       macro avg       1.00      0.99      0.99      9560   |        macro avg       0.96      0.93      0.94      4117
    weighted avg       0.99      0.99      0.99      9560   |     weighted avg       0.95      0.95      0.94      4117
-----------------------------------------------------------------------------------------------------------------------

InceptionResnetV2 - Adulto - Adolescente - Criança
-----------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------    
     Performance do treinamento                             |     Performance do teste
    
                  precision    recall  f1-score   support   |                   precision    recall  f1-score   support
           Adult       0.99      1.00      0.99      5759   |            Adult       0.86      0.99      0.92      2472
           Child       1.00      0.98      0.99      3801   |            Child       0.96      0.88      0.92      1645
            Teen       0.99      0.98      0.99       997   |             Teen       0.81      0.40      0.53       437

        accuracy                           0.99     10557   |         accuracy                           0.89      4554
       macro avg       0.99      0.99      0.99     10557   |        macro avg       0.88      0.75      0.79      4554
    weighted avg       0.99      0.99      0.99     10557   |     weighted avg       0.89      0.89      0.88      4554
-----------------------------------------------------------------------------------------------------------------------

### Matriz de Confusão - InceptionResnetV2

####  1 - Adulto / Criança
     |  Adulto | Criança |
     |   2457  |   15    |
     |    209  |  1436   |
     
####  2 - Adulto / Adolescente / Criança

    | Adulto  | Criança | Adolescente | 
    |   2440  |   15    |      17     |
    |    176  |  1444   |      25     |
    |    225  |   38    |     174     |
                              
<table>
<tr>
<th><img src="https://drive.google.com/uc?id=1FClq6Z1EA8cxGF4T0DCv_Qv1lRQ1aXY0"/></th>
<th><img src="https://drive.google.com/uc?id=1scFXb7hkM372oZ6aWI3aC6PiHhuAvuAk"/></th>
</tr></table>

Comparando os dois modelos vemos pouca diferença na performance. Destacamos que a classe "adolescente" apresenta uma acurácia menor, tal qual já verificado em Levi e Hassner (2015), e conforme mencionado anteriormente.


### Utilização de Face Embeddings - Facenet

<p style='text-align: justify;'>
Adicionalmente, efetuamos a análise do dataset mediante:</p>
<ol>
<li>Detecção Facial com a biblioteca MTCNN;</li>
<li>Extração de Features (Face Embeddings), utilizando Facenet;</li>
<li>Utilização da biblioteca Pycaret para explorar o potencial modelo de Aprendizado de Máquina mais adequado ao problema de classificação;</li>
<li>Aplicação do algoritmo Xgboost.</li>
</ol>
 <p style='text-align: justify;'>
O sistema Facenet foi desenvolvido em 2015, por pesquisadores do Google, e representa o estado da arte em datasets benchmark de reconhecimento facial.
Ele pode ser utilizado para extrair features de alta qualidade das faces, chamados <i>face embeddings</i>, representados por 128 embeddings vetoriais. Utilizamos uma implementação desse modelo para a biblioteca Keras, desenvolvido por Hiroki Taniai
</p>
<p style='text-align: justify;'>
Após a extração dos <i>face embeddings</i>, utilizamos a biblioteca Pycaret, que permite testar vários modelos de uma só vez no dataset, a fim de escolher o que apresente os melhores resultados. Posteriormente, implementamos o modelo escolhido, Xgboost, que possui biblioteca própria.
</p>
  https://medium.com/clique-org/how-to-create-a-face-recognition-model-using-facenet-keras-fd65c0b092f1; https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/; https://github.com/nyoki-mtl; https://github.com/pycaret/pycaret.


<p style='text-align: justify;'>Verificamos que os resultados obtidos com a extração de <i>face embeddings</i> seguida pela classificação utilizando o algoritmo XGBoost revelaram-se ligeiramente superiores aos que obtivemos utilizando <i>transfer learning</i> com Redes Neurais Convolucionais.</p> 
<p style='text-align: justify;'>O grande diferencial, no entanto, é o esforço computacional necessário. Se o treinamento das redes convolucionais levou <b>dias</b> em nossos computadores, o processamento completo (extração de face embeddings com a utilização do Facenet seguido do treinamento utilizando o XGBoost) realizou-se em <b>poucas horas</b>, revelando-se uma metodologia muito mais efetiva e eficiente.</p>   

## Resultado do modelo XGBoost

<p style='text-align: justify;'>
XGBoost é uma biblioteca otimizada de aumento de gradiente distribuída projetada para ser altamente eficiente, flexível e portátil. Ele implementa algoritmos de aprendizado de máquina sob a estrutura Gradient Boosting. O XGBoost fornece um reforço de árvore paralela (também conhecido como GBDT, GBM) que resolve muitos problemas de ciência de dados de maneira rápida e precisa. O mesmo código é executado em grandes ambientes distribuídos (Hadoop, SGE, MPI) e pode resolver problemas além de bilhões de exemplos.</p>  https://xgboost.readthedocs.io/en/latest/ 

-----------------------------------------------------------------------------------------------------------------------    
     Performance do treinamento                             |     Performance do teste
    
                  precision    recall  f1-score   support   |                   precision    recall  f1-score   support
           Adult       1.00      1.00      1.00      6098   |            Adult       0.94      0.99      0.96      2607
           Child       1.00      1.00      1.00      4075   |            Child       0.98      0.97      0.97      1715
            Teen       1.00      0.99      1.00      1141   |             Teen       0.93      0.68      0.79       480

        accuracy                           1.00     11314   |         accuracy                           0.92      4802
       macro avg       1.00      1.00      1.00     11314   |        macro avg       0.91      0.81      0.85      4802
    weighted avg       1.00      1.00      1.00     11314   |     weighted avg       0.92      0.92      0.91      4802
-----------------------------------------------------------------------------------------------------------------------

![matrix confusao](https://drive.google.com/uc?id=1CSdSLzNPFVVgVETuNeNJLbdUPUmefjpN)

<p style='text-align: justify;'>Verificamos que os resultados obtidos com a extração de <i>face embeddings</i> seguida pela classificação utilizando o algoritmo XGBoost revelaram-se ligeiramente superiores aos que obtivemos utilizando <i>transfer learning</i> com Redes Neurais Convolucionais.</p> 
<p style='text-align: justify;'>O grande diferencial, no entanto, é o esforço computacional necessário. Se o treinamento das redes convolucionais levou <b>dias</b> em nossos computadores, o processamento completo (extração de face embeddings com a utilização do Facenet seguido do treinamento utilizando o XGBoost) realizou-se em <b>poucas horas</b>, revelando-se uma metodologia muito mais efetiva e eficiente.</p>   


**Referências**

<p>G. Levi and T. Hassncer, "Age and gender classification using convolutional neural networks," 2015 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Boston, MA, 2015, pp. 34-42, doi: 10.1109/CVPRW.2015.7301352.</p>

<p>(1, 2) Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.</p>

<p>Jason Brownlee on December 20, 2017 in Deep Learning for Computer Vision https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/.</p>
<p> Raimi Karim on July, 29, 2019: "Illustrated: 10 CNN Architectures
A compiled visualisation of the common convolutional neural networks" in Towards Data Science. https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d, acesso em 01/10/2020.</p> 

<p>TY-JOUR, AU-Philipp, George,AU-Song, Dawn,AU-Carbonell, Jaime,PY-2017/12/15 SP-T1-Gradients explode-Deep Networks are shallow - ResNet explained.</p> 
<p> Schroff, F., Kalenichenko, D., Philbin, J.. FaceNet: A Unified Embedding for Face Recognition and Clustering. Disponível em: https://arxiv.org/abs/1503.03832, acesso em 01/10/2020.</p>

<p>Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. 2015. Disponível em: https://arxiv.org/abs/1512.03385, acesso em 01/10/2020.</p>

<p>Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. 2016. Disponível em: https://arxiv.org/abs/1602.07261, acesso em: 01/10/2020.</p>

<p>J. Deng, W. Dong, R. Socher, L. Li, Kai Li and Li Fei-Fei, "ImageNet: A large-scale hierarchical image database," 2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, 2009, pp. 248-255, doi: 10.1109/CVPR.2009.5206848.</p>




