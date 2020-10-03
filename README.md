# Age_classification
 Trabalho de conclusão de Curso 


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
<li>e diversas outras etapas.</li>
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

**Referências**
#### Referências

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


