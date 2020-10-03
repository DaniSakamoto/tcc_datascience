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
  




**Referências**

<p>

<p>Tal Hassner, Shai Harel, Eran Paz and Roee Enbar, Effective Face Frontalization in Unconstrained Images, IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015.
    Disponível em: https://ieeexplore.ieee.org/document/7301352, acesso em 29/09/2020.

<p> Gil Levi and Tal Hassner, Age and Gender Classification Using Convolutional Neural Networks, IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015.
    Disponível em: https://talhassner.github.io/home/projects/frontalize/CVPR2015_frontalize.pdf, acesso em 29/09/2020.

<p>Eran Eidinger, Roee Enbar, and Tal Hassner, Age and Gender Estimation of Unfiltered Faces, Transactions on Information Forensics and Security (IEEE-TIFS), special issue on Facial Biometrics in the Wild, Volume 9, Issue 12, pages 2170 - 2179, Dec. 2014.
    Disponível em: https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf, acesso em 29/09/2020.
