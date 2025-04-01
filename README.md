# Aprendizagem Profunda: Trabalho Prático 1

## Autores

- Diogo Ferreira, PG55932 (DiogoUM)
- Guilherme Barbosa, PG55946 (Kard9876)
- João Carvalho, PG55959 (Hiicecream)
- Rafael Peixoto, PG55998 (rafapeixoto16)
- Rodrigo Ralha, PG56005 (rodrigo0345)

## Código Base

- O código base usado para a nossa implementação da DNN foi o fornecido na aula PL03
- O código base usado para a nossa implementação da regressão logística foi o fornecido na aula PL01

## Estrutura do Repositório

- Na pasta `Code` encontra-se o código dos modelos desenvolvidos à mão pelo grupo
  - A pasta `Code/DNN` possui o código do modelo Redes Neuronais Densas
  - A pasta `Code/LogisticRegression` possui o código do modelo de Regressão Logística
  - A pasta `Code/RNN` possui o código implementado para o modelo de Redes Neuronais Recorrentes
  - A pasta `Code/SVM` possui uma tentativa de implementação do modelo de Support Vector Machine
  - A pasta `Code/utils` possui alguns utilitários desenvolvidos pelo grupo, tais como a leitura dos datasets e o código para guardar os objetos dos modelos

- Na Pasta `Dataset` tem o código necessário para gerar os datasets de treino, teste e validação
  - O ficheiro `Dataset/dataset2.py` ainda está em desenvolvimento, pelo que deve ser ignorado
  - Possui diversos datasets base com o formato .csv
  - Possui uma pasta (`Dataset/DatasetsGerados`) com o resultado de processar os dataset base a partir do script Datasets/dataset.py
  - A pasta `Dataset/gerador textos` possui o código para gerar datasets mais fidedignos a este trabalho

- As pastas `DNNOriginal`, `LogisticRegressionOriginal` e `RegularizationOriginal(PL02)` são pastas com o código base fornecido pelo docente nos quais nos baseamos para a construção de alguns modelos. o grupo decidiu mantê-las caso precise de backup de algum código

- A pasta `Notebooks` é onde estão os diversos jupyter-notebooks dos modelos implementados
  - A pasta `Notebooks/Implemented` é onde estão guardados os notebooks dos modelos implementados de raíz pelo grupo
  - A pasta `Notebooks/Tensorflow` é onde estão guardados os notebooks dos modelos tensorflow usados pelo grupo para as próximas fases do trabalho

- A pasta `LLMs` é onde estão os notebooks que implementam a comunicação com as API's do deepseek e together.ai

- A pasta `Submissao1` é onde o grupo colocou os notebooks e as previsões necessárias à primeira submissão do trabalho
  - O grupo desejaria guardar os modelos para não precisar de os treinar a cada vez. No entanto, apesar de o poder fazer localmente, não consegue enviar os ficheiros com os modelos por ocuparem mais espaço do que o permitido pelo GitHub

- Na pasta `Submissao2` encontram-se os notebooks e as previsões necessárias à segunda submissão do trabalho

- A pasta `Submissao3` possui os notebooks e as previsões necessárias à terceira e última submissão do trabalho

- Os ficheiros `Apresentação.pdf` e `Apresentação.pptx` possuem os slides apresentados no dia 28/03/2025, nos formatos PDF e Powerpoint, respetivamente

- Por fim, o ficheiro `Relatório.pdf` possui o relatório desenvolvido pelo grupo

## Dependências

- Numpy
- Pandas
- Scikit-learn (usado para pre-processar os dados)
- nltk (deve usar-se Python 3.12)
- gensim
- keras-tuner
- tensorflow (2.18.0)
- tf-keras (2.18.0)

## Submissões

Para garantir reproduzibilidade, o grupo indica a seguir qual o ambiente e dependências de cada uma das submissões:

- A 1ª submissão foi executada num ambiente python 3.12.6, com numpy==1.26.4, pandas==2.2.3, scikit-learn==1.5.2, nltk==3.9.1, gensim==4.3.3
- A 2ª submissão foi executada num ambiente python 3.12.6, com numpy==1.26.4, pandas==2.2.3, scikit-learn==1.5.2, nltk==3.9.1, gensim==4.3.3 e tensorflow==2.18.0
- Na 3ª submissão:  
  - O ficheiro `submissao3-grupo007-s1.ipynb` foi executado num ambiente python 3.12.6 com requests==2.32.3.
  - O ficheiro `submissao3-grupo007-s2.ipynb` foi executado num ambiente pythin 3.12.6 com numpy==1.26.4, tensorflow==2.19.0, nltk==3.9.1, keras==3.9.0, pandas==2.2.3
