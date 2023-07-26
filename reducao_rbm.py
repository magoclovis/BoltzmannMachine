import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

base = datasets.load_digits()
previsores = np.asarray(base.data, 'float32')
classe = base.target

normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.2, random_state=0)

rbm = BernoulliRBM(random_state = 0)

# numero de epocas
rbm.n_iter = 25

# numero de neuronios na camada de saida
rbm.n_components = 50

# não é necessário passar a quantidade da camada de entrada pois o progama ja sabe que 
# são 64 entradas por causa da base de dados

naive_rbm = GaussianNB()

classificador_rbm = Pipeline(steps = [('rbm', rbm), ('naive', naive_rbm)])

# treinamento
classificador_rbm.fit(previsores_treinamento, classe_treinamento)

plt.figure(figsize=(20,20))
for i, comp in enumerate(rbm.components_):
    # 10, 10 tamanho da imagem no consolo
    # i+1 indice da imagem
    plt.subplot(10, 10, i + 1)
    
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()

# respostas
# fazer o comparativo com o classe_teste
previsoes_rbm = classificador_rbm.predict(previsores_teste)

# saber a taxa de acerto
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste)

# sem utilizar o RBM
naive_simples = GaussianNB()
naive_simples.fit(previsores_treinamento, classe_treinamento)
previsoes_naive = naive_simples.predict(previsores_teste)
precisao_naive = metrics.accuracy_score(previsoes_naive, classe_teste)

# com RBM é 7,5% mais preciso nesse exercicio