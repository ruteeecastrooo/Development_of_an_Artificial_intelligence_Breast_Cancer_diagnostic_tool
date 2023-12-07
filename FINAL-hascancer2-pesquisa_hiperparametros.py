import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pickle


# Carregar os dados
df = pd.read_csv("final4.csv", index_col=0)

# Filtrar para equilibrar as classes da coluna 'hasCancer'
df# Assuming your target column is 'hasCancer'
has_cancer_rows = df[df['hasCancer'] == 1].sample(n=144, random_state=42)
other_rows = df[df['hasCancer'] != 1]

# Combinar os 2
df = pd.concat([has_cancer_rows, other_rows], ignore_index=True)

# Separa os atributos das etiquetas
y = df["hasCancer"]
 #y=y.sample(frac=1)
X = df.drop(["hasCancer"], axis=1)

# Divide o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Código para visualização TSNE 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.manifold import TSNE



X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
X_embedded.shape
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 4)

sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, legend='full', palette=palette)
fig = px.scatter_3d(x=X_embedded[:, 0], y=X_embedded[:, 1], z=X_embedded[:, 2], color=y, opacity=0.8)
fig.show()

# Seleção de características usando F-valor
# X = SelectKBest(f_classif, k=1500).fit_transform(X, y)
algoritmo_pre_processing = SelectKBest(f_classif, k=1500)
algoritmo_pre_processing.fit(X_train, y_train)

# Transforma os conjuntos de treino e teste
X_train = algoritmo_pre_processing.transform(X_train)
X_test = algoritmo_pre_processing.transform(X_test)

# Salvar os conjuntos de dados em arquivos pickle
with open('X_train_validation_fvttt.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('y_train_validation_fvttt.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('X_test_fvttt.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('y_test_fvttt.pkl', 'wb') as f:
    pickle.dump(y_test, f)

# Recuperar os f-value e os nomes das características correspondentes
valores_f = algoritmo_pre_processing.scores_
nomes_caracteristicas = X.columns

# Selecionar os índices das k características + importantes
indices_selecionados = algoritmo_pre_processing.get_support(indices=True)

# Ordenar os f-value e os nomes das características
indices_ordenados = np.argsort(valores_f[indices_selecionados])[::-1]
nomes_caracteristicas_ordenados = nomes_caracteristicas[indices_selecionados][indices_ordenados]
valores_f_ordenados = valores_f[indices_selecionados][indices_ordenados]

# Salvar os nomes das características f-value ordenadoas em um ficheiro txt
with open('sorted_f_value_hascancer.txt', 'w') as f:
    for nome, valor in zip(nomes_caracteristicas_ordenados, valores_f_ordenados):
        f.write(f"{nome}\t{valor}\n")

    

# Ler os dados para treino, validação e teste
# Neste caso, estamos a usar os dados que foram pré-processados, vamos utilizar o critério 'f-value' para seleção de características para a pesquisa de hiperparametros

with open(f'X_train_validation_fvttt.pkl','rb') as f:
    X_train_validation = pickle.load(f)
with open(f'y_train_validation_fvttt.pkl','rb') as f:
    y_train_validation = pickle.load(f)
with open(f'X_test_fvttt.pkl','rb') as f:
    X_test = pickle.load(f)
with open(f'y_test_fvttt.pkl','rb') as f:
    y_test = pickle.load(f)

from sklearn.model_selection import RandomizedSearchCV

"""
Classe CustomRandomizedSearchCV:
A CustomRandomizedSearchCV é uma adaptação do RandomizedSearchCV providenciado pelo Scikit-learn.
Ela foi projetada com a intenção de simplificar e potencializar a busca aleatória de hiperparâmetros em diversos modelos de aprendizado de máquina. 
A classe foi desenvolvida para permitir uma avaliação simultânea e comparativa de diferentes modelos e as combinações de seus hiperparâmetros.

Ao concluir a pesquisa, a classe proporciona uma análise detalhada, revelando os hiperparâmetros mais eficazes para cada modelo avaliado.
Além disso, apresenta o best score de cada modelo baseando-se na métrica F1 micro-averaged.

Desta forma, otimiza-se o processo de seleção e ajuste de modelo, garantindo que os melhores hiperparâmetros
sejam identificados de maneira eficiente e sistemática.

"""
class CustomRandomizedSearchCV:
    
    def __init__(self, X, y, algorithm, parameter_distribution, cv_folds = 10, num_iters_allowed=100, score_function='f1_micro', n_jobs=25) -> None:
        self.X = X
        self.y = y
        self.algorithm = algorithm
        self.parameter_distribution = parameter_distribution
        self.cv_folds = cv_folds
        self.num_iters_allowed = num_iters_allowed
        self.score_function = score_function
        self.search = None
        self.n_jobs = n_jobs
        
    # Calcula o total de combinações de parâmetros possíveis
    def calc_total_parameters(self, params_list: list) -> int: 
        tmp = [len(lista) for lista in params_list.values()]
        res = 1
        for x in tmp:
            res *= x
        return res
    
    # Executa RandomizedSearchCV
    def random_search_cv(self):
        if self.search == None:
            percentage = float(self.num_iters_allowed) * 100 / self.calc_total_parameters(self.parameter_distribution)
            print(f"Going to train and test {self.num_iters_allowed} out of a total of {self.calc_total_parameters(self.parameter_distribution)} parameter combinations ({percentage:.2f}%)")
            random_search_cv = RandomizedSearchCV(self.algorithm, self.parameter_distribution, random_state=0, n_iter=self.num_iters_allowed, cv=self.cv_folds, n_jobs=self.n_jobs, scoring=self.score_function)
            self.search = random_search_cv.fit(self.X, self.y)
    
    # Retorna os melhores parâmetros encontrados
    def get_best_parameters(self):
        self.random_search_cv()
        return self.search.best_params_
    
    def get_best_score(self):
        self.random_search_cv()
        return self.search.best_score_
    
    def get_average_train_time(self):
        self.random_search_cv()
        return np.mean(self.search.cv_results_['mean_fit_time'])
    
    def get_average_test_time(self):
        self.random_search_cv()
        return np.mean(self.search.cv_results_['mean_score_time'])
    
    def get_best_estimator(self):
        self.random_search_cv()
        return self.search.best_estimator_
    
    def get_all_cv_results(self):
        self.random_search_cv()
        return self.search.cv_results_

    def print_details(self):
        print(f"Best score: {self.get_best_score()}")
        print(f"Best parameters: {self.get_best_parameters()}")
        print(f"Average train time (secs): {self.get_average_train_time():.2}")
        print(f"Average test time (secs): {self.get_average_test_time():.2}")

from sklearn.svm import SVC
# O algoritmo SVM é útil para encontrar fronteiras não lineares complexas em conjuntos de dados de alta dimensão.

svm_random_search_cv = CustomRandomizedSearchCV(X_train_validation, y_train_validation, algorithm=SVC(class_weight='balanced'),
                               parameter_distribution=dict(C=np.logspace(np.log10(0.1),np.log10(1000), 1000), 
                                                           kernel=['linear', 'poly', 'rbf'], 
                                                           gamma=np.logspace(np.log10(0.001),np.log10(10), 1000), 
                                                           max_iter=[100, 200, 500, 1000]),
                               num_iters_allowed=1000)

svm_random_search_cv.print_details()


from sklearn.neighbors import KNeighborsClassifier
# KNN é um algoritmo de aprendizagem baseado em instância e é usado tanto para classificação quanto regressão.

knn_random_search_cv = CustomRandomizedSearchCV(X_train_validation, y_train_validation, algorithm=KNeighborsClassifier(),
                               parameter_distribution=dict(n_neighbors=[int(x) for x in np.linspace(1, 30, 30)], 
                                                           algorithm=['ball_tree', 'kd_tree', 'brute'],
                                                           weights=['uniform', 'distance'], 
                                                           metric=['euclidean', 'manhattan', 'minkowski']), 
                               num_iters_allowed=1000)


knn_random_search_cv.print_details()


from sklearn.linear_model import LogisticRegression
# A regressão logística é usada quando a variável dependente é categórica.

log_reg_random_search_cv = CustomRandomizedSearchCV(X_train_validation, y_train_validation, algorithm=LogisticRegression(),
                               parameter_distribution=dict(penalty=['l1', 'l2', 'elasticnet', 'none'],
                                                           solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                                           max_iter=[100, 200, 500, 1000, 2000, 3500, 5000],
                                                           multi_class=['auto', 'ovr', 'multinomial'],
                                                           C=np.linspace(0.5, 2, 30)), 
                               num_iters_allowed=1000)


log_reg_random_search_cv.print_details()


from sklearn.tree import DecisionTreeClassifier
# As árvores de decisão são usadas na tomada de decisões e para resolver problemas de classificação e regressão.

dt_random_search_cv = CustomRandomizedSearchCV(X_train_validation, y_train_validation, algorithm=DecisionTreeClassifier(),
                               parameter_distribution=dict(
                               criterion=['gini', 'entropy'],
                               splitter=['best', 'random'],
                               max_depth=[None, 2, 4, 8, 16],
                               min_samples_split=[2, 4, 8, 16, 32],
                               # min_samples_leaf=[1],
                               max_features=['auto', 'sqrt', 'log2'],
                               ), 
                               num_iters_allowed=1000)


dt_random_search_cv.print_details()

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
# Um random forest é um meta-estimador que se ajusta a uma série de árvores de decisão.

rf_random_search_cv = CustomRandomizedSearchCV(X_train_validation, y_train_validation, algorithm=RandomForestClassifier(n_estimators=100),
                               parameter_distribution=dict(
                                   criterion=['gini', 'entropy'],
                                    max_depth=[None, 2, 4, 8, 16],
                                    min_samples_split=[2, 4, 8, 16, 32],
                                    max_features=['auto', 'sqrt', 'log2'],
                               ), 
                               num_iters_allowed=1000)

rf_random_search_cv.print_details()

from sklearn.ensemble import AdaBoostClassifier
# AdaBoost é um algoritmo de boosting

ada_b_random_search_cv = CustomRandomizedSearchCV(X_train_validation, y_train_validation, algorithm=AdaBoostClassifier(),
                               parameter_distribution=dict(algorithm=['SAMME', 'SAMME.R'],
                                                           n_estimators=[10, 50, 100, 200],
                                                           learning_rate= np.logspace(np.log10(0.01),np.log10(1), 100)
                                                           
                               ), 
                               num_iters_allowed=1000)


ada_b_random_search_cv.print_details()

from sklearn.neural_network import MLPClassifier
# MLP é uma rede neural artificial que pode ser usada tanto para classificação quanto para regressão.

mlp_single_layer_random_search_cv = CustomRandomizedSearchCV(X_train_validation, y_train_validation, algorithm=MLPClassifier(),
                               parameter_distribution=dict(hidden_layer_sizes=[(128,),
                                                                               (256,),
                                                                               (512,),
                                                                               (1024,),
                                                                               (2048,)],
                                                           solver=['lbfgs', 'sgd', 'adam'],
                                                           learning_rate=['constant', 'invscaling', 'adaptive'],
                                                           learning_rate_init=np.logspace(np.log10(0.001),np.log10(1), 1000),
                                                           max_iter=[200, 500, 2000],
                                                           batch_size=["auto", 64, 512]
                                                           ), 
                               num_iters_allowed=1000,
                               )



mlp_single_layer_random_search_cv.print_details()

from sklearn.neural_network import MLPClassifier


# algorithm = MLPClassifier()
mlp_dnn_random_search_cv = CustomRandomizedSearchCV(X_train_validation, y_train_validation, algorithm=MLPClassifier(),
                               parameter_distribution=dict(hidden_layer_sizes=[(512, 256, 128, 128, 128, 128, 128, 128),
                                                                               (256, 128, 128, 128, 128, 128, 128, 128),
                                                                               (512, 256, 128, 128, 128, 128, 128),
                                                                               (512, 256, 128, 128, 128, 128),
                                                                               (256, 128, 128, 128, 128), 
                                                                               (512, 128, 128, 128, 128), 
                                                                               (512, 128, 128, 128),
                                                                               (256, 128, 128, 128),
                                                                               (256, 128, 128), 
                                                                               (512, 128),
                                                                               (256, 128)],
                                                           solver=['lbfgs', 'sgd', 'adam'],
                                                           learning_rate=['constant', 'invscaling', 'adaptive'],
                                                           learning_rate_init=np.logspace(np.log10(0.001),np.log10(1), 1000),
                                                           max_iter=[200, 500, 2000],
                                                           batch_size=["auto", 64, 512]
                                                           ), 
                               num_iters_allowed=1000,
                               )

mlp_dnn_random_search_cv.print_details()
