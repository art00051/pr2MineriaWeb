import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer

dataset = 'news_reducido.csv'

semillas = [42, 640, 5300742]
pesos = ["uniform", "distance"]
valor_p = [1, 2] # [1, 2, 3, 5, 7, 10]
valor_k = [3, 4, 5, 6, 7, 10]

# Leer los datos en formato csv
data = pd.read_csv(dataset, on_bad_lines='skip')
# Nos quedamos con el texto (puedes quedarte con más información si quieres)
X = data['text'].fillna('').astype(str).to_numpy() # Ponco con un espacio los espacio nulos

enc = OrdinalEncoder()
y = enc.fit_transform(np.reshape(data['category'], (-1,1))).reshape(-1)

text_clasifier_binario = Pipeline([
    ('vect', CountVectorizer(binary=True)),
    ('clf', KNeighborsClassifier()) #Preguntar los randoms state
])

text_clasifier_tf = Pipeline([
    ('vect', CountVectorizer(binary=False)),
    ('clf', KNeighborsClassifier()) #Preguntar el random_state
])

text_clasifier_tfidf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', KNeighborsClassifier())
])

results = []

pipelines = {
    'binary': text_clasifier_binario,
    'tf':     text_clasifier_tf,
    'tfidf':  text_clasifier_tfidf
}

for semilla in semillas:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)

    for nombre_pipe, pipeline in pipelines.items():
        for peso in pesos:
            for p in valor_p:
                for k in valor_k:
                    # Actualizar hiperparámetros del clasificador dentro del pipeline
                    pipeline.set_params(
                        clf__weights=peso,
                        clf__p=p,
                        clf__n_neighbors=k
                    )

                    accuracies = np.zeros(5)
                    for i, (tra, tst) in enumerate(skf.split(X, y)):
                        pipeline.fit(X[tra], y[tra])
                        predicted = pipeline.predict(X[tst])
                        accuracies[i] = np.mean(predicted == y[tst])

                    avg_acc = np.average(accuracies)
                    results.append({
                        'semilla':     semilla,
                        'vectorizer':  nombre_pipe,
                        'peso':        peso,
                        'p':           p,
                        'k':           k,
                        'avg_acc':     avg_acc
                    })
                    print(f'semilla={semilla} | vec={nombre_pipe} | peso={peso} | p={p} | k={k} → acc={avg_acc:.4f}')

# Convertir a DataFrame para analizar mejor los resultados
results_df = pd.DataFrame(results)
best = results_df.loc[results_df['avg_acc'].idxmax()]
print("\n=== Mejor configuración ===")
print(best)