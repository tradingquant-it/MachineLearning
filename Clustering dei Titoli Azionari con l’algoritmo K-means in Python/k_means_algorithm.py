
import numpy as np
import pandas as pd
import yfinance as yf
from math import sqrt
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Legge url ed estrae i dati dei ticker
data_table = pd.read_html(sp500_url)
tickers = data_table[0][1:]['Symbol'].to_list()
prices_list = []
# Scarica i dati storici dei ticker
for ticker in tickers:
    try:
        prices = yf.download(ticker, '2017-01-01')['Adj Close']
        prices = pd.DataFrame(prices)
        prices.columns = [ticker]
        prices_list.append(prices)
        prices.to_csv("data/"+ticker+".csv", sep="\t")
    except Exception as e:
        print(str(e))
        pass
prices_df = pd.concat(prices_list,axis=1)

prices_df.sort_index(inplace=True)
print(prices_df.head())

# Calcola la media e la volatilità annuale dei rendimenti percentuali
returns = prices_df.pct_change().mean() * 252
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = prices_df.pct_change().std() * sqrt(252)
returns.dropna(inplace=True)

# Formattazione dei dati come un array numpy per l'algoritmo K-Means
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
X = data
distorsions = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k, n_init=20)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.show()

# Calcolo del K-Means con K = 5 (5 clusters)
centroids,_ = kmeans(data,5)
# Assegnazione di ogni titolo a un cluster
idx,_ = vq(data,centroids)
# Grafici usando la logica di indicazzazione di numpy
plt.plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
plt.show()

details = [(name, cluster) for name, cluster in zip(returns.index, idx)]
for detail in details:
    print(detail)
