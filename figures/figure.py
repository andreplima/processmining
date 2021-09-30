"""
"""

import sys
import pandas as pd
from matplotlib  import pyplot as plt
from collections import defaultdict, namedtuple

ECO_FONT_SIZE = 15

# map from k (number of clusters) to colour
#k2colour = ['#F7FCB9', '#D9F0A3', '#FED976', '#FEB24C', '#FF9826', '#FC4E2A', '#E3211C', '#BD1A26', '#800F26']

Marker = namedtuple('Marker', ['type', 'size'])

k2colour = ['#FEB24C', '#FEB24C',
            '#FF9826', '#FF9826',
            '#FC4E2A', '#FC4E2A',
            '#E3211C', '#BD1A26',
            '#800F26']

# map from VSM to category
vsm2cat = {'af':         'Activity Frequency (AF)',
           'af-ilf':     'Activity Frequency (AF)',
           'bin':        'Binary VSM (BIN)',
           'emb(2,1)':   'Embeddings-based VSM (EMB)',
           'emb(2,2)':   'Embeddings-based VSM (EMB)',
           'emb(2,3)':   'Embeddings-based VSM (EMB)',
           'emb(m,1)':   'Embeddings-based VSM (EMB)',
           'emb(m,2)':   'Embeddings-based VSM (EMB)',
           'emb(m,3)':   'Embeddings-based VSM (EMB)',
           'emb(m/2,1)': 'Embeddings-based VSM (EMB)',
           'emb(m/2,2)': 'Embeddings-based VSM (EMB)',
           'emb(m/2,3)': 'Embeddings-based VSM (EMB)',
           'emb(m/4,1)': 'Embeddings-based VSM (EMB)',
           'emb(m/4,2)': 'Embeddings-based VSM (EMB)',
           'emb(m/4,3)': 'Embeddings-based VSM (EMB)',
           }

# map from category to marker
cat2marker = {'Activity Frequency (AF)':    Marker('3', 80),
              'Binary VSM (BIN)':           Marker('4', 80),
              'Embeddings-based VSM (EMB)': Marker('o', 30),
              }

def main(tablename):

  title = 'plot title'
  xlabel = 'Silhouette Index (log)'
  ylabel = 'Coefficient of Variation (%)'

  # reads data file with the table results
  df = pd.read_csv(tablename + '.tsv', sep='\t')
  df.info()
  # transform column VSM

  # plots the data
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  plt.title('(for the {0} log)'.format(tablename), fontsize=ECO_FONT_SIZE)
  plt.gca().patch.set_facecolor('0.95')
  plt.gca().invert_yaxis()
  ax.tick_params(axis='both', which='major', labelsize=12)
  ax.set_xlabel(xlabel, labelpad=ECO_FONT_SIZE, fontsize=ECO_FONT_SIZE)
  ax.set_ylabel(ylabel, labelpad=ECO_FONT_SIZE, fontsize=ECO_FONT_SIZE)

  xs = defaultdict(list)
  ys = defaultdict(list)
  cs = defaultdict(list)

  for idx, row in df.iterrows():
    (SID, VSM, k, CV, SI) = row
    category = vsm2cat[VSM]
    xs[category].append(SI)
    ys[category].append(CV)
    cs[category].append(k2colour[k-2])

  for category in xs:
    ax.scatter(xs[category],
               ys[category],
               color=cs[category],
               marker=cat2marker[category].type,
               s=cat2marker[category].size,
               label=category)

  ax.grid(True)
  ax.legend()

  plt.show()

if __name__ == "__main__":

  tablename = sys.argv[1]

  main(tablename)
