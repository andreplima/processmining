# using SRESPIM environment

"""
  represent - obtain numerical representation of activities from a log
  python represent.py <sample file>
    <sample file> is a filename where the sample is stored. This is a pickle file.

  Example:
  python represent.py sample_E1
"""

import sys
import numpy as np

from   copy          import deepcopy
from   collections   import defaultdict
from   sharedDefs    import tsprint, saveAsText, serialise, deserialise

START = '*'
STOP  = '+'

def countEdges(sample):

  M = defaultdict(lambda: defaultdict(int))
  W = defaultdict(lambda: defaultdict(int))
  for trace in sample:
    ub = len(trace) - 1
    for i in range(ub):
      (left, right) = trace[i:i+2]
      M[left][right] += 1 # M is a sparse matrix storing edge-counts
                          # (e.g., M['A']['B'] is the number of occurrences 'AB' in the log)
      W[right][left] += 1 # W is M-transposed

  rows = sorted(M.keys())
  cols = sorted(W.keys())

  return M, W, rows, cols

def normalise(M, rows, cols):
  M_ = deepcopy(M)

  # each row in the matrix is normalised to unit sum (i.e., its elements add up to 1)
  for row in rows:
   acc = sum(M[row].values())
   for col in M[row].keys():  # since M is sparse, only the relevant columns are visited
     M_[row][col] = M[row][col] / acc

  return M_

def represent(M_, W_, rows, cols):

  V = defaultdict(list) # V is a dense matrix in which each row stores
                        # the representation of an activity (easily converted to numpy.array)

  rows_ = rows + [STOP]
  cols_ = cols + [START]

  # determines the first (N+1) cols, with N as the number of distinct activities in the log
  # (plus 1 due to STOP)
  for row in rows_:
    for col in cols:
      V[row].append(M_[row][col])

  # determines the last (N+2) cols
  for col in cols_:
    for row in rows:
      V[col].append(W_[col][row])

  return V

def representTraces(sample_, M_, V_, rows, cols):

  # inserts the activities in the sample, for improved inspection of results
  sample = (['*{0}+'.format(a) for a in rows] +
            ['*{0}+'.format(a) for a in cols] +
            sample_ +
            ['*{0}+'.format(a) for a in sample_]
           )

  Y = {}
  for trace_ in sample:
    trace = trace_[1:-1]

    if trace not in Y:

      # initialises working variables
      k    = len(V_[trace[0]]) // 2 # length of the "half-vectors" _v (precedes v) and v_ (succeeds v)
      peta = 1.0                    # the accumulated probability of the left-vector eta
      last = 'eta'                  # the name of the first left-vector, represented by eta
                                    # (this is a hypothetical activity; not supposed to exist in the alphabet)

      # the initial left-vector, as [ <k ones> | <last k values of the first activity> ]
      eta  = np.hstack((peta * np.ones(k), V_[trace[0]][k:]))

      # obtains the matrix P with probabilities of transitions by complementing the M_ matrix
      P = deepcopy(M_)
      P[last][trace[0]] = 1.0

      # iterates from left to right to obtain composition of representations
      # (e.g., ABCD is obtained by ( ( ( (eta o A) o B) o C) o D)
      for i in range(len(trace)):
        p     = P[last][trace[i]]
        peta  = peta * p
        left  = np.hstack((peta * np.ones(k), eta[k:]))
        right = np.hstack((V_[trace[i]][0:k], p * np.ones(k)))
        eta   = left @ np.diag(right)
        last  = trace[i]

      Y[trace] = eta

  return Y

def distanceMatrix(V):
  V_ = {}
  for row in V:
    V_[row] = np.array(V[row])

  D = defaultdict(lambda: defaultdict(int))
  rows = sorted(V.keys())
  ub = len(rows)
  for i in range(ub - 1):
    for j in range(i, ub):
      val = np.linalg.norm(V_[rows[i]] - V_[rows[j]])
      D[rows[i]][rows[j]] = val
      D[rows[j]][rows[i]] = val

  return V_, D

def saveMatrixAsText(M, rows, cols, filename, orientation = 'left'):

  header  = [orientation] + cols
  content = ['\t'.join(header)]
  for row in rows:
    buffer = [row]
    for col in cols:
      buffer.append(str(M[row][col]))
    content.append('\t'.join(buffer))

  saveAsText('\n'.join(content), filename)

def saveRepAsText(V, rows, cols, filename):

  rows_ = rows + [STOP]
  header  = ['both'] + ['_{0}'.format(col) for col in cols] + ['{0}_'.format(row) for row in rows]
  content = ['\t'.join(header)]
  for row in rows_:
    buffer = [row] + [str(v) for v in V[row]]
    content.append('\t'.join(buffer))

  saveAsText('\n'.join(content), filename)

def saveTraceRepAsText(Y, rows, cols, filename):

  rows_ = rows + [STOP]
  header  = ['both'] + ['_{0}'.format(col) for col in cols] + ['{0}_'.format(row) for row in rows]
  content = ['\t'.join(header)]
  for trace in Y:
    buffer = [trace] + [str(v) for v in Y[trace]]
    content.append('\t'.join(buffer))

  saveAsText('\n'.join(content), filename)

def main(filename):

  tsprint('Recovering log from file {0}.pkl'.format(filename))
  sample = deserialise(filename)

  # step 1: creates the counting matrices
  M, W, rows, cols = countEdges(sample)

  # step 2: creates the column-normalised matrix
  M_ = normalise(M, rows, cols)

  # step 3: creates the row-normalised matrix
  W_ = normalise(W, cols, rows)

  # step 4: creates representations for each activity (by collating M_ and W_ images)
  # -- including START and STOP activities
  V = represent(M_, W_, rows, cols)

  # step 5: computes a distance matrix for learned representations
  V_, D = distanceMatrix(V)


  # after Apr 2nd meeting, we wanted to explore representations for traces

  # first try : traces represented in the same space as activities, using average as composition
  # -- issue .: in example 2, *ABCD+ gets the same representation as *ACBD+
  #            also, the trace representation do not keep the meaning of each dimension
  #    cause .: average is based on sum, which, being a symmetric operation, destroys order
  #    soln ..: maybe some asymmetric operation will be sensitive to the order of activities?

  # second try: traces represented in the same space as activities, using matrix multiplication
  #             meaning is expected to remain valid for both activity and trace representations

  Y = representTraces(sample, M_, V_, rows, cols)

  tsprint('Saving results.')

  serialise(dict(M),  'M')
  serialise(dict(W),  'W')
  serialise(dict(M_), 'M_')
  serialise(dict(W_), 'W_')
  serialise(V,  'V')
  serialise(V_, 'V_')
  serialise(dict(D),  'D')
  serialise(Y, 'Y')

  saveMatrixAsText(M,  rows, cols, 'M.csv')
  saveMatrixAsText(W,  cols, rows, 'W.csv',  'right')
  saveMatrixAsText(M_, rows, cols, 'M_.csv')
  saveMatrixAsText(W_, cols, rows, 'W_.csv', 'right')

  saveRepAsText(V, rows, cols, 'V.csv')
  saveTraceRepAsText(Y, rows, cols, 'Y.csv')

  nodes = rows + [STOP]
  saveMatrixAsText(D, nodes, nodes, 'D.csv', 'dist')

  tsprint('Done.')

if __name__ == "__main__":

  filename = sys.argv[1]

  main(filename)
