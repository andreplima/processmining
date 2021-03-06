# using SRESPIM environment

"""
  generate - abstract state machines to produce sample data (from a ground truth model)
  python generate.py <template process> <sample size>
    <template process> are defined in the template_hub
    <sample size> is an integer larger than 0

  Example:
  python generate.py E1 10
"""

import sys
from random     import seed, random
from sharedDefs import tsprint, stimestamp, stimediff, saveAsText, serialise

START = '*'
STOP  = '+'

ECO_SEED = 23

def template_hub(template):

  if(template   == 'E1'):
    res = modelE1

  elif(template == 'E2'):
    res = modelE2

  elif(template == 'E3'):
    res = modelE3

  elif(template == 'X1'):
    res = modelX1

  else:
    raise ValueError('No template process named {0}'.format(template))

  return res

def modelE1(ss, alpha = 0.5):
  # state machine from example 1

  sample = []
  for _ in range(ss):

    state = START
    trace = [state]

    while state != STOP:

      if(state == START):
        state = 'A'

      elif(state == 'A'):
        if(random() < alpha):
          state = 'B'
        else:
          state = 'C'

      elif(state in ['B', 'C']):
        state = 'D'

      elif(state == 'D'):
        state = STOP

      else:
        raise ValueError('Error during model sampling')

      trace.append(state)

    sample.append(''.join(trace))

  return sample

def modelE2(ss, alpha = 0.5):
  # state machine from example 1

  sample = []
  for _ in range(ss):

    state = START
    trace = [state]

    while state != STOP:

      if(state == START):
        state = 'A'

      elif(state == 'A'):
        if(random() < alpha):
          state = 'BC'
        else:
          state = 'CB'

      elif(state in ['BC', 'CB']):
        state = 'D'

      elif(state == 'D'):
        state = STOP

      else:
        raise ValueError('Error during model sampling')

      trace.append(state)

    sample.append(''.join(trace))

  return sample

def modelE3(ss, alpha = 0.2):
  # state machine from example 1

  sample = []
  for _ in range(ss):

    state = START
    trace = [state]

    while state != STOP:

      if(state == START):
        state = 'A'

      elif(state == 'A'):
        if(random() < alpha):
          state = 'BC'
        else:
          state = 'D'

      elif(state == 'BC'):
        state = 'A'

      elif(state == 'D'):
        state = STOP

      else:
        raise ValueError('Error during model sampling')

      trace.append(state)

    sample.append(''.join(trace))

  return sample

def modelX1(ss, alpha = 0.5):
  # state machine from example 1

  sample = []
  for _ in range(ss):

    state = START
    trace = [state]

    while state != STOP:

      if(state == START):
        state = 'A'

      elif(state == 'A'):
        if(random() < alpha):
          state = 'A'
        else:
          state = STOP

      else:
        raise ValueError('Error during model sampling')

      trace.append(state)

    sample.append(''.join(trace))

  return sample

def main(template, ss):

  # initialises the random number generator
  seed(ECO_SEED)

  # generates a sample of the underlying model (i.e., a finite state machine)
  tsprint('Generating {0} samples from template {1}.'.format(ss, template))
  generator = template_hub(template)
  sample    = generator(ss)

  # saves the results
  tsprint('Saving results.')
  saveAsText('\n'.join(sample), 'sample.csv')
  serialise(sample, 'sample')

  tsprint('Done.')


if __name__ == "__main__":

  template    = sys.argv[1]
  sample_size = int(sys.argv[2])

  main(template, sample_size)
