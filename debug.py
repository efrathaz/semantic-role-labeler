from Sentence import *
from Word import *

def print_domain(s, d):
  string = 'Domain for: ' + s.text + '\n'
  for w in d:
    string = string + w.to_string() + '\n'
  print(string)


def print_range(s, r):
  string = 'Range for: ' + s.text + '\n'
  for w in r:
    if w is None:
      string = string + 'None\n'
    else:
      string = string + w.to_string() + '\n'
  print(string)
  
  
def print_alignment(a, s):
  string = 'Alignment:\n'
  for pair in a:
    if pair[1] is None:
      string = string + '[ ' + pair[0].to_string() + ' , None ]\n'
    else:
      string = string + '[ ' + pair[0].to_string() + ' , ' + pair[1].to_string() + ' ]\n'
  string = string + 'Score = ' + str(s)
  print(string)