import gzip
import re
import numpy as np
import math
from functools import reduce

zeroValue =  float("inf")
infoExtractor = re.compile(r'(\S+)=(?:"((?:[^\\"]+|\\.)*)"|(\S+))')
floatRounder = lambda x: "%.2f" % x


def get_spoken_words(textFile):
  data = []
  lines = []
  with open(textFile) as f:
    lines = [line.split() for line in f]
  del lines[0:5]
  for element in lines:
    del element[0:6]
  for line in lines:
    for word in line:
      data.append(word)
  return data

def read_ctm_headers(ctmFile):
  header = []
  indices = []
  with open(ctmFile) as f:
    lines = [line.split() for line in f]
  header.append(lines[0])
  for line in lines:
    if (line[0] == ";;" and lines.index(line) != 0):
      i = lines.index(line)
      indices.append(i)
      indices.append(i + 1)
      lines[i + 1][-1] = None
  for index in indices:
    header.append(lines[index])

  return header


def log_addition(a, b):
  if a < b:
    return log_addition(b, a)
  if b == zeroValue:
    return a
  else:
    x = math.exp(b - a)
    if x < 0.01:
      return a + x
    else:
      return a + np.log1p(x)

def n_log_addition(values):
  sum = zeroValue
  for i in range(len(values)):
    sum = log_addition(sum, values[i])
  return sum

def get_intersection_extrems(a, b):
  # a and b are two lists of length 2 having extrems
  return max(a[0], b[0]), min(a[1], b[1])

def is_intersected(interval, timePoint):
  return (timePoint >= interval[0] and timePoint <= interval[1])


