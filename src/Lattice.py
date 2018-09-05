from utilities import *


class Graph():
  def __init__(self, filename, label, headerArgs):
    # Attributes
    self.label = label
    self.data = []
    self.nodes = []
    self.edges = []
    self.timeStep = 0.01
    self.lmScale = None
    self.prior = None
    self.recogFileHeaders = {"attributesLine": headerArgs[0], "segmentId": headerArgs[1], "firstRow": headerArgs[2]}
    self.text = ""
    self.read_file(filename)
    self.construct_graph()
    self.propagate_graph()
    self.set_best_hypothese_and_recogFile()

  class Node(object):
    def __init__(self, label, time):
      self.label = label
      self.time = time
      self.outs = []  # edge labels
      self.ins = []  # edge labels
      self.score = {"f": 0.0, "b": 0.0}
      self.backPointer = None

  class Edge(object):
    def __init__(self, label, start, end, word, amScore, lmScore, lmScale):
      self.label = label
      self.start = start
      self.end = end
      self.word = word
      self.amScore = amScore
      self.lmScore = lmScore
      self.weight = amScore + lmScore * lmScale  # w(e)
      self.posterior = None  # p(e|X)
      self.confidence = None  # conf(e)

  def read_file(self, filename):
    file = gzip.open(filename)
    for element in file:
      stringElement = element.decode("utf-8")
      self.data.append(stringElement)
      if stringElement.startswith('lmscale='):
        self.lmScale = float(stringElement[8:])
      if stringElement.startswith('NODES='):
        self.nodeNum = int(stringElement[6:])
      if stringElement.startswith('LINKS='):
        self.arcNum = int(stringElement[6:])
      if stringElement.startswith('I'):
        n = infoExtractor.findall((stringElement))
        node = self.Node(int(n[0][2]), float(n[1][2]))
        self.nodes.append(node)
      if stringElement.startswith('J'):
        n = infoExtractor.findall((stringElement))
        edge = self.Edge(int(n[0][2]), int(n[1][2]), int(n[2][2]), str(n[3][1]), -float(n[5][2]), -float(n[6][2]),
                         self.lmScale)
        self.edges.append(edge)

  def construct_graph(self):
    for e in self.edges:
      self.nodes[e.start].outs.append(e.label)
      self.nodes[e.end].ins.append(e.label)

  def get_node_list_topological(self, reverse):
    nodeListCopy = self.nodes.copy()
    newList = sorted(nodeListCopy, key=lambda x: x.time, reverse=reverse)
    return newList

  def forward(self):
    self.nodes[0].score["f"] = 0.0
    newList = self.get_node_list_topological(False)
    for node in newList:
      score = zeroValue
      if len(node.ins) > 0:
        for arcInLabel in node.ins:
          arcIn = self.edges[arcInLabel]
          prevNod = self.nodes[arcIn.start]
          localCost = prevNod.score["f"] + arcIn.weight
          score = log_addition(score, localCost)
        node.score["f"] = score

  def backward(self):
    self.nodes[-1].score["b"] = 0.0
    newList = self.get_node_list_topological(True)
    for node in newList:
      score = zeroValue
      if len(node.outs) > 0:
        for arcOutLabel in node.outs:
          arcOut = self.edges[arcOutLabel]
          nextNode = self.nodes[arcOut.end]
          localCost = nextNode.score["b"] + arcOut.weight
          score = log_addition(score, localCost)
        node.score["b"] = score

  def set_prior(self):
    self.prior = self.nodes[-1].score["f"]

  def set_arc_posteriors(self):
    for e in self.edges:
      numerator = self.nodes[e.start].score["f"] + e.weight + self.nodes[e.end].score["b"]
      e.posterior = numerator - self.prior

  def propagate_graph(self):
    self.forward()
    self.backward()
    self.set_prior()
    self.set_arc_posteriors()

  def set_best_hypothese_and_recogFile(self):
    self.text = ""
    recogFileSegment = [self.recogFileHeaders["attributesLine"], self.recogFileHeaders["segmentId"]]
    bestWordSequence = []
    timePointer = float(self.recogFileHeaders["segmentId"][-1][1:6])
    actualNode = self.nodes[0]

    while (actualNode != self.nodes[-1]):
      # initializations
      bestScore = zeroValue
      bestArc = None
      for edgeLabel in actualNode.outs:
        arc = self.edges[edgeLabel]
        if (arc.posterior < bestScore):
          bestScore = arc.posterior
          bestArc = edgeLabel
      word = self.edges[bestArc].word
      recogLine = self.recogFileHeaders["firstRow"].copy()
      recogLine[-1] = word
      # chosse to add noise and silence only in the recognition segment list(?!)
      # recogFileSegment.append(recogLine)
      if (word != "[NOISE]" and word != "[SILENCE]" and word != "!NULL"):
        recogFileSegment.append(recogLine)
        self.text += word
        self.text += " "
        bestWordSequence.append(self.edges[bestArc])
      actualNode = self.nodes[self.edges[bestArc].end]
      actualNode.backPointer = bestArc

    self.recognitionSegment = recogFileSegment
    self.bestWordSequence = bestWordSequence

  def get_arc_interval(self, arcLabel):
    return [self.nodes[self.edges[arcLabel].start].time, self.nodes[self.edges[arcLabel].end].time]

  def get_intersected_arcs(self, time):
    return [e for e in self.edges if is_intersected(self.get_arc_interval(e.label), time)]

  def get_interval_time_steps(self, nodeStart, nodeEnd):
    if nodeStart == nodeEnd:
      return [nodeStart]
    interval = np.arange(nodeStart, nodeEnd, 0.01)
    return [float(floatRounder(x)) for x in interval]

  def set_intersect_dic(self):

    totalTimeSteps = self.get_interval_time_steps(self.nodes[0].time, self.nodes[-1].time)
    self.frameWiseDic = dict.fromkeys(totalTimeSteps)

    for time in totalTimeSteps:
      intersectedArcs = self.get_intersected_arcs(time)
      wordSet = list(set([e.word for e in intersectedArcs]))
      wordDic = dict.fromkeys(wordSet)
      self.frameWiseDic[time] = wordDic

      for w in wordSet:
        p = n_log_addition([e.posterior for e in list(filter(lambda x: x.word == w, intersectedArcs))])
        self.frameWiseDic[time][w] = p

  def best_path_conf(self):
    for arc in self.bestWordSequence:
      conf = zeroValue
      start = self.nodes[arc.start].time
      end = self.nodes[arc.end].time
      duration = end - start
      timeStepsInterval = self.get_interval_time_steps(start, end)
      for time in timeStepsInterval:
        conf = log_addition(conf, self.frameWiseDic[time][arc.word])
      arc.confidence = conf - duration

  def rescore(self):
    for arc in self.edges:
      conf = zeroValue
      start = self.nodes[arc.start].time
      end = self.nodes[arc.end].time
      timeStepsInterval = self.get_interval_time_steps(start, end)
      for time in timeStepsInterval:
        conf = log_addition(conf, self.frameWiseDic[time][arc.word])
      arc.posterior = conf
