from venv.src.Lattice import Graph
from utilities import *

if __name__ == '__main__':

  spokenWords = get_spoken_words("transcriptions.txt")
  recogFileHeaders = read_ctm_headers("example.ctm")


  G1 = Graph("lattice.1.htk.gz", "G1", [recogFileHeaders[0], recogFileHeaders[1], recogFileHeaders[2]])
  G2 = Graph("lattice.2.htk.gz", "G2", [recogFileHeaders[0], recogFileHeaders[3], recogFileHeaders[4]])
  G3 = Graph("lattice.3.htk.gz", "G3", [recogFileHeaders[0], recogFileHeaders[5], recogFileHeaders[6]])
  G4 = Graph("lattice.4.htk.gz", "G4", [recogFileHeaders[0], recogFileHeaders[7], recogFileHeaders[8]])
  G5 = Graph("lattice.5.htk.gz", "G5", [recogFileHeaders[0], recogFileHeaders[9], recogFileHeaders[10]])
  graphs = [G1, G2, G3, G4, G5]

  print("Forward and Backward probabilities for:")
  for g in graphs:
    print(g.label, ": ", g.nodes[-1].score["f"], ",", g.nodes[0].score["b"])

  print("Recognized text for:")
  for g in graphs:
    print(g.label, ": ", g.text)

  # write the ctm file
  outputCtmData = []
  for g in graphs:
    for row in g.recognitionSegment:
      row = " ".join(map(str, row))
      outputCtmData.append(row)
  with open("recognition.ctm", "w") as outputCtmFile:
    for listitem in outputCtmData:
      outputCtmFile.write('%s\n' % listitem)

  for g in graphs:
    g.set_intersect_dic()
    g.best_path_conf()
    g.rescore()
    g.set_best_hypothese_and_recogFile()

  outputCtmData = []
  for g in graphs:
    for row in g.recognitionSegment:
      row = " ".join(map(str, row))
      outputCtmData.append(row)
  with open("rescored.ctm", "w") as outputCtmFile:
    for listitem in outputCtmData:
      outputCtmFile.write('%s\n' % listitem)