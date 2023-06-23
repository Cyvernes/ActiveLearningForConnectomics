import numpy as np
from tools import *


def singleStrat(current_strategy : int, input_points : list, input_labels : list) -> int:
    return(0)

def changeAtFirstMito(current_strategy : int, input_points : list, input_labels : list) -> int:
    return(int((current_strategy == 1) or (True in input_labels)))

def changeAfterAGivenAmountOfSeed(current_strategy : int, input_points : list, input_labels : list) -> int:
    if ((current_strategy == 1) or (len(input_points) >= 15 and (True in input_labels))):
        return(1)
    else:
        return(0)

