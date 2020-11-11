import numpy as np 
import argparse
from itertools import product, combinations
from functools import reduce
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.etree import ElementTree
from xml.dom import minidom
from tqdm import tqdm

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def make_char(x):
	return chr(x+65)

def tup_print(tups, joiner='', f=None):
	tup_strs = [joiner.join(list(map(make_char, tup))) for tup in tups]
	# print(' '.join(tup_strs),file=f)
	return tup_strs






# Create Combined Patient Adherence POMDP
# n = number of patients
# k = patients per day one can call
# T = n-length vector of transition matrices, which are each num_actions x num_states x num_states
# O = n-length vector of observation matrices, which are each num_actions x num_states x num_observations=3
# dicsount = discount in infinite horizon case
# method = {'exact', 'DESPOT'}






