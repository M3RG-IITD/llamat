import os
import argparse
import random
import pandas as pd
import numpy as np
from pymatgen.core import Element
from pymatgen.core.structure import Structure

import matgl
from matgl.ext.ase import Relaxer


def find_similar_elements(target_element, elements, tolerance=0.1):
    similar_elements = []
    for state, radius in target_element.ionic_radii.items():
        for el in elements:
            if state in el.ionic_radii:
                radius_diff = abs(radius - el.ionic_radii[state])
                if radius_diff < tolerance and el.symbol != target_element.symbol:
                    similar_elements.append((el.symbol, state, radius_diff))
    return sorted(similar_elements, key=lambda x: x[2])

def make_swap_table(tolerance=0.1):
    elements = [Element(el) for el in Element]

    swap_table = {}

    for el in elements:
        swap_table[el.symbol] = [
            x[0] for x in find_similar_elements(el, elements, tolerance=tolerance)
        ]

    return swap_table