import pandas as pd
import random
import re

def remove_spaces_and_convert_to_upper(input_string, remove_first = 5):
    #input_string = input_string.upper()
    input_string = input_string.replace(" ", "")
    input_string = input_string.replace("\u200b", "")
    return input_string[remove_first:]

def remove_spaces(input_string):
    #input_string = input_string.upper()
    input_string = input_string.replace(" ", "")
    input_string = input_string.replace("\u200b", "")
    return input_string


def change_name( input_sequence, seed ):
    
    return input_sequence + f'_seed_{seed}'



def check_char(sequence):
    mapping = {'A': 0, 'a': 0, 'T': 3, 't': 3, 'C': 1, 'c':1, 'G': 2, 'g':2}
    flag = True
    for char in sequence:
        if char not in mapping:
             flag = False

    return flag

def check_if_string(sequence):
    if isinstance(sequence, str):
        return True
    else:
        return False


def check_structure(input_string):
    pattern = r'^.*?____(.*?)____.*?$'
    if re.match(pattern, input_string):
        return True
    else:
        return False
    
def check_empty_or_short(input_string, threshold = 20): 
    if len(input_string) >=threshold:
        return True
    else:
        return False
    
def find_substrings(input_string):
    return re.findall(r'____(.*?)____', input_string)[0]

def attach_stems(main_string):
    return 'GGGACGAC' + main_string + 'GTCGTCCC'