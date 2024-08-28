
"""Predict nucleic acid secondary structure"""

import numpy as np
from typing import List, Tuple

import sys
import os

# Example path you want to add
new_path = r'./src/'
# Check if the path already exists to avoid duplication
if new_path not in sys.path:
    sys.path.append(new_path)
from Types import Comp, MultiBranch, BpEnergy, LoopEnergy, Energies
from dna import DNA_ENERGIES
from trash.substructures import substructure_ID



def _pair(s: str, i: int, i1: int, j: int, j1: int) -> str:
    """Return a stack representation, a key for the NN maps

    Args:
        s: Sequence being folded
        i: leftmost index
        i1: index to right of i
        j: rightmost index
        j1: index to left of j

    Returns:
        str: string representation of the pair
    """

    return (
        (s[i] if i >= 0 else ".")
        + (s[i1] if i1 >= 0 else ".")
        + "/"
        + (s[j] if j >= 0 else ".")
        + (s[j1] if j1 >= 0 else ".")
    )


def _j_s(query_len: int, known_len: int, d_g_x: float, temp: float) -> float:
    """Estimate the free energy of length query_len based on one of length known_len.

    The Jacobson-Stockmayer entry extrapolation formula is used
    for bulges, hairpins, etc that fall outside the 30nt upper limit
    for pre-calculated free-energies. See SantaLucia and Hicks (2004).

    Args:
        query_len: Length of element without known free energy value
        known_len: Length of element with known free energy value (d_g_x)
        d_g_x: The free energy of the element known_len
        temp: Temperature in Kelvin

    Returns:
        float: The free energy for a structure of length query_len
    """

    gas_constant = 1.9872e-3
    return d_g_x + 2.44 * gas_constant * temp * np.log(query_len / float(known_len))


def _d_g(d_h: float, d_s: float, temp: float) -> float:
    """Find the free energy given delta h, s and temp

    Args:
        d_h: The enthalpy increment in kcal / mol
        d_s: The entropy increment in cal / mol
        temp: The temperature in Kelvin

    Returns:
        The free energy increment in kcal / (mol x K)
    """

    return d_h - temp * (d_s / 1000.0)



def _hairpin(seq: str, i: int, j: int, temp: float, emap: Energies) -> float:
    """Calculate the free energy of a hairpin.

    Args:
        seq: The sequence we're folding
        i: The index of start of hairpin
        j: The index of end of hairpin
        temp: Temperature in Kelvin
        emap: Map of energies

    Returns:
        float: The free energy increment from the hairpin structure
    """

    #if np.abs(j - i) < 4:
      #  return np.inf

    hairpin = seq[i : j + 1]
    hairpin_len = len(hairpin) - 2
    pair = _pair(seq, i, i + 1, j, j - 1)

    if emap.COMPLEMENT[hairpin[0]] != hairpin[-1]:
        # not known terminal pair, nothing to close "hairpin"
        raise RuntimeError

    d_g = 0.0
    if emap.TRI_TETRA_LOOPS and hairpin in emap.TRI_TETRA_LOOPS:
        # it's a pre-known hairpin with known value
        d_h, d_s = emap.TRI_TETRA_LOOPS[hairpin]
        d_g = _d_g(d_h, d_s, temp)

    # add penalty based on size
    if hairpin_len in emap.HAIRPIN_LOOPS:
        d_h, d_s = emap.HAIRPIN_LOOPS[hairpin_len]
        d_g += _d_g(d_h, d_s, temp)
    else:
        # it's too large, extrapolate
        d_h, d_s = emap.HAIRPIN_LOOPS[30]
        d_g_inc = _d_g(d_h, d_s, temp)
        d_g += _j_s(hairpin_len, 30, d_g_inc, temp)

    # add penalty for a terminal mismatch
    if hairpin_len > 3 and pair in emap.TERMINAL_MM:
        if pair in emap.TERMINAL_MM:
            d_h, d_s = emap.TERMINAL_MM[pair]
            d_g += _d_g(d_h, d_s, temp)

    # add penalty if length 3 and AT closing, formula 8 from SantaLucia, 2004
    if hairpin_len == 3 and (hairpin[0] == "A" or hairpin[-1] == "A"):
        d_g += 0.5  # convert to entropy

    return d_g

def _stack(seq: str, i: int, i1: int, j: int, j1: int, temp: float, emap: Energies) -> float:
    """Get the free energy for a stack.

    Using the indexes i and j, check whether it's at the end of
    the sequence or internal. Then check whether it's a match
    or mismatch, and return.

    Two edge-cases are terminal mismatches and dangling ends.
    The energy of a dangling end is added to the energy of a pair
    where i XOR j is at the sequence's end.

    Args:
        seq: The full folding sequence
        i: The start index on left side of the pair/stack
        i1: The index to the right of i
        j: The end index on right side of the pair/stack
        j1: The index to the left of j
        temp: Temperature in Kelvin

    Returns:
        float: The free energy of the NN pairing
    """

    if any(x >= len(seq) for x in [i, i1, j, j1]):
        return 0.0

    pair = _pair(seq, i, i1, j, j1)
    if any(x == -1 for x in [i, i1, j, j1]):
        # it's a dangling end
        d_h, d_s = emap.DE[pair]
        return _d_g(d_h, d_s, temp)

    if i > 0 and j < len(seq) - 1:
        # it's internal
        d_h, d_s = emap.NN[pair] if pair in emap.NN else emap.INTERNAL_MM[pair]
        return _d_g(d_h, d_s, temp)

    if i == 0 and j == len(seq) - 1:
        # it's terminal
        d_h, d_s = emap.NN[pair] if pair in emap.NN else emap.TERMINAL_MM[pair]
        return _d_g(d_h, d_s, temp)

    if i > 0 and j == len(seq) - 1:
        # it's dangling on left
        d_h, d_s = emap.NN[pair] if pair in emap.NN else emap.TERMINAL_MM[pair]
        d_g = _d_g(d_h, d_s, temp)

        pair_de = seq[i - 1] + seq[i] + "/." + seq[j]
        if pair_de in emap.DE:
            d_h, d_s = emap.DE[pair_de]
            d_g += _d_g(d_h, d_s, temp)
        return d_g

    if i == 0 and j < len(seq) - 1:
        # it's dangling on right
        d_h, d_s = emap.NN[pair] if pair in emap.NN else emap.TERMINAL_MM[pair]
        d_g = _d_g(d_h, d_s, temp)

        pair_de = "." + seq[i] + "/" + seq[j + 1] + seq[j]
        if pair_de in emap.DE:
            d_h, d_s = emap.DE[pair_de]
            d_g += _d_g(d_h, d_s, temp)
        return d_g

    return 0


def _bulge( seq: str, i: int, i1: int, j: int, j1: int, temp: float, emap: Energies) -> float:
    """Calculate the free energy associated with a bulge.

        seq: The full folding DNA sequence
        i: The start index of the bulge
        i1: The index to the right of i
        j: The end index of the bulge
        j1: The index to the left of j
        temp: Temperature in Kelvin
        emap: Map to DNA/RNA energies

    Returns:
        float: The increment in free energy from the bulge
    """

    loop_len = max(i1 - i - 1, j - j1 - 1)
    if loop_len <= 0:
        raise RuntimeError

    # add penalty based on size
    if loop_len in emap.BULGE_LOOPS:
        d_h, d_s = emap.BULGE_LOOPS[loop_len]
        d_g = _d_g(d_h, d_s, temp)
    else:
        # it's too large for pre-calculated list, extrapolate
        d_h, d_s = emap.BULGE_LOOPS[30]
        d_g = _d_g(d_h, d_s, temp)
        d_g = _j_s(loop_len, 30, d_g, temp)

    if loop_len == 1:
        # if len 1, include the delta G of intervening NN (SantaLucia 2004)
        pair = _pair(seq, i, i1, j, j1)
        assert pair in emap.NN
        d_g += _stack(seq, i, i1, j, j1, temp, emap)

    # penalize AT terminal bonds
    if any(seq[k] == "A" for k in [i, i1, j, j1]):
        d_g += 0.5

    return d_g


def _internal_loop(seq: str, i: int, i1: int, j: int, j1: int, temp: float, emap: Energies) -> float:
    """Calculate the free energy of an internal loop.

    The first and last bp of both left and right sequences
    are not themselves parts of the loop, but are the terminal
    bp on either side of it. They are needed for when there's
    a single internal looping bp (where just the mismatching
    free energies are used)

    Note that both left and right sequences are in 5' to 3' direction

    This is adapted from the "Internal Loops" section of SantaLucia/Hicks, 2004

    Args:
        seq: The sequence we're folding
        i: The index of the start of structure on left side [(i,j), (i1,j1)]
        i1: The index to the right of i
        j: The index of the end of structure on right side
        j1: The index to the left of j
        temp: Temperature in Kelvin
        emap: Dictionary mapping to energies for DNA/RNA

    Returns:
        float: The free energy associated with the internal loop
    """

    loop_left = i1 - i - 1
    loop_right = j - j1 - 1
    loop_len = loop_left + loop_right

    if loop_left < 1 or loop_right < 1:
        raise RuntimeError

    # single bp mismatch, sum up the two single mismatch pairs
    if loop_left == 1 and loop_right == 1:
        mm_left = _stack(seq, i, i1, j, j1, temp, emap)
        mm_right = _stack(seq, i1 - 1, i1, j1 + 1, j1, temp, emap)
        
        if  emap.COMPLEMENT[seq[i+1]]  == seq[j-1]:#MODIFICATION to avoid inner loops  with one bp that matches
                return 1600
        else:
            return mm_left + mm_right

    # apply a penalty based on loop size
    if loop_len in emap.INTERNAL_LOOPS:
        d_h, d_s = emap.INTERNAL_LOOPS[loop_len]
        d_g = _d_g(d_h, d_s, temp)
    else:
        # it's too large an internal loop, extrapolate
        d_h, d_s = emap.INTERNAL_LOOPS[30]
        d_g = _d_g(d_h, d_s, temp)
        d_g = _j_s(loop_len, 30, d_g, temp)

    # apply an asymmetry penalty
    loop_asymmetry = abs(loop_left - loop_right)
    d_g += 0.3 * loop_asymmetry

    # apply penalty based on the mismatching pairs on either side of the loop 
    pair_left_mm = _pair(seq, i, i + 1, j, j - 1)

    d_h, d_s = emap.TERMINAL_MM[pair_left_mm]
    d_g += _d_g(d_h, d_s, temp)

    pair_right_mm = _pair(seq, i1 - 1, i1, j1 + 1, j1)
    if pair_right_mm not in  emap.TERMINAL_MM:
        return np.inf
    else:   
        d_h, d_s = emap.TERMINAL_MM[pair_right_mm]
        d_g += _d_g(d_h, d_s, temp)

    return d_g


def _multi_branch(seq: str, structure: str,  temp: float,branches: list, emap: Energies, helix: bool = False,
) -> float:
    """Calculate a multi-branch energy penalty using a linear formula.

    From Jaeger, Turner, and Zuker, 1989.
    Found to be better than logarithmic in Ward, et al. 2017

    Args:
        seq: The sequence being folded
        i: The left starting index
        j: The right ending index
        temp: Folding temp
        v_cache: Structs of energies where V(i,j) bond
        w_cache: Structs of min energy of substructures between W(i,j)
        helix: Whether this multibranch is enclosed by a helix
        emap: Map to DNA/RNA energies

    Keyword Args:
        helix: Whether V(i, j) bond with one another in a helix

    Returns:
        Struct: A multi-branch structure
    """
    i,j = branches[0]
    if i >0 and j < len(seq) -1 and structure[i-1] == '(' and structure[j+1] == ')':
         helix = True


    # count up unpaired bp and asymmetry
    unpaired = 0
    e_sum = 0.0
    for index, (i2, j2) in enumerate(branches):
        _, j1 = branches[(index - 1) % len(branches)]
        i3, j3 = branches[(index + 1) % len(branches)]

        # add energy from unpaired bp to the right
        # of the helix as though it was a dangling end
        # if there's only one bp, it goes to whichever
        # helix (this or the next) has the more favorable energy
        unpaired_left = 0
        unpaired_right = 0
        de = 0.0
        
        if index == len(branches) - 1 and not helix:
            pass
        elif (i3, j3) == (i, j):
            unpaired_left = i2 - j1 - 1
            unpaired_right = j3 - j2 - 1

            if unpaired_left and unpaired_right:
                de = _stack(seq, i2 - 1, i2, j2 + 1, j2, temp, emap)
            elif unpaired_right:
                de = _stack(seq, -1, i2, j2 + 1, j2, temp, emap)
                if unpaired_right == 1:
                    de = min(_stack(seq, i3, -1, j3, j3 - 1, temp, emap), de)
        elif (i2, j2) == (i, j):
            unpaired_left = j2 - j1 - 1
            unpaired_right = i3 - i2 - 1

            if unpaired_left and unpaired_right:
                de = _stack(seq, i2 - 1, i2, j2 + 1, j2, temp, emap)
            elif unpaired_right:
                de = _stack(seq, i2, i2 + 1, j2, -1, temp, emap)
                if unpaired_right == 1:
                    de = min(_stack(seq, i3 - 1, i3, -1, j3, temp, emap), de)
        else:
            unpaired_left = i2 - j1 - 1
            unpaired_right = i3 - j2 - 1

            if unpaired_left and unpaired_right:
                de = _stack(seq, i2 - 1, i2, j2 + 1, j2, temp, emap)
            elif unpaired_right:
                de = _stack(seq, -1, i2, j2 + 1, j2, temp, emap)
                if unpaired_right == 1:
                    de = min(_stack(seq, i2 - 1, i2, j2 + 1, j2, temp, emap), de)
        #print('it',index, unpaired_right)
        e_sum += de
        unpaired += unpaired_right
        assert unpaired_right >= 0
        '''
        if (i2, j2) != (i, j):  # add energy
            e_sum += _w(seq, i2, j2, temp, v_cache, w_cache, emap).e
            
        ''' 
    assert unpaired >= 0

    # penalty for unmatched bp and multi-branch
    a, b, c, d = emap.MULTIBRANCH
    e_multibranch = a + b * len(branches) + c * unpaired

    if unpaired == 0:
        e_multibranch = a + d

    # energy of min-energy neighbors
    e = e_multibranch + e_sum 

    return e


def compute_energy(string : str, seq: str, temp: float= 37.0, emap: Energies = DNA_ENERGIES) -> float:
 
    dct = substructure_ID(seq)
    temp = temp + 273.15
    energy = 0.0
    #structs = []
    #stacks
    sorted_stack = sorted(dct['stacks'], key=lambda x: x[0])
    for s in sorted_stack:
        energy += round(_stack(string, s[0], s[0]+1, s[1],s[1]-1, temp, emap),1)
        #structs.append((f'STACK {string[s[0]], string[s[0]+1], string[s[1]], string[s[1]-1]}', s, round(_stack(string, s[0], s[0]+1, s[1],s[1]-1, temp, emap),1)))
    #hairpins
    for s in dct['hairpins']:
        energy += round(_hairpin(string, s[0], s[1], temp, emap), 1)
        #structs.append((f'HAIRPIN {string[s[0]],  string[s[1]]}', s,round(_hairpin(string, s[0], s[1], temp, emap), 1)))
        
    #bulges
    for bulge in dct['left_bulges']:
        i_j, i1_j1 = bulge
        energy += round(_bulge( string , i_j[0], i1_j1[0],  i_j[1], i1_j1[1], temp, emap), 1)
        #structs.append(('L_BULGE', bulge, _bulge( string , i_j[0], i1_j1[0],  i_j[1], i1_j1[1], temp, emap), 1))
    for bulge in dct['right_bulges']:
        i_j, i1_j1 = bulge
        energy += round(_bulge( string , i_j[0], i1_j1[0],  i_j[1], i1_j1[1], temp, emap), 1)
        #structs.append(('R_BULGE', bulge,round( _bulge( string , i_j[0], i1_j1[0],  i_j[1], i1_j1[1], temp, emap), 1)))
    
    #internal loops
    for loop in dct['inner_loops']:
            i_j, i1_j1 = loop   
            energy += round(_internal_loop( string , i_j[0], i1_j1[0],  i_j[1], i1_j1[1], temp, emap), 1)
            #structs.append(('Loop', loop, round(_internal_loop( string , i_j[0], i1_j1[0],  i_j[1], i1_j1[1], temp, emap), 1)))
            
    #multibranches 
    for multi_brach in dct['multi-branches']:
        energy += round(_multi_branch(string,seq,  temp,multi_brach, emap,  False), 1)
        #structs.append(('multi-brach', multi_brach, round(_multi_branch(string, seq, temp,multi_brach, emap,  False), 1)))

    return energy#, structs