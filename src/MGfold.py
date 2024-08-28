"""Python implementation of MGfold to predict nucleic acid secondary structure."""


import sys
import math
from typing import List, Tuple
import numpy as np
from dna import DNA_ENERGIES
from Types import Energies, Cache
from graph_matching import  Aptamer_match
from mg_energy_functions import _bulge, _hairpin, mg_multi_branch, all_combinations, open_ending_branch, _stack, _pair, _internal_loop, Struct




STRUCT_DEFAULT = Struct(-math.inf)
STRUCT_NULL = Struct(math.inf)


Structs = List[List[Struct]]

def mg_s_matrix(seq: str, structs: List[Struct], size = None):
    """Compute structural matrix of the DNA sequence.
    Args:
        seq: The sequence to fold
        structs: The list of structures defining the secondary structure of the input sequence
        size: Size of the structural matrix. To be used in case padding with zeros is required. Default is len(seq)
    Returns:
        M: numpy array with shape size X seize defining the structural matrix of the input sequence
    """

    if size:
        n = size
    else:
        n = len(seq)
        
    M = np.zeros((n,n), dtype = int)

    if structs[0].desc[:24] == "OPEN_ENDING_MULTI_BRANCH":
        structs = structs[1:]
        
    for s in structs: 
            if len(s.ij) == 1:
                i, j = s.ij[0]
                M[i,j] = 1
                M[j,i] = 1
    return M

def mgfold(seq: str, temp: float = 37.0, l_fix = 0, n_branches = 4) -> List[Struct]:
    """Fold the DNA sequence and return the lowest free energy score.
    Args:
        seq: The sequence to fold.
        temp: The temperature the fold takes place in, in Celcius.
        l_fix: length of initial stem. Forces the first l_fix nucleotides to base pair with the last l_fix nucleotides.
        n_branches: amount of branches to consider when computing multi-branch loops
    Returns:
        List[Struct]: A list of structures. Stacks, bulges, hairpins, etc.
    """
    
    # Solve graph matching problem
    APT = Aptamer_match()
    APT.fit_fold( sequence=seq ,  n_tmpl=4, l_fix= l_fix )
    S = APT.dict_Sij
    D = APT.dict_d
    bps = APT.bps

    # Fill e_cache
    e_cache,   min_struct, min_ene= _cache(seq, temp, S, D, l_fix=l_fix, n_branches= n_branches)
    
    n = len(seq)
    emap = DNA_ENERGIES 
    
    branches = []   
    # Consider configurations ending with an open loop    
    if (e_cache[0][n-1].e > min_ene or e_cache[0][n-1]== STRUCT_DEFAULT) and l_fix == 0:
        
        # Exclude branches with positive energy value
        for bp in list(set(bps)):
                if e_cache[bp[0]][bp[1]].e < 0:
                    branches.append(bp)
                    
        # Compute all possible compatible branches configurations consisting up to n_branches
        combos = all_combinations(branches, r2 =n_branches)  
        for combo in combos:                         
                        e3_test =  open_ending_branch( seq,0,n-1,temp, e_cache, emap, S, combo)
                        if e3_test and e3_test.e < e_cache[min_struct[0]][min_struct[1]].e:
                                e_cache[0][n-1] = e3_test
                                min_struct = (0,n-1)           

    return mg_traceback(min_struct[0], min_struct[1], e_cache, l_fix)




def mg_dot_bracket(seq: str, structs: List[Struct]) -> str:
    """Get the dot bracket notation for a secondary structure.
    Args:
        structs: The list of structures defining the secondary structure of the input sequence
    Returns:
        str: The dot bracket notation of the secondary structure
    """

    result = ["."] * len(seq)
    #If ending with open loop do not consider the first structure
    if structs[0].desc[:24] == "OPEN_ENDING_MULTI_BRANCH":
        structs = structs[1:]
        
    for s in structs:
            if len(s.ij) == 1:
                i, j = s.ij[0]
                result[i] = "("
                result[j] = ")"
    return "".join(result)


def _cache(seq: str, temp: float = 37.0, S= None, D = None, l_fix = 0, n_branches = None) :
    """Create and fill the e_cache
    Args:
        seq: The sequence to fold
        temp: The temperature to fold at
        S: dictionary containing, for each base pair (i,j) all possible base pairs (i',j') such that i<i'<j'<j.
        D: dictionary containing for each possible length d all base pairs with length d identified solving the subgraph matching problem
        l_fix: length of initial stem. Forces the first l_fix nucleotides to base pair with the last l_fix nucleotides.
        n_branches: amount of branches to consider when computing multi-branch loops

    Returns:
        Structs: filled  e_cache 
    """

    seq = seq.upper()
    temp = temp + 273.15  # kelvin

    # Set DNA energies. By changing these energies the following code can ptentially be used for RNA sequences
    emap = DNA_ENERGIES 
    n = len(seq)
    e_cache: Structs = []

    for _ in range(n):
        e_cache.append([STRUCT_DEFAULT] * n)
    
    # fill the cache
    non_maximal = set([])
    min_ene = np.inf
    min_struct = None
    
    # if not stem
    if l_fix ==0:
        for d in sorted(list(D.keys()), key= int):
            for bp in set(D[d]):
                e, non_maximal =  _e(seq, bp[0], bp[1], temp, e_cache, emap, S, non_maximal, n_branches)
                if e.e  < min_ene:
                    min_struct = bp
                    min_ene = e.e
    #if stem
    if l_fix >0:
        fixed_bp = [(k,n-1-k) for k in range(l_fix-1)]
        for d in sorted(list(D.keys()), key= int):
            for bp in set(D[d]):
                if bp in fixed_bp:
                    i = bp[0]
                    j = bp[1]
                    i1 = i+1
                    j1 = j-1
                    # i1 and j1 must match
                    if emap.COMPLEMENT[seq[i1]] != seq[j1]:
                        continue
                    e2 = Struct(math.inf)
                    pair = _pair(seq, i, i1, j, j1)
                    e2_test, e2_test_type = math.inf, ""
                    # it's a neighboring/stacking pair in a helix
                    e2_test = _stack(seq, i, i1, j, j1, temp, emap)
                    e2_test_type = f"STACK:{pair}"
                    if i > 0 and j == n - 1 or i == 0 and j < n - 1:
                        # there's a dangling end
                        e2_test_type = f"STACK_DE:{pair}"
                    e2_test += e_cache[i1][j1].e
                    if e2_test != -math.inf and e2_test < e2.e:
                        e2= Struct(e2_test, e2_test_type, [(i1, j1)])
                    e_cache[i][j] = e2
                else:
                    e, non_maximal =  _e(seq, bp[0], bp[1], temp, e_cache, emap, S, non_maximal, n_branches)
                if e.e  < min_ene:
                    min_struct = bp
                    min_ene = e.e

    return e_cache,  min_struct, min_ene



def _e(
    seq: str,
    i: int,
    j: int,
    temp: float,
    e_cache: Structs,
    emap: Energies,
    S,
    non_maximal = set,
    n_branches = None
):
    """Find, store and return the minimum free energy of the structure between i and j

    If i and j don't bp, store and return INF.
    See: Figure 2B of Zuker, 1981

    Args:
        seq: The sequence being folded
        i: The start index
        j: The end index (inclusive)
        temp: The temperature in Kelvin
        e_cache: Free energy cache for if i and j bp. INF otherwise
        
        emap: Energy map for DNA

    Returns:
        float: The minimum energy folding structure possible between i and j on seq
    """
    
    if e_cache[i][j] != STRUCT_DEFAULT:
        return e_cache[i][j]
    
    # the ends must basepair for V(i,j)
    if emap.COMPLEMENT[seq[i]] != seq[j]:
        e_cache[i][j] = STRUCT_NULL
        return e_cache[i][j]
    
    # if the basepair is isolated, and the seq large, penalize at 1,600 kcal/mol
    # heuristic for speeding this up
    # from https://www.ncbi.nlm.nih.gov/pubmed/10329189
    isolated_outer = True
    if i and j < len(seq) - 1:
        isolated_outer = emap.COMPLEMENT[seq[i - 1]] != seq[j + 1]
    isolated_inner = emap.COMPLEMENT[seq[i + 1]] != seq[j - 1]

    if isolated_outer and isolated_inner:
        e_cache[i][j] = Struct(1600)
        return e_cache[i][j]
    
    # E1 = FH(i, j); hairpin
    pair = _pair(seq, i, i + 1, j, j - 1)
    e1 = Struct(_hairpin(seq, i, j, temp, emap), "HAIRPIN:" + pair)
    if j - i == 4:  # small hairpin; 4bp
        e_cache[i][j] = e1
        return e_cache[i][j], non_maximal

    # E2 = min{FL(i, j, i', j') + V(i', j')}, i<i'<j'<j
    # stacking region or bulge or interior loop; Figure 2A(2)
    # j-i=d>4; various pairs i',j' for j'-i'<d
    n = len(seq)
    e2 = Struct(math.inf)
    key = '({}, {})'.format(*(i,j))
    for bp in S[key]:
            i1 = bp[0]
            j1 = bp[1]
            # i1 and j1 must match
            if emap.COMPLEMENT[seq[i1]] != seq[j1]:
                continue

            pair = _pair(seq, i, i1, j, j1)
            pair_left = _pair(seq, i, i + 1, j, j - 1)
            pair_right = _pair(seq, i1 - 1, i1, j1 + 1, j1)
            pair_inner = pair_left in emap.NN or pair_right in emap.NN

            stack = i1 == i + 1 and j1 == j - 1
            bulge_left = i1 > i + 1
            bulge_right = j1 < j - 1

            e2_test, e2_test_type = math.inf, ""
            if stack:
                # it's a neighboring/stacking pair in a helix
                e2_test = _stack(seq, i, i1, j, j1, temp, emap)
                e2_test_type = f"STACK:{pair}"

                if i > 0 and j == n - 1 or i == 0 and j < n - 1:
                    # there's a dangling end
                    e2_test_type = f"STACK_DE:{pair}"
            elif bulge_left and bulge_right and not pair_inner:
                # it's an interior loop
                e2_test = _internal_loop(seq, i, i1, j, j1, temp, emap)
                e2_test_type = f"INTERIOR_LOOP:{str(i1 - i)}/{str(j - j1)}"

                if i1 - i == 2 and j - j1 == 2:
                    loop_left = seq[i : i1 + 1]
                    loop_right = seq[j1 : j + 1]
                    # technically an interior loop of 1. really 1bp mismatch
                    e2_test_type = f"STACK:{loop_left}/{loop_right[::-1]}"
            elif bulge_left and not bulge_right:
                # it's a bulge on the left side
                e2_test = _bulge(seq, i, i1, j, j1, temp, emap)
                e2_test_type = f"BULGE:{str(i1 - i)}"
            elif not bulge_left and bulge_right:
                # it's a bulge on the right side
                e2_test = _bulge(seq, i, i1, j, j1, temp, emap)
                e2_test_type = f"BULGE:{str(j - j1)}"
            else:
                # it's basically a hairpin, only outside bp match
                continue

            # add V(i', j')
            e2_test += e_cache[i1][j1].e
            if e2_test != -math.inf and e2_test < e2.e:
                e2 = Struct(e2_test, e2_test_type, [(i1, j1)])
            

    # E3 = min{W(i+1,i') + W(i'+1,j-1)}, i+1<i'<j-2
    
    e3 = STRUCT_NULL
    
    if not isolated_outer or not i or j == len(seq) - 1:
        branches = list( set(S[key])-non_maximal)
            
        if len(branches)>=2:
            combos = all_combinations(branches, r1=2, r2=n_branches )
            
            for combo in combos: 
                    e3_test = mg_multi_branch(seq, i, j, temp, e_cache, emap, S, list(combo), n_branches)
                    
                    if e3_test and e3_test.e < e3.e:
                            e3 = e3_test            
    e = _min_struct(e1, e2, e3) 
    
    if e.e >0 or e == e3 or e ==e1:
        non_maximal.add((i,j))
    
    e_cache[i][j] = e

    return e, non_maximal





def _min_struct(*structs: Struct) -> Struct:
    """Return the struct with the lowest free energy that isn't -inf (undef)

    Args:
        structs: Structures being compared

    Returns:
        struct: The min free energy structure
    """

    s: Struct = STRUCT_NULL
    for struct in structs:
        if struct.e != -math.inf and struct.e < s.e:
            s = struct
    return s

def mg_traceback(i: int, j: int, e_cache: Structs, l_fix) -> List[Struct]:
    """Traceback thru the V(i,j) and W(i,j) caches to find the structure

    For each step, get to the lowest energy W(i,j) within that block
    Store the structure in W(i,j)
    Inc i and j
    If the next structure is viable according to V(i,j), store as well
    Repeat

    Args:
        i: The leftmost index to start searching in
        j: The rightmost index to start searching in
        e_cache: Energies where i and j bond
        w_cache: Energies/sub-structures between or with i and j

    Returns:
        A list of Structs in the final secondary structure
    """

    # move i,j down-left to start coordinates
    structs: List[Struct] = []
    if l_fix > 1:
  
        for k in range(1,l_fix):
            
            s = e_cache[k-1][len(e_cache)-k]
            s.ij[0] = k, len(e_cache)-1-k
  
        i = 0 
        j = len(e_cache)-1
            
    elif l_fix == 1:
        i = 0
        j = len(e_cache)-1
       
    #structs = _trackback_energy(structs)
    while True:
        s = e_cache[i][j]

        structs.append(s.with_ij([(i, j)]))

        # it's a hairpin, end of structure
        if not s.ij:
            # set the energy of everything relative to the hairpin
            return _trackback_energy(structs)

        # it's a stack, bulge, etc
        # there's another single structure beyond this
        if len(s.ij) == 1:
            i, j = s.ij[0]
            continue
        
        # it's a multibranch
        e_sum = 0.0
        structs = _trackback_energy(structs)
        branches: List[Struct] = []
        for i1, j1 in s.ij:
            tb = mg_traceback(i1, j1, e_cache, 0)
            if tb and tb[0].ij:
                e_sum += e_cache[i1][j1].e
                branches += tb
        
        last = structs[-1]
        structs[-1] = Struct(round(last.e - e_sum, 1), last.desc, list(last.ij))
        
        
        return structs + branches

    return _trackback_energy(structs)

def _trackback_energy(structs: List[Struct]) -> List[Struct]:
    """Add energy to each structure, based on how it's W(i,j) differs from the one after

    Args:
        structs: The structures for whom energy is being calculated

    Returns:
        List[Struct]: Structures in the folded DNA with energy
    """

    structs_e: List[Struct] = []
    for index, struct in enumerate(structs):
        e_next = 0.0 if index == len(structs) - 1 else structs[index + 1].e
        structs_e.append(
            Struct(round(struct.e - e_next, 1), struct.desc, list(struct.ij))
        )
    return structs_e
