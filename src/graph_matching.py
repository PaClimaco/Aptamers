import numpy as np
import sys
# Example path you want to add
new_path = r'../src/'
# Check if the path already exists to avoid duplication
if new_path not in sys.path:
    sys.path.append(new_path)

import copy



class Aptamer_match():
    """Class function to solve graph matching problem 
    """
    def __init__(self, sequence='ATA', n_tmpl=4):
        self.l_fix= None # number of fixed base pairs in the initial stem.
        self.n_tmpl =  n_tmpl
        self.n_wrld = None
        self.sequence = sequence # orginal sequnce in capital letters
        self.sqnc_num = None # original sequence converted in numbers
        self.B_g = None #matrix  storing all posible bonds between base-pairs
        self.dlt_g  = None #matrix sotring backbone bonds
        self.motifs = []  #list of all tmpl graphs found
        self.bps = [] #list of non isolated base pairs



    def convert_sequence_to_numeric(self,):
        """
        Converts a nucleotide string into a list of numeric representations.
        """
        mapping = {'A': 0, 'a': 0, 'T': 3, 't': 3, 'C': 1, 'c':1, 'G': 2, 'g':2}
        self.sqnc_num = []
        for char in self.sequence:
            if char in mapping:
                self.sqnc_num.append(mapping[char])
            else:
                print('Invalid character. Only allowed nucleotides are ACTG')
                break
        return 

    def build_BG_matrix(self,):
        """
        Builds a binary adjacency matrix representing the BG motif structure.
        """
        self.B_g = np.zeros((self.n_wrld, self.n_wrld), dtype=int)
        for i in range(self.n_wrld):
            for j in range(i+self.n_tmpl//2 +1,self.n_wrld):
                if (i <= self.l_fix -1 or j <= self.l_fix-1) and (i + j == self.n_wrld - 1):
                    self.B_g[i, j] = 1
                    self.B_g[j, i] = 1
                if self.n_wrld - self.l_fix -1 >= i > self.l_fix -1 and self.n_wrld - self.l_fix -1>= j > self.l_fix -1 and self.sqnc_num[i] + self.sqnc_num[j] in [3] and np.abs(i - j) >= 4:
                    self.B_g[i, j] = 1
                    self.B_g[j, i] = 1
        return


    def apt_filter(self, ):
            """
            Filters candidate vertices based on aptamer criteria.
            """
            for i in range(self.n_wrld):
                if self.l_fix - 2 <= i <= self.n_wrld - self.l_fix :
                        for j in range(i + self.n_tmpl//2 +1,self.n_wrld-self.n_tmpl // 2 +1):
                            if  j <= self.n_wrld - self.l_fix -1:
                                if (np.all(np.diag(self.B_g[list(range(i, i + self.n_tmpl // 2))][:, list(range(j + self.n_tmpl // 2 - 1, j - 1, -1))]) == np.ones(self.n_tmpl // 2, dtype=int))):                   
                                    self.motifs.append([(i + k, j + self.n_tmpl // 2 - 1 - k) for k in range(self.n_tmpl // 2)])   
            return 

    
    def create_dict_d(self,):
        '''Create dictionary containing for each possible length d = |i-j| all base pairs with length d 
            identified solving the subgraph matching problem
        '''
        self.dict_d = {}
        
        for bp in self.bps:
                    try:
                        self.dict_d[str(bp[1]-bp[0])].add(bp)
                        
                    except KeyError:
                        self.dict_d[str(bp[1]-bp[0])] = set([])
                        
                        self.dict_d[str(bp[1]-bp[0])].add(bp)
        return  
    
    def create_dict_Sij(self, ):
        '''Create a dictionary containing for each base pair (i,j),
            identified by solving the graph matching problem, all possible base pairs (i',j') such that i<i'<j'<j.
        '''
        self.dict_Sij = {}

        for count, bp1 in enumerate(self.bps):
                if '({}, {})'.format(*bp1) not in list(self.dict_Sij.keys()):
                    self.dict_Sij[str(bp1)]  =[]
                for bp2 in self.bps[count+1:]:
                    if bp1[0]<bp2[0]<bp2[1]<bp1[1]:
                        try:
                            self.dict_Sij[str(bp1)].append(bp2)
                            
                        except KeyError:
                            self.dict_Sij[str(bp1)]  =[]
                            self.dict_Sij[str(bp1)].append(bp2)
                            
                    elif bp2[0]<bp1[0]<bp1[1]<bp2[1]:
                        try:
                            self.dict_Sij[str(bp2)].append(bp1)
                            
                        except KeyError:  
                            self.dict_Sij[str(bp2)]  =[]
                            self.dict_Sij[str(bp2)].append(bp1)
        return  

    
    def fit_fold(self,sequence=[] , n_tmpl=4, l_fix=0):
            """
            Solves the subgraph matching problem.

            Args:
                seq (str): The sequence to be folded.
                n_tmpl (int): The template length used for subgraph matching. Default is 4.
                l_fix (int): The number of fixed base pairs in the initial stem. Default is 0.

            Results:
                self.bps (set): A set containing all non-isolated base pairs.
                self.dict_Sij (dict): A dictionary where each key is a base pair (i, j) identified by solving the graph matching problem. 
                                    The corresponding value is a list of all possible base pairs (i', j') such that i < i' < j' < j.
                self.dict_d (dict): A dictionary where each key is a possible distance d = |i - j| (the length of a base pair) and the value 
                                    is a list of all base pairs of that length identified by solving the subgraph matching problem.
            """

            self.l_fix= l_fix # number of fixed base pairs 
            self.n_tmpl = n_tmpl # lenght template 
            self.n_wrld = len(sequence) # length sequence
            self.sequence = sequence # orginal sequnce in capital letters
            self.convert_sequence_to_numeric() 
            self.build_BG_matrix()
            self.apt_filter()                
            self.bps = set([])
            
            #Store base pairs found solving the graph matching problem
            for mot in self.motifs:
                for bp in mot:
                    self.bps.add(bp)
                    
            
            # Include initial stem in base pair list        
            if l_fix >0:
                for k in range(l_fix):
                    self.bps.add((k, self.n_wrld-1-k))                
        
            self.bps = list(self.bps) 
            # Organize base pairs in dictionary based on their length
            self.create_dict_d()
            # Organize base pairs in dictionary. For each base pair find all base pairs included in it.
            self.create_dict_Sij()    
        

