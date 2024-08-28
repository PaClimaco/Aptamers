
import numpy as np

def compute_descriptor(input_string):
    descriptor = []
    sum = 0
    for count, char in enumerate(input_string):
        if char == '(':
            sum +=1
            descriptor.append(sum)
        elif char == ')':
            descriptor.append(sum)
            sum -= 1
        elif char == '.':
            descriptor.append(sum)      
    return descriptor


def substructure_ID(string):
    w= compute_descriptor(string)
    # mask with 1 when closed or open parantesis
    mask = np.isin( np.array(list(string)), ['(', ')'])
    p = mask.astype(int)
    
    dct = {'hairpins' : [],
    'multi-branches' : [],
    'stacks' : [],
    'left_bulges' : [],
    'right_bulges' : [],
    'inner_loops' : [] }
    
    stacks = []
    for count, s in enumerate(string):
        #print(stacks)
        if s == '(':
            stacks.append(count)
            
        elif s == ')':
            if stacks:
                i = stacks.pop()
                j = count
            
            if w[i]==w[i+1]-1 and w[j]==w[j-1]-1:
                dct['stacks'].append((i,j))
                
            else:
                l = w[i]
                L = np.where(np.logical_and(np.array(w[i:j], dtype = int)== l+1 , np.array(p[i:j])==1  ))[0] + i
                T = int(len(L)/2)
                if T == 0 and np.abs(i-j)>=4:
                    dct['hairpins'].append((i,j))
                elif T == 1:
                    stat = [i, L[0], L[1],j]
                    if w[i]==w[i+1]-1 and w[j]!=w[j-1]-1: #right buldge
                        dct['right_bulges'].append([(stat[k], stat[-k-1]) for k in range(2)]) #[(i, j), (i1,j1)]
                    elif w[i]!=w[i+1]-1 and w[j]==w[j-1]-1: #left buldge
                        # indices of the starting (i,j) and closing (i1,j1) bonded bp 
                        dct['left_bulges'].append([(stat[k], stat[-k-1]) for k in range(2)])#[(i, j), (i1,j1)]
                    else:
                        # indices of the starting (i,j) and closing (i1,j1) bonded bp   
                        dct['inner_loops'].append([(stat[k], stat[-k-1]) for k in range(2)])
                elif T > 1:
                    #[(i, j), (i1,j1) (i2,j2)]  intial bonded bp of each branch are saved
                    stat =   [i, j] + list(L)
                    dct['multi-branches'].append( [(stat[2*k], stat[2*k+1]) for k in range(T+1)]  )           
    return dct
