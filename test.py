from math import exp  # exp(x) gives e^x

ngrams = []
n = 2
seq = ['I', 'am', 'afraid', 'David']

for i in range(len(seq)):
        word = []
        if (i+n-1)<len(seq):
                for j in range(n):
                        if ((i+j) >= len(seq)):
                                continue
                        else:
                                word.append(seq[i+j]) 
                ngrams.append(word)

print(ngrams)