from math import exp  # exp(x) gives e^x

def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    ngrams = []
    
    for i in range(len(seq)):
        word = []
        if (i+n-1)<len(seq):
                for j in range(n):
                        if ((i+j) >= len(seq)):
                                continue
                        else:
                                word.append(seq[i+j]) 
                ngrams.append(word)

    return (ngrams)
n = 2
candidate = ['I', 'fear', 'afraid', 'david']
reference = ['I', 'am', 'afraid', 'david']
if not candidate:
        p_n = 0
        

reference_ngram = grouper(reference, n)
candidate_ngram = grouper(candidate, n)
print(reference_ngram)
print(candidate_ngram)
c = len(candidate_ngram)
p = 0
for word in candidate_ngram:
        if word in reference_ngram:
                p += 1
p_n = p/c
print(c)
print(p)
print(p_n)