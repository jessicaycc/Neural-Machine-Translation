# Copyright 2020 University of Toronto, all rights reserved

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

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


def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    if not candidate:
        p_n = 0
        return(p_n)

    reference_ngram = grouper(reference, n)
    candidate_ngram = grouper(candidate, n)
    N = len(candidate_ngram)
    C = 0

    for word in candidate_ngram:
        if word in reference_ngram:
            C += 1
    if (N == 0):
        p_n = 0
    else:
        p_n = C/N
    return (p_n)

def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    if not candidate:
       BP = 0
       return(BP)
    
    c = len(candidate)
    r = len(reference)
    brevity = r/c

    if brevity >= 1:
        BP = exp(1-brevity)
    else:
        BP = 1
    return(BP)

def BLEU_score(reference, hypothesis, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    BP = brevity_penalty(reference, hypothesis)
    p_total = 1.00
    for i in range(n):
        p = n_gram_precision(reference, hypothesis, i+1)
        p_total = p_total * p
    bleu = BP * p_total**(1/n)
    # print("bleu: ", bleu)
    return (bleu) 
