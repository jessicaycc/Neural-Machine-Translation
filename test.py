from math import exp  # exp(x) gives e^x

ngrams = []
n = 2
seq = "I am afraid Dave"
sentence = seq.split()
for i, word in enumerate(sentence):
        token = ""
        if (i+n-1)<len(sentence):
                for j in range(n):
                        if ((i+j) >= len(sentence)):
                                continue
                        elif j < n-1:
                                token += sentence[i+j] + " "
                        else:
                                token += sentence[i+j]
                ngrams.append(token)

print(ngrams)