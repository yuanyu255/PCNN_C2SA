def proceess_data(data, threshold=0, UNKid=0):
    print 'process word dict ..'
    w_t = [0 for i in range(300000)]
    for bag in data:
        for sentence in bag.sentences:
            for word in sentence:
                w_t[word] += 1
    tt = 0

    for i in range(300000):
        if w_t[i] >= threshold:
            tt += 1

    print 'non_UNK = ', tt
    for bag in data:
        for sentence in bag.sentences:
            for i, word in enumerate(sentence):
                if w_t[word]<threshold:
                    sentence[i] = UNKid
    print 'finished ..'
    return data







