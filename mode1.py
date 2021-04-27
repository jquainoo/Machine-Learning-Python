def mode(y):
    if len(y) == 1:
        return y[0]
    else:
        counts = {}
        for i in range(len(y)):
            if y[i] in counts:
                counts[y[i]] +=1
            else:
                counts[y[i]] =1
        m = max(counts, key=counts.get)
        c = counts[m]
        del counts[m]
        if len(counts) == 0:
            return m
        else:
            second_highest = max(counts, key=counts.get)

            if counts[second_highest] == c:
                return mode(y[:(len(y)-1)])
            else:
                return m

print(mode([3,3,4,5,5,5,6,7,8]))
