def countApplesAndOranges(s, t, a, b, apples, oranges):
    count = [0,0]
    for i in range(len(apples)):
        apples[i] = apples[i] + a

    for i in range(len(oranges)):
        oranges[i] = oranges[i] + b
        
    for num in apples:
        if(num >= s and num <= t):
            count[0] += 1

    for num in oranges:
        if(num >= s and num <= t):
            count[1] += 1
            
    print(count[0])
    print(count[1])
countApplesAndOranges(7, 10, 4, 12, [2,3,-4], [3,-2, -4])
