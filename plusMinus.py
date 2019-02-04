def plusMinus(arr):
    new_list = []
    count_list = [0, 0, 0]
    for num in arr:
        if(num > 0):
            count_list[0] += 1
        elif(num < 0):
            count_list[1] += 1
        else:
            count_list[2] += 1
    frac_list = []
    print(count_list)
    for counter in count_list:
        frac_list.append(counter/(len(arr)))
    for frac in frac_list:
        print("%.6f" % frac)
            
plusMinus([-4, 3, -9, 0, 4 ,1 , ])


def staircase(n):
    for i in range(n):
        for x in range(n):
            if((i+x) > n):
                print("#",end="")
            else:
                print()

staircase(6)
