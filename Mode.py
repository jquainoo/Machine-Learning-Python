def Mode():
    mode = -1
    arr = [3,5,5,8,8,5]
    num = 0
    n = 0
    for i in range(len(arr)):
        num = arr.count(arr[i])
        if num >= n:
            n = num
            mode = arr[i]
        else:
            continue
    print(mode)

Mode()
