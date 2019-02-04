"""Given two lists, both having integer elements, write a python program using
python lists to create and return a new list as per the rule given below:
If the double of an element in list1 is present in list2, then add it to
the new list."""

def DoubleElement(arr1, arr2):
    newList = []
    for num in arr1:
        for number in arr2:
            if(num * 2 == number):
                newList.append(num)
    return newList
print(DoubleElement([11, 8,23,7,25, 15], [6, 33, 50,31, 46, 78, 16,34]))


#Optimized & Better
def DoubleElement(arr1, arr2):
    newList = []
    for num in arr1:
        if((num * 2) in arr2):
            newList.append(num)
    return newList
print(DoubleElement([11, 8,23,7,25, 15], [6, 33, 50,31, 46, 78, 16,34]))

