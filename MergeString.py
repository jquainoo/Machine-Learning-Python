"""Given two lists, both having String elements, write a python program using
python lists to create a new string as per the rule given below: The first
element in list1 should be merged with last element in list2, second element
in list1 should be merged with second last element in list2 and so on. If an
element in list1/list2 is None, then the corresponding element in the other
list should be kept as it is in the merged list.
eg.
'['A', 'app','a', 'd', 'ke', 'th', 'doc', 'awa']'
'['y','tor','e','eps','ay',None,'le','n']'
should give you 'An apple a day keeps the doctor away' """

def MergeToString(arr1, arr2):
    newList = []
    length = len(arr2) - 1
    sentence = ""
    for i in range(len(arr1)):
        if(arr2[length-i] == None):
            newList.append(arr1[i])
        elif (arr1[i] == None):
            newList.append(arr2[length-i])
        else:
              newList.append(arr1[i] + arr2[length-i])
              
    for word in newList:
        sentence += word + " "
    return sentence

print(MergeToString(['A', 'app','a', 'd', 'ke', 'th', 'doc', 'awa'],['y','tor','e','eps','ay',None,'le','n']))
