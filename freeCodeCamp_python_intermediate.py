#  1 Interediate Python Programming Course
#  2 https://www.youtube.com/watch?v=HGOBQPFzWKo
#  3 ~ 6 hoursm

# Outline of the course

# list created with :put =range(1,21)
# 1 - Lists - 00:00:56
# 2 - Tuples - 00:16:30
# 3 - Dictionary - 00:29:49
# 4 - Sets - 00:42:40
# 5 - Strings - 0058:44
# 6 - Collections - 01:22.50
# 7 - Itertools - 01:36:43
# 8 - Lambda Functions - 01:51:50
# 9 - Exceptions and Errors - 02:04:03
# 10 - Logging - 02:20:10
# 11 - JSON - 02:42:20
# 12 - Random Numbers - 02:59:42
# 13 - Decorators - 03:14:23
# 14 - Generators - 03:35:32
# 15 - Threading vs Multiprocessing - 03:53:29
# 16 - Multithreading - 04:07:59
# 17 - Multiprocessing - 04:31:05
# 18 - Function Arguments - 04:53:26
# 19 - The Asterisk (*) Operator - 05:17:28
# 20 - Shallow vs Deep Copying - 05:30:19
# 21 - Context Managers - 05:40:07



# 1 - Lists - 00:00:56

# square brackets[]
# mix data types, duplicate data types allowed
# index the lists
# index over your lists with a for group
mylist = ["banana", "cherry", "apple"]

for i in list:
    print(i)

if "apple" in mylist:
    print("yes")
else:
    print("no")

print(len(myList))  # check the list length

mylist.append("lemon")  # add an item to the list

mylist.insert(1, "blueberry")  # add an item to a specified index position

mylist.pop()  # remove the last value in the list

mylist.remove("cherry")  # remove the specified item from the list

mylist.clear()  # remove all items from the list

mylist.reverse()  # reverse the list order

mylist.sort()  # change the original list

    new_list = sorted(mylist)  # doesn't change the original order, but gives you a sorted list

    mylist = [0] * 5  # new list but with 5 zeroes

    # you can add lists with +

mylist = [1,2,3,4,5,6,7,8,9]
a = mylist[1:5]
print(a)  # will only print the section sliced by the specified index 
            # note: it's inclusive of the first index value, exclusive of the second
            # if you do not include a start or end index, it will include everyting from the start or to the end away from your specified index

# pick up at 00:11:51
