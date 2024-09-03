import random  # for the random number generation with out 10K long list
import time  # so we can time how long our operations take


# This is a naive search example
# Check every item in the list and ask if it's equal to the target
    # if yes, return the index
    # if no, then return -1
def naive_search (l, target):  # our naive search function will accept l and target variables
    # example l = [1, 3, ]
    for i in range(len(l)): # for every single index
        if l[i] == target:  #if the index at l is the target, 
            return i
    return -1  # otherwise we've gone through the whole list and nothing is there

# Binary search example
# Use the fact that our list is sorted to make it faster
def binary_search (l, target, low=None, high=None):  #assigning a floor and ceiling that will change over each iteraiton of the search
    if low is None:
        low = 0
    if high is None:
        high = len(l) - 1

    if high < low:
        return -1  # this should only happen if we've moved outside of the list, like we couldn't find it

    midpoint = (low + high) // 2  # what's the length of the list, divided by 2, rounded down?

    if l[midpoint] == target:
        return midpoint
    elif target < l[midpoint]:
        return binary_search (l, target, low, midpoint-1)  # the new high is the midpoint -1 if the target is lower than the midpoint
    else:
        # if target > l[midpoint]
        return binary_search(l, target, midpoint+1, high)

if __name__=='__main__':
    l = [1, 3, 5, 10, 12]
    target = 10
    print(naive_search(l, target))
    print(binary_search(l, target))

# Examining which method is faster

    length = 10000

    # build a sorted list of length 10000
    sorted_list = set()
    while len(sorted_list) < length:
        sorted_list.add(random.randint( -3*length, 3*length))  # choose a number -30,000 to 30,000 to populate this 10,000 item long list
    sorted_list = sorted(list(sorted_list))  # make this into a list with list(), then sort it with sorted()

# Examining Naive Search

    start = time.time()  # the start time is the time at that time
    for target in sorted_list:  # every item is the target for one iteration of the list
        naive_search(sorted_list, target)  # run the naive search 10K times
    end = time.time()  # the end time is the time at the time it's done
    print("Naive search time: ", (end - start)/length, "seconds")  # the average time it took

    start = time.time()  # the start time is the time at that time
    for target in sorted_list:  # every item is the target for one iteration of the list
        binary_search(sorted_list, target)  # run the naive search 10K times
    end = time.time()  # the end time is the time at the time it's done
    print("Binary search time: ", (end - start)/length, "seconds")  # the average time it took

Naive search time:  0.0002758157968521118 seconds
Binary search time:  3.5634756088256838e-06 seconds
Binary search time: 0.0000035634756088256838 seconds
