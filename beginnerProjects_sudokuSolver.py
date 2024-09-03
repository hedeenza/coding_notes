def find_next_empty(puzzle):
    # find the next row that's not filled
    # for this program -1 is the representation of a negative space, so find that
    # return the row, col tuple (the row, then the column), now that we're in the right row, which index to we need to go to?
        # if the whole board is filled, return (None, None)

    # our indices are 0-8, but the puzzle values will be 1-9
    for row in range(9):  #check the rows
        for column in range(9):  # check the columns
            if puzzle[row][column] == -1:  # if it's blank
                return row, column  # return that index
    return None, None  # if none are blank, return None, None

def is_valid(puzzle, guess, row, col):  
    # determines if the guess is valid at that row/column
    # returns True if it is

    # Checking against the rows
    row_vals = puzzle[row]
    if guess in row_vals:
        return False  # if it's the same as an existing value, it can't be the answer

    # Checking against the columns
    col_vals = [puzzle[i][col] for i in range(9)]  # index to the row, then the column for every row in the range
    if guess in col_vals:
        return False  # if it's the same as an existing value, it can't be the answer

    # Checking against the 3x3 grid
    # find the first row index in the 3x3, then the first column index in teh 3x3, and iterate over all of those
    row_start = (row // 3) * 3  # lets us know if it's in the first, second, or third set of 3 rows; *3 to get the index of that row
    col_start = (col // 3) * 3  # lets us know if it's in the first, second, or third set of 3 rows; *3 to get the index of that row

    for row in range(row_start, row_start + 3):  # +3 because we want to iterate over 3 rows
            for col in range(col_start, col_start + 3):
                if puzzle[row][col] == guess:
                    return False  # if the guess is already in the grid, it can't be the answer for that square

    # if we pass the row test, column test, and grid test, then our guess is valid and we can return True
    return True   



def solve_sudoku(puzzle):
    # solve sudoku using backtracking technique
        # puzzle = list of lists
            # each inner list is a row in the sudoku puzzle
        # return whether the solution exists
        # mutates puzzle to be the solution, if it iexists


# 1: choose where on the puzzle to make a guess
    # creat a function to find the next empty 
    row, col = find_next_empty(puzzle)

#2: If the puzzle grid is full, we need to do validation checks to make sure it's correct
    if row is None:
        return True # if None, None got returned, we've solved the puzzle

#3: If there's a blank, make a guess 1-9
    for guess in range(1, 10): # because we want the values 1-9 with the 0 index, not exclusive of the second value
        # check if it's a valid guess
        if is_valid(puzzle, guess, row, col):  # all four values are needed to determine validity of the guess
            # If we passed all the validity checks, then place the guess at that position
            puzzle[row][col] = guess # we're actually mutating our puzzle array
            # now recurse using this puzzle, which has been updated by one value

#4: recursively call our function
            if solve_sudoku(puzzle):
                return True

#5: if our guess was not valid OR if our guess does not solve the puzzle
        # then we need to backtrack and try a new number
        puzzle[row][col] = -1  # set the value of the square back to -1

#6: if none of the numbers we try work, the puzzle is unsolvable, because our loop tires 1-9 for every square
    return False


#7: Testing
if __name__=='__main__':
    example_board = [
            [3, 9, -1,  -1, 5, -1,  -1, -1, -1],
            [-1, -1, -1,  2, -1, -1,  -1, -1, 5],
            [-1, -1, -1,  7, 1, 9,  -1, 8, -1],

            [-1, 5, -1,  -1, 6, 8,  -1, -1, -1],
            [2, -1, 6,  -1, -1, 3,  -1, -1, -1],
            [-1, -1, -1,  -1, -1, -1,  -1, -1, 4],

            [5, -1, -1,  -1, -1, -1,  -1, -1, -1],
            [6, 7, -1,  1, -1, 5,  -1, 4, -1],
            [1, -1, 9,  -1, -1, -1,  2, -1, -1],
    ]
    print(solve_sudoku(example_board))
    print(example_board)


