# Bash Scripting Tutorial for Beginners
# https://www.youtube.com/watch?v=tK9Oc6AEnR4

# Bash scripts have the file-typ .sh
# we can run bash scripts with the command "bash SCRIPTNAME.sh"

# The first shell script
# file-name = shelltest.sh
echo Hello World!

# in the command line
echo $SHELL 
# > /bin/bash

#!/bin/bash   -   the shebang, telling it what shell to use
# then you can simply run `./shelltest.sh` and the program will run 

# Updating the permissions so it can run as an executable

# Display the permissions with 
# `$ ls -l`

# Add the execute permission for the user with 
# `$ chmod u+x shelltest.sh`



# VARIABLES IN BASH

# file-name = hellothere.sh

#!/bin/bash

FIRST_NAME=Herbert  # it's important that there are NO SPACES here
LASE_NAME=Lindemans
echo Hello $FIRST_NAME $LASE_NAME


# USER INPUT IN BASH

# file-name = interactiveshell.sh

#!/bin/bash

echo What is your first name?
read FIRST_NAME  # the `read` command will prompt us for input
echo What is your last name?
read LAST_NAME

echo Hello $FIRST_NAME $LAST_NAME


# POSITIONAL ARGUMENTS IN BASH AKA COMMAND LINE ARGUMENTS

# file-name = postarg.sh

#!/bin/bash

echo hello $1 $2  # the $1 and $2 will be what the input was in those positions in the command line.
                        # REMEMBER the command name is position 0


# OUTPUT / INPUT REDIRECTION 

# piping output to the next input using `|`

# > will be used to write to a file !!! WILL OVERWRITE ANY EXISTING CONTENT !!!
# >> will append to the end of an existing file

# e.g. `$ echo Hello world! > hello.txt`
#        `$ cat hello.txt`
#        > Hello world!

# passing a file to a command rather than it acting as a positional argument
# $ wc -w hello.txt
# > 6 hello.txt
# 
# $ wc -w < hello.txt
# >6

# Passing multiple lines back with `<<`
        # it will prompt you for an input
        # you can pass in multiple lines until you write that input again 

# $ cat << EOF
# > I will
# > write some
# > text here
# > EOF
# 
# I will 
# write some
# text here

# Single strings of text to the command line with `<<<`
        # you need to have the string in double quotes " or it will not work

# wc -w <<< "Hello there word count!"
# > 3


# TEST OPERATORS - determine whether a statement is true or if one text is equal to another

# $ [ hello = hello ]  # the spaces are important in this case
# $ echo $?  # $? returns the exit code of the last command
# > 0  # this indicates the last code run was executed without any issues

# $ [ 1 = 0 ]
# $ echo $?
# > 1  # indicates this is false, because there was an error running it

# use -eq instead of = to make sure the values are in fact numerical
# would throw an error if you used letters


# IF / ELSE / ELIF

# file-name = ifelifelse.sh

#!/bin/bash

if [ ${1,,} = herbert ]; then  # the commas and curly braces = parameter expansion, allowing us to ignore upper and lower cases when comparing the two values
        echo "Oh, you're the boss here. Welcome!"
elif [ ${1,,} = help ]; then 
        echo "Just enter your username!"
else
        echo "I don't know who you are!"
fi  # closes the block of if/elif/else statements


# CASE STATEMENTS - CHECKING FOR MULTIPLE STATES, PERFORMING ACTIONS BASED ON WHICH CASE IS TRUE

#!/bin/bash

case ${1,,} in
        herbert | administrator)  # the pipe operator `|` acts like "OR" here, `)` acts to end the list of conditions
                echo "Hello, you're the boss!"
                ;;  # double semi-colon ends this set of options and actions
        help) 
                echo "Enter your username!"
                ;;
        *)  # catch all, everyitng else
                echo "Hello there. Enter a valid username!"
esac  # closes this case block.


# ARRAYS - MULTIPLE VAULES ASSIGNED TO ONE VARIABLE IN A LIST

$ MY_FIRST_LIST=(one two three four five)  # `()` to contain the list, space is the delimiter

# Printing the list in the command line

echo $MY_FIRST_LIST  # will just return `one`
echo ${MY_FIRST_LIST[@]}  # will return the whole array
echo ${MY_FIRST_LIST[2]}  # will return the value at index 2, in this case `three`


# FOR LOOP 
$ for item in ${MY_FIRST_LIST[@]}; do echo -n $item | wc -c; done

# for each item in the array
# echo the item, not including any of the new line characters '\n\
# pipe that into the word count function, with the -c flag to ask for the number of characters in the item
# tell the loop that we're done

# returns
3
3
5
4
4


# FUNCTIONS
# when there's a lot of repeated code,
# when you want to run things in a specific order
# when you want to run through if/else statements multiple times
# re-using parts of your script over and over, saving time re-writing

# file-name = firstfunction.sh

#!/bin/bash

showuptime(){  # define our function name
        up=$(uptime -p | cut -c4- )  # define our first variable
        since=$(uptime -s)  # define our second variable, create a nice output with `cat` 
        cat << EOF
-----
This machine has been up for ${up}
It has been running since ${since}
-----
EOF
}
showuptime  # call the function by re-typing the function name under where the function was defined


# define local variables inside the functions so they're not available to the entire script, which may cause issues for larger scripts
# you may accidentally override local variables with global variables with the same name

# define them as local variables in side your function with `local `

#e.g. 
up="before"
since="function"
echo $up
echo $since
showuptime(){  # define our function name
        local up=$(uptime -p | cut -c4- )  # define our first variable
        local since=$(uptime -s)  # define our second variable, create a nice output with `cat` 
        cat << EOF
-----
This machine has been up for ${up}
It has been running since ${since}
-----
EOF
}
up
since
echo $up
echo $since
showuptime  # call the function by re-typing the function name under where the function was defined

# in this case, the global variable was not overwritten by the local variables inside the function


# EXIT CODES

#!/bin/bash

showname(){
        echo hello $1
        if [ ${1,,} = herbert ]; then 
                return 0
        else 
                return 1
        fi
}
showname $1
if [ $? = 1 ]; then  # if anything except herbert is typed in, the showname() will return 1, the "error" exit code
        echo "someone unknown called the function!"
fi


# AWK

# getting file contents, printing the most essential parts of a file

$ echo one two three four >> test_text.txt
$ awk '{print $1}' test_text.txt
> one

$ echo one,two,three,four > test_csv.csv
$ awk -F, '{print $1}' test_csv.csv  # use `-F` to call a different delimiter, use `,` as the delimiter
> one



# SED -  CHANGING CERTAIN VALUES IN TEXT FILES

cat sedtest.txt
> The fly flies like no fly flies.
> A fly is an insect that has wings and a fly likes to eat leftovers. 

$ sed 's/fly/grasshopper/g' sedtest.txt # replaces every instance of 'fly' with 'grasshopper'
# s/ = substitution mode
# fly = the word we want to substitute out
# grasshopper = the word we want to substitute in
# /g = we want to do this globally 
# sedtest.txt = the file we want to make the change in 

$ sed -i.ORIGINAL
# create a backup file that retains the original configuration before the substitution













