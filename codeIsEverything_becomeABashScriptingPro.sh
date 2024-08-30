# https://ww.youtube.com/watch?v=4ygaA_y1wvQw


#!/bin/bash/env bash   - will check where the bash shell is located

#!/bin/bash  - note every system will have this path, so a bash scrip with just this may not function

# bash has acess to everything you usually do in the command line

# single quotes ' will output the literal text

name="YouTube"
echo 'Hello, $name!'

> Hello, $name!


name="YouTube"
echo "Hello, $name!"

> Hello, YouTube

# you can use the curly braces as a format in strings

echo "Hello, ${name}!"

> Hello, YouTube

# you can add a # right before the variable name with the curly brace
# to show how many characters there are

echo "Hello, ${#name}!"

> Hello, 7!

# you can set a default value if the varable is undefined

name=""
echo "Hello, ${name:-"Anonymous"}!"
echo $name

> Hello, Anonymous!
>  # No output

# we can set the Anonymous to the default variable with = instead of -

echo "Hello, ${name:="Anonymous"}!"
echo $name

> Hello, Anonymous!
> Anonymous

# The Subshell - used most often for command substitution

() <- Subshell  # you can chain commands together

$ pwd  # will get you the working directory
$ (cd ..; pwd)  # will get you the working directory, but one level up because we cd'd into that directory first
$ pwd  # will get you the same working directory as the first time we ran pwd becuase we didn't actually go anywhere

$() <- Command Substitution

# you can run a command in a subshell, then use that wherever you'd normally use a value

$ var=$(pwd)
$ echo $var
> /home/logan/workspace/videos/bash # or whatever

# fetch from an api, then using jq to extract the value
$ image_url=$curl -s "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY" | jq -r '.hdurl')

# then we're taking the final output from that command and using it later
$ curl -s "$image_url" -o "apod.jpg"


<() <- Process substitition # retrieving the output and treating it like a file

$ diff <(ls ./v1) <(ls ./v2)  # diff is expecting two files, but this will compare the differences between these two ls commands AS THOUGH THEY ARE FILES!

$(()) <- Arithmetic expansion  # for running a basic mathematical operation

$(( 3 + 4 ))
> 7


# Getting user input, usually command line arguments

./mergeav.sh input.mp4 input.wav output.mp4
    ($0)        ($1)      ($2)     ($3)


# If Statements

# structure
if [[ some_condition ]]; then
        echo "This condition is true"
elif [[ some_other_condition ]]; then
        echo "This other condition is true"
else
        echo "None of the conditions are true"
fi

# Exit Codes - rather than TRUE/FALSE
Exit code 0 -> Success!
Exit code n (with n being anything else) -> Failure


# Conditionals 

# for strings
==
! = (no space)

# for numbers
-eq # equal
-ne # not equal
-lt # less than 
-le # less than or equal to
-gt # greater than 
-ge # greater than or equal to

# for existance
[[ -z $variable ]] # variable is null (empty)
[[ -n $variable ]] # variable is not-null 

# file checks
file=".hello"
[[ -f $file ]] # file exists
[[ -d $file ]] # directory exist
[[ -e $file ]] # File/dir exists

# Permission checks
file="./hello"
[[ -r $file ]] # File is readable
[[ -w $file ]] # File is writable
[[ -x $file ]] # File is executable

# Combinations

#internal
[[ $varible -gt 5 -a $val -lt 10 ]] # -a -> logical AND
[[ $varible -gt 5 -o $val -lt 3 ]] # -a -> logical OR

# external 
[[ $varible -gt 5 ]] && [[ $val -lt 10 ]] # -a -> logical AND
[[ $varible -gt 5 ]] || [[ $val -lt 3 ]] # -a -> logical OR


# USEFUL COMMANDS YOU'LL USE OFTEN

sleep n (in seconds)  # how many seconds you want to sleep for
sleep 30

read -r name # ask the user for input, assign the value to the named variable
echo "Hello, $name!"


# Building in good behavior

set -euo pipefail
        # -e = Exit on error
                # by default, if there's an error, BASH just keeps on running
                # if something fails, you don't want the later things to run without that success

        # -u = Exit on unset variable
                # if you reference something that wasn't defined, it will exit

        # -o pipefail = Exit on pipe fail 
                #

# good practice to set this line right under the shebang in all your scripts

# Arrays

echo ${#my_arr[@]}  # prints the length of the array because of the #

# For loops

# working with a range
for i in {1..10}; do 
        echo "$i"
done

# pattern matching with a loop
for item in ./content/*.md; do
        echo "$item"
done

# command results
for item in $(ls); do
        echo "$item"
done


# Breaking out

# you can run bash scripts within other bash scripts!
# you can even pass in arguments like you would in the command line

# provisioner.sh
echo "Provisioning environment"
./instance.sh nyc3 2gb
./dns.sh example.com
echo "Provisioning complete!"

# instance.sh
region=$1
size=$2
doctl ... # pretend this does something to provision a VM

# dns.sh
domain=$1
doctl ... # pretend this configures the DNS


# Functions

# Temporary files and directories

# you can make temporary files and directories that wipe themselves when they finish running

tempfile=$(mktemp)  # will make a file in the temporary directory
trap "rm -f $tempfile" EXIT # trap executes a command on a give signal, in this case, removing the file, as soon as we get the EXIT signal, which is when the script completes

echo "Hello, YouTube!" > $tempfile

tempdir=$(mktemp -d)  # does the same, but makes a temporary directory
trap "rm -rf $tempdir" EXIT

echo "Hello, YouTube (from inside a tempdir)!" > "$tempdir/hello"

# THE $PATH

# the path is an environment variable in the shell that tells it which directories to look through when it's looking for a command

# add script directory to the path 
export PATH=$PATH:$HOME/.scripts/bin  #or wherever you want your scripts to be held

# make sure to add the export line above to your ~/.bashrc or ~/.zshrc file to make it permanent!


# The General Process for creating a bash script

# 0. Figure out what you want to do
# 1. Identify the tools you'll use
# 2. sketch it out in the terminal
# 3. copy it into a script
# 4. pull out variables and inputs
# 5. add checks (guards, etc.)
# 
# optional
# 6. add loops and other more "advanced" functionality
# 7. If it is growing too large, break it into multiple files















