#!/bin/bash
# fyx - 2024-05-04
'''
usage: ./auto_fetch.sh [merge|pull]
example: 
  ./auto_pull.sh merge
  ./auto_pull.sh pull
  If "merge" is chosen, it will fetch and then manually merge.
  If "pull" is chosen, it will just pull the changes.
'''

# Script beginning here

# Function to perform git merge
do_merge() {
  echo "Fetching and manually merging changes from the remote repository..."
  git fetch origin master
  git merge origin/master
}

# Function to perform git pull
do_pull() {
  echo "Pulling the latest changes from the remote repository..."
  git fetch origin master
  git pull origin master
}

# Check if an argument is provided
if [ $# -eq 0 ]; then
  echo "Please choose either 'merge' or 'pull'."
  exit 1
fi

# Check the choice and perform the corresponding action
if [ "$1" = "merge" ]; then
  do_merge
elif [ "$1" = "pull" ]; then
  do_pull
else
  echo "Invalid option: $1. Please choose either 'merge' or 'pull'."
  exit 1
fi

# Check the exit status of the last command
if [ $? -eq 0 ]; then
  echo "Successfully updated the local repository."
else
  echo "Failed to update the local repository."
  exit 1
fi