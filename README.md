# Group 2 exam project.
<<<<<<< HEAD
<<<<<<< HEAD
Pull the repository and read up on branching so we avoid push/pull issues throughout the project! NEVER WORK ON THE SAME FILES CONCURRENTLY.
=======
=======
>>>>>>> origin/branch-guide-in-readme
## YOU CANNOT PUSH TO THE MASTER BRANCH, YOU HAVE TO MAKE YOUR OWN BRANCH FOR WHATEVER FEATURE YOU'RE WORKING ON, AND MERGE WHEN YOUR WORK IS DONE. SEE GUIDE BELOW.

## Branching Guide:

### 1. Open up terminal
### 2. Write ```git branch```. (git branch lists all the branches on your computer in the working repository)
```
git branch
```
### 3. Next, run the following commands:
```
git status
git checkout -b myBranch
git status
```
The first command, git status reports you are currently on branch master, and it is up to date with origin/master, which means all the files you have on your local copy of the branch master are also present on GitHub. There is no difference between the two copies. All commits are identical on both the copies as well.

The next command, ```git checkout -b myBranch```, -b tells Git to create a new branch and name it myBranch, and checkout switches us to the newly created branch. Enter the third line, ```git status```, to verify you are on the new branch you just created.

The last ```git status``` reports you are on branch myBranch and there is nothing to commit. This is because there is neither a new file nor any modification in existing files.

If you want to see a visual representation of branches, run the command ```gitk```. If the computer complains bash: gitk: command not found…, then install gitk. Google it.

### 4. Now let's create a new file on our branch myBranch and let's observe terminal output. Run the following commands:
```
echo "Creating a newFile on myBranch" > newFile
cat newFile
git status
```
The first command, echo, creates a file named newFile, and cat newFile shows what is written in it. git status tells you the current status of our branch myBranch. In the terminal screenshot below, Git reports there is a file called newFile on myBranch and newFile is currently untracked. That means Git has not been told to track any changes that happen to newFile.

### 5. The next step is to add, commit, and push newFile to myBranch.

```
git add newFile
git commit -m "Adding newFile to myBranch"
git push origin myBranch
```
In these commands, the branch in the push command is myBranch instead of master. Git is taking newFile, pushing it to your working repository in GitHub, and telling you it's created a new branch on GitHub that is identical to your local copy of myBranch. 

If you go to GitHub, you can see there are two branches to pick from in the branch drop-down.

### 6. Merging branch with master
After you're done with the feature you're working on, you'll want to merge your branch with the master, to upload the work to the project. First go to your branch and then pull what's on the master:
```
git checkout myBranch
git pull origin master
git checkout master
git merge myBranch
```
If no complications come up, they should be ready for a Pull Request on the GitHub website. On there you can see what changes will be made to the files in the master branch, and another user has to approve and do the actual merge of the branches.

When this is done, the master branch will now be updated with whatever feature you're working on. 
Please make sure you're NOT working on the same feature as someone else, or complications will arise.
<<<<<<< HEAD
>>>>>>> c79524148d403e8e122433ebff4524836dc731a6
=======
>>>>>>> origin/branch-guide-in-readme

