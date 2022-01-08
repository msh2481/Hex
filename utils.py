from time import time

log = open('log.txt', 'w')

def git_save():
    from os import system as cmd
    cmd('git switch autosave >git.txt')
    cmd('git add . >git.txt'),
    cmd('git commit -m' + str(int(time())) + '>git.txt')
    cmd('git push >git.txt')

