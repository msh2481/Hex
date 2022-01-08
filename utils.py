from time import time

log = open('log.txt', 'w')

def git_save():
    from os import system as cmd
    cmd('git add . >git.txt'),
    cmd('git commit --amend --no-edit >git.txt')
    cmd('git push -f >git.txt')

