from time import time

log = open('log.txt', 'w')

def git_save():
    global log
    from os import system as cmd
    log.close()
    cmd('git switch autosave >log.txt')
    cmd('git add . >log.txt'),
    cmd('git commit -m' + str(int(time())) + '>log.txt')
    cmd('git push >log.txt')
    log = open('log.txt', 'a')

