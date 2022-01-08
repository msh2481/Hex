from time import time

log = open('log.txt', 'w')

def git_save():
    from os import system as cmd
    print(cmd('git switch autosave'), file=log)
    print(cmd('git add .'), file=log)
    print(cmd('git commit -m' + str(int(time()))), file=log)
    print(cmd('git push'), file=log)
    print(flush=True, file=log)

