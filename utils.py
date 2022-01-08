def git_save():
    from os import system as cmd
    print(cmd('git switch autosave'))
    print(cmd('git add .'))
    print(cmd('git commit -m' + str(int(time()))))
    print(cmd('git push'))

log = open('log.txt', 'w')
