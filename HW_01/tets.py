import time
def meow():
    meowseq = "Meow"
    while True:
        yield meowseq
        meowseq += " Meow"

for i in meow():
    print(i, end="\n")
    time.sleep(10)