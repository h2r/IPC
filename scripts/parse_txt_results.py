flip_bit = False
f = open('experiment-results/prob-09-expers.txt', 'r')
for line in f:
    if 'real' in line:
        if flip_bit:
            print(line[7:-2])
        flip_bit = not(flip_bit)
