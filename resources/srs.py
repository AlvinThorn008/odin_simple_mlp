def srs(x):
    stack = [0] * 4
    v = 0.0
    p = 0
    for i in range(0, len(x)):
        v = x[i]
        
        b = 1
        print(i, ":", stack, v)
        while i & b != 0:
            v += stack[p - 1]
            b <<= 1
            p -= 1
        stack[p] = v
        p += 1
        print(i, ":", stack, "\n")

    print(stack)

srs([1, 2, 3, 4, 5, 6, 7, 8])
    
