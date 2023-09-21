def central_difference_rate(y):
    x=range(1,31)
    n = len(x)
    table = [[0] * 6 for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]

    for j in range(1, 6):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    k=int(len(x)/2)
    for row in table:
        print(row)
    h=x[1]-x[0]
    rate=1/h*((table[k][1]+table[k-1][1])/2-1/12*(table[k-1][3]+table[k-2][3])+1/60*(table[k-2][5]+table[k-3][5]))
    return rate

arr=[3*i*i for i in range(1,31)]
f=1
f+=2
if(f==3):
    print(f)
print(central_difference_rate(arr))