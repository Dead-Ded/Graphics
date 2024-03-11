def fib(a=1, b=1):
    fib.a, fib.b = b, a + b
    return fib.b


fib.a, fib.b = 1, 1
for i in range(10):
    print(fib(fib.a, fib.b))
