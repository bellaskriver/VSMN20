def main():
    n = 3
    r = 0.05
    P = 1000
    results = intrest(A)
    print(results)

def intrest(n, r, P):
   A = P * (1 + r/100) ** (n)
    return A
