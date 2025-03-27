def main():
    n = 3 # Number of years
    r = 5 # Interest rate
    P = 1000 # Initial amount
    print()
    print('Amount after three years:', "{:.2f}".format(intrest(n,r,P)), 'SEK')
    print()

def intrest(n, r, P):
   A = P * (1 + r/100) ** (n)
   return A

main()