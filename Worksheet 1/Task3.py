def main():
    n = 3 # Number of years
    r = 5 # Interest rate
    P = 1000 # Initial amount
    print('After three years:',intrest(n,r,P))

def intrest(n, r, P):
   A = P * (1 + r/100) ** (n)
   return A

main()