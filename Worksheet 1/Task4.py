def main(): 
    print('Fahrenheit to Celsius:', temperture())

def temperture():
    matrix_C = []
    for n in range(10):
        F = n * 10
        C = 5 / 9 * (F - 32)
        matrix_C.append((F, C))
    return matrix_C

main()