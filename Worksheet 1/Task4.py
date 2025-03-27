def main(): 
    print()
    print(f"{'Fahrenheit':<12}{'Celsius':<12}")
    print("-" * 20)
    temperture()
    print()

def temperture():
    for n in range(11):
        F = n * 10
        C = 5 / 9 * (F - 32)
        print(f"{F:<12}{C:<12.2f}")

main()