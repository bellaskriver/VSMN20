def main():
    n = 640
    results = conversion(n)
    print(results)

def conversion(n):
    meters_to_inches = n / 0.0254
    meters_to_feet = n / 0.3048
    meters_to_yards = n / 0.9144
    meters_to_miles = n / 1609.344
    return meters_to_inches, meters_to_feet, meters_to_yards, meters_to_miles

main()