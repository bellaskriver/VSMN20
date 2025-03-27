def main():
    n = 640 # Length in meters
    inches, feet, yards, miles = conversion(n)
    print()
    print(n, 'meter(s) is equal to:')
    print("{:.2f}".format(inches), 'inche(s)')
    print("{:.2f}".format(feet), 'feet')
    print("{:.2f}".format(yards), 'yard(s)')
    print("{:.4f}".format(miles), 'mile(s)')
    print()

def conversion(n):
    meters_to_inches = n / 0.0254
    meters_to_feet = n / 0.3048
    meters_to_yards = n / 0.9144
    meters_to_miles = n / 1609.344
    return meters_to_inches, meters_to_feet, meters_to_yards, meters_to_miles

main()