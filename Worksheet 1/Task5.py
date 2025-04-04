import numpy as np

try:
    file_path = './Worksheet 1/numbers.txt'
    total_sum = 0
    with open(file_path, 'r') as file:
        for line in file:
            try:
                total_sum += float(line.strip())
            except ValueError as e:
                print()
                print(f"Error: {e}")
    print()
    print(f"The total sum is: {total_sum}")
    print()

except FileNotFoundError:
    print()
    print(f"Error: The file '{file_path}' was not found.")

except Exception as e:
    print()
    print(f"An unexpected error occurred: {e}")