try:
    file_path = '/Users/bellaskriver/Documents/GitHub/VSMN20/Worksheet 1/numbers.txt'
    total_sum = 0
    with open(file_path, 'r') as file:
        for line in file:
            try:
                total_sum += float(line.strip())
            except ValueError:
                print(f"Invalid data found in file: {line.strip()}")
    
    print(f"The total sum is: {total_sum}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")