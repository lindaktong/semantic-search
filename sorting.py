import json

try:
    # Read the data from a text file
    with open('allusers.txt', 'r') as file:
        data = file.read()

    # Convert the string to a dictionary
    data_dict = json.loads(data)

    # Extract the list of users
    users_list = data_dict.get('users', [])

    # Sort the list of dictionaries by 'id'
    sorted_users = sorted(users_list, key=lambda x: x['id'])

    # Update the dictionary with the sorted list
    data_dict['users'] = sorted_users

    # Write the sorted data to a text file
    # with open('sorted.txt', 'w') as file:
    #     file.write(json.dumps(data_dict, indent=2))

    print("Sorted data written to sorted.txt")

except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Extract all the ids
ids = [user['id'] for user in sorted_users]

# Print the list of ids
print(ids)
print(len(ids))