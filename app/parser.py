# app/parser.py
def parse_states_data(file_path):
    states_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines[2:]:  # Skip the first two header lines
        parts = line.split()
        if len(parts) == 6:
            states_data.append({
                'state_1': parts[0],
                'postal_1': parts[1],
                'fips_1': parts[2],
                'state_2': parts[3],
                'postal_2': parts[4],
                'fips_2': parts[5],
            })
    return states_data

if __name__ == '__main__':
    data = parse_states_data('states_data.txt')
    for entry in data:
        print(entry)
