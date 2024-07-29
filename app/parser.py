
import re

def parse_states_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip the first three lines which are headers
    lines = lines[3:]
    data = []

    state_regex = re.compile(r'([A-Za-z ]+?)\s{2,}([A-Z]{2})\s{2,}(\d{2})')

    for line in lines:
        line = line.strip()
        parts = line.split()

        if len(parts) >= 6:  # Line with two states
            state1_data = " ".join(parts[:3])
            state2_data = " ".join(parts[3:])

            match1 = state_regex.match(state1_data)
            match2 = state_regex.match(state2_data)
            
            if match1 and match2:
                state1, postal1, fips1 = match1.groups()
                state2, postal2, fips2 = match2.groups()
                
                data.append({"State": state1.strip(), "Abbr. Postal": postal1, "FIPS Code": fips1})
                data.append({"State": state2.strip(), "Abbr. Postal": postal2, "FIPS Code": fips2})
            else:
                print(f"Unexpected format in line: {line}")
        
        elif len(parts) >= 3:  # Line with one state
            state_data = " ".join(parts)
            match = state_regex.match(state_data)
            if match:
                state, postal, fips = match.groups()
                data.append({"State": state.strip(), "Abbr. Postal": postal, "FIPS Code": fips})
            else:
                print(f"Unexpected format in line: {line}")
        
        else:
            print(f"Unexpected format in line: {line}")

    return data

