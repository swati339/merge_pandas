def parse_states_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove headers and filter out empty lines
    lines = [line for line in lines if line.strip()][2:]
    data = []

    for line in lines:
        parts = line.split()
        if len(parts) == 6:
            state1, postal1, fips1, state2, postal2, fips2 = parts
            data.append({"State": state1, "Abbr. Postal": postal1, "FIPS Code": fips1})
            data.append({"State": state2, "Abbr. Postal": postal2, "FIPS Code": fips2})
        elif len(parts) == 3:
            state, postal, fips = parts
            data.append({"State": state, "Abbr. Postal": postal, "FIPS Code": fips})
        else:
            print(f"Unexpected format in line: {line}")

    return data
