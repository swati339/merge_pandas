import os
import pytest
from app.parser import parse_states_data

def test_parse_states_data():
    test_data = """State    Postal
Abbr.    FIPS
Code    State    Postal
Abbr.    FIPS
Code
Alabama    AL    01    Nebraska    NE    31
Alaska    AK    02    Nevada    NV    32"""
    
    test_file_path = 'test_states_data.txt'
    
    with open(test_file_path, 'w') as test_file:
        test_file.write(test_data)
    
    expected_output = [
        {
            'State': 'Alabama',
            'Abbr. Postal': 'AL',
            'FIPS Code': '01'
        },
        {
            'State': 'Nebraska',
            'Abbr. Postal': 'NE',
            'FIPS Code': '31'
        },
        {
            'State': 'Alaska',
            'Abbr. Postal': 'AK',
            'FIPS Code': '02'
        },
        {
            'State': 'Nevada',
            'Abbr. Postal': 'NV',
            'FIPS Code': '32'
        }
    ]

    result = parse_states_data(test_file_path)
    assert result == expected_output

    os.remove(test_file_path)
