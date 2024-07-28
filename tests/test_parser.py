# tests/test_parser.py
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
            'state_1': 'Alabama',
            'postal_1': 'AL',
            'fips_1': '01',
            'state_2': 'Nebraska',
            'postal_2': 'NE',
            'fips_2': '31'
        },
        {
            'state_1': 'Alaska',
            'postal_1': 'AK',
            'fips_1': '02',
            'state_2': 'Nevada',
            'postal_2': 'NV',
            'fips_2': '32'
        }
    ]

    result = parse_states_data(test_file_path)
    assert result == expected_output

    os.remove(test_file_path)
