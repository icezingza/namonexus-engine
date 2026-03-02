import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from verify_headers import verify_file, REQUIRED_HEADER

@pytest.fixture
def temp_python_file():
    """Creates a temporary python file and cleans it up after test."""
    fd, path = tempfile.mkstemp(suffix=".py", text=True)
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_verify_file_valid(temp_python_file):
    content = f"{REQUIRED_HEADER}\n\ndef main():\n    pass\n"
    with open(temp_python_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    assert verify_file(temp_python_file) is True

def test_verify_file_missing_header(temp_python_file):
    content = "def main():\n    pass\n"
    with open(temp_python_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Should fail and print error
    assert verify_file(temp_python_file) is False

def test_verify_file_syntax_error(temp_python_file):
    # Header is present, but code is broken
    content = f"{REQUIRED_HEADER}\n\ndef main()  # Missing colon\n    pass\n"
    with open(temp_python_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Should fail due to syntax error
    assert verify_file(temp_python_file) is False