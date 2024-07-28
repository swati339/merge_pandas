# tests/test_app.py
import requests
from bs4 import BeautifulSoup
from app.__init__ import main

def test_main(monkeypatch):
    class MockResponse:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

    def mock_get(*args, **kwargs):
        return MockResponse('<html><head><title>Mock Title</title></head><body></body></html>')

    monkeypatch.setattr(requests, 'get', mock_get)

    main()
