import pytest
from src.tools import ToolRegistry

@pytest.fixture
def tools(tmp_path):
    return ToolRegistry(tmp_path/'support.db')
