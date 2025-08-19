"""
Basic tests to verify project setup and environment.
These tests ensure the Digital Lending Accelerator project is properly configured.
"""

import os
import sys
import pytest


def test_project_structure():
    """Test that essential project directories exist."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_dirs = ['api', 'data', 'ml_model', 'salesforce', 'tests', 'docs']
    
    for directory in required_dirs:
        dir_path = os.path.join(project_root, directory)
        assert os.path.exists(dir_path), f"Directory {directory} should exist"


def test_requirements_file():
    """Test that requirements.txt exists and contains essential packages."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    requirements_path = os.path.join(project_root, 'requirements.txt')
    
    assert os.path.exists(requirements_path), "requirements.txt should exist"
    
    with open(requirements_path, 'r') as f:
        content = f.read()
        
    essential_packages = ['scikit-learn', 'pandas', 'numpy', 'Flask']
    
    for package in essential_packages:
        assert package in content, f"Package {package} should be in requirements.txt"


def test_python_version():
    """Test that Python version is compatible."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required for this project"


def test_env_template_exists():
    """Test that environment template exists."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_template_path = os.path.join(project_root, '.env.template')
    
    assert os.path.exists(env_template_path), ".env.template should exist"


def test_readme_exists():
    """Test that README.md exists and contains project info."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme_path = os.path.join(project_root, 'README.md')
    
    assert os.path.exists(readme_path), "README.md should exist"
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    assert "Digital Lending Accelerator" in content, "README should contain project name"
    assert "92%" in content, "README should contain accuracy target"
    assert "40%" in content, "README should contain automation target"


def test_import_basic_packages():
    """Test that essential packages can be imported."""
    try:
        import pandas
        import numpy
        import sklearn
        import flask
        assert True, "All essential packages imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import essential package: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
