#!/usr/bin/env python3
"""
Test script to verify DarkBottomLine installation
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(f"{package_name}.{module_name}")
        else:
            importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False

def main():
    print("Testing DarkBottomLine installation...")
    print("=" * 50)
    
    # Core Python packages
    core_packages = [
        "numpy", "scipy", "matplotlib", "pandas", "yaml"
    ]
    
    # Physics packages
    physics_packages = [
        "awkward", "uproot", "correctionlib"
    ]
    
    # Coffea packages
    coffea_packages = [
        "coffea", "dask", "distributed"
    ]
    
    # ML packages
    ml_packages = [
        "torch", "sklearn"
    ]
    
    # Plotting packages
    plotting_packages = [
        "hist", "plotly"
    ]
    
    # DarkBottomLine modules
    darkbottomline_modules = [
        "objects", "selections", "corrections", "weights", 
        "histograms", "processor", "regions", "analyzer",
        "dnn_trainer", "dnn_inference", "plotting", 
        "combine_tools", "combine_inputs", "cli"
    ]
    
    all_passed = True
    
    print("Core packages:")
    for pkg in core_packages:
        if not test_import(pkg):
            all_passed = False
    
    print("\nPhysics packages:")
    for pkg in physics_packages:
        if not test_import(pkg):
            all_passed = False
    
    print("\nCoffea packages:")
    for pkg in coffea_packages:
        if not test_import(pkg):
            all_passed = False
    
    print("\nMachine learning packages:")
    for pkg in ml_packages:
        if not test_import(pkg):
            all_passed = False
    
    print("\nPlotting packages:")
    for pkg in plotting_packages:
        if not test_import(pkg):
            all_passed = False
    
    print("\nDarkBottomLine modules:")
    for module in darkbottomline_modules:
        if not test_import(module, "darkbottomline"):
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All packages installed successfully!")
        print("\nYou can now run the DarkBottomLine framework.")
        print("To activate the environment: source venv/bin/activate")
        print("To run analysis: darkbottomline --help")
    else:
        print("✗ Some packages failed to install.")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
