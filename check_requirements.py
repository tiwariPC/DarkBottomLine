#!/usr/bin/env python3
"""
Script to check if packages from environment.yml are installed.
If not, installs them locally (for use on lxplus).

Creator: ptiwari (main creator)

Created based on instructions to:
- Check if packages from environment.yml are installed
- Install missing packages locally (not in home directory)
- Show summary of installed/missing packages with names and versions
- Suppress pip output during installation
"""

import sys
import re
import subprocess
import argparse
import importlib
import importlib.metadata
from pathlib import Path


def get_package_version(package_name, local_dir=None):
    """Get installed package version, or None if not installed."""
    # Check local directory first if provided
    if local_dir:
        local_dir = Path(local_dir)
        if local_dir.exists():
            local_dir_str = str(local_dir.absolute())
            if local_dir_str not in sys.path:
                sys.path.insert(0, local_dir_str)

            try:
                import_map = {'pyyaml': 'yaml', 'scikit-learn': 'sklearn'}
                module_name = import_map.get(package_name, package_name)
                module = importlib.import_module(module_name)
                if hasattr(module, '__version__'):
                    return module.__version__
                return "installed"
            except ImportError:
                pass

    # Check system installation
    try:
        dist = importlib.metadata.distribution(package_name)
        return dist.version
    except importlib.metadata.PackageNotFoundError:
        try:
            import_map = {'pyyaml': 'yaml', 'scikit-learn': 'sklearn'}
            module_name = import_map.get(package_name, package_name)
            module = importlib.import_module(module_name)
            if hasattr(module, '__version__'):
                return module.__version__
            return "installed"
        except (ImportError, AttributeError):
            return None


def parse_requirements(requirements_path):
    """Parse requirements.txt and return list of (package_name, requirement_line, version_constraint)."""
    requirements = []
    with open(requirements_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract base package name (handle extras like dask[complete])
                pkg_match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)', line)
                if pkg_match:
                    package_spec = pkg_match.group(1)
                    # Extract base package name (remove extras)
                    base_package = re.match(r'^([a-zA-Z0-9_-]+)', package_spec).group(1)
                    # Extract version constraint (everything after package spec)
                    version_part = line[len(package_spec):].strip()
                    requirements.append((base_package, line, version_part))
    return requirements


def _add_requirement(pkg_line, requirements, conda_to_pip, skip_names):
    """Parse a conda/pip package line and append to requirements list."""
    if not pkg_line or pkg_line.startswith('#'):
        return
    pkg_match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)', pkg_line)
    if not pkg_match:
        return
    package_spec = pkg_match.group(1)
    base_package = re.match(r'^([a-zA-Z0-9_-]+)', package_spec).group(1).lower()
    if base_package in skip_names:
        return
    pip_name = conda_to_pip.get(base_package, base_package)
    extras = package_spec[len(base_package):]  # e.g. '[complete]'
    version_part = pkg_line[len(package_spec):].strip()
    # Normalize conda exact-version syntax (=X.Y.Z) to pip syntax (==X.Y.Z)
    version_part = re.sub(r'^=(?!=)', '==', version_part)
    req_line = f"{pip_name}{extras}{version_part}"
    # Replace existing entry for same base package (pip section overrides conda section)
    for i, (name, _, _) in enumerate(requirements):
        if name == pip_name:
            requirements[i] = (pip_name, req_line, version_part)
            return
    requirements.append((pip_name, req_line, version_part))


def parse_environment_yml(env_path):
    """Parse environment.yml and return list of (package_name, requirement_line, version_constraint)."""
    # conda uses different package names than pip for some packages
    conda_to_pip = {'pytorch': 'torch'}
    skip_names = {'python', 'pip'}
    requirements = []

    with open(env_path, 'r') as f:
        lines = f.readlines()

    in_dependencies = False
    in_pip_section = False

    for line in lines:
        stripped = line.rstrip()

        if stripped == 'dependencies:':
            in_dependencies = True
            continue

        if not in_dependencies:
            continue

        # A new top-level key ends the dependencies section
        if stripped and stripped[0] not in (' ', '\t'):
            break

        if stripped.strip() == '- pip:':
            in_pip_section = True
            continue

        if in_pip_section:
            if stripped.startswith('    - '):
                _add_requirement(stripped[6:].strip(), requirements, conda_to_pip, skip_names)
                continue
            elif stripped.startswith('  - '):
                in_pip_section = False
                # fall through to conda handling

        if not in_pip_section and stripped.startswith('  - '):
            pkg_line = stripped[4:].strip()
            if pkg_line != 'pip:':
                _add_requirement(pkg_line, requirements, conda_to_pip, skip_names)

    return requirements

def install_missing_packages(local_dir, missing_lines):
    """Install missing packages locally."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables to force installation to target directory only
    import os
    env = os.environ.copy()
    # Set PYTHONUSERBASE to local_dir to ensure both packages AND scripts go to cwd .local
    # This is better than --target because --target doesn't install scripts to the target
    env['PYTHONUSERBASE'] = str(local_dir.absolute())
    # Unset PYTHONHOME to prevent interference
    env.pop('PYTHONHOME', None)

    # Verify Python and pip are working before attempting installation
    try:
        import subprocess as sp_test
        test_result = sp_test.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if test_result.returncode != 0:
            print(f"Error: pip is not working with {sys.executable}")
            print(f"pip error: {test_result.stderr}")
            print(f"\nTry using a different Python installation or fix your Python environment.")
            return False
    except Exception as e:
        print(f"Error: Cannot verify pip installation: {e}")
        print(f"Python executable: {sys.executable}")
        print(f"\nYour Python installation may be corrupted. Try:")
        print(f"  1. Use a virtual environment: python3 -m venv venv")
        print(f"  2. Use conda/miniforge Python")
        print(f"  3. Reinstall Python")
        return False

    # Install packages individually to ensure only missing packages are installed
    # Using --user with PYTHONUSERBASE ensures both packages and scripts go to cwd .local
    failed_packages = []
    for req_line in missing_lines:
        req_line = req_line.strip()
        if not req_line:
            continue

        # Extract package name for display
        pkg_match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)', req_line)
        pkg_name = pkg_match.group(1) if pkg_match else req_line

        # Install package individually using --user with PYTHONUSERBASE
        # This ensures packages go to local_dir/lib/pythonX.X/site-packages
        # and scripts go to local_dir/bin
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--prefix", str(local_dir.absolute()),  # Use --prefix instead of --user
            "--no-cache-dir",  # Avoid cache issues
            "--upgrade-strategy", "only-if-needed",  # Only upgrade if needed
            req_line,  # Install this specific package (and its dependencies)
            "--quiet",
            "--disable-pip-version-check",
            "--no-warn-script-location"  # Suppress warnings since we control the location
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # Capture stderr to show if there's an error
            env=env  # Use modified environment with PYTHONUSERBASE
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
            print(f"⚠ Failed to install {pkg_name}: {error_msg[:200]}")
            failed_packages.append((pkg_name, req_line))
        else:
            print(f"✓ Installed {pkg_name}")

    if failed_packages:
        print(f"\n✗ Failed to install {len(failed_packages)} package(s):")
        for pkg_name, req_line in failed_packages:
            print(f"  - {pkg_name}")
        return False

    # Add to sys.path
    local_dir_str = str(local_dir.absolute())
    if local_dir_str not in sys.path:
        sys.path.insert(0, local_dir_str)
    return True

def main():
    parser = argparse.ArgumentParser(description="Check and install packages from environment.yml")
    parser.add_argument('--install', action='store_true', help='Install missing packages locally')
    parser.add_argument('--local-dir', default='./.local', help='Local installation directory')
    parser.add_argument('--requirements', default=None, help='Path to environment.yml (default: environment.yml)')
    args = parser.parse_args()

    # Find environment.yml
    script_dir = Path(__file__).parent
    requirements_path = Path(args.requirements) if args.requirements else script_dir / "environment.yml"

    if not requirements_path.exists():
        print(f"Error: environment.yml not found at {requirements_path}")
        sys.exit(1)

    # Check local directory (only if it exists, don't create it unless installing)
    local_dir = Path(args.local_dir)
    local_dir_for_check = None

    if local_dir.exists():
        local_dir_str = str(local_dir.absolute())
        if local_dir_str not in sys.path:
            sys.path.insert(0, local_dir_str)
        local_dir_for_check = local_dir
    elif args.install:
        # Only create directory if we're installing
        local_dir.mkdir(parents=True, exist_ok=True)
        local_dir_str = str(local_dir.absolute())
        if local_dir_str not in sys.path:
            sys.path.insert(0, local_dir_str)
        local_dir_for_check = local_dir

    # Parse and check requirements
    if requirements_path.suffix in ('.yml', '.yaml'):
        print(f"Reading packages from: {requirements_path.name}")
        requirements = parse_environment_yml(requirements_path)
    else:
        print(f"Reading packages from: {requirements_path.name}")
        requirements = parse_requirements(requirements_path)
    missing = []
    installed = []

    for package_name, requirement_line, version_constraint in requirements:
        version = get_package_version(package_name, local_dir_for_check)
        if version:
            installed.append((package_name, version, version_constraint))
        else:
            missing.append((package_name, requirement_line, version_constraint))

    # Show summary
    total = len(requirements)
    installed_count = len(installed)
    missing_count = len(missing)

    print(f"Found {total} packages in {requirements_path.name}")
    print(f"✓ Installed: {installed_count}/{total}")
    print(f"✗ Missing: {missing_count}/{total}")
    print()

    if installed:
        print("Installed packages:")
        for pkg_name, version, version_constraint in installed:
            constraint_info = f" (required: {version_constraint})" if version_constraint else ""
            print(f"  ✓ {pkg_name} (version: {version}){constraint_info}")
        print()

    if missing:
        print("Missing packages:")
        for pkg_name, requirement_line, version_constraint in missing:
            if version_constraint:
                print(f"  ✗ {pkg_name} (will install: {version_constraint})")
            else:
                print(f"  ✗ {pkg_name} (will install: latest)")
        print()

    if not missing:
        print("All packages are installed.")
        return

    if not args.install:
        print(f"To install missing packages, run:")
        print(f"  python {Path(__file__).name} --install")
        return

    print(f"Installing {missing_count} missing packages...")
    missing_lines = [req_line for _, req_line, _ in missing]
    if install_missing_packages(args.local_dir, missing_lines):
        local_packages_path = Path(args.local_dir).absolute()
        print(f"✓ Packages installed to: {local_packages_path}")
        print(f"\nTo use these packages, add to your Python path:")
        print(f"  export PYTHONPATH={local_packages_path}:$PYTHONPATH")
        print(f"\nOr in your Python scripts, add:")
        print(f"  import sys")
        print(f"  sys.path.insert(0, '{local_packages_path}')")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
