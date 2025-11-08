#!/usr/bin/env python
"""
Demonstration of Advanced Dependency Management Features

This script showcases the new features added to pyproject_updater.py:
1. Vulnerability checking
2. Conflict detection
3. Pre-release policies
4. Impact analysis

Usage:
    python scripts/demo_advanced_features.py
"""

from pathlib import Path

from pyproject_updater import (
    Options,
    check_vulnerabilities,
    detect_conflicts,
    upgrade_with_prerelease_policy,
)


def demo_vulnerability_checking():
    """Demonstrate vulnerability checking."""
    print("\n" + "=" * 70)
    print("DEMO 1: Vulnerability Checking")
    print("=" * 70)

    pyproject = Path("pyproject.toml")
    vulns = check_vulnerabilities(pyproject)

    if vulns:
        print(f"\nFound {len(vulns)} packages with known vulnerabilities:\n")
        for pkg, issues in vulns.items():
            print(f"  Package: {pkg}")
            for issue in issues:
                print(f"    - {issue}")
    else:
        print("\n[OK] No known vulnerabilities detected in current dependencies")

    print(
        "\nNote: This uses a demo vulnerability database. In production,"
    )
    print("      integrate with pip-audit, Safety DB, or OSV API.")


def demo_conflict_detection():
    """Demonstrate dependency conflict detection."""
    print("\n" + "=" * 70)
    print("DEMO 2: Dependency Conflict Detection")
    print("=" * 70)

    pyproject = Path("pyproject.toml")
    conflicts = detect_conflicts(pyproject)

    if conflicts:
        print(f"\nFound {len(conflicts)} potential conflicts:\n")
        for conflict_id, message in conflicts.items():
            print(f"  Conflict: {conflict_id}")
            print(f"    {message}\n")
    else:
        print("\n[OK] No dependency conflicts detected")

    print("Note: Conflicts include both version mismatches across groups")
    print("      and known ecosystem conflicts (e.g., numpy version issues).")


def demo_prerelease_policies():
    """Demonstrate pre-release policy support."""
    print("\n" + "=" * 70)
    print("DEMO 3: Pre-release Policy Support")
    print("=" * 70)

    policies = {
        "conservative": {
            "desc": "No pre-releases, no major version bumps",
            "use_case": "Production environments requiring maximum stability",
        },
        "moderate": {
            "desc": "No pre-releases, allow major version bumps",
            "use_case": "Development with controlled updates",
        },
        "aggressive": {
            "desc": "Allow pre-releases and major version bumps",
            "use_case": "Cutting-edge development, early testing",
        },
        "security_only": {
            "desc": "Only update packages with known vulnerabilities",
            "use_case": "Emergency security patches, minimal change",
        },
    }

    print("\nAvailable update policies:\n")
    for policy_name, info in policies.items():
        print(f"  {policy_name}:")
        print(f"    Description: {info['desc']}")
        print(f"    Use case: {info['use_case']}\n")

    print("Example usage:")
    print('  upgrade_with_prerelease_policy(pyproject, opts, policy="conservative")')


def demo_usage_example():
    """Show complete usage example."""
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE: Security-Only Upgrade")
    print("=" * 70)

    example_code = '''
# Example: Automated security updates in CI/CD

from pathlib import Path
from pyproject_updater import Options, upgrade_with_prerelease_policy

pyproject = Path("pyproject.toml")

# Create options for dry-run check
opts = Options(
    strategy="caret",
    allow_major=False,
    include_prerelease=False,
    groups=["main", "dev"],
    only=None,
    check=True,  # Dry-run mode
    file=pyproject,
    timeout=8.0,
)

# Run security-only upgrade
result = upgrade_with_prerelease_policy(
    pyproject,
    opts,
    policy="security_only"
)

# In production: set check=False to actually write changes
    '''

    print(example_code)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Advanced Dependency Management Features Demo")
    print("=" * 70)

    demo_vulnerability_checking()
    demo_conflict_detection()
    demo_prerelease_policies()
    demo_usage_example()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nFor more information, see scripts/pyproject_updater.py")
    print()


if __name__ == "__main__":
    main()
