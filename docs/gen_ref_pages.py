"""Generate API reference pages automatically."""

from pathlib import Path

import mkdocs_gen_files

# Modules to document
modules = [
    ("heavytails.heavy_tails", "Core Distributions"),
    ("heavytails.extra_distributions", "Extra Distributions"),
    ("heavytails.discrete", "Discrete Distributions"),
    ("heavytails.tail_index", "Tail Estimators"),
    ("heavytails.plotting", "Plotting Utilities"),
    ("heavytails.utilities", "Utilities"),
]

# Generate reference pages
for module_path, title in modules:
    # Create markdown file path
    filename = module_path.split(".")[-1]
    doc_path = Path("reference") / f"{filename}.md"

    # Write the reference page
    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"::: {module_path}\n")
        f.write("    options:\n")
        f.write("      show_root_heading: true\n")
        f.write("      show_source: true\n")
        f.write("      heading_level: 2\n")

    # Update navigation
    mkdocs_gen_files.set_edit_path(doc_path, Path("..") / "heavytails" / f"{filename}.py")
