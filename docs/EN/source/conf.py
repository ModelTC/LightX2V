"""Sphinx configuration for the EN documentation."""

import ast
from pathlib import Path

# Read the package version without importing GPU dependencies.
package_init = Path(__file__).resolve().parents[3] / "lightx2v" / "__init__.py"
version = next(
    ast.literal_eval(node.value)
    for node in ast.parse(package_init.read_text(encoding="utf-8")).body
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
)
release = version

project = "LightX2V"
copyright = "2026, LightX2V Team"
author = "the LightX2V Team"
language = "en"

extensions = ["sphinx_copybutton", "sphinx.ext.mathjax", "myst_parser"]
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 4
exclude_patterns = ["**/*.template.rst"]

copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

html_title = project
html_theme = "sphinx_book_theme"
html_logo = "../../../assets/img_lightx2v.png"
html_theme_options = {
    "path_to_docs": "docs/EN/source",
    "repository_url": "https://github.com/ModelTC/LightX2V",
    "use_repository_button": True,
}
navigation_with_keys = False

epub_identifier = "https://github.com/ModelTC/LightX2V/tree/main/docs/EN"
epub_scheme = "URL"
