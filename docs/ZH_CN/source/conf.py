"""Sphinx configuration for the ZH_CN documentation."""

project = "LightX2V"
copyright = "2026, LightX2V Team"
author = "the LightX2V Team"
language = "zh_CN"

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
    "path_to_docs": "docs/ZH_CN/source",
    "repository_url": "https://github.com/ModelTC/LightX2V",
    "use_repository_button": True,
}
navigation_with_keys = False

epub_identifier = "https://github.com/ModelTC/LightX2V/tree/main/docs/ZH_CN"
epub_scheme = "URL"
