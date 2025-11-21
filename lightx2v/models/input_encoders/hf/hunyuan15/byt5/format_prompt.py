import json


def closest_color(requested_color):
    import webcolors

    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def convert_rgb_to_names(rgb_tuple):
    try:
        import webcolors

        color_name = webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        color_name = closest_color(rgb_tuple)
    return color_name


class MultilingualPromptFormat:
    def __init__(
        self,
        font_path: str = "assets/glyph_sdxl_assets/multilingual_10-lang_idx.json",
        color_path: str = "assets/glyph_sdxl_assets/color_idx.json",
    ):
        with open(font_path, "r") as f:
            self.font_dict = json.load(f)
        with open(color_path, "r") as f:
            self.color_dict = json.load(f)

    def format_prompt(self, texts, styles):
        """
        Text "{text}" in {color}, {type}.
        """

        prompt = ""
        for text, style in zip(texts, styles):
            text_prompt = f'Text "{text}"'

            attr_list = []

            # format color
            if style["color"] is not None:
                import webcolors

                hex_color = style["color"]
                rgb_color = webcolors.hex_to_rgb(hex_color)
                color_name = convert_rgb_to_names(rgb_color)
                attr_list.append(f"<color-{self.color_dict[color_name]}>")

            # format font
            if style["font-family"] is not None:
                attr_list.append(f"<{style['font-family'][:2]}-font-{self.font_dict[style['font-family']]}>")
                attr_suffix = ", ".join(attr_list)
                text_prompt += " in " + attr_suffix
                text_prompt += ". "
            else:
                text_prompt += ". "

            prompt = prompt + text_prompt
        return prompt
