# flake8: noqa
# pyright: ignore


# The hard-coded font cache is broken in the pyodide matplotlib package.
# This works around the issue, saving at least a second on startup time.


font_cache = """
{
    "_version": 330,
    "_FontManager__default_weight": "normal",
    "default_size": null,
    "defaultFamily": {"ttf": "DejaVu Sans", "afm": "Helvetica"},
    "afmlist": [],
    "ttflist": [
        {
            "fname": "fonts/ttf/STIXSizFourSymBol.ttf",
            "name": "STIXSizeFourSym",
            "style": "normal",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSansDisplay.ttf",
            "name": "DejaVu Sans Display",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/cmex10.ttf",
            "name": "cmex10",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXGeneralBolIta.ttf",
            "name": "STIXGeneral",
            "style": "italic",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/cmmi10.ttf",
            "name": "cmmi10",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/cmss10.ttf",
            "name": "cmss10",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/cmsy10.ttf",
            "name": "cmsy10",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/cmtt10.ttf",
            "name": "cmtt10",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSans.ttf",
            "name": "DejaVu Sans",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXNonUniBolIta.ttf",
            "name": "STIXNonUnicode",
            "style": "italic",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSerif-BoldItalic.ttf",
            "name": "DejaVu Serif",
            "style": "italic",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXSizThreeSymBol.ttf",
            "name": "STIXSizeThreeSym",
            "style": "normal",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSansMono.ttf",
            "name": "DejaVu Sans Mono",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXSizOneSymBol.ttf",
            "name": "STIXSizeOneSym",
            "style": "normal",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSans-BoldOblique.ttf",
            "name": "DejaVu Sans",
            "style": "oblique",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXSizOneSymReg.ttf",
            "name": "STIXSizeOneSym",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXGeneral.ttf",
            "name": "STIXGeneral",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/cmb10.ttf",
            "name": "cmb10",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/cmr10.ttf",
            "name": "cmr10",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXNonUni.ttf",
            "name": "STIXNonUnicode",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXGeneralItalic.ttf",
            "name": "STIXGeneral",
            "style": "italic",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSans-Oblique.ttf",
            "name": "DejaVu Sans",
            "style": "oblique",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSerifDisplay.ttf",
            "name": "DejaVu Serif Display",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXNonUniIta.ttf",
            "name": "STIXNonUnicode",
            "style": "italic",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSerif-Italic.ttf",
            "name": "DejaVu Serif",
            "style": "italic",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSerif-Bold.ttf",
            "name": "DejaVu Serif",
            "style": "normal",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXSizFourSymReg.ttf",
            "name": "STIXSizeFourSym",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXNonUniBol.ttf",
            "name": "STIXNonUnicode",
            "style": "normal",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/Humor-Sans.ttf",
            "name": "Humor Sans",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSerif.ttf",
            "name": "DejaVu Serif",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXSizFiveSymReg.ttf",
            "name": "STIXSizeFiveSym",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXGeneralBol.ttf",
            "name": "STIXGeneral",
            "style": "normal",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXSizTwoSymBol.ttf",
            "name": "STIXSizeTwoSym",
            "style": "normal",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXSizTwoSymReg.ttf",
            "name": "STIXSizeTwoSym",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSansMono-BoldOblique.ttf",
            "name": "DejaVu Sans Mono",
            "style": "oblique",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/STIXSizThreeSymReg.ttf",
            "name": "STIXSizeThreeSym",
            "style": "normal",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSansMono-Bold.ttf",
            "name": "DejaVu Sans Mono",
            "style": "normal",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSansMono-Oblique.ttf",
            "name": "DejaVu Sans Mono",
            "style": "oblique",
            "variant": "normal",
            "weight": 400,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        },
        {
            "fname": "fonts/ttf/DejaVuSans-Bold.ttf",
            "name": "DejaVu Sans",
            "style": "normal",
            "variant": "normal",
            "weight": 700,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry"
        }
    ],
    "__class__": "FontManager"
}
"""


with open("/lib/python3.12/site-packages/matplotlib/fontlist.json", "w") as f:
    f.write(font_cache)


# Imports need to happen *after* the file workaround above.


import micropip


await micropip.install("bench_runner=={{bench_runner_version}}", deps=False)


import io


from bench_runner import plot
from bench_runner import result


def plot_diff(base_url, base_data, head_url, head_data):
    """
    Plot the diff between two benchmark results.
    """

    # Convert the JSON data to Result objects

    base_result = result.Result.from_online_json(base_url, base_data)
    head_result = result.Result.from_online_json(head_url, head_data)

    compare = result.BenchmarkComparison(base_result, head_result, "base")
    output = io.StringIO()
    plot.plot_diff(compare.get_timing_diff(), output, "timings", ("slower", "faster"))

    return output.getvalue()


plot_diff
