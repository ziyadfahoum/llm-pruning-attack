import re
from pathlib import Path

import pandas as pd


def escape_latex(s):
    if isinstance(s, str):
        return s.replace("%", r"\%").replace("_", r"\_")
    elif isinstance(s, tuple):
        return tuple(escape_latex(val) for val in s)
    return s


def to_latex(df: pd.DataFrame, outpath: Path) -> None:
    df = df.map(escape_latex)
    df.index = df.index.map(escape_latex)
    df.columns = df.columns.map(escape_latex)
    num_id_col = len(df.index[0]) if isinstance(df.index, pd.MultiIndex) else len(df.index)
    latex_string = df.to_latex(
        index=True,
        multicolumn=True,
        multirow=True,
        float_format="%.1f",
        escape=False,
        na_rep="",
        column_format="l" * num_id_col + "c" * len(df.columns),
    )

    # post processing
    # \multicolumn{N}{r} -> \multicolumn{N}{c}
    latex_string = re.sub(r"\\multicolumn\{(\d+)\}\{r\}", r"\\multicolumn{\1}{c}", latex_string)
    # cline -> cmidrule
    latex_string = latex_string.replace(r"\cline", r"\cmidrule")
    # remove final cline before bottomrule
    final_rule_pattern = r"\\cmidrule\{[^}]*\}\n\\bottomrule"
    latex_string = re.sub(final_rule_pattern, r"\\bottomrule", latex_string)
    # \multirow[t] -> \multirow
    latex_string = latex_string.replace(r"\multirow[t]", r"\multirow")

    with open(outpath, "w") as f:
        f.write(latex_string)

    print(f"Wrote LaTeX table to {outpath}")
    return latex_string
