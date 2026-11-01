"""LaTeX -> GitHub Markdown (REPORT.md) via pandoc, after flattening the
constructs pandoc does not understand (adjustbox, custom callout macros)."""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def inline_inputs(text):
    def rep(m):
        if m.group(1).startswith("generated/tab_"):
            return "\n\nMDTABLE" + m.group(1)[len("generated/tab_"):].replace("_", "QQ") + "MDEND\n\n"
        p = os.path.join(HERE, m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))
        return inline_inputs(open(p).read())
    return re.sub(r"\\input\{([^}]+)\}", rep, text)


def replace_macro(text, name, label):
    out, i = [], 0
    key = "\\" + name + "{"
    while True:
        j = text.find(key, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k, depth = j + len(key), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        body = text[j + len(key):k - 1]
        out.append("\n\\begin{quote}\\textbf{%s} %s\\end{quote}\n" % (label, body))
        i = k
    return "".join(out)


def main():
    src = open(os.path.join(HERE, "main.tex")).read()
    src = re.sub(r"\\newcommand\{\\(keyidea|sayit|warn)\}.*\n", "", src)
    src = inline_inputs(src)
    src = re.sub(r"\\begin\{adjustbox\}\{[^}]*\}", "", src).replace("\\end{adjustbox}", "")
    for name, label in [("keyidea", "Key idea."), ("sayit", "Say it in the interview:"), ("warn", "Careful:")]:
        src = replace_macro(src, name, label)
    src = src.replace("\\includegraphics[", "\\includegraphics[").replace("{pipeline_", "{figures/pipeline_")
    src = re.sub(r"\\includegraphics\[[^\]]*\]\{(?!figures/)([^}]+)\}", r"\\includegraphics{figures/\1}", src)
    src = src.replace("\\includegraphics{figures/figures/", "\\includegraphics{figures/")
    tmp = os.path.join(HERE, "_flat.tex")
    open(tmp, "w").write(src)
    subprocess.run(["pandoc", tmp, "-f", "latex", "-t", "gfm", "--toc", "--toc-depth=2", "-s",
                    "-o", os.path.join(HERE, "REPORT.md")], check=True)
    os.remove(tmp)
    md = open(os.path.join(HERE, "REPORT.md")).read()
    md = re.sub(r"MDTABLE(\w+?)MDEND", lambda m: open(os.path.join(HERE, "generated", "tab_" + m.group(1).replace("QQ", "_") + ".md")).read(), md)
    open(os.path.join(HERE, "REPORT.md"), "w").write(md)


if __name__ == "__main__":
    main()
