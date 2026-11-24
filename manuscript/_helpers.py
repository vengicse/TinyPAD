def initials(given):
    parts = re.split(r"[\s\-]+", given.strip())
    return " ".join(p[0] + "." for p in parts if p)


NAME_OVERRIDE = {  # names whose Crossref given/family split does not follow Western order
    ("Alimov Abdulboriy Abdulkhay", "ugli"): "Alimov Abdulboriy Abdulkhay ugli",
    ("Nasreen Fathima", "A H"): "Nasreen Fathima A H",
    ("Syed Ibrahim", "S P"): "Syed Ibrahim S P",
    ("Alfredo", "Jr. Gomez"): "A. Gomez Jr.",
    ("H.", "Vincent Poor"): "H. V. Poor",
}


def fmt_ref(doi):
    v = REFV[doi]
    au = [NAME_OVERRIDE.get((g, f), f"{initials(g)} {f}".strip()) for g, f in v["authors"]]
    if len(au) > 6:
        a = ", ".join(au[:6]) + ", et al."
    elif len(au) > 1:
        a = ", ".join(au[:-1]) + ", and " + au[-1]
    else:
        a = au[0]
    title = re.sub(r"<[^>]+>", "", v["title"]).replace("&amp;", "&")
    cont = v["container"].replace("&amp;", "&")
    s = f"{a}, “{title},” *{cont}*"
    if v.get("volume"):
        s += f", vol. {v['volume']}"
    if v.get("issue"):
        s += f", no. {v['issue']}"
    pg = v.get("page") or v.get("article_number")
    if pg:
        s += f", {'pp.' if '-' in str(pg) else 'Art. no.'} {str(pg).replace('-', '–')}"
    s += f", {v['year']}, doi: {doi}."
    return s


def number_citations(text):
    order = []

    def rep(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in KEY2DOI and k not in MANUAL:
                raise KeyError("unknown citation " + k)
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        nums = sorted(nums)
        # compress runs: 1,2,3 -> 1–3
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
            i = j + 1
        return "[" + ", ".join(out) + "]"

    text = re.sub(r"\[(@[^\]]+)\]", rep, text)
    refs = []
    for i, k in enumerate(order, 1):
        refs.append(f"{i}. " + (fmt_ref(KEY2DOI[k]) if k in KEY2DOI else MANUAL[k]))
    return text, refs, order



# ------------------------------------------------------------------ table helpers
def best_idx(vals, higher=True):
    v = np.array(vals, dtype=float)
    return int(np.nanargmax(v) if higher else np.nanargmin(v))


def pipe_table(header, rows, caption, align=None, widths=None):
    align = align or ["l"] + ["c"] * (len(header) - 1)
    widths = list(widths or [10] * len(header))    # relative column widths (dash counts)
    if header[0] == "Dataset":
        widths[0] = max(widths[0], 14)
    if len(header) > 1 and header[1] in ("Method", "Variant"):
        widths[1] = max(widths[1], 18)
    sep = lambda al, w: {"l": ":" + "-" * w, "c": ":" + "-" * w + ":", "r": "-" * w + ":"}[al]
    s = "| " + " | ".join(header) + " |\n|" + "|".join(sep(x, w) for x, w in zip(align, widths)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s + f"\nTable: {caption}\n"


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "–"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"



def fill(text, V):
    def rep(m):
        k = m.group(1)
        if k not in V:
            raise KeyError("missing value " + k)
        return V[k]
    return re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", rep, text)


# ------------------------------------------------------------------ docx post-processing
def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders"); tcPr.append(borders)
    for edge, val in kwargs.items():
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}"); borders.append(el)
        for k, v in val.items():
            el.set(qn(f"w:{k}"), str(v))


def postprocess(path):
    d = Document(path)
    for st in ["Normal", "Body Text", "First Paragraph", "Compact"]:
        if st in [s.name for s in d.styles]:
            s = d.styles[st]; s.font.name = "Times New Roman"; s.font.size = Pt(11)
            s.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    for st in d.styles:
        if st.name.startswith("Heading") or st.name in ("Title", "Subtitle", "Table Caption", "Image Caption", "Caption"):
            st.font.name = "Times New Roman"; st.font.color.rgb = RGBColor(0, 0, 0)
    for p in d.paragraphs:
        if p.style.name in ("Body Text", "First Paragraph"):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for t in d.tables:
        t.alignment = 1
        nrows = len(t.rows)
        for i, row in enumerate(t.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0)
                    for r in p.runs:
                        r.font.size = Pt(8); r.font.name = "Times New Roman"
                        if i == 0:
                            r.font.bold = True
                b = {"val": "single", "sz": 8, "color": "000000"}
                set_cell_border(cell, top=b if i == 0 else {"val": "nil"}, bottom=b if i in (0, nrows - 1) else {"val": "nil"})
    d.save(path)


