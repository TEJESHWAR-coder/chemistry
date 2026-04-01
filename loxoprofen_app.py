import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import pandas as pd
from io import BytesIO
import base64

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loxoprofen · Molecular Explorer",
    page_icon="🧪",
    layout="wide",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d1117; }

    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 700; color: #e6edf3; margin: 0 0 .4rem 0; }
    .hero p  { color: #8b949e; font-size: 1.05rem; margin: 0; }
    .hero .badge {
        display: inline-block;
        background: #238636;
        color: #aff5b4;
        font-size: .75rem;
        font-weight: 600;
        padding: .2rem .6rem;
        border-radius: 20px;
        margin-right: .4rem;
        margin-top: .5rem;
    }
    .badge-blue  { background: #1f6feb !important; color: #a5d6ff !important; }
    .badge-purple{ background: #6e40c9 !important; color: #d2a8ff !important; }
    .badge-red   { background: #b91c1c !important; color: #fca5a5 !important; }

    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card h3 { color: #e6edf3; font-size: 1.05rem; font-weight: 600; margin: 0 0 .8rem 0; }

    .smiles-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: .8rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: .85rem;
        color: #79c0ff;
        word-break: break-all;
        margin: .4rem 0;
    }
    .inchi-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: .8rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: .78rem;
        color: #a5d6ff;
        word-break: break-all;
        margin: .4rem 0;
    }

    .prop-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: .8rem;
    }
    .prop-item {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: .7rem .9rem;
    }
    .prop-label { font-size: .72rem; color: #8b949e; text-transform: uppercase; letter-spacing: .05em; }
    .prop-value { font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-top: .2rem; }

    .stereo-table th { color: #8b949e !important; font-size: .8rem; }
    .stereo-table td { color: #e6edf3 !important; }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 1.5rem 0 .8rem 0;
        padding-bottom: .4rem;
        border-bottom: 1px solid #21262d;
    }

    .rs-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .rs-symbol {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
    }
    .rs-r { color: #79c0ff; }
    .rs-s { color: #ffa657; }
    .rs-label { font-size: .8rem; color: #8b949e; margin-top: .3rem; }
    .rs-desc  { font-size: .85rem; color: #c9d1d9; margin-top: .5rem; }
</style>
""", unsafe_allow_html=True)

# ─── Data ─────────────────────────────────────────────────────────────────────
MOLECULES = {
    "Loxoprofen (free acid)": {
        "smiles": "CC(C(O)=O)c1ccc(CC2CCCC2=O)cc1",
        "cas":    "68767-14-6",
        "formula": "C₁₅H₁₈O₃",
        "mw":    "246.30 g/mol",
        "inchi":  "InChI=1S/C15H18O3/c1-10(15(17)18)12-7-5-11(6-8-12)9-13-3-2-4-14(13)16/h5-8,10,13H,2-4,9H2,1H3,(H,17,18)",
        "inchikey": "YMBXTVYHTMGZDW-UHFFFAOYSA-N",
        "stereo_center": "C-2 (α-methyl)",
        "note": "Racemic prodrug — both R and S enantiomers present.",
        "badges": ["NSAID", "Prodrug", "COX-1/2 Inhibitor"],
        "badge_colors": ["", "badge-blue", "badge-purple"],
    },
    "Loxoprofen Sodium": {
        "smiles": "[Na+].CC(C([O-])=O)c1ccc(CC2CCCC2=O)cc1",
        "cas":    "80382-23-6",
        "formula": "C₁₅H₁₇NaO₃",
        "mw":    "268.28 g/mol",
        "inchi":  "InChI=1S/C15H18O3.Na/c1-10(15(17)18)12-7-5-11(6-8-12)9-13-3-2-4-14(13)16;/h5-8,10,13H,2-4,9H2,1H3,(H,17,18);/q;+1/p-1",
        "inchikey": "WORCCYVLMMTGFR-UHFFFAOYSA-M",
        "stereo_center": "C-2 (α-methyl, chiral)",
        "note": "Marketed sodium salt (Loxonin®). Racemic mixture.",
        "badges": ["Salt", "Marketed", "Loxonin®"],
        "badge_colors": ["badge-purple", "badge-blue", ""],
    },
    "Loxoprofen Sodium Dihydrate": {
        "smiles": "O.O.[Na+].CC(C([O-])=O)c1ccc(CC2CCCC2=O)cc1",
        "cas":    "226721-96-6",
        "formula": "C₁₅H₂₁NaO₅",
        "mw":    "304.32 g/mol",
        "inchi":  "InChI=1S/C15H18O3.Na.2H2O/c1-10(15(17)18)12-7-5-11(6-8-12)9-13-3-2-4-14(13)16;;;/h5-8,10,13H,2-4,9H2,1H3,(H,17,18);;2*1H2/q;+1;;/p-1",
        "inchikey": "BAZQYVYVKYOAGO-UHFFFAOYSA-M",
        "stereo_center": "C-2 (α-methyl, chiral)",
        "note": "Dihydrate pharmaceutical form used in oral tablets.",
        "badges": ["Dihydrate", "Oral Tablet Form"],
        "badge_colors": ["badge-red", "badge-blue"],
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def mol_to_svg(smiles: str, width: int = 500, height: int = 380) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "<p style='color:red'>Invalid SMILES</p>"
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().padding = 0.15
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # Dark background compatible
    svg = svg.replace("white", "transparent").replace("#FFFFFF", "transparent")
    return svg

def compute_props(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "Mol. Weight":     f"{Descriptors.MolWt(mol):.2f}",
        "Exact Mass":      f"{Descriptors.ExactMolWt(mol):.4f}",
        "LogP (cLogP)":    f"{Descriptors.MolLogP(mol):.2f}",
        "H-Bond Donors":   str(rdMolDescriptors.CalcNumHBD(mol)),
        "H-Bond Acceptors":str(rdMolDescriptors.CalcNumHBA(mol)),
        "TPSA (Å²)":       f"{Descriptors.TPSA(mol):.1f}",
        "Rotatable Bonds": str(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Ring Count":      str(rdMolDescriptors.CalcNumRings(mol)),
        "Aromatic Rings":  str(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "Heavy Atoms":     str(mol.GetNumHeavyAtoms()),
    }

def get_stereo_info(smiles: str) -> list[dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    rows = []
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            cip = atom.GetPropsAsDict().get("_CIPCode", "?")
            rows.append({
                "Atom Index": atom.GetIdx(),
                "Element": atom.GetSymbol(),
                "CIP Code": cip,
                "Chiral Tag": str(atom.GetChiralTag()).split(".")[-1],
            })
    return rows

# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧪 Loxoprofen · Molecular Explorer</h1>
  <p>Interactive visualization of SMILES strings, InChI identifiers, RS stereochemistry,
     and computed physicochemical properties for loxoprofen and its pharmaceutical forms.</p>
  <span class="badge">NSAID</span>
  <span class="badge badge-blue">Propionic Acid Derivative</span>
  <span class="badge badge-purple">COX-1 / COX-2 Inhibitor</span>
  <span class="badge badge-red">Racemic Prodrug</span>
</div>
""", unsafe_allow_html=True)

# ─── Molecule Selector ────────────────────────────────────────────────────────
selected = st.selectbox(
    "**Select molecular form:**",
    list(MOLECULES.keys()),
    index=0,
)
data = MOLECULES[selected]

# ─── Layout: 2 columns ────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown('<div class="section-title">2D Structure</div>', unsafe_allow_html=True)
    svg = mol_to_svg(data["smiles"], width=520, height=360)
    st.markdown(
        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;'
        f'padding:1rem;text-align:center">{svg}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Rendered with RDKit · stereo annotations enabled")

with col_right:
    # Identifiers
    st.markdown('<div class="section-title">Chemical Identifiers</div>', unsafe_allow_html=True)

    badges_html = "".join(
        f'<span class="badge {c}">{b}</span>'
        for b, c in zip(data["badges"], data["badge_colors"])
    )

    st.markdown(f"""
    <div class="card">
      <h3>{selected}</h3>
      {badges_html}
      <div style="margin-top:.9rem">
        <div style="color:#8b949e;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em">CAS Number</div>
        <div class="smiles-box">{data["cas"]}</div>
        <div style="color:#8b949e;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;margin-top:.5rem">Molecular Formula</div>
        <div class="smiles-box">{data["formula"]} · MW {data["mw"]}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**SMILES**")
    st.code(data["smiles"], language="text")

    st.markdown("**InChI**")
    st.code(data["inchi"], language="text")

    st.markdown("**InChIKey**")
    st.code(data["inchikey"], language="text")

# ─── RS Stereochemistry ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">RS Stereocentre Analysis</div>', unsafe_allow_html=True)

stereo_rows = get_stereo_info(data["smiles"])
rs_col1, rs_col2, rs_col3 = st.columns([1.2, 1, 1.8])

with rs_col1:
    st.markdown(f"""
    <div class="rs-card">
      <div class="rs-symbol rs-r">R</div>
      <div class="rs-label">( R )-Loxoprofen</div>
      <div class="rs-desc">Inactive enantiomer in its native state.<br>
      Contributes to pharmacokinetic profile.</div>
    </div>
    """, unsafe_allow_html=True)

with rs_col2:
    st.markdown(f"""
    <div class="rs-card">
      <div class="rs-symbol rs-s">S</div>
      <div class="rs-label">( S )-Loxoprofen</div>
      <div class="rs-desc">Metabolically converted to the active<br>
      trans-OH metabolite (SRS form).</div>
    </div>
    """, unsafe_allow_html=True)

with rs_col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Chiral Centre Details**")
    st.markdown(f"""
    <div class="prop-item" style="margin-bottom:.6rem">
      <div class="prop-label">Stereocentre</div>
      <div class="prop-value" style="font-size:.9rem">{data["stereo_center"]}</div>
    </div>
    <div class="prop-item">
      <div class="prop-label">Note</div>
      <div class="prop-value" style="font-size:.85rem;font-weight:400">{data["note"]}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if stereo_rows:
    st.markdown("**RDKit-detected stereocentres:**")
    df_stereo = pd.DataFrame(stereo_rows)
    st.dataframe(df_stereo, use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ No explicit stereocentres encoded in this SMILES (racemic / unspecified). "
            "Loxoprofen is marketed as a racemate — both R and S forms are present.")

# ─── Physicochemical Properties ───────────────────────────────────────────────
st.markdown('<div class="section-title">Computed Physicochemical Properties</div>', unsafe_allow_html=True)

props = compute_props(data["smiles"])
cols = st.columns(5)
for i, (label, value) in enumerate(props.items()):
    with cols[i % 5]:
        st.markdown(f"""
        <div class="prop-item">
          <div class="prop-label">{label}</div>
          <div class="prop-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# Lipinski Rule of 5
st.markdown("")
mw_val   = float(props["Mol. Weight"])
logp_val = float(props["LogP (cLogP)"])
hbd      = int(props["H-Bond Donors"])
hba      = int(props["H-Bond Acceptors"])

lip_pass = (mw_val <= 500) and (logp_val <= 5) and (hbd <= 5) and (hba <= 10)
lip_color = "#238636" if lip_pass else "#b91c1c"
lip_text  = "✅ Passes Lipinski Rule of Five" if lip_pass else "❌ Violates Lipinski Rule of Five"

st.markdown(f"""
<div style="background:{lip_color}22;border:1px solid {lip_color};border-radius:8px;
            padding:.7rem 1.1rem;margin-top:.5rem;color:#e6edf3;font-size:.9rem">
  <strong>{lip_text}</strong> &nbsp;|&nbsp;
  MW {mw_val:.1f} ≤ 500 &nbsp;·&nbsp;
  LogP {logp_val:.2f} ≤ 5 &nbsp;·&nbsp;
  HBD {hbd} ≤ 5 &nbsp;·&nbsp;
  HBA {hba} ≤ 10
</div>
""", unsafe_allow_html=True)

# ─── SMILES Comparison Table ──────────────────────────────────────────────────
st.markdown('<div class="section-title">All Forms — SMILES &amp; InChIKey Reference</div>', unsafe_allow_html=True)

rows = []
for name, d in MOLECULES.items():
    rows.append({
        "Form": name,
        "CAS": d["cas"],
        "SMILES": d["smiles"],
        "InChIKey": d["inchikey"],
        "Formula": d["formula"],
        "MW": d["mw"],
    })
df_all = pd.DataFrame(rows)
st.dataframe(df_all, use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#21262d;margin:2rem 0 1rem">
<div style="text-align:center;color:#8b949e;font-size:.8rem">
  Data sourced from DrugBank · PubChem (CID 3965) · NCATS Inxight Drugs · 2D rendering via RDKit
</div>
""", unsafe_allow_html=True)
