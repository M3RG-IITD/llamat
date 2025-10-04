import io
import os
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import py3Dmol
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read as ase_read, write as ase_write

try:
    from .model_client import LocalCompletionClient, ModelConfig
    from .parsing_fn import parse_fn
except ImportError:
    from model_client import LocalCompletionClient, ModelConfig
    from parsing_fn import parse_fn


APP_TITLE = "CIF Generation and Explorer"

# System messages for different task types (from notebook)
SYSTEM_MESSAGES = {
    "conditional_generation": "Apply your skills as a Material Science expert, particularly in managing CIF files, to provide insights into generating stable materials despite incomplete descriptive data.",
    "infill": "Utilize your proficiency in crystallographic file analysis, coupled with your background in Material Science, to respond to questions regarding the prediction of the masked element in a CIF file.",
    "formula_compute": "You are a Material Science expert who works with crystallographic files (CIF files). Use your expertise to find the chemical formula of the crystal whose unit cell is described below.",
    "element_generation": "Employ your expertise in Material Science, particularly in the realm of CIF files, to address inquiries related to the creation of stable materials despite incomplete data.",
    "unconditional": "Apply your skills as a Material Science expert, particularly in managing CIF files, to provide insights into generating stable materials despite incomplete descriptive data."
}

# Input templates for different task types
INPUT_TEMPLATES = {
    "conditional_generation": "Below is a description of a bulk material. {description} Generate a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice.\nThe output should be of the following format ONLY:\nl1, l2, l3\na1, a2, a3\natom1\nx, y, z\natom2\nx, y, z\n ...\n\nl1, l2, l3 should be the predicted cell lengths.\na1, a2, a3 should be the predicted cell angles.\natom1, atom2, atom3, and so on, should be replaced with atom names and corresponding x, y, z with their coordinates in the lattice.\n",
    "infill": "Below is a partial description of a bulk material where one element has been replaced with the string \"[MASK]\":\n{structure}\nGenerate an element that could replace [MASK] in the bulk material:\n",
    "formula_compute": "Below is a description dimensions, as well as of the element type and coordinates for each atom within the lattice of a stable crystal:\n{structure}\nFind the chemical formula of the stable crystal which has the above unit cell.\n",
    "element_generation": "Consider the following elements:\n{elements}\nYou need to create a stable crystal that contains at least one instance of each of these elements. It should not contain elements other than the specified ones. Generate a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice:\nThe output should be of the following format ONLY:\nl1, l2, l3\na1, a2, a3\natom1\nx, y, z\natom2\nx, y, z\n ...\n\nl1, l2, l3 should be the predicted cell lengths.\na1, a2, a3 should be the predicted cell angles.\natom1, atom2, atom3, and so on, should be replaced with atom names and corresponding x, y, z with their coordinates in the lattice.\n",
    "unconditional": "Below is a description of a bulk material. Generate a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice.\nThe output should be of the following format ONLY:\nl1, l2, l3\na1, a2, a3\natom1\nx, y, z\natom2\nx, y, z\n ...\n\nl1, l2, l3 should be the predicted cell lengths.\na1, a2, a3 should be the predicted cell angles.\natom1, atom2, atom3, and so on, should be replaced with atom names and corresponding x, y, z with their coordinates in the lattice.\n"
}

def build_prompt_from_notebook_format(task_type: str, **kwargs) -> str:
    """
    Build prompt using the exact format from the notebook: {system} input-{input} output-
    """
    system = SYSTEM_MESSAGES.get(task_type, SYSTEM_MESSAGES["unconditional"])
    
    if task_type == "conditional_generation":
        description = kwargs.get("description", "")
        input_text = INPUT_TEMPLATES[task_type].format(description=description)
    elif task_type == "infill":
        structure = kwargs.get("structure", "")
        input_text = INPUT_TEMPLATES[task_type].format(structure=structure)
    elif task_type == "formula_compute":
        structure = kwargs.get("structure", "")
        input_text = INPUT_TEMPLATES[task_type].format(structure=structure)
    elif task_type == "element_generation":
        elements = kwargs.get("elements", "")
        input_text = INPUT_TEMPLATES[task_type].format(elements=elements)
    else:  # unconditional
        input_text = INPUT_TEMPLATES[task_type]
    
    return f"{system} input-{input_text} output-"


CONDITIONAL_TEMPLATES = {
    # Updated to match notebook format
    "Composition": "The chemical formula is {composition}. The elements are {elements}.",
    "Composition + Spacegroup": (
        "The chemical formula is {composition}. The elements are {elements}. The spacegroup number is {spacegroup}."
    ),
    "Lattice + Atoms": (
        "The approximate lattice parameters are a={a}, b={b}, c={c}, "
        "alpha={alpha}, beta={beta}, gamma={gamma}. The elements are {atom_list}."
    ),
}


def instruction_from_conditional(
    mode: str,
    composition: str = "",
    spacegroup: str = "",
    a: float = 0.0,
    b: float = 0.0,
    c: float = 0.0,
    alpha: float = 90.0,
    beta: float = 90.0,
    gamma: float = 90.0,
    atom_list: str = "",
) -> str:
    # Extract elements from composition for the elements field
    elements = ""
    if composition:
        # Simple element extraction - could be improved
        import re
        element_pattern = r'[A-Z][a-z]?'
        elements = ', '.join(re.findall(element_pattern, composition))
    
    if mode == "Composition":
        return CONDITIONAL_TEMPLATES[mode].format(composition=composition, elements=elements)
    if mode == "Composition + Spacegroup":
        return CONDITIONAL_TEMPLATES[mode].format(composition=composition, elements=elements, spacegroup=spacegroup)
    if mode == "Lattice + Atoms":
        return CONDITIONAL_TEMPLATES[mode].format(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            atom_list=atom_list,
        )
    return composition


def build_ase_from_parsed(
    lengths: List[float],
    angles: List[float],
    species: List[str],
    coords: List[List[float]],
) -> Atoms:
    if not (len(lengths) == 3 and len(angles) == 3 and len(species) == len(coords) and len(species) > 0):
        raise ValueError("Parsed structure incomplete for ASE build")

    cell = cellpar_to_cell([lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2]])
    atoms = Atoms(symbols=species, cell=cell, pbc=True)
    atoms.set_scaled_positions(np.array(coords))
    return atoms


def atoms_to_cif_string(atoms: Atoms) -> str:
    # Use BytesIO to avoid text/binary issues in ASE's CIF writer
    bbuf = io.BytesIO()
    ase_write(bbuf, atoms, format="cif")
    cif_bytes = bbuf.getvalue()
    if isinstance(cif_bytes, str):
        return cif_bytes
    return cif_bytes.decode("utf-8", errors="ignore")


def extract_elements_from_cif(cif_str: str) -> List[str]:
    """Extract unique element symbols from CIF string"""
    elements = set()
    lines = cif_str.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('_atom_site_type_symbol'):
            # This is the header line, elements will be in subsequent lines
            continue
        elif line and not line.startswith('_') and not line.startswith('#'):
            # This might be a data line with element information
            parts = line.split()
            if len(parts) > 0:
                # First column is usually the element symbol
                element = parts[0].strip('0123456789+-')  # Remove numbers and charges
                if element and element.isalpha():
                    elements.add(element)
    return sorted(list(elements))


def create_atom_legend(elements: List[str], colorscheme: str = "Jmol") -> str:
    """Create an HTML legend showing element symbols and their colors"""
    if not elements:
        return ""
    
    # Define color mappings for different schemes
    color_mappings = {
        "Jmol": {
            "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", 
            "F": "#90E050", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F",
            "Br": "#A62929", "I": "#940094", "Li": "#CC80FF", "Na": "#AB5CF2",
            "K": "#8F40D4", "Rb": "#8F40D4", "Cs": "#8F40D4", "Be": "#C2FF00",
            "Mg": "#8AFF00", "Ca": "#3DFF00", "Sr": "#00FF00", "Ba": "#00C900",
            "Al": "#BFA6A6", "Si": "#F0C8A0", "Ti": "#BFC2C7", "V": "#A6A6AB",
            "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0",
            "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F",
            "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Zr": "#94E0E0",
            "Mo": "#54B5B5", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573",
            "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00", "W": "#2194D6",
            "Au": "#FFD123", "Hg": "#B8B8D0", "Pb": "#575961", "Bi": "#9E4FB5"
        },
        "default": {
            "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
            "F": "#90E050", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F"
        }
    }
    
    colors = color_mappings.get(colorscheme, color_mappings["Jmol"])
    
    legend_html = """
    <div style="position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.9); 
                padding: 10px; border-radius: 5px; border: 1px solid #ccc; font-family: Arial, sans-serif; 
                font-size: 12px; z-index: 1000;">
        <strong>Elements:</strong><br>
    """
    
    for element in elements:
        color = colors.get(element, "#CCCCCC")  # Default gray if element not found
        legend_html += f'<span style="color: {color}; font-weight: bold;">●</span> {element}<br>'
    
    legend_html += "</div>"
    return legend_html


def show_structure_py3dmol(
    cif_str: str,
    width: int = 700,
    height: int = 500,
    style: str = "stick",
    colorscheme: str = "Jmol",
    show_unit_cell: bool = True,
    show_labels: bool = False,
    show_legend: bool = True,
    background: str = "0xFFFFFF",
    spin: bool = False,
):
    v = py3Dmol.view(width=width, height=height)
    v.setBackgroundColor(background)
    v.addModel(cif_str, "cif")
    style_map = {
        "stick": {"stick": {"colorscheme": colorscheme}},
        "ball&stick": {"stick": {"colorscheme": colorscheme}, "sphere": {"scale": 0.25, "colorscheme": colorscheme}},
        "sphere": {"sphere": {"scale": 0.35, "colorscheme": colorscheme}},
        "line": {"line": {"linewidth": 2}},
    }
    v.setStyle(style_map.get(style, style_map["stick"]))
    if show_unit_cell:
        v.addUnitCell()
    # Add element labels if requested (best-effort)
    if show_labels:
        try:
            # Method 1: Try addLabels with element property
            v.addLabels(
                {"model": 0},  # Select all atoms in model 0
                {"fontSize": 10, "fontColor": "black", "showBackground": True, "backgroundColor": "white"}
            )
        except Exception:
            try:
                # Method 2: Try addPropertyLabels with correct parameter order
                v.addPropertyLabels(
                    "elem",  # Property to display (element symbol)
                    {"model": 0},  # Which atoms to label
                    {"fontSize": 10, "fontColor": "black", "showBackground": True, "backgroundColor": "white"}
                )
            except Exception:
                try:
                    # Method 3: Try addLabels with different parameter structure
                    v.addLabels(
                        {"model": 0, "elem": True},  # Select atoms and specify element property
                        {"fontSize": 10, "fontColor": "black", "showBackground": True, "backgroundColor": "white"}
                    )
                except Exception:
                    # If all methods fail, silently continue without labels
                    pass
    if spin:
        v.spin(True)
    v.zoomTo()
    html = v._make_html()
    
    # Add legend if requested
    if show_legend:
        elements = extract_elements_from_cif(cif_str)
        legend_html = create_atom_legend(elements, colorscheme)
        # Combine the 3D visualization with the legend
        combined_html = f"""
        <div style="position: relative; width: {width}px; height: {height}px;">
            {html}
            {legend_html}
        </div>
        """
        st.components.v1.html(combined_html, height=height, scrolling=False)
    else:
        st.components.v1.html(html, height=height, scrolling=False)


def generate_with_model(client: LocalCompletionClient, instruction: str, task_type: str = "conditional_generation") -> str:
    # Use the notebook format for prompt construction
    prompt = build_prompt_from_notebook_format(task_type, description=instruction)
    return client.generate(prompt)

def generate_with_model_debug(client: LocalCompletionClient, instruction: str, task_type: str = "conditional_generation") -> tuple[str, str]:
    # Use the notebook format for prompt construction with debug info
    prompt = build_prompt_from_notebook_format(task_type, description=instruction)
    result = client.generate(prompt)
    return result, prompt


def parse_and_visualize(gen_str: str, key_prefix: str = ""):
    lengths, angles, species, coords = parse_fn(gen_str or "")
    st.write({"lengths": lengths, "angles": angles, "species": species, "coords": coords})
    if len(lengths) == 3 and len(angles) == 3 and len(species) > 0 and len(species) == len(coords):
        try:
            atoms = build_ase_from_parsed(lengths, angles, species, coords)
            cif_str = atoms_to_cif_string(atoms)
            # Persist latest conditional generation only (unconditional handled at caller-level)
            if key_prefix.startswith("cond_"):
                st.session_state["cond_last_gen_str"] = gen_str
                st.session_state["cond_last_cif_str"] = cif_str
            with st.expander("Visualization settings", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    style = st.selectbox("Style", ["stick", "ball&stick", "sphere", "line"], index=0, key=f"{key_prefix}style")
                    show_unit_cell = st.checkbox("Show unit cell", value=True, key=f"{key_prefix}uc")
                with col2:
                    colorscheme = st.selectbox("Colors", ["Jmol", "default", "chain", "cyanCarbon", "greenCarbon"], index=0, key=f"{key_prefix}colors")
                    # show_labels = st.checkbox("Show element labels", value=False)
                with col3:
                    background = st.text_input("Background (hex or 0xRGB)", value="0xFFFFFF", key=f"{key_prefix}bg")
                    show_legend = st.checkbox("Show element legend", value=True, key=f"{key_prefix}legend")
                    spin = st.checkbox("Spin", value=False, key=f"{key_prefix}spin")
            show_structure_py3dmol(
                cif_str,
                style=style,
                colorscheme=colorscheme,
                show_unit_cell=show_unit_cell,
                # show_labels=show_labels,
                show_legend=show_legend,
                background=background,
                spin=spin,
            )
            st.download_button("Download CIF", data=cif_str, file_name="generated_structure.cif", mime="chemical/x-cif", key=f"{key_prefix}download")
        except Exception as e:
            st.warning(f"Could not visualize structure: {e}")
    else:
        st.warning("Parsed output incomplete; cannot visualize.")


def page_generate(client: LocalCompletionClient):
    # Task type selection
    task_type = st.selectbox(
        "Task Type", 
        ["conditional_generation", "unconditional", "infill", "formula_compute", "element_generation"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    # Debug option
    show_debug = st.checkbox("Show prompt debug info", value=False)
    
    if task_type == "unconditional":
        num_samples = st.number_input("Number of samples", min_value=1, max_value=50, value=3)
        base_instruction = st.text_area(
            "Instruction (optional)",
            value="Generate a plausible crystal structure for an inorganic material.",
            help="You may leave this empty for fully unconditional generation.",
        )
        if st.button("Generate"):
            gens: List[str] = []
            for idx in range(num_samples):
                instruction = base_instruction.strip() or "Generate a plausible crystal structure for an inorganic material."
                if show_debug:
                    gen_str, prompt = generate_with_model_debug(client, instruction, task_type)
                    st.subheader(f"Sample {idx+1} - Prompt Sent to Model")
                    st.code(prompt, language="text")
                else:
                    gen_str = generate_with_model(client, instruction, task_type)
                gens.append(gen_str)
            # Persist all generated samples for re-rendering across reruns
            st.session_state["uncond_last_gens"] = gens
        # Render any previously generated samples (including from this run)
        if st.session_state.get("uncond_last_gens"):
            for i, g in enumerate(st.session_state["uncond_last_gens"], 1):
                st.subheader(f"Sample {i}")
                st.code(g)
                parse_and_visualize(g, key_prefix=f"uncond_vis_{i}_")

    elif task_type == "conditional_generation":
        cond_mode = st.selectbox("Condition Type", list(CONDITIONAL_TEMPLATES.keys()))
        composition = ""
        spacegroup = ""
        a = b = c = 0.0
        alpha = beta = gamma = 90.0
        atom_list = ""

        if cond_mode in ("Composition", "Composition + Spacegroup"):
            composition = st.text_input("Composition (e.g., AlN or LiFePO4)")
        if cond_mode == "Composition + Spacegroup":
            spacegroup = st.text_input("Space Group (e.g., Fm-3m)")
        if cond_mode == "Lattice + Atoms":
            a = st.number_input("a", min_value=0.1, value=4.2)
            b = st.number_input("b", min_value=0.1, value=4.2)
            c = st.number_input("c", min_value=0.1, value=6.7)
            alpha = st.number_input("alpha", min_value=1.0, max_value=179.0, value=90.0)
            beta = st.number_input("beta", min_value=1.0, max_value=179.0, value=90.0)
            gamma = st.number_input("gamma", min_value=1.0, max_value=179.0, value=120.0)
            atom_list = st.text_input("Atoms list (e.g., Al, N, O)")

        if st.button("Generate"):
            instruction = instruction_from_conditional(
                cond_mode, composition, spacegroup, a, b, c, alpha, beta, gamma, atom_list
            )
            if show_debug:
                gen_str, prompt = generate_with_model_debug(client, instruction, task_type)
                st.subheader("Prompt Sent to Model")
                st.code(prompt, language="text")
            else:
                gen_str = generate_with_model(client, instruction, task_type)
            st.subheader("Raw Generation")
            st.code(gen_str)
            # Persist the latest generation so visualization persists across reruns
            st.session_state["cond_last_gen_str"] = gen_str
        # If a previous generation exists, render it so that visualization controls persist across reruns
        if st.session_state.get("cond_last_gen_str"):
            parse_and_visualize(st.session_state["cond_last_gen_str"], key_prefix="cond_vis_")
    
    elif task_type == "infill":
        st.write("**Infill Task**: Predict the masked element in a crystal structure")
        structure = st.text_area(
            "Crystal Structure (with [MASK] for unknown elements)",
            value="5.4 7.7 11.4\n75 90 90\nGd\n0.60 0.59 0.97\n[MASK]\n0.82 0.24 0.29\n[MASK]\n0.32 0.14 0.32",
            height=200
        )
        if st.button("Generate"):
            if show_debug:
                gen_str, prompt = generate_with_model_debug(client, structure, task_type)
                st.subheader("Prompt Sent to Model")
                st.code(prompt, language="text")
            else:
                gen_str = generate_with_model(client, structure, task_type)
            st.subheader("Predicted Element")
            st.code(gen_str)
    
    elif task_type == "formula_compute":
        st.write("**Formula Compute Task**: Find the chemical formula from crystal structure")
        structure = st.text_area(
            "Crystal Structure",
            value="5.4 7.7 11.4\n75 90 90\nGd\n0.60 0.59 0.97\nGd\n0.97 0.59 0.47\nTm\n0.54 0.09 0.97\nO\n0.82 0.24 0.29\nO\n0.32 0.14 0.32",
            height=200
        )
        if st.button("Generate"):
            if show_debug:
                gen_str, prompt = generate_with_model_debug(client, structure, task_type)
                st.subheader("Prompt Sent to Model")
                st.code(prompt, language="text")
            else:
                gen_str = generate_with_model(client, structure, task_type)
            st.subheader("Chemical Formula")
            st.code(gen_str)
    
    elif task_type == "element_generation":
        st.write("**Element Generation Task**: Generate crystal structure from given elements")
        elements = st.text_input(
            "Elements (comma-separated)", 
            value="Gd, Tm, Lu, W, O",
            help="Enter elements separated by commas"
        )
        if st.button("Generate"):
            if show_debug:
                gen_str, prompt = generate_with_model_debug(client, elements, task_type)
                st.subheader("Prompt Sent to Model")
                st.code(prompt, language="text")
            else:
                gen_str = generate_with_model(client, elements, task_type)
            st.subheader("Generated Structure")
            st.code(gen_str)
            parse_and_visualize(gen_str)


def page_explore(cif_root: str):
    st.write("Explore local CIF examples and visualize them.")
    if not os.path.isdir(cif_root):
        st.warning(f"CIF directory not found: {cif_root}")
        return
    # List CIF files recursively under example/
    candidates: List[str] = []
    for root, _, files in os.walk(cif_root):
        for f in files:
            if f.lower().endswith(".cif"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        st.warning("No CIF files found.")
        return
    selected = st.selectbox("Select CIF file", candidates)
    if selected:
        try:
            atoms = ase_read(selected)
            # ase_read can return a list; take the first structure
            if isinstance(atoms, list):
                atoms = atoms[0]
            cif_str = atoms_to_cif_string(atoms)
            if isinstance(cif_str, bytes):
                cif_str = cif_str.decode("utf-8", errors="ignore")
            with st.expander("Visualization settings", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    style = st.selectbox("Style", ["stick", "ball&stick", "sphere", "line"], index=0, key="exp_style")
                    show_unit_cell = st.checkbox("Show unit cell", value=True, key="exp_uc")
                with col2:
                    colorscheme = st.selectbox("Colors", ["Jmol", "default", "chain", "cyanCarbon", "greenCarbon"], index=0, key="exp_colors")
                    # show_labels = st.checkbox("Show element labels", value=False, key="exp_labels")
                with col3:
                    background = st.text_input("Background (hex or 0xRGB)", value="0xFFFFFF", key="exp_bg")
                    show_legend = st.checkbox("Show element legend", value=True, key="exp_legend")
                    spin = st.checkbox("Spin", value=False, key="exp_spin")
            show_structure_py3dmol(
                cif_str,
                style=style,
                colorscheme=colorscheme,
                show_unit_cell=show_unit_cell,
                # show_labels=show_labels,
                show_legend=show_legend,
                background=background,
                spin=spin,
            )
            st.download_button("Download CIF", data=cif_str, file_name=os.path.basename(selected), mime="chemical/x-cif")
        except Exception as e:
            st.error(f"Failed to load/visualize CIF: {e}")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    
    st.info("""
    **Updated to use the exact prompt format from the notebook!** 
    
    This app now uses the same prompt construction as `cif_condtional_gen_zero_shot.ipynb`:
    - Format: `{system} input-{input} output-`
    - System messages match the notebook exactly
    - Supports all task types: conditional generation, infill, formula compute, element generation, and unconditional
    - Enable "Show prompt debug info" to see the exact prompts being sent to the model
    """)

    # Sidebar: model configuration
    st.sidebar.header("Model Settings")
    endpoint = st.sidebar.text_input("Endpoint", value="http://localhost:8000/v1/completions")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="model_path",
    )
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    max_tokens = st.sidebar.number_input("Max tokens", min_value=64, max_value=4096, value=1024, step=64)
    timeout_s = st.sidebar.number_input("Timeout (s)", min_value=5, max_value=300, value=60, step=5)

    client = LocalCompletionClient(
        ModelConfig(
            model_endpoint=endpoint,
            model_name=model_path,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            timeout_s=int(timeout_s),
        )
    )

    tab = st.tabs(["Generate CIFs", "Explore CIFs"])  # type: ignore
    with tab[0]:
        page_generate(client)

    with tab[1]:
        # Expected CIF root: crystal-text-llm/cif_db/example
        base_dir = os.path.dirname(__file__)
        cif_root = os.path.join(base_dir, "cif_db", "example")
        page_explore(cif_root)


if __name__ == "__main__":
    main()



