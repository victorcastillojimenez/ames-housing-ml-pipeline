import os
import glob

palette_replacements = {
    # Seaborn themes
    'style="darkgrid", palette="muted"': 'style="whitegrid"',
    'style="darkgrid", palette="muted",': 'style="whitegrid",',
    'style="darkgrid", palette="deep"': 'style="whitegrid"',
    
    # Specific colors
    '"#4C72B0"': '"#007B92"',     # Base Teal instead of Deep Blue
    '"#C44E52"': '"#357083"',     # Tertiary instead of Red
    '"#55A868"': '"#3DB9D3"',     # Highlight Cyan instead of Green
    '"#DD8452"': '"#9AA5AE"',     # Neutral instead of Orange
    '"#555"': '"#9AA5AE"',        # Subtitles Neutral instead of dark grey
    'color="blue"': 'color="#007B92"',
    'color="green"': 'color="#3DB9D3"',
    'color="red"': 'color="#3DB9D3"',  # Use star color instead of red
    
    # Palettes
    'palette="Blues"': 'palette=sns.light_palette("#007B92", as_cmap=False)',
    'palette="viridis"': 'palette=["#357083", "#007B92", "#3DB9D3"]',
    'palette="coolwarm"': 'palette=sns.blend_palette(["#357083", "#007B92", "#3DB9D3"], as_cmap=False)',
    'cmap="RdBu_r"': 'cmap=sns.diverging_palette(220, 190, s=80, l=45, as_cmap=True)',
}

files = glob.glob("scripts/*.py") + ["ames_housing_notebook.py", "agents/agent_02_data_scientist.py"]

for f in files:
    if not os.path.exists(f): continue
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
        
    orig = content
    for old, new in palette_replacements.items():
        content = content.replace(old, new)
        
    if orig != content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print("Updated", f)
