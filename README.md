# Autoplace
This is a kicad plugin to help in the intial layout of a board. It "solves" a component placement by minimizing wirelength while smoothing out density.

This isn't meant to one-shot a component layout, but as a helpful first step to group related components.

**It works best when Edge.Cuts is well defined, and static components (like ports) are locked.**

<img width="2906" height="1192" alt="placement" src="https://github.com/user-attachments/assets/ecb5b47f-14c4-4c67-8778-cb3d45fdacd6" />

## Installation
Symlink `autoplace/` into the plugins folder:

### MacOS
```bash
ln -s "$(pwd)/autoplace" ~/Documents/KiCad/10.0/plugins/autoplace  # KiCad 10
```

### Linux
```bash
ln -s "$(pwd)/autoplace" ~/.local/share/KiCad/10.0/plugins/autoplace  # KiCad 10
```

The idea is based on the DreamPlace algorithm: https://github.com/limbo018/DREAMPlace
