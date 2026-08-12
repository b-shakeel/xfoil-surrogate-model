import subprocess, os

polar_file = "polar_naca0012.txt"

if os.path.exists(polar_file):
    os.remove(polar_file)

commands = "\n".join([
    "NACA 0012",
    "OPER",
    "VISC 200000",
    "MACH 0.1",
    "PACC",
    polar_file,
    "",
    "ASEQ -5 15 0.5",
    "PACC",
    "",
    "QUIT",
]) + "\n"

result = subprocess.run(
    ["xfoil"],
    input=commands,
    capture_output=True,
    text=True,
    timeout=30,
)

with open("xfoil_debug_output.txt", "w") as f:
    f.write(result.stdout)

print("Done. Check polar_naca0012.txt and xfoil_debug_output.txt")