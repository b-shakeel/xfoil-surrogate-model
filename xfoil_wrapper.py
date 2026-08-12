import subprocess, re, os, tempfile, shutil
import numpy as np

def _build_commands(airfoil, alpha_start, alpha_end, alpha_step, reynolds, mach, polar_path, n_iter, naca=True):
    """Enters the parameters into xfoil the way you would by hand"""
    if naca:
        airfoil_cmd = f"NACA {airfoil}"
    else:
        airfoil_cmd = f"LOAD {airfoil}"
    return [
        airfoil_cmd,
        "OPER",
        f"VISC {reynolds}",
        f"MACH {mach}",
        f"ITER {n_iter}",
        "PACC",
        polar_path,
        "",
        f"ASEQ {alpha_start} {alpha_end} {alpha_step}",
        "PACC",
        "",
        "QUIT",
    ]

def _call_xfoil(xfoil_path, commands, timeout, cwd=None):
    """Runs xfoil with the given commands"""
    script = "\n".join(commands) + "\n"
    try:
        result = subprocess.run(
            [xfoil_path],
            input = script,
            capture_output = True,
            text = True,
            timeout = timeout,
            cwd = cwd,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"XFOIL did not finish within {timeout}s, likely due to non-converging viscous solve. Try smaller alpha range, lower Re, or longer timeout"
        )
    return result.stdout

def _parse_polar(polar_path):
    """Reads an xfoil polar file and returns alpha, CL, CD, CDp, CM as numpy arrays. skips anything that didn't converge"""
    with open(polar_path, "r",) as f:
        lines = f.readlines()
    data_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*-+\s+-+", line):
            data_start = i + 1
            break
    alpha, CL, CD, CDp, CM = [], [], [], [], []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            a, cl, cd, cdp, cm = [float(x) for x in parts[:5]]
        except ValueError:
            continue
        alpha.append(a)
        CL.append(cl)
        CD.append(cd)
        CDp.append(cdp)
        CM.append(cm)

    if not alpha:
        raise RuntimeError("Polar file was created but contains no converged data points. Try narrower alpha range, lower Re, or higher n_iter")
    
    return {
        "alpha": np.array(alpha),
        "CL": np.array(CL),
        "CD": np.array(CD),
        "CDp": np.array(CDp),
        "CM": np.array(CM),
    }

def run_polar(airfoil, alpha_start, alpha_end, alpha_step, reynolds, mach=0.0, xfoil_path="xfoil", n_iter=100, timeout=60, naca=True):
    """Runs an xfoil alpha sweep for a NACA airfoil and returns the resulting polar as a dict of numpy arrays. Meant to be called outside this file"""
    with tempfile.TemporaryDirectory() as workdir:
        polar_filename = "polar.txt"
        polar_path = os.path.join(workdir, polar_filename)

        if naca:
            airfoil_arg = airfoil
        else:
            local_name = os.path.basename(airfoil)
            shutil.copy(airfoil, os.path.join(workdir, local_name))
            airfoil_arg = local_name
        
        commands = _build_commands(airfoil_arg, alpha_start, alpha_end, alpha_step, reynolds, mach, polar_filename, n_iter, naca=naca)

        _call_xfoil(xfoil_path, commands, timeout, cwd=workdir)

        if not os.path.exists(polar_path):
            raise RuntimeError("XFOIL did not produce polar file. Check airfoil code/file & Re/Mach values are valid.")

        return _parse_polar(polar_path)

if __name__ == "__main__":
    polar = run_polar(airfoil="0012", alpha_start=-5, alpha_end=15, alpha_step=0.5, reynolds=200000, mach=0.1, n_iter=100, naca=True)
    
    print("Rows parsed:", len(polar["alpha"]))
    print("First row:", polar["alpha"][0], polar["CL"][0], polar["CD"][0])
    print("Last row:", polar["alpha"][-1], polar["CL"][-1], polar["CD"][-1])