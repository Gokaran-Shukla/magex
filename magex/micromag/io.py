"""OVF 2.0 reader/writer (OOMMF / mumax compatible) and table output."""
from __future__ import annotations

import numpy as np


def write_ovf(path, m, mesh, title="m", binary=True, desc=None):
    """Write a vector field (nx,ny,nz,3) as OVF 2.0 rectangular mesh ('Binary 8' or 'Text')."""
    nx, ny, nz = mesh.n
    dx, dy, dz = mesh.d
    hdr = ["# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment", "# Begin: Header",
           f"# Title: {title}", "# meshtype: rectangular", "# meshunit: m",
           "# xmin: 0", "# ymin: 0", "# zmin: 0",
           f"# xmax: {nx * dx:.10g}", f"# ymax: {ny * dy:.10g}", f"# zmax: {nz * dz:.10g}",
           "# valuedim: 3", "# valuelabels: m_x m_y m_z", "# valueunits: 1 1 1"]
    for line in (desc or "").splitlines():
        hdr.append(f"# Desc: {line}")
    hdr += [f"# xbase: {dx / 2:.10g}", f"# ybase: {dy / 2:.10g}", f"# zbase: {dz / 2:.10g}",
            f"# xnodes: {nx}", f"# ynodes: {ny}", f"# znodes: {nz}",
            f"# xstepsize: {dx:.10g}", f"# ystepsize: {dy:.10g}", f"# zstepsize: {dz:.10g}", "# End: Header"]
    data = np.ascontiguousarray(np.transpose(m, (2, 1, 0, 3)).reshape(-1, 3))   # x fastest
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode())
        if binary:
            f.write(b"# Begin: Data Binary 8\n")
            f.write(np.array([123456789012345.0], "<f8").tobytes())
            f.write(data.astype("<f8").tobytes())
            f.write(b"\n# End: Data Binary 8\n")
        else:
            f.write(b"# Begin: Data Text\n")
            np.savetxt(f, data, fmt="%.8g")
            f.write(b"# End: Data Text\n")
        f.write(b"# End: Segment\n")


def read_ovf(path):
    """Read OVF 1/2 text or binary 4/8 files. Returns (m (nx,ny,nz,3), header dict)."""
    raw = open(path, "rb").read()
    hdr = {}
    pos = 0
    while True:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode(errors="replace").strip()
        pos = end + 1
        if line.lower().startswith("# begin: data"):
            kind = line.split(":", 1)[1].strip().lower()
            break
        if ":" in line:
            k, v = line[1:].split(":", 1)
            hdr[k.strip().lower()] = v.strip()
    nx, ny, nz = (int(hdr[k]) for k in ("xnodes", "ynodes", "znodes"))
    dim = int(hdr.get("valuedim", 3))
    n = nx * ny * nz * dim
    if "binary 8" in kind:
        arr = np.frombuffer(raw, "<f8", count=n + 1, offset=pos)[1:]
    elif "binary 4" in kind:
        arr = np.frombuffer(raw, "<f4", count=n + 1, offset=pos)[1:].astype(float)
    else:
        txt = raw[pos:].decode().split("#")[0]
        arr = np.array(txt.split(), float)[:n]
    m = arr.reshape(nz, ny, nx, dim).transpose(2, 1, 0, 3).copy()
    return m, hdr


def write_table(path, rows, header):
    np.savetxt(path, np.asarray(rows), fmt="%.10g", header="\t".join(header), comments="# ", delimiter="\t")
