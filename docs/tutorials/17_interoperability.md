# 17 · Interoperability: TB2J in, VAMPIRE / UppASD / Spirit / mumax³ out

## Use TB2J results with every magex engine

```bash
magex import-tb2j path/to/TB2J_results --out fe_tb2j.json      # reads TB2J.pickle (or exchange.out)
magex tc fe_tb2j.json                                         # MF / RPA / quantum estimates
```

and then `results = "fe_tb2j.json"` in any `[spinmodel]` (LLG, S(q,ω), GNEB, Monte Carlo) or
`magex params fe_tb2j.json` for micromagnetics.  Isotropic J, DMI vectors and the symmetric
anisotropic exchange are imported (eV → meV); TB2J uses the same Hamiltonian convention as magex.
The pickle and the text parser give identical models (checked on bcc Fe and SrMnO3).

## Export to other codes

```bash
magex export results.json --to vampire --rcut 8 --outdir vampire_input   # vampire.UCF, vampire.mat, input
magex export results.json --to uppasd  --outdir asd_input                # posfile, momfile, jfile, dmfile
magex export results.json --to spirit  --outdir spirit_input             # input.cfg + couplings.txt
magex export device.toml  --to mumax3  --outdir device.mx3               # micromagnetic script
```

| target | content | convention |
|--------|---------|-----------|
| VAMPIRE | tensorial UCF (J), materials (μ_s, uniaxial K, initial direction), Curie-temperature input | J_VAMPIRE = 2 J_magex, joules; identical (10⁻⁶) to TB2J's own VAMPIRE export for the same couplings |
| UppASD | posfile, momfile, isotropic jfile and DMI dmfile in mRy, inpsd.dat template | LKAG convention (same as magex) |
| Spirit | unique pairs with J and DMI (meV), anisotropy, geometry | J, D ×2 (pairs counted once); symmetric anisotropic exchange is dropped with a warning |
| mumax³ | grid, geometry (rectangle/disk/ellipse), material, field, initial state, relax/minimise/run | first region only; stages without a mumax³ equivalent become comments |
