# cli/

Command-line interface for LNPBO. Entry point: `lnpbo` (registered in pyproject.toml).

Subcommands:
- `lnpbo suggest` — Suggest the next batch of LNP formulations given existing data
- `lnpbo encode` — Encode a raw CSV into molecular features
- `lnpbo checkpoint` — Save/load optimization state
- `lnpbo propose-ils` — Generate novel ionizable lipid candidates via SELFIES mutation
