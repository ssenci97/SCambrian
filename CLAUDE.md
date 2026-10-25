# CLAUDE.md

##################################################################### CODE REQUIREMENTS
- One code box for each script, starting with a shebang.
- No comments. The only exceptions are brief Google-style docstrings: one short docstring at the top of each Python module and one on each function.
- Type hints on every function: all arguments and the return type.
- Split code into sections, and mark each with a banner of exactly this form (no other comment-like separators): `###################################### <section-name>`
- Put all dependencies and constants at the top, sorted.
- Put hardcoded configuration and parameters at the top of the script, right after the dependencies.
- Use `SEED = 42` wherever a seed applies.
- Keep the code compact, with almost no blank lines, without going overboard.
- Don't add file saving to a script that doesn't already save files.
- Use `tqdm` for long iterations.
- Give every script two arguments that work on `configs/param_configs.yaml`:
  - `--yaml-in`: read the parameters from the YAML file.
  - `--yaml-out`: find or create the YAML file and update it with the parameters, nested under the script name.
- When editing a script that already exists, end the message with a short plain-text changelog.

##################################################################### LOGGING
- Start every log line with a tag such as `[PYTHON-INFO]` or `[SLURM-INFO]`.
- End longer scripts with a one-liner that logs, in human-readable form: `{date} {execution time} {max memory allocation}`.

##################################################################### REPO NOTES
- Every slurm script uses the single conda env `./.env`, which `esmc_config.sh` creates. Add new dependencies there.
- Colours: use `PALETTE` (alternating) for categorical data and the `#76BBE2` → `#7b2c6d` gradient for quantitative data.
