# Local data

Downloaded and workstation data files are ignored by Git. Run
`python scripts/build_artifacts.py` from the project root to obtain the housing
archive and rebuild training artifacts. The script reads the CSV directly from
the archive and does not need to extract the local `housing/` directory.
