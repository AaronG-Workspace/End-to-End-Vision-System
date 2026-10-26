# Challenges

Corrections from Tom during the build: what Claude proposed, what Tom caught, what changed.

- 2026-10-25, data.py: Claude proposed checking only the per-folder image counts. Tom caught that the data came from a third-party mirror, so matching counts do not prove these are the original files. data.py now asserts every image is 900x900 before hashing it.
