# Compliance and licensing notes

## Third-party Python packages

The project depends on the following libraries (see `pyproject.toml` / `requirements.txt`):

| Package | License |
| --- | --- |
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| nibabel | MIT |
| numba | BSD-2-Clause |
| hnswlib | Apache-2.0 |
| faiss-cpu | MIT |

To regenerate this table:

```powershell
pip install pip-licenses
pip-licenses --from=mixed --format=markdown > THIRD_PARTY_LICENSES.md
```

Keep the generated `THIRD_PARTY_LICENSES.md` alongside releases as part of your compliance documentation.

## Dataset usage

- **BrainWeb**: governed by the BrainWeb terms of use. Data must be downloaded directly from the provider; do not redistribute.
- **IXI**: governed by the IXI dataset licence. Use `scripts/download_ixi_dataset.ps1` to obtain copies on demand; never commit the volumes.
- **Synthetic phantoms**: generated at runtime and safe to include in public artifacts.

Document any new datasets in `data/real_dataset_manifest.md`, including the licence and acquisition steps. Keep raw MRI volumes, DICOM files, and other PHI outside of version control.

## Security reporting

If you discover a vulnerability, email `jorgeruizwilliams@gmail.com` with "SECURITY" in the subject. Please avoid filing public issues until a fix is available.
