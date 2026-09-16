# Packaged processed data

These compact files are frozen preprocessing inputs for the benchmark. The panel-building scripts use them to reconstruct `data/derived/` without requiring the historical research-project directory. Full raw GEO archives are not included; optional download URLs are listed in `configs/datasets/download_manifest.json`.

| File | Dataset | SHA-256 |
|---|---|---|
| `GSE140802/gse140802_in_vivo_t2_t9.npz` | GSE140802 | `d1a32c25c9e887f912b55f7bf596622c58973fb31ecd40d65eb528219560fed9` |
| `GSE140802/gse140802_in_vivo_t2_t16.npz` | GSE140802 | `f7918686fb75fa3cabb5d76f570a15d9bbbc7eda6b7b208d8acd2a280be399cc` |
| `GSE239651/gse239651_lineage_transition_records.csv` | GSE239651 | `8860687b83c954204836bc2ca008296d50dbc5c7b07a7f014159293d5a07f7bb` |
| `GSE239651/macsgestalt_locked_transition_records.csv` | macsGESTALT | `49ea89a65924f83560fcc68744683c56458b11fb1769d1ca3d75b0644d700e17` |
| `GSE239651/gse239651_expt1_lineage_panel.parquet` | GSE239651, experiment 1 | `87a7f8331a241966ba2e5dbeb2854fc75eb1d77d50df8e9be8cd0d6c1cd5c85c` |
| `GSE239651/gse239651_expt2_lineage_panel.parquet` | GSE239651, experiment 2 | `c95beebc5e6d783e7a3463e0f3475dad394b64caff2049650b52458bd2206c58` |

The benchmark package license applies to code only. Each underlying dataset remains subject to its original source terms and should be cited using the accession and source publication recorded by GEO or the dataset authors.
