# Data files

`source/flair_photophysical_property_dataset.csv` is the main photophysical-property dataset. `source/ae_pre_holdout_candidate_pool.csv` is the additional candidate pool used when constructing the reliability calibration/held-out split.

`splits/deployment/deployment.csv` is the deployment dataset and contains the complete fixed five-fold assignment in `cv_fold`. For Fold *k*, rows with `cv_fold != k` are training data and rows with `cv_fold == k` are validation data. `splits/deployment/deployment_test.csv` is its fixed test dataset and is shared across all five folds.

The main-model deployment partition uses the two files in `splits/deployment/`. Step 2 also reads the candidate pool in `source/`; reliability training and prediction require generated feature tables under `../results/intermediate/`. The alternative Random/Scaffold partitions in `splits/development/` support additional analyses. Split-generation scripts produce scaffold maps, selection tables, and assignment summaries.

The held-out data form an internal fixed holdout, not a completely independent external dataset. See `../docs/data_dictionary.csv` for definitions and units.

`../spec_solv.csv` is the held-out-solvent benchmark after PBS exclusion: 144 records across 12 solvent identities. Of the 165 reserved records reported in the manuscript, 21 PBS (phosphate-buffered saline) records were excluded because the current model requires a single solvent SMILES and PBS is a buffered mixture. Preserve this CSV as a separate evaluation dataset; do not use its inherited `split` labels to add records to main-model training or Reliability Calibration85/Test15.

`source/negative_solvatochromism_38_compounds.csv` contains 38 compounds and 76 paired DCM/TOL records. Its columns are `Compound_ID`, `SMILES`, `Solvent`, `Solvent_SMILES`, `Exp_Abs`, `Exp_Emi`, and `DOI`. `Compound_ID` follows the supplied compound order from 1 to 38; both solvent records share the same ID. Experimental absorption and emission wavelengths are in nm. `DOI` identifies the source article for each compound. All headers and text entries are in English. Use this cohort for separate evaluation, not for main-model training or Reliability calibration.

The source assignments are listed below. For IDs 1-22, the compound labels are indexed in [Telegin et al. (2023), Table 1 and references](https://doi.org/10.3390/ijms24021217); the CSV cites the corresponding original articles.

| Compound_ID | Source article DOI |
|---|---|
| 1, 2 | [10.1016/j.dyepig.2018.05.009](https://doi.org/10.1016/j.dyepig.2018.05.009) |
| 3, 7 | [10.1039/c5tc02954f](https://doi.org/10.1039/c5tc02954f) |
| 4, 8, 12, 13, 15, 18 | [10.1021/jp305551w](https://doi.org/10.1021/jp305551w) |
| 5 | [10.1039/c0cp01147a](https://doi.org/10.1039/c0cp01147a) |
| 6, 22 | [10.1021/acssensors.0c02275](https://doi.org/10.1021/acssensors.0c02275) |
| 9, 10, 17, 20 | [10.1039/c9dt02309g](https://doi.org/10.1039/c9dt02309g) |
| 11, 14, 16 | [10.1039/c5ra17419h](https://doi.org/10.1039/c5ra17419h) |
| 19 | [10.1016/j.optmat.2018.08.007](https://doi.org/10.1016/j.optmat.2018.08.007) |
| 21 | [10.1002/slct.201701897](https://doi.org/10.1002/slct.201701897) |
| 23 | [10.3762/bjoc.21.37](https://doi.org/10.3762/bjoc.21.37) |
| 24 | [10.1021/acs.jpcc.3c04543](https://doi.org/10.1021/acs.jpcc.3c04543) |
| 25, 26, 27, 28, 29 | [10.1002/chem.202203464](https://doi.org/10.1002/chem.202203464) |
| 30, 31, 32, 33, 34, 35, 36, 37, 38 | [10.1039/d4tc04850d](https://doi.org/10.1039/d4tc04850d) |

Rows include literature provenance fields such as `reference(doi)`. Before public release, the authors should confirm that the applicable licenses and journal policy permit redistribution of each source dataset and derived table.

## Generated reliability inputs

This distribution omits `results/`. Step 2 generates Calibration85/Test15 splits, calibration/test features, and test labels there. These Reliability splits differ from the fixed main-model deployment folds. Step 3 also requires the matching historical `test15_evaluated_rows.csv` comparison table from the separate Colab artifacts. It cannot run from the source CSVs alone.

The default predictor rebuilds deployment caches under `cache/` after calibration features are available. No deployment cache is bundled.
