python src/foam/opt_graph_ga_fc/run_opt.py \
    --seed 42 \
    --num-workers 64 \
    --device cuda:0 \
    --gpu-workers 8 \
    --batch-size 32 \
    --wandb disable \
    \
    --gen-model-ckpt ../ms-models/iceberg_results_20240630/dag_nist20/split_1_rnd1/version_0/best.ckpt \
    --inten-model-ckpt ../ms-models/iceberg_results_20240630/dag_inten_nist20/split_1_rnd1/version_0/best.ckpt \
    --ignore-precursor-peak \
    --max-nodes 100 \
    \
    --oracle-type Cos_SA_ \
    --criteria entropy \
    --multiobj \
    --eval-names NDSBestMol TopNDSScore NDSParetoRanking NDSFronts InchiKeyMatch \
    \
    --spec-id nist_3253773 \
    --spec-lib-dir path/to/nist_data/nist23/spec_files.hdf5 \
    --spec-lib-label path/to/nist_data/nist23/labels.tsv \
    --seed-lib-dir /home/gridsan/mmanjrekar/coley_lab/foam/data/pubchem/pubchem_formulae_inchikey.hdf5 \
    --max-seed-sim 0.95 \
    \
    --top-k 1 5 10 \
    --max-calls 2000 \
    --keep-population 2000 \
    --population-size 200 \
    --offspring-size 600 \
    --num-islands 1 \
    --threshold 0.0 \
    --starting-seed-size 600 \
    \
    --selection-sorting-type cand_crowding \
    --parent-tiebreak cand_crowding \
    --truncate \
    `--use-clustered-evs \` \
    `--mutate-parents \` \
    `--use-iceberg-spectra `\ \
    \
    --save-dir results/debug_ga_test \
    --tags debug-via-vscode popsize100 parent-tiebreak=cand_crowding selection-sorting-type=cand_crowding truncate \
    