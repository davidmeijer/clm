# Training Data Flow

This document explains, step by step, how data moves through Harvest from the raw input file to a trained model checkpoint.

It focuses on the training path invoked by:

```bash
harvest train --configfile ...
```

and covers the three supported conditioning modes:

- `none`
- `descriptors`
- `graph`

For the lipopeptide setup in [workflow/config/config_lipopeptides.yml](workflow/config/config_lipopeptides.yml), the active path is currently `graph`.

## 1. Top-level entry point

The CLI entry point is [src/harvest/cli.py](src/harvest/cli.py). The `train` subcommand calls [src/harvest/train.py](src/harvest/train.py), which builds and runs a Snakemake command against:

- [workflow/Snakefile](workflow/Snakefile)
- [workflow/Snakefile_data](workflow/Snakefile_data)

The command-line override:

```bash
--snakemake-args --config 'paths={output_dir: ..., dataset: ...}'
```

replaces values under `paths` in the YAML config before Snakemake expands the workflow.

## 2. Config decides the training mode

The main config used for the lipopeptide run is [workflow/config/config_lipopeptides.yml](workflow/config/config_lipopeptides.yml).

The keys that control the data path are:

- `model_params.conditional.type`
- `paths.dataset`
- `paths.output_dir`
- `paths.preprocess_output`
- `paths.preprocess_output_graph`
- `paths.train_file`
- `paths.train_file_graph`
- `paths.condition_test_file`
- `paths.condition_test_file_graph`
- `paths.condition_vocab_file`
- `paths.model_file`

The workflow resolves the conditioning mode in [_conditioning_type_from_config()](workflow/Snakefile_data) inside [workflow/Snakefile_data](workflow/Snakefile_data). That choice determines:

- which raw input format is expected
- which preprocessed file path is used
- which fold-specific training input path is used
- whether a graph-label vocabulary is built
- which model class is instantiated later

### Input format by mode

`none`

- Input file is a SMILES text/CSV/TSV file.
- Only the `smiles` column is used.

`descriptors`

- Input file is a SMILES table.
- `smiles` is required.
- Numeric columns after `smiles` are treated as conditioning descriptors.

`graph`

- Input file is a JSONL file.
- Each row must contain `smiles` and a graph field.
- Preferred field name is `condition_graph`.
- Legacy `graph` is also accepted and normalized by [src/clm/graph_conditional.py](src/clm/graph_conditional.py).

## 3. Preprocess step

The Snakemake rule is `preprocess` in [workflow/Snakefile_data](workflow/Snakefile_data). It runs:

```bash
clm preprocess ...
```

The implementation is in [src/clm/commands/preprocess.py](src/clm/commands/preprocess.py).

### Non-graph modes

For `none` and `descriptors`, preprocess uses [read_file()](src/clm/functions.py) from [src/clm/functions.py](src/clm/functions.py) to read the tabular input.

It then:

1. parses SMILES with RDKit
2. removes invalid molecules
3. optionally removes salts and solvents
4. optionally neutralizes charges
5. filters on valid atoms
6. canonicalizes SMILES
7. adds `inchikey`
8. optionally removes duplicates
9. optionally removes molecules with very rare SMILES tokens

The output is a CSV-style file at `paths.preprocess_output`.

### Graph mode

For `graph`, preprocess uses [read_graph_condition_file()](src/clm/graph_conditional.py) from [src/clm/graph_conditional.py](src/clm/graph_conditional.py).

Each row is normalized by [load_data_sample()](src/clm/graph_conditional.py) and [normalize_condition_graph()](src/clm/graph_conditional.py), then the SMILES string is cleaned exactly as above.

Important graph-mode normalization rules:

- `graph` is accepted as a legacy alias for `condition_graph`
- empty graphs are replaced by a single `<UNK>` node
- nodes with no identified `name` and no valid `name_candidates` are converted to `name: "<UNK>"`
- `max_input_smiles: 0` means "no limit", not "read zero rows"

The graph-mode output is JSONL at `paths.preprocess_output_graph`. Each surviving row keeps:

- `smiles`
- `inchikey`
- `condition_graph`
- any extra metadata already present in the row

### Why your earlier run failed in `create_training_sets`

The error:

```text
ValueError: You are trying to merge on float64 and object columns for key 'inchikey'
```

was a downstream symptom. The real issue was that graph-mode preprocessing interpreted `max_input_smiles: 0` as zero lines, so it wrote an empty preprocessed JSONL. That left `create_training_sets` trying to build plain `train0/test0` side files from an empty graph dataset.

That behavior is now fixed in [src/clm/graph_conditional.py](src/clm/graph_conditional.py) and [src/clm/functions.py](src/clm/functions.py).

## 4. Create training folds

The Snakemake rule is `create_training_sets` in [workflow/Snakefile_data](workflow/Snakefile_data). It runs:

```bash
clm create_training_sets ...
```

The implementation is in [src/clm/commands/create_training_sets.py](src/clm/commands/create_training_sets.py).

This step is responsible for:

1. reading the preprocessed dataset
2. optionally filtering by `min_tc`
3. splitting into cross-validation folds
4. optionally enumerating randomized SMILES
5. writing the fold-specific train/test files
6. writing the sequence vocabulary
7. writing the graph-condition vocabulary in graph mode

### Non-graph outputs

For `none` and `descriptors`, the important outputs are:

- `train_file`
- `train0_file`
- `test0_file`
- `vocab_file`

Descriptor-conditioned heldout data is also written for conditional sampling.

### Graph outputs

For `graph`, the important outputs are:

- `train_file_graph`
- `condition_test_file_graph`
- `condition_vocab_file`
- `train0_file`
- `test0_file`
- `vocab_file`

The distinction is important:

- `train_file_graph` is the real model input used during training
- `condition_test_file_graph` is the held-out graph-conditioned input used later during conditional sampling
- `train0_file` and `test0_file` are plain SMILES side files kept for downstream evaluation rules

In other words, for graph-conditioned training, the model does not train from `train0_file`. It trains from `train_file_graph`.

### Enumeration behavior in graph mode

If `enum_factor > 0`, [src/clm/commands/create_training_sets.py](src/clm/commands/create_training_sets.py) randomizes only the SMILES string. The attached `condition_graph` is copied unchanged to each enumerated training example.

## 5. Sequence and condition vocabularies

This project uses two different vocabularies in graph mode.

### Sequence vocabulary

The sequence vocabulary is built from the training SMILES in [src/clm/commands/create_training_sets.py](src/clm/commands/create_training_sets.py) via [vocabulary_from_representation()](src/clm/datasets.py), and written to:

- `vocab_file`

This vocabulary is used by the SMILES tokenizer and the recurrent model.

### Graph-condition vocabulary

The graph-label vocabulary is built from training graphs by [build_condition_vocab()](src/clm/graph_conditional.py) and written to:

- `condition_vocab_file`

This vocabulary always includes `<UNK>`. Unknown or unidentified graph nodes map to this label unless they are represented as weighted `name_candidates`.

## 6. Dataset loading for training

The Snakemake rule is `train_models_RNN` in [workflow/Snakefile_data](workflow/Snakefile_data). It runs:

```bash
clm train_models_RNN ...
```

The implementation is in [src/clm/commands/train_models_RNN.py](src/clm/commands/train_models_RNN.py).

That command delegates data loading to [load_dataset()](src/clm/functions.py) in [src/clm/functions.py](src/clm/functions.py).

### Non-graph modes

`load_dataset()` returns:

- [SmilesDataset](src/clm/datasets.py) for `SMILES`
- [SelfiesDataset](src/clm/datasets.py) for `SELFIES`

For descriptor-conditioned training, numeric columns are stored as `descriptors`.

### Graph mode

`load_dataset()` reads the fold JSONL with [read_graph_condition_file()](src/clm/graph_conditional.py) and returns [GraphSequenceDataset](src/clm/datasets.py) from [src/clm/datasets.py](src/clm/datasets.py).

`GraphSequenceDataset` does three things:

1. tokenizes the `smiles` sequence using the sequence vocabulary
2. loads the graph-label vocabulary from `condition_vocab_file`
3. converts each `condition_graph` into tensor form via [graph_to_condition_graph()](src/clm/graph_conditional.py)

At batching time, [ConditionGraphCollate](src/clm/graph_conditional.py) builds a [ConditionGraphBatch](src/clm/graph_conditional.py) object containing packed tensors for:

- graph connectivity
- per-node label candidates
- candidate weights
- graph membership for pooling

## 7. Model selection

The model class is chosen in [src/clm/commands/train_models_RNN.py](src/clm/commands/train_models_RNN.py) based on `conditioning_type`.

- `none` -> [RNN](src/clm/models.py)
- `descriptors` -> [ConditionalRNN](src/clm/models.py)
- `graph` -> [GraphConditionalRNN](src/clm/models.py)

The graph-conditioned model is implemented in [src/clm/models.py](src/clm/models.py). It uses [BiosynthesisGraphEncoder](src/clm/graph_conditional.py) to turn each graph batch into a fixed-width vector.

That vector is then passed into the same conditional RNN pathways already used for descriptor conditioning.

Conceptually:

```text
condition_graph -> graph encoder -> conditioning vector -> conditional RNN
```

## 8. Training loop

The actual training loop is in [src/clm/commands/train_models_RNN.py](src/clm/commands/train_models_RNN.py).

At a high level it does:

1. build the dataset and dataloader
2. instantiate the model
3. instantiate the optimizer
4. sample validation batches from the held-out portion of the dataset
5. compute `model.loss(batch)`
6. backpropagate
7. update parameters
8. track validation loss
9. save the best checkpoint via [EarlyStopping](src/clm/loggers.py)

Outputs written during training:

- `model_file`
- `loss_file`

In the lipopeptide config these expand to patterns like:

```text
{output_dir}/{enum_factor}/prior/models/{dataset}_{repr}_{fold}_{train_seed}_model.pt
```

## 9. What files exist after training

For a single graph-conditioned fold, the most important files are:

- raw input JSONL: `paths.dataset`
- preprocessed JSONL: `paths.preprocess_output_graph`
- fold training JSONL: `paths.train_file_graph`
- fold held-out conditioning JSONL: `paths.condition_test_file_graph`
- sequence vocabulary: `paths.vocab_file`
- graph-label vocabulary: `paths.condition_vocab_file`
- trained model checkpoint: `paths.model_file`
- loss log: `paths.loss_file`

## 10. Where sampling fits in

Sampling is not required to train the model, but the next workflow step uses:

- [src/clm/commands/sample_molecules_RNN.py](src/clm/commands/sample_molecules_RNN.py)

In graph mode it reloads:

- `model_file`
- `vocab_file`
- `condition_vocab_file`
- `condition_test_file_graph`

and then samples molecules conditioned on the held-out graphs.

## 11. How RetroMol training data should be generated

The helper script for turning RetroMol results into Harvest training JSONL is:

- [scripts/harvest/retromol_results_to_training_data.py](scripts/harvest/retromol_results_to_training_data.py)

It now writes rows in the schema expected by graph-conditioned training:

```json
{
  "smiles": "...",
  "coverage": 0.76,
  "condition_graph": {
    "nodes": [{"id": 0, "name": "glycine"}],
    "links": []
  }
}
```

Important generator rules:

- unknown nodes are retained as `<UNK>`
- completely empty path sets become a one-node `<UNK>` graph
- the preferred field name is `condition_graph`

## 12. Quick debugging checklist for graph-conditioned runs

If training fails early, check these files in order:

1. [workflow/config/config_lipopeptides.yml](workflow/config/config_lipopeptides.yml)
2. the raw dataset JSONL passed as `paths.dataset`
3. the preprocessed JSONL written to `paths.preprocess_output_graph`
4. the fold training JSONL written to `paths.train_file_graph`
5. the graph label vocab written to `paths.condition_vocab_file`

Common failure causes:

- the raw JSONL uses neither `condition_graph` nor `graph`
- the raw JSONL contains invalid SMILES and everything gets filtered out
- the graph contains nodes with no `name` and no `name_candidates`
- the graph is completely empty and the generator did not preserve an `<UNK>` node
- an old code checkout still treats `max_input_smiles: 0` as "read zero lines"

## 13. Minimal end-to-end path for your lipopeptide run

For the current graph-conditioned lipopeptide setup, the concrete flow is:

1. [workflow/config/config_lipopeptides.yml](workflow/config/config_lipopeptides.yml) sets `model_params.conditional.type: graph`
2. `harvest train` in [src/harvest/train.py](src/harvest/train.py) launches Snakemake
3. `preprocess` in [workflow/Snakefile_data](workflow/Snakefile_data) reads the raw JSONL and writes `prior/raw/...jsonl`
4. `create_training_sets` in [workflow/Snakefile_data](workflow/Snakefile_data) writes per-fold `train_...jsonl`, `test_condition_...jsonl`, vocabularies, and `train0/test0`
5. `train_models_RNN` in [workflow/Snakefile_data](workflow/Snakefile_data) loads `train_...jsonl`
6. [GraphSequenceDataset](src/clm/datasets.py) converts each `condition_graph` into tensors
7. [GraphConditionalRNN](src/clm/models.py) encodes those graphs into conditioning vectors
8. the recurrent model is trained and the best checkpoint is saved to `prior/models/..._model.pt`
