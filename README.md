# dm-data-subsetter
Creates a statistically representative subset of a larger dataset while preserving data relationships. Uses stratified sampling based on key fields to maintain distribution fidelity. Relies on Pandas and NumPy. - Focused on Tools designed to generate or mask sensitive data with realistic-looking but meaningless values

## Install
`git clone https://github.com/ShadowStrikeHQ/dm-data-subsetter`

## Usage
`./dm-data-subsetter [params]`

## Parameters
- `-h`: Show help message and exit
- `--subset_size`: Number of rows to include in the subset.
- `--stratify_by`: List of column names to use for stratified sampling.
- `--random_state`: Random state for reproducibility.
- `--mask_columns`: List of column names to mask with fake data.

## License
Copyright (c) ShadowStrikeHQ
