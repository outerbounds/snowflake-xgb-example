<img style="display: block; float: left; max-width: 20%; height: auto; margin: auto; float: none!important;" src="static/snowflow.png"/>

## Setup

### Create the environment
```bash
mamba env create -f env.yml
```

### Activate the environment 
```bash
mamba activate metaflow-snowflake-xgb-example
```

### Create the demo DB in Snowflake
In a Snowflake SQL worksheet, run the queries in `setup-preds-db.sql`.

## Run the flow
```bash
python flow.py --environment=conda run
```