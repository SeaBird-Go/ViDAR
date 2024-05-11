# The XWorld Predictive World Model

## Prepare Dataset
```bash
cd data
ln -s /path/to/openscene-v1.1 ./


# for the occupancy labels
cd ..
mkdir dataset
ln -s /path/to/openscene-v1.0 ./
```

### Preprocess the Dataset
```bash
python tools/collect_vidar_mini_split.py
python tools/update_pickle.py
```

## Train the Model
```bash

```

## Evaluate the Model