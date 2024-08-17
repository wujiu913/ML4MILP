## ML4MILP: A Benchmark Dataset for Machine Learning-based Mixed-Integer Linear Programming

### Overview

**ML4MILP is the first benchmark dataset specifically designed to test ML-based algorithms for solving MILP problems**, consisting of three main components: Similarity Evaluation, Benchmark Datasets, and Baseline Library. Based on this structure, we conducted uniform training and testing of baseline algorithms, followed by a comprehensive evaluation and ranking of the results.

![Framework](./Picture/Framework.png)

### Benchmark Datasets

We have meticulously assembled a substantial collection of mixed integer linear programming (MILP) instances from a variety of sources, including open-source, comprehensive datasets, domain-specific academic papers and competitions related to MILP.  Additionally, we generated a substantial number of standard problem instances based on four canonical MILP problems: the Maximum Independent Set (MIS) problem, the Minimum Vertex Covering (MVC) problem,  and the Set Covering (SC) problem. For each type of problem, we generated instances at three levels of difficulty—easy, medium, and hard.

The sizes of each categorized datasets are as follows, and the download links are detailed in `./Benchmark Datasets/README.md`.

| Name(Path)                     | Number of Instances | Avg.Vars | Avg.Constrains |
| ------------------------------ | ------------------- | -------- | -------------- |
| MIS_easy                       | 50                  | 20000    | 60000          |
| MIS_medium                     | 50                  | 100000   | 300000         |
| MIS_hard                       | 50                  | 1000000  | 3000000        |
| MVC_easy                       | 50                  | 20000    | 60000          |
| MVC_medium                     | 50                  | 100000   | 300000         |
| MVC_hard                       | 50                  | 1000000  | 3000000        |
| SC_easy                        | 50                  | 40000    | 40000          |
| SC_medium                      | 50                  | 200000   | 200000         |
| SC_hard                        | 50                  | 2000000  | 2000000        |
| nn\_verification               | 3622                | 7144.02  | 6533.58        |
| item\_placement                | 10000               | 1083     | 195            |
| load\_balancing                | 10000               | 61000    | 64307.19       |
| anonymous                      | 138                 | 34674.03 | 44498.19       |
| HEM\_knapsack                  | 10000               | 720      | 72             |
| HEM\_mis                       | 10002               | 500      | 1953.48        |
| HEM\_setcover                  | 10000               | 1000     | 500            |
| HEM\_corlat                    | 1984                | 466      | 486.17         |
| HEM\_mik                       | 90                  | 386.67   | 311.67         |
| vary\_bounds\_s1               | 50                  | 3117     | 1293           |
| vary\_bounds\_s2               | 50                  | 1758     | 351            |
| vary\_bounds\_s3               | 50                  | 1758     | 351            |
| vary\_matrix\_s1               | 50                  | 802      | 531            |
| vary\_matrix\_rhs\_bounds\_s1  | 50                  | 27710    | 16288          |
| vary\_matrix\_rhs\_bounds\_obj | 50                  | 7973     | 3558           |
| vary\_obj\_s1                  | 50                  | 360      | 55             |
| vary\_obj\_s2                  | 50                  | 745      | 26159          |
| vary\_obj\_s3                  | 50                  | 9599     | 27940          |
| vary\_rhs\_s1                  | 50                  | 12760    | 1501           |
| vary\_rhs\_s2                  | 50                  | 1000     | 1250           |
| vary\_rhs\_s3                  | 50                  | 63009    | 507            |
| vary\_rhs\_s4                  | 50                  | 1000     | 1250           |
| vary\_rhs\_obj\_s1             | 50                  | 90983    | 33438          |
| vary\_rhs\_obj\_s2             | 50                  | 4626     | 8274           |
| Aclib                          | 99                  | 181      | 180            |
| Coral                          | 279                 | 18420.92 | 11831.01       |
| Cut                            | 14                  | 4113     | 1608.57        |
| ECOGCNN                        | 44                  | 36808.25 | 58768.84       |
| fc.data                        | 20                  | 571      | 330.5          |
| MIPlib                         | 50                  | 7719.98  | 6866.04        |
| Nexp                           | 77                  | 9207.09  | 7977.14        |
| Transportation                 | 32                  | 4871.5   | 2521.467       |
| MIRPLIB_Original               | 72                  | 36312.2  | 180.0          |
| MIRPLIB_Maritime_Group1        | 40                  | 13919.5  | 180.0          |
| MIRPLIB_Maritime_Group2        | 40                  | 24639.8  | 180.0          |
| MIRPLIB_Maritime_Group3        | 40                  | 24639.8  | 180.0          |
| MIRPLIB_Maritime_Group4        | 20                  | 4343.0   | 180.0          |
| MIRPLIB_Maritime_Group5        | 20                  | 48330.0  | 180.0          |
| MIRPLIB_Maritime_Group6        | 20                  | 48330.0  | 180.0          |

### Baseline Library

To validate the effectiveness of the proposed dataset, we organized the existing mainstream methods into a Baseline Library and conducted comparisons using Benchmark Datasets against these mainstream baselines. The algorithms in the Baseline Library are as follows. 

| Baseline                                                   | Code                                   |
| ---------------------------------------------------------- | -------------------------------------- |
| Gurobi                                                     | ./Baseline Library/Gurobi/             |
| SCIP                                                       | ./Baseline Library/SCIP/               |
| Large Neighborhood Searc                                   | ./Baseline Library/LNS/                |
| Adaptive Constraint Partition Based Optimization Framework | ./Baseline Library/ACP/                |
| Learn to Branch                                            | ./Baseline Library/Learn2Branch/       |
| GNN&GBDT-Guided Fast Optimizing Framework                  | ./Baseline Library/GNN&GBDT/           |
| GNN-Guided Predict-and-Search Framework                    | ./Baseline Library/Predict&Search      |
| Neural Diving                                              | ./Baseline Library/Neural Diving       |
| Hybrid Learn to Branch                                     | ./Baseline Library/Hybrid_Learn2Branch |
| Graph Neural Networks with Random Feat                     | ./Baseline Library/GNN_MILP            |

### Similarity Evaluation

#### Similarity Evaluation Metrics

##### Structure Embedding

The codes are shown in `./Similarity Evaluation/Similarity/Structure Similarity`. The following bash command can be run to calculate the structural embedding similarity of a given dataset containing several instances of MILP in `.lp` format.

```bash
base_dir="<The dataset folder>"

# Traverse all subfolders under _latest_datasets
find "$base_dir" -type d -name LP | while read lp_dir; do
		# Get the directory where the LP is located, i.e. the instance directory
    instance_dir=$(dirname "$lp_dir")  
    # Get the question name
    problem_name=$(basename "$instance_dir")

    echo "Processing problem: $problem_name, directory: $instance_dir"

    # Create the Test0 folder in the instance directory
    mkdir -p "$instance_dir/Test0"

    # Run MILP_utils.py
    python MILP_utils.py --mode=model2data \
        --input_dir="$lp_dir" \
        --output_dir="$instance_dir/Test0" \
        --type=direct

    # Run graph_statistics.py
    python graph_statistics.py --input_dir="$instance_dir/Test0" \
        --output_file="$instance_dir/statistics"

    # Run calc_sim.py and output the results
    python calc_sim.py  --input_file1="$instance_dir/statistics" > "$instance_dir/result.txt"
done
```

##### Neural Embedding

The codes are shown in `./Similarity Evaluation/Similarity/Neural Embedding Similarity`. The following bash command can be run to calculate the neural embedding similarity of a given dataset containing several instances of MILP in `.lp` format.

```bash

special_dir="<Your Dataset Folder>"


find "$special_dir" -type d -name LP | while read lp_dir; do
    process_instance "$lp_dir"
done

process_instance() {
    lp_dir=$1
    # Get the directory where the LP is located, i.e. the instance directory
    instance_dir=$(dirname "$lp_dir")  
    # Get the question name
    problem_name=$(basename "$instance_dir") 

    echo "Processing problem: $problem_name, directory: $instance_dir"

    # Create the Test0 folder in the instance directory
    mkdir -p "$instance_dir/Test0"

    # Run MILP_utils.py
    python MILP_utils.py --mode=model2data \
        --input_dir="$lp_dir" \
        --output_dir="$instance_dir/Test0" \
        --type=direct
		
		# Run inference.py to inference
    python src/inference.py --dataset=MILP \
    --cfg_path=./experiments/configs/test.yml \
    --seed=1 \
    --device=2 \
    --model_path=./experiments/weights/encoder.pth \
    --input_dir="$instance_dir/Test0" \
    --output_file="$instance_dir/embedding" \
    --filename="$instance_dir/namelist" || echo "Inference failed"

    # Run calc_sim.py and output the results
    python calc_sim.py  --input_file1="$instance_dir/embedding" > "$instance_dir/result_embedding.txt"
}

```

#### Classification Algorithm

The codes are shown in `./Similarity Evaluation/Classification`. The following bash command can be run to build, train, inference and cluster. For classification, we can use our trained model and just run build, inference and cluster in turn.

```bash
case $1 in
  build|train|inference|cluster)
    echo "Valid argument: $1"
    ;;
  *)
    echo "Invalid argument: $1. Please enter correct choice."
    exit 1
    ;;
esac


if [ "$1" = "build" ]; then
    python src/MILP_utils.py --mode=model2data --input_dir=dataset/model --output_dir=dataset/data --type=direct
fi

if [ "$1" = "train" ]; then
    python src/train.py --dataset=MILP --cfg_path=experiments/configs/test.yml --seed=1 --device=0 --model_path=experiments/weights/encoder.pth --dataset_path=dataset/data || echo "Training failed"
fi

if [ "$1" = "inference" ]; then
    python src/inference.py --cfg_path=experiments/configs/test.yml --seed=1 --device=0 --model_path=experiments/weights/encoder.pth --input_dir="<Your Dataset Folder>" --output_file=tmp.pkl --filename=namelist.pkl || echo "Inference failed"
fi

if [ "$1" = "cluster" ]; then
    python src/clustering.py --filename=namelist.pkl --input_file=tmp.pkl
fi
```

