import argparse
import json
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from bwa import bwa
from ebcc import ebcc_vb
from ibcc import ibcc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default=r"similarity.csv")
    parser.add_argument("--method", type=str, default="ebcc")

    args = parser.parse_args()

    return args



def preprocess_annotations(csv_path: str) -> Tuple[np.ndarray, Dict, Dict]:
    df = pd.read_csv(csv_path)

    new_df = df[["task_id", "completed_by_id", "result"]].drop_duplicates(keep="first")
    
    new_df = new_df.to_records(index=False).tolist()
    
    new_df = list(map(lambda x: (x[0], x[1], json.loads(x[2])), new_df))
    
    new_df = list(filter(lambda x: x[2] != [], new_df))
    
    new_df = list(map(lambda x: (x[0], x[1], x[2][0]["value"]["choices"][0]), new_df))
    
    task_list = list(sorted(list(set(list(map(lambda x: x[0], new_df)))), key=lambda x: int(x)))
    workers = list(sorted(list(set(list(map(lambda x: x[1], new_df)))), key=lambda x: int(x)))
    categories = list(set(list(map(lambda x: x[2], new_df))))
    
    task_list_to_index = dict(zip(task_list, range(len(task_list))))
    workers_to_index = dict(zip(workers, range(len(workers))))
    categories_to_index = dict(zip(categories, range(len(categories))))
    
    
    new_df = list(map(lambda x: (task_list_to_index[x[0]], workers_to_index[x[1]], categories_to_index[x[2]]), new_df))
    
    new_df = list(set(new_df))
    
    new_df.sort(key=lambda x: x[0])
    
    new_array = np.array(new_df, dtype=np.int)

    index_to_task_list = {v: k for k, v in task_list_to_index.items()}
    index_to_categories = {v: k for k, v in categories_to_index.items()}

    return new_array, index_to_task_list, index_to_categories


if __name__ == "__main__":
    args = get_args()

    new_array, index_to_task_list, index_to_categories = preprocess_annotations(csv_path=args.csv_path)

    if args.method == "bwa":
        prediction = bwa(new_array)
    elif args.method == "ebcc":
        elbos = []
        seeds = []
        results = []
        for _ in range(40):
            seed = np.random.randint(1e8)
            prediction, elbo = ebcc_vb(new_array, num_groups=10, seed=seed, empirical_prior=True)
            elbos.append(elbo)
            results.append((prediction, seed, elbo))
        
        prediction, seed, elbo = results[np.argmax(elbos)]
    elif args.method == "ibcc":
        prediction = ibcc(new_array)

    prediction = np.argmax(prediction, axis=-1)

    result = dict(zip(list(map(lambda x: index_to_task_list[x], range(prediction.shape[0]))), list(map(lambda x: index_to_categories[x], prediction.tolist()))))
    result = list(map(lambda x: {"task_id": x[0], "label": x[1]}, result.items()))

    print(result)
    