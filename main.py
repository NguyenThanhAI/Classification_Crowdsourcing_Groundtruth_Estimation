import json
import argparse
from typing import List, Dict, Tuple
import pandas as pd
from itertools import groupby

from pandas.io import parsers

from utils import MulticlassDawidSkeneEM, majority_vote, post_process


#csv_path = r"C:\Users\Thanh\Downloads\similarity\similarity.csv"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default=r"similarity.csv")

    args = parser.parse_args()

    return args


def preprocess_annotations(csv_path: str) -> Tuple[List[Dict], List]:
    df = pd.read_csv(csv_path)

    new_df = df[["task_id", "completed_by_id", "result"]]

    new_df = new_df.to_records(index=False).tolist()

    new_df = list(map(lambda x: (x[0], x[1], json.loads(x[2])), new_df))

    new_df = list(filter(lambda x: x[2] != [], new_df))

    new_df = list(map(lambda x: (x[0], x[1], x[2][0]["value"]["choices"][0]), new_df))

    task_list = list(map(lambda x: x[0], new_df))
    categories = list(set(list(map(lambda x: x[2], new_df))))

    #print(len(task_list), len(list(set(task_list))), categories)

    new_df.sort(key=lambda x: x[0])

    dset_to_annotations = []
    for task_id, records in groupby(new_df, key=lambda x: x[0]):
        #print(len(list(records)))
        records = list(records)
        annotations = []
        for rec in records:
            annotations.append({"workerId": rec[1], "annotationData": {"content": rec[2]}})
        dset_annotations = {"datasetObjectId": task_id, "annotations": annotations}
        dset_to_annotations.append(dset_annotations)
    return dset_to_annotations, categories


if __name__ == "__main__":
    args = get_args()

    csv_path = args.csv_path

    dset_to_annotations, categories = preprocess_annotations(csv_path=csv_path)
    print(len(dset_to_annotations))

    dawid_skene = MulticlassDawidSkeneEM("classification")

    responses = dawid_skene.update(annotation_payload=dset_to_annotations, label_categories=categories,
                                label_attribute_name="categories")

    responses = post_process(responses) # List of dictionary {'task_id': 490250, 'label': 'not similar'}
    """print(responses)
    print(len(responses))

    responses = majority_vote(dset_objects=dset_to_annotations)
    responses = post_process(responses)
    print(len(responses))"""