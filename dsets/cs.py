import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"


class CommonSenseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        noise_token: str,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        cf_loc = "/work/anshitagupta_umass_edu/allenai_inp_study/memit_commonsense/tracing/editing/gpt2-xl_20q_normal.json"

        with open(cf_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]
        for i in self.data:
            words = i["requested_rewrite"]["prompt"].split()
            i["requested_rewrite"]["prompt"] = " ".join([w if w!=i["requested_rewrite"][noise_token] else "{}" for w in words])

        print(f"Loaded dataset with {len(self)} elements {self.data[0]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]