import json
import typing
from pathlib import Path

import re
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
        cf_loc = Path(data_dir)

        with open(cf_loc, "r") as f:
            data = json.load(f)

        final_data = []
        for i in data:
            substring = i["requested_rewrite"][noise_token]
            matches = len(re.findall(rf"\b{substring}\b", i["requested_rewrite"]["prompt"]))
            if matches > 1:
                continue
            elif not matches:
                i["requested_rewrite"]["prompt"] = i["requested_rewrite"]["prompt"].replace(substring, "{}")
            else:
                i["requested_rewrite"]["prompt"] = re.sub(rf"\b{substring}\b", "{}", i["requested_rewrite"]["prompt"])
            final_data.append(i)
        
        self.data = final_data
        if size is not None:
            self.data = self.data[:size]
        print(f"Loaded dataset with {len(self)} elements {self.data[0]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]