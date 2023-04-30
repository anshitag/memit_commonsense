import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from dsets import (
    AttributeSnippets,
    get_tfidf_vectorizer,
    CommonSenseDataset
)
from experiments.py.eval_utils_cs import compute_rewrite_quality_cs, evaluate_model, compare_models
from memit import MEMITHyperParams, apply_memit_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT_CS": (MEMITHyperParams, apply_memit_to_model)
}

DS_DICT = {
    "cs": (CommonSenseDataset, compute_rewrite_quality_cs)
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    model_checkpoint: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    edit_token: str,
    edit_location : str,
    edit_file: str,
    inference_file: str,
    max_layer: int,
    layer_size: int,
    v_lr: float,
    kl_factor: float,
    wandb_active: bool = False,
    num_edits: int = 1,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )

    hparams = params_class.from_json(params_path)
    if edit_token is None:
        edit_token = hparams.fact_token.split("_")[0]
    if edit_location is not None and edit_token is not None:
        hparams.fact_token = f"{edit_token}_{edit_location}"
    if max_layer is not None:
        min_layer = max_layer-(layer_size-1) if max_layer-(layer_size-1)>0 else 1
        hparams.layers = list(range(min_layer, max_layer+1))
    if v_lr is not None:
        hparams.v_lr = v_lr
    if kl_factor is not None:
        hparams.kl_factor = kl_factor

    if not (run_dir / "params.json").exists():
        with open(run_dir / "params.json", 'w') as f:
            json.dump(dict(vars(hparams)), f, indent=4)

    if wandb_active:
        wandb.config.update(dict(vars(hparams)), allow_val_change=True)

    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        if model_checkpoint:
            model = AutoModelForCausalLM.from_pretrained(model_checkpoint).cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    if inference_file is not None:
        unedited_eval_metrics, unedited_output_data = evaluate_model(model, tok, inference_file, "unedited")
        if wandb_active:
            wandb.log(unedited_eval_metrics)

    # Load data
    # print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(edit_file, tok=tok, size=dataset_size_limit, noise_token=edit_token)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}_{edit_token}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        start = time()
        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            noise_token = edit_token,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        print("Execution took", exec_time)

        if SAVE_MODEL:
            edited_model.save_pretrained(str(run_dir)+"/edited_model")
            
        # Evaluate new model
        if inference_file is not None:
            edited_eval_metrics, edited_output_data = evaluate_model(edited_model, tok, inference_file, "edited")
            compare_metrics = compare_models(unedited_output_data, edited_output_data)
            compare_metrics["f1_difference"] = edited_eval_metrics["edited_f1Score"] - unedited_eval_metrics["unedited_f1Score"]
            if wandb_active:
                wandb.log(edited_eval_metrics)
                wandb.log(compare_metrics)


        if SAVE_SUMMARY:
            start = time()
            gen_test_vars = [snips, vec]
            for record in record_chunks:
                out_file = Path(case_result_template.format(num_edits, record["case_id"]))
                if out_file.exists():
                    print(f"Skipping {out_file}; already exists")
                    continue

                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "time": exec_time,
                    "post": ds_eval_method(
                        edited_model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                        noise_token = edit_token,
                    ),
                }

                # Dump metrics in .json
                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)

        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT_CS", "ROME", "FT", "MEND"],
        default="MEMIT_CS",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        help="Fine tuned Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cs"],
        default="cs",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=-1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--edit_token",
        choices=["subject","verb","object"],
        default=None,
        help="Use this token to edit model",
    )
    parser.add_argument(
        "--edit_location",
        choices=["first","last"],
        default=None,
        help="Use this location of edit token to edit model",
    )
    parser.add_argument(
        "--save_model",
        dest="save_model",
        action="store_true",
        help="Save edited model",
    )
    parser.add_argument(
        "--save_summary",
        dest="save_summary",
        action="store_true",
        help="Save summary data for edited model",
    )
    parser.add_argument(
        "--edit_file",
        type=str,
        default=None,
        help="The commonsense file to edit model using MEMIT",
        required=True,
    )
    parser.add_argument(
        "--inference_file",
        type=str,
        default=None,
        help="The commonsense file to compare edited model",
    )
    parser.add_argument(
        "--max_layer",
        type=int,
        default=None,
        help="Maximum value of layer to edit",
    )
    parser.add_argument(
        "--layer_size",
        type=int,
        default=5,
        help="Size of layers to edit from the max_layer",
    )
    parser.add_argument(
        "--v_lr",
        type=float,
        default=None,
        help="Editing learning rate",
    )
    parser.add_argument(
        "--kl_factor",
        type=float,
        default=None,
        help="Editing KL factor",
    )
    parser.add_argument('--wandb_account', type=str, default="anshitag")
    parser.add_argument('--wandb_project', type=str, default="memit_commonsense_edit")
    parser.add_argument('--wandb_active', action='store_true')

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    if args.wandb_active:
        wandb.init(project=args.wandb_project, entity=args.wandb_account, config=args)

    SAVE_MODEL = args.save_model
    SAVE_SUMMARY = args.save_summary

    main(
        args.alg_name,
        args.model_name,
        args.model_checkpoint,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        edit_token = args.edit_token,
        edit_location = args.edit_location,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        edit_file=args.edit_file,
        inference_file = args.inference_file,
        max_layer = args.max_layer,
        layer_size = args.layer_size,
        v_lr = args.v_lr,
        kl_factor = args.kl_factor,
        wandb_active= args.wandb_active,
    )
