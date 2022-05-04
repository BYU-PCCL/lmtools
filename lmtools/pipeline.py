import argparse
from datetime import date

from lmtools.experiment import Experiment
from lmtools.postprocessor import Postprocessor


def run_experiment(ds_path, model_name, n=500):
    print("Passing data through model...")

    in_fname = ds_path
    date_str = date.today().strftime("%Y-%m-%d")

    # replace_str
    replace_str = f"_exp_results_{model_name.replace('/', '-')}_{date_str}.pkl"

    # replace .pkl with replace_str
    out_fname = in_fname.replace(".pkl", replace_str)
    Experiment(
        model_name=model_name,
        in_fname=in_fname,
        out_fname=out_fname,
    )

    model_name = model_name.replace("/", "-")

    # Postprocessing
    print("Postprocessing...")
    date_str = date.today().strftime("%d-%m-%Y")
    processed_in = out_fname
    # replace .pkl with _processed.pkl
    processed_out = processed_in.replace(".pkl", "_processed.pkl")

    Postprocessor(
        results_fname=processed_in,
        save_fname=processed_out,
        matching_strategy="startswith",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="pipeline",
    )
    # MUST be present, cannot accept none
    parser.add_argument(
        "-d", "--ds_path", type=str, help="path to dataset", required=True
    )
    parser.add_argument(
        "-m", "--model_name", type=str, help="path to model", required=True
    )
    args = parser.parse_args()
    ds_path = args.ds_path
    model_name = args.model_name
    print(f"ds_path: {ds_path}")
    print(f"model_name: {model_name}")
    run_experiment(ds_path, model_name)


if __name__ == "__main__":
    main()
