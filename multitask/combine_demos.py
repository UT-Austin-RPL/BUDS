"""This is to combine several demos. It is very painful to collect hundreds of demos straight."""
import os
import h5py
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
    )


    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_f = h5py.File(f"{args.out_dir}/demo.hdf5", "w")

    grp = out_f.create_group("data")
    num_eps = 0
    env_name = None
    env_info = None
    for folder in args.folders:
        f = h5py.File(f"{folder}/demo.hdf5", "r")
        if env_name is None:
            env_name = f["data"].attrs["env"]
        if env_info is None:
            env_info = f["data"].attrs["env_info"]
        demos = list(f["data"].keys())
        for demo in demos:
            num_eps += 1
            ep_data_grp = grp.create_group(f"demo_{num_eps}")
            ep_data_grp.attrs["model_file"] = f[f"data/{demo}"].attrs["model_file"]
            ep_data_grp.attrs["task_id"] = f[f"data/{demo}"].attrs["task_id"]
            ep_data_grp.create_dataset("states", data=f[f"data/{demo}/states"])
            ep_data_grp.create_dataset("actions", data=f[f"data/{demo}/actions"])

        f.close()

    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    grp.attrs["num_eps"] = num_eps
    print(list(out_f["data"].keys()))

    out_f.close()
if __name__ == "__main__":
    main()
