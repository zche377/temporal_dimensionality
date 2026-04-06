from dotenv import load_dotenv
load_dotenv()

import os
from osfclient.api import OSF
from osfclient.models.file import Folder
from bonner.caching import BONNER_CACHING_HOME

PROJECT_ID = "y6znd"

PATH_DICT = {
    "behavior": BONNER_CACHING_HOME / "behavior",
}


def download_osf_project(project_id, path_dict):
    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage('osfstorage')
    for top_level_item in storage.folders:
        if isinstance(top_level_item, Folder) and top_level_item.name in path_dict:
            local_destination = path_dict[top_level_item.name]
            os.makedirs(local_destination, exist_ok=True)
            print(f"Downloading '{top_level_item.name}' to '{local_destination}'")
            download_folder(top_level_item, local_destination)


def download_folder(folder, local_path):
    for item in folder.folders:
        item_path = os.path.join(local_path, item.name)
        os.makedirs(item_path, exist_ok=True)
        download_folder(item, item_path)
    for item in folder.files:
        item_path = os.path.join(local_path, item.name)
        print(f"Downloading '{item.name}' to '{item_path}'")
        with open(item_path, 'wb') as f:
            item.write_to(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output base directory (default: BONNER_CACHING_HOME)")
    args = parser.parse_args()

    if args.output_dir:
        path_dict = {k: os.path.join(args.output_dir, k) for k in PATH_DICT}
    else:
        path_dict = PATH_DICT

    download_osf_project(PROJECT_ID, path_dict)
