import json
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Final

PROJECT_ROOT_DIR: Final[Path] = Path(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR: Final[Path] = PROJECT_ROOT_DIR.joinpath("results")
CHECKPOINTS_DIR: Final[Path] = PROJECT_ROOT_DIR.joinpath("checkpoints")
TEST_RESULTS_DIR: Final[Path] = RESULTS_DIR.joinpath("test_results")
HP_SEARCHES_DIR: Final[Path] = RESULTS_DIR.joinpath("hp_searches")
RESULTS_TEXTFILE_NAME: Final[str] = "results.txt"


def load_json_content(full_path: Path):
    """
    Reads the content of a file with JSON content and returns it as dict as a string.

    Args:
        full_path: Path of the file

    Returns:
        Content as a string
    """
    json_content: str
    with open(full_path, 'r') as file:
        json_content = json.load(file)
    return json_content


def create_train_validation_test_directories_for_db():
    """Creates the folder 'data' if it does not exist already.
    Afterwards it creates the folder 'training', 'validation' and 'testing' if they also do not exist."""

    # create root folder for the data and the images
    img_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'diagnostik_bilanz', 'img')
    os.makedirs(img_path, exist_ok=True)

    # create three folders for training/validation/testing
    train_val_test = ['training', 'validation', 'testing']
    for foldername in train_val_test:
        path = os.path.join(img_path, foldername)
        if not os.path.isdir(path):
            os.mkdir(path)


def load_csv_containing_the_mapping_for_db_patient_to_CVFold(path:str):
    """
    Loads the CSV that maps each patient from the Diagnostikbilanz Dataset to one of the 4 Folds.
    The CSV File should be located inside /projectname/data/diagnostik_bilanz

    Args:
        path: The Path to the CSV File

    Returns:
        A Dataframe containing the content of the CSV file.
    """
    assert os.path.isfile(path), f'No file at {path} path'
    mapping_df = pd.read_csv(path, sep=':')
    return mapping_df


def map_vertebra_patch_to_train_val_test_directory(path_to_all_vertebra:str, path_to_map_csv:str, train_fold1:int = 2, train_fold2:int = 3, validation_fold:int = 1, test_fold:int = 0):
    """
    1. Creates the needed directories if they do not exist already
    2. Reads the CSV File that contains the mapping
    3. Puts each vertebra patch in the corresponding directory

    Args:
        path_to_all_vertebra:
        path_to_map_csv:
    """
    # Creates the needed directories if they do not exist already
    create_train_validation_test_directories_for_db()

    # Reads the CSV File that contains the mapping
    mapping_df = load_csv_containing_the_mapping_for_db_patient_to_CVFold(path_to_map_csv)

    # Define the directories for training, validation and testing
    training_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'diagnostik_bilanz', 'img', 'training')
    validation_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'diagnostik_bilanz', 'img', 'validation')
    testing_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'diagnostik_bilanz', 'img', 'testing')

    for vertebra in os.listdir(path_to_all_vertebra): # iterate over all vertebra
        if not vertebra.endswith('nii.gz'):  # Only move nifti files that can be used as input for the model
            continue

        full_vertebra_path = os.path.join(path_to_all_vertebra, vertebra)

        # get the patient number and the fold of the current vertebra
        patient_number_of_current_vertebra = int(vertebra[:4])
        fold_of_current_vertebra = int(mapping_df.loc[mapping_df['patient'] == patient_number_of_current_vertebra]['fold'])

        # Depending on the fold of the vertebra, move it to train/validation/test directory
        if fold_of_current_vertebra == train_fold1 or fold_of_current_vertebra == train_fold2:
            shutil.copy(full_vertebra_path, training_path)
        elif fold_of_current_vertebra == validation_fold:
            shutil.copy(full_vertebra_path, validation_path)
        elif fold_of_current_vertebra == test_fold:
            shutil.copy(full_vertebra_path, testing_path)


def sanity_check_label_files(path_to_excel_file:str, path_to_csv_label_file:str):
    """
    This method computes how many samples for each grade exists.
    The -g columns seem to be the genant grade columns as the numbers match the paper:
    https://pubs.rsna.org/doi/epdf/10.1148/ryai.2020190138
    There are 1725 vertebra in total but only 1469 are annotated.
    Args:
        path_to_excel_file: The filepath to the excel file from the supplemental material
        path_to_csv_label_file: The filepath to the label file used for training to check whether the numbers match

    Returns:
        none
    """
    df = pd.read_excel(path_to_excel_file)
    total_number_of_vertebra = df['N_vertebrae'].sum()
    print(f'Total number of vertebra = {total_number_of_vertebra}')

    important = []
    for ax in df.axes[1].values:
        if ax.endswith('-g'):
            important.append(ax)
    df_grades = df[important]
    official_count0 = 0
    official_count1 = 0
    official_count2 = 0
    official_count3 = 0
    official_countx = 0

    for ax in important:
        official_count0 += len(df_grades[ax][df_grades[ax] == 0])
        official_count1 += len(df_grades[ax][df_grades[ax] == 1])
        official_count2 += len(df_grades[ax][df_grades[ax] == 2])
        official_count3 += len(df_grades[ax][df_grades[ax] == 3])
        official_countx += len(df_grades[ax][df_grades[ax] == 'x'])

    print(f'count0 from excel file = {official_count0}')
    print(f'count1 from excel file = {official_count1}')
    print(f'count2 from excel file = {official_count2}')
    print(f'count3 from excel file = {official_count3}')
    print(f'countx from excel file = {official_countx}')

    label = pd.read_csv(path_to_csv_label_file, sep=';')
    label_count0 = label['Score'][label['Score'] == 0].count()
    label_count1 = label['Score'][label['Score'] == 1].count()
    label_count2 = label['Score'][label['Score'] == 2].count()
    label_count3 = label['Score'][label['Score'] == 3].count()

    print(f'count0 from self made csv file = {label_count0}')
    print(f'count1 from self made csv file = {label_count1}')
    print(f'count2 from self made csv file = {label_count2}')
    print(f'count3 from self made csv file = {label_count3}')



if __name__ == '__main__':
    pass
    #create_train_validation_test_directories()
    #mapping_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'diagnostik_bilanz', 'patient_fold_mapping.csv')
    #all_vertebrba_path = '/data/vertebra_patches/db'
    #map_vertebra_patch_to_train_val_test_directory(all_vertebrba_path, mapping_path)
