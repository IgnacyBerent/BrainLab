import prep
import prsa
import peaks_detection
import os
import numpy as np
import biosppy

SAMPLING_RATE = 200

NORMO_DIR = "./data_2/normokapnia"
HYPER_DIR = "./data_2/hiperkapnia"

ABP_COLUMN = "abp_finger_mm_hg_[abp_finger_mm_Hg_]"
ALTERNATIVE_ABP_COLUMN = "abp_finger[abp_finger]"


def main():
    for ws in [3, 5, 7]:
        print(
            f"\n-------------------------WINDOW SIZE: {ws}-------------------------\n"
        )
        folder_name = (
            f"corrected_results_L{ws}"  # folder to save results in .csv format
        )

        normo_dc_data = {}
        normo_ac_data = {}
        hyper_dc_data = {}
        hyper_ac_data = {}

        normo_dc_w = {}
        normo_ac_w = {}
        hyper_dc_w = {}
        hyper_ac_w = {}

        for filename in os.listdir(NORMO_DIR):
            if os.path.isfile(os.path.join(NORMO_DIR, filename)):
                # read data
                file_path = os.path.join(NORMO_DIR, filename)
                try:
                    dc, ac, dc_w, ac_w = prsa_pipeline(file_path, ws)
                except ValueError:
                    print(f"Error in {filename}")
                    continue
                # saving results
                filename = filename[-7:-5]
                normo_dc_data[filename] = dc
                normo_ac_data[filename] = ac
                normo_dc_w[filename] = dc_w
                normo_ac_w[filename] = ac_w
                print(f"Finished normo: {filename}")

        for filename in os.listdir(HYPER_DIR):
            if os.path.isfile(os.path.join(HYPER_DIR, filename)):
                # read data
                file_path = os.path.join(HYPER_DIR, filename)

                try:
                    dc, ac, dc_w, ac_w = prsa_pipeline(file_path, ws)
                except ValueError:
                    print(f"Error in {filename}")
                    continue

                # saving results
                filename = filename[-7:-5]
                hyper_dc_data[filename] = dc
                hyper_ac_data[filename] = ac
                hyper_dc_w[filename] = dc_w
                hyper_ac_w[filename] = ac_w
                print(f"Finished hyper: {filename}")

        save_to_csv(
            file_tittle="AC_DC_Windows",
            folder_name=folder_name,
            AC_normo_W=normo_ac_w,
            AC_hyper_W=hyper_ac_w,
            DC_normo_W=normo_dc_w,
            DC_hyper_W=hyper_dc_w,
        )

        save_to_csv(
            file_tittle="AC_DC_Values",
            folder_name=folder_name,
            AC_normo=normo_ac_data,
            AC_hyper=hyper_ac_data,
            DC_normo=normo_dc_data,
            DC_hyper=hyper_dc_data,
        )


def save_to_csv(file_tittle: str, folder_name: str, **columns: dict[str:float]):
    """
    Saves results to csv file
    :param file_tittle: name of the file
    :param folder_name: name of the folder
    :param columns: column names and their values for given file
    """

    # Get the column names and values
    columns_names = list(columns.keys())
    columns_values = list(columns.values())

    # Get the filenames
    filenames = set()
    for col in columns_values:
        filenames.update(list(col.keys()))

    filenames = list(filenames)
    filenames.sort(key=lambda x: int(x))
    # Create the directory if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    with open(f"{folder_name}/{file_tittle}.csv", "w") as f:
        f.write(f'file number;{";".join(columns_names)}\n')
        for filename in filenames:
            row_values = [row.get(filename, -np.inf) for row in columns_values]
            if -np.inf in row_values:
                continue
            f.write(f'{filename};{";".join([str(val) for val in row_values])}\n')


def prsa_pipeline(file_path: str, window_size: int):
    try:
        df = prep.read_data(file_path, ABP_COLUMN, sep=";")
    except KeyError:
        df = prep.read_data(file_path, ALTERNATIVE_ABP_COLUMN, sep=";")

    # data preprocessing
    df.interpolate(method="linear", inplace=True)
    try:  # few files have too much noise to be filtered
        filtered_abp = biosppy.signals.abp.abp(
            signal=df["Values"], sampling_rate=SAMPLING_RATE, show=False
        )[1]
        df["Values"] = filtered_abp
    except ValueError:
        raise ValueError("Too much noise")

    # finding peaks
    rr = peaks_detection.rr_intervals(df["Values"], sampling_rate=SAMPLING_RATE)

    # calculating PRSA
    prsa_dc, dc_w = prsa.calculate(rr, window_size, "DC", 0.2)
    prsa_ac, ac_w = prsa.calculate(rr, window_size, "AC", 0.2)

    # calculating capacity
    dc = prsa.capacity(prsa_dc)
    ac = prsa.capacity(prsa_ac)

    return dc, ac, dc_w, ac_w


if __name__ == "__main__":
    main()
