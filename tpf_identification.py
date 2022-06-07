import matplotlib.pyplot as plt
import lightkurve as lk


if __name__ == "__main__":
    from pathlib import Path
    from utils import plot_identification
    from astropy.units import UnitsWarning
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UnitsWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    result_path = Path.cwd() / "results"
    result_path.mkdir(exist_ok=True)

    file_path_list = [i for i in (Path.cwd() / "tpfs").glob("*.fit*")]
    file_path_list.sort()

    index = 0
    for file_path in file_path_list:
        print("-------------------------")
        print(file_path)
        tpf = lk.read(file_path)

        index += 1
        label = tpf.get_header(ext=0).get("OBJECT")
        print("Num {}, {}".format(index, label))

        mission = tpf.mission
        if mission == "TESS":
            part = "S{:0>2d}".format(tpf.get_header(ext=0).get("SECTOR"))
        elif mission == "K2":
            part = "C{:0>2d}".format(tpf.get_header(ext=0).get("CAMPAIGN"))
        elif mission == "Kepler":
            part = "Q{:0>2d}".format(tpf.get_header(ext=0).get("QUARTER"))
        else:
            raise ValueError

        fig, ax = plt.subplots(figsize=(9, 4))
        plot_identification(ax, tpf)

        plt.savefig(result_path / Path("{}-{}.pdf".format(label.replace(" ", ""), part)), bbox_inches="tight")
