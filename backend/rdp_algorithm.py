from firebase import FirebaseDownloader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rdp

if __name__ == "__main__":
    downloader_1 = FirebaseDownloader(
        "iosLoggerDemo/tarQueue",
        "training_ua-5ad6b6714d5e52c97ca53c47f38b9655_Test_3-13-occamlab-3rd_floor.tar",
    )
    downloader_1.extract_ios_logger_tar()
    # breakpoint()
    localization_phase_list = downloader_1.extracted_data.sensors_extracted[
        "localization_phase"
    ]["poses"]

    df = pd.DataFrame(localization_phase_list)
    # print(df["translation"])
    # print(df["translation"].to_numpy())
    positions = df["translation"].to_numpy()  # .reshape(870, 3)
    # positions_rounded = np.array(
    #     [[round(val, 8) for val in sublist] for sublist in positions]
    # )
    # for sublist in positions:
    #     for idx, num in enumerate(sublist):
    #         sublist[idx].replace(np.round(num, 8))
    # positions = positions_rounded
    # print(positions)
    positions = np.array([np.array(sublist) for sublist in positions])

    abs_positions = rdp.rdp(positions, 0.6)
    #     print(abs_positions)
    #     abs_timestamp = []
    #     for position in abs_positions:
    #         print(df["translation"].size, position.size)
    #         abs_timestamp.append(df.loc[df["translation"] == position]["timestamp"])
    #     print(df["translation"])
    #     print(abs_positions[0])
    # print(abs_timestamp)
    # Plot the original path
    x, y, z = zip(*positions)
    plt.plot(z, x, marker="o", linestyle="-", color="b")

    # Plot the original path
    x, y, z = zip(*abs_positions)
    plt.plot(z, x, marker="o", linestyle="-", color="r")
    plt.axis("equal")

    plt.show()
