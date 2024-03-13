from firebase import FirebaseDownloader
import pandas as pd
import numpy as np
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
    positions = np.array([np.array(sublist) for sublist in positions])

    print(rdp.rdp(positions, 0.1))
