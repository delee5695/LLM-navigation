from firebase import FirebaseDownloader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import os
from pathlib import Path
import rdp

<<<<<<< HEAD
TAR_FILE_NAME = "training_ua-9843530f4d1fbbdbd0cc26bb3e655da1_HAL_lab_to_206"
=======
TAR_FILE_NAME = (
    "training_ua-9843530f4d1fbbdbd0cc26bb3e655da1_HAL_lab_to_206"  # don't include .tar
)
>>>>>>> 3bf18d31b43d0c2ff8f1a0f791dff12f49a9e2cd
EXTRACTED_IMAGES = (
    f"backend/.cache/firebase_data/{TAR_FILE_NAME}/extracted/localization-video/"
)
critical_images = {}  # {index of image} : timestamp (float)

if __name__ == "__main__":
    downloader_1 = FirebaseDownloader(
        "iosLoggerDemo/tarQueue",
        f"{TAR_FILE_NAME}.tar",
    )
    downloader_1.extract_ios_logger_tar()
    # breakpoint()
    localization_phase_list = downloader_1.extracted_data.sensors_extracted[
        "localization_phase"
    ]["poses"]

    df = pd.DataFrame(localization_phase_list)
    positions = np.array(
        [np.array(sublist) for sublist in df["translation"].to_numpy()]
    )

    abs_positions = rdp.rdp(positions, 0.6)
    abs_positions_bool = rdp.rdp(positions, 0.6, return_mask=True)
    abs_timestamp = []
    indexes = []
    for idx, abs_position in enumerate(abs_positions_bool):
        if abs_position:

            critical_images[idx] = df.iloc[idx]["timestamp"]
    print(abs_positions)
    print(abs_timestamp)
    print(critical_images)
    # image plotting

    fig = plt.figure(figsize=(3, len(critical_images) // 3 + 1))

    backend = Path(__file__).parent
    shutil.rmtree(f"{backend}/critical_images/{TAR_FILE_NAME}")
    os.mkdir(f"{backend}/critical_images/{TAR_FILE_NAME}")

    for idx, image_number in enumerate(critical_images.keys()):
        img = Image.open(f"{EXTRACTED_IMAGES}{image_number}.jpg")
        fig.add_subplot(3, len(critical_images) // 3 + 1, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Timestamp: {str(critical_images[image_number]-30)[8:10]} (seconds)")

    # # route plotting
    # x, y, z = zip(*positions)
    # plt.plot(z, x, marker="o", linestyle="-", color="b")

    # # Plot the original path
    # x, y, z = zip(*abs_positions)
    # plt.plot(z, x, marker="o", linestyle="-", color="r")
    # plt.axis("equal")

        shutil.copy(
            f"{EXTRACTED_IMAGES}{image_number}.jpg",
            f"{backend}/critical_images/{TAR_FILE_NAME}",
        )
    plt.show()
