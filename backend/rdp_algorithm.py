from firebase import FirebaseDownloader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rdp

TAR_FILE_NAME = (
    "training_ua-5ad6b6714d5e52c97ca53c47f38b9655_Test_3-13-occamlab-3rd_floor"
)
EXTRACTED_IMAGES = (
    f"backend/.cache/firebase_data/{TAR_FILE_NAME}/extracted/localization-video/"
)
critical_images = {}

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

    abs_positions = rdp.rdp(positions, 0.6, return_mask=True)

    abs_timestamp = []
    indexes = []
    for idx, abs_position in enumerate(abs_positions):
        if abs_position:

            critical_images[idx] = df.iloc[idx]["timestamp"]

    fig = plt.figure(figsize=(3, len(critical_images) // 3 + 1))
    for idx, key in enumerate(critical_images.keys()):
        img = Image.open(f"{EXTRACTED_IMAGES}{key}.jpg")
        fig.add_subplot(3, len(critical_images) // 3 + 1, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Timestamp: {critical_images[key]}")

    plt.show()
