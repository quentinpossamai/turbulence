#!/usr/bin/env python3

"""
Winter Guerra <winterg@mit.edu>
April 13th, 2020

Usage: sequenceDownloader.py -flight ampersand/yawForward/maxSpeed2p0/
                             -environment Small_Apartment
                             -datasetFolder ~/BlackbirdDatasetData
"""

from pathlib import Path

import fire
import os
import shutil
import urllib
import wget

# S3 address as URI (yes, /// is on purpose)
serverAddress = "ijrr20-blackbird-dataset.s3-accelerate.amazonaws.com/BlackbirdDatasetData"

# Flight files that can be used for playback / analysis
flightFileList = [
    "rosbag.bag",
    "flightNormalizationOffset.csv"]
# "groundTruthPoses.csv"
# ]

# These are not required, but may be helpful
csvFiles = [
    "blackbird_slash_imu.csv",
    "blackbird_slash_pose_ref.csv",
    "blackbird_slash_pwm.csv",
    "blackbird_slash_rotor_rpm.csv",
    "blackbird_slash_state.csv",
    "camera_d_slash_camera_info.csv",
    "camera_l_slash_camera_info.csv",
    "camera_r_slash_camera_info.csv",
    "tf.csv",
]

globalFiles = [
    "trajectoryOffsets.yaml"
]

environmentFiles = [
    "120hzTimestamps.csv"
]

cameras = [
    "Camera_Left_RGB",
    "Camera_Down_RGB",
    "Camera_Left_Gray",
    "Camera_Right_Gray",
    "Camera_Down_Gray",
    "Camera_Left_Depth",
    "Camera_Down_Depth",
    "Camera_Left_Segmented",
    "Camera_Down_Segmented",
]

# Files next to movie files.
# @TODO: change this to the IJRR filename in the PDF
timestampFile = "nSecTimestamps.txt"


def getCamFile(camName):
    return "lossless.tar" if ("Depth" in camName or "Segmented" in camName) else "lossless.mov"


def append(root, files):
    return [root / f for f in files]


# def download(relPaths, datasetFolder):
#     for relPath in relPaths:
#         # Make sure that directory exists
#         relPath.parent.mkdir(parents=True, exist_ok=True)
#         # Download the file
#         wget(serverAddress / relPath, datasetFolder / relPath)

def downloadSequence(flight, environment, datasetFolder):
    flight = Path(flight)
    environment = Path(environment)
    datasetFolder = Path(datasetFolder).expanduser()

    print(flight, environment, datasetFolder)

    # Inline definition
    def download(relPaths):
        for relPath in relPaths:
            relPath = Path(relPath)
            dest = datasetFolder / relPath
            src = "http://" + str(serverAddress / relPath)
            # Make sure that directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Check that the file or extracted directory does not already exist
            if ((dest.suffix != ".tar" and not dest.exists()) or (
                    dest.suffix == ".tar" and not dest.with_name("images").exists())):

                # Download the file
                print(f"Downloading {relPath}")
                try:
                    wget.download(src, out=str(dest))
                except urllib.error.HTTPError as err:
                    print(err)

                # If needed, extract the file
                if (dest.suffix == ".tar"):
                    print("Extracting tarball")
                    shutil.unpack_archive(dest, extract_dir=dest.with_name("images"))
                    os.remove(dest)

            else:
                print(f"Skipping {relPath}")

    # download(globalFiles)
    # download(append(flight / "csv", csvFiles))
    download(append(flight, flightFileList))  # Download only trajectories
    # download(append(flight / environment, environmentFiles))

    # # Download camera files
    # for cam in cameras:
    #     download( [
    #         flight / environment / cam / getCamFile(cam),
    #         flight / environment / cam / timestampFile
    #     ])


def main():
    datasetFolder = "~/phd/turbulence/data/blackbird/raw"
    environment = "Small_Apartment"

    flights_combinations = {"3dFigure8": {"yawConstant": [0.5, 1, 2, 3, 4, 5],
                                          "yawForward": []},
                            "ampersand": {"yawConstant": [1, 2, 3],
                                          "yawForward": [1, 2]},
                            "bentDice": {"yawConstant": [1, 2, 3, 4],
                                         "yawForward": [0.5, 1, 2, 3]},
                            "clover": {"yawConstant": [1, 2, 3, 4, 5, 6],
                                       "yawForward": [0.5, 1, 2, 3, 4, 5]},
                            "dice": {"yawConstant": [2, 3, 4],
                                     "yawForward": [1, 2, 3]},
                            "egg": {"yawConstant": [13.8],
                                    "yawForward": [3, 4, 5, 6, 7, 8, 9]},
                            "figure8": {"yawConstant": [0.5, 1, 2, 3.25, 5],
                                        "yawForward": []},
                            "halfMoon": {"yawConstant": [1, 2, 3, 4],
                                         "yawForward": [1, 1.5, 2, 3, 4, "maxSpeed1p0_run_2"]},
                            "mouse": {"yawConstant": [1, 2, 3, 4, 5, 6, 7],
                                      "yawForward": [0.5, 1, 2, 3, 4, 5, 6, 7, "maxSpeed0p5_run_2"]},
                            "oval": {"yawConstant": [2, 3, 4],
                                     "yawForward": [1, 2, 3, 4]},
                            "patrick": {"yawConstant": [1, 2, 3, 4, 5],
                                        "yawForward": [0.5, 1, 2, 3, 4]},
                            "picasso": {"yawConstant": [0.5, 1, 2, 3, 4, 5, 6],
                                        "yawForward": [0.5, 1, 3, 4, 5]},
                            "sid": {"yawConstant": [1, 2, 3, 4, 5, 6, 7],
                                    "yawForward": [0.5, 1, 2, 3, 4, 5]},
                            "sphinx": {"yawConstant": [1, 2, 3, 4],
                                       "yawForward": [1, 2, 3, 4]},
                            "star": {"yawConstant": [1, 2, 3, 4, 5, "maxSpeed4p0_run_2"],
                                     "yawForward": [0.5, 1, 2, 3, 4, 5]},
                            "thrice": {"yawConstant": [1, 2, 3, 4, 5, 6, 7],
                                       "yawForward": [0.5, 1, 2, 3, 4, 5, 6]},
                            "tiltedThrice": {"yawConstant": [1, 2, 3, 4, 5, 6, 7],
                                             "yawForward": [0.5, 1, 2, 3, 4, 5, 6]},
                            "winter": {"yawConstant": [1, 2, 3, 4, 5],
                                       "yawForward": [0.5, 2, 3, 4, "maxSpeed3p0_run_2"]}
                            }
    flight_list = []
    for flight_name, traj in flights_combinations.items():
        for flight_yaw, speeds in traj.items():
            for speed in speeds:
                if type(speed) == int:
                    flight_speed = f"maxSpeed{speed}p0"
                elif type(speed) == float:
                    unit, dec = str(speed).split(".")
                    flight_speed = f"maxSpeed{int(unit)}p{int(dec)}"
                else:
                    # speed must be str
                    flight_speed = speed
                flight_list.append(f"{flight_name}/{flight_yaw}/{flight_speed}/")

    for flight in flight_list:
        downloadSequence(flight=flight, environment=environment, datasetFolder=datasetFolder)


# CLI
if __name__ == "__main__":
    # fire.Fire(downloadSequence)
    main()
