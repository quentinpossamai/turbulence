import posture_error_estimation
import util


def main():
    f = util.DataFolder('euroc_mav')
    f.get_files_paths('.csv', f.folders['raw'][0])
    f.get_unique_file_path('.csv', f.folders['raw'][0])
    estimator = posture_error_estimation.ErrorEstimation()


if __name__ == '__main__':
    main()
