import os
import raha


def main():
    dataset_name = "toy"
    dataset_dictionary = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join("datasets", dataset_name, "dirty.csv")),
        "clean_path": os.path.abspath(os.path.join("datasets", dataset_name, "clean.csv")),
    }

    detector = raha.Detection()
    detector.LABELING_BUDGET = 5
    detected_cells = detector.run(dataset_dictionary)

    data = raha.Dataset(dataset_dictionary)
    print("检测结果 ===", detected_cells)
    data.detected_cells = detected_cells

    corrector = raha.Correction()
    corrector.LABELING_BUDGET = 5
    #corrector.USE_VICINITY_BASED_MODEL = False
    corrector.NUM_WORKERS = 1
    corrected_cells = corrector.run(data)
    print("纠正结果 ===", corrected_cells)
    data.create_repaired_dataset(corrected_cells)
    output_path = os.path.abspath(os.path.join("datasets", dataset_name, "repaired.csv"))
    data.write_csv_dataset(output_path, data.repaired_dataframe)

    print(output_path)
    print(data.get_data_cleaning_evaluation(corrected_cells))


if __name__ == "__main__":
    main()
