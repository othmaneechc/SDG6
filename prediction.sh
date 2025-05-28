#!/bin/bash

# List of countries
COUNTRIES=(
    "Liberia"
    "Madagascar"
    "Malawi"
    "Burkina Faso"
    "Burundi"
    "Cameroon"
    "Seychelles"
    "Congo"
    "Cabo Verde"
    "Ethiopia"
    "Togo"
    "Chad"
    "Mauritania"
    "Mali"
    "Niger"
    "Mauritius"
    "Zambia"
    "Côte d'Ivoire"
    "Gambia"
    "Nigeria"
    "Angola"
    "Central African Rep."
    "Egypt"
    "Mayotte"
    "Tanzania"
    "Sierra Leone"
    "Libya"
    "Namibia"
    "Djibouti"
    "Algeria"
    "Comoros"
    "Benin"
    "Lesotho"
    "Senegal"
    "Tunisia"
    "eSwatini"
    "Ghana"
    "Kenya"
    "South Africa"
    "Botswana"
    "Eritrea"
    "Eq. Guinea"
    "Gabon"
    "Dem. Rep. Congo"
    "Guinea-Bissau"
    "Mozambique"
    "Zimbabwe"
    "Rwanda"
    "Sao Tome and Principe"
    "Guinea"
    "Uganda"
)

# Define constant paths for the model checkpoints
CHECKPOINT_PATH="/dkucc/home/oe23/MODELS/prediction/weights/imbalanced_sew/checkpoint.pth"
KNN_CLASSIFIER_PATH="/dkucc/home/oe23/MODELS/prediction/weights/imbalanced_sew/knn_classifier.pth"

# Loop through each country and run the updated python script
for COUNTRY in "${COUNTRIES[@]}"; do
    # Construct the image directory and output CSV file paths.
    DIRECTORY="/work/lamlab/${COUNTRY}/"
    CSV_PATH="/dkucc/home/oe23/MODELS/prediction/${COUNTRY}/sewage.csv"

    # Special handling for countries with special characters (e.g., Côte d'Ivoire)
    if [ "$COUNTRY" == "Côte d'Ivoire" ]; then
        DIRECTORY="/work/lamlab/Côte d'Ivoire/"
        CSV_PATH="/dkucc/home/oe23/MODELS/prediction/Côte d'Ivoire/sewage.csv"
    fi

    CMD="python /dkucc/home/oe23/MODELS/prediction/predict.py \
         --checkpoint_path '${CHECKPOINT_PATH}' \
         --knn_classifier_path '${KNN_CLASSIFIER_PATH}' \
         --csv_path '${CSV_PATH}' \
         --directory '${DIRECTORY}'"

    echo "Running: $CMD"
    eval $CMD
done
