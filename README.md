# Hotel Attribute Extraction & Tiering

This project builds an end-to-end Machine Learning pipeline that extracts hotel quality signals from unstructured TripAdvisor reviews and converts them into structured hotel attribute ratings.

The system identifies hotel-related attributes from review text, predicts sentiment for each attribute, aggregates evidence, and assigns quality tiers that can be used in a corporate travel booking platform.

Selected Attributes

The following hotel attributes were analyzed:

Cleanliness
Room Size
Staff Service
Bed Quality
Wi-Fi Quality

## Pipeline

Review Text
→ Sentence Segmentation
→ Attribute Extraction
→ Weak Label Generation
→ TF-IDF Feature Engineering
→ Logistic Regression Classification
→ Sentiment Prediction
→ Score Aggregation
→ Confidence Estimation
→ Tier Assignment
→ Evidence Extraction

Dataset

TripAdvisor Hotel Reviews Dataset

Source:
https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews

Installation

```bash
pip install -r requirements.txt
```

Running the Project

Open and run:

```bash
notebooks/Hotel_Attribute_Extraction.ipynb
```

Run all cells from top to bottom.

The notebook performs:

1. Data loading
2. Data cleaning
3. Sentence segmentation
4. Attribute extraction
5. Weak supervision labeling
6. Model training
7. Model evaluation
8. Tier assignment
9. Evidence extraction

Outputs

Generated outputs include:

* attribute_scores.csv
* final_output.csv

Models

The project trains separate models for:

* Cleanliness
* Staff Service
* Wi-Fi
* Room Size
* Bed Quality

Each model uses:

* TF-IDF Vectorization
* Logistic Regression Classifier

Limitations

* Attribute extraction relies on keyword matching.
* Weak supervision introduces label noise.
* Dataset does not contain hotel identifiers.
* Review-level aggregation is used as a proxy for hotel-level aggregation.

For detailed discussion, see WRITEUP.md.

