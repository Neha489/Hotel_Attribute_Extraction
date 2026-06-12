Hotel Attribute Extraction and Tiering Using Machine Learning

1. Introduction

Online hotel reviews contain valuable information about the actual experiences of travellers. While ratings provide an overall opinion, review text often contains detailed feedback about specific hotel attributes such as cleanliness, room size, staff behaviour, Wi-Fi quality, and bed comfort. Extracting these signals manually is difficult because of the large volume of reviews.

The goal of this project is to build a machine learning pipeline that extracts hotel attribute information from unstructured review text, predicts sentiment for each attribute, and assigns quality tiers that can be used in a corporate travel booking platform.

 2. Dataset and Exploratory Data Analysis

The TripAdvisor Hotel Reviews dataset from Kaggle was used for this project. The dataset contains hotel review text and overall review ratings.

Dataset Overview
Total Reviews: 20,491
Features: Review Text and Rating
Rating Scale: 1 to 5

Rating Distribution

Rating        Count 
5                 9,054
4                 6,039 
3                 2,184 
2                 1,793 
1                 1,421 

The distribution shows that most reviews are positive, which is common in hotel review datasets.

Review Length Analysis

Review length statistics:

Average Length: 725 characters
Minimum Length: 44 characters
Maximum Length: 13,501 characters

The review length distribution was highly right-skewed. Most reviews were relatively short, while a small number of reviews were extremely long and detailed.

Challenges Identified

During exploration, several challenges were observed:

•	Reviews discuss multiple attributes in a single paragraph.
•	The dataset does not contain hotel identifiers.
•	Attribute-level labels are not available.
•	Review ratings represent overall experience and may not reflect sentiment for individual attributes.

These challenges influenced the design of the labelling and modelling approach.

3. Attribute Selection and Labeling Strategy

Five hotel attributes were selected:
1. Cleanliness
2. Room Size
3. Staff Service
4. Bed Quality
5. Wi-Fi Quality

The selected attributes provide a combination of service quality, comfort, and facility-related information.

Review Segmentation
Each review was split into smaller segments using NLTK sentence tokenization. Working at the sentence level allows the model to focus on attribute-specific information rather than the entire review.

Example:
Review:
"The room was spacious and clean. Staff were friendly and helpful."
Segments:
The room was spacious and clean.
Staff were friendly and helpful.

Attribute Extraction
A keyword-based matching approach was used to identify attribute-related segments.

Examples:
Cleanliness Keywords:
[clean, Dirty, Hygiene, spotless]
Staff Service Keywords:
[staff, Service, Receptionist, manager]
Room Size Keywords:
[room size, Spacious, Large, small]

Only segments containing attribute-related keywords were retained.
This process produced approximately 56,000 attribute-specific segments.

Sentiment Labeling
Since attribute-level sentiment labels were unavailable, weak supervision was used.

Review ratings were converted into sentiment labels:
•	Rating 4–5 → Positive
•	Rating 3 → Neutral
•	Rating 1–2 → Negative

Although simple, this approach enabled creation of a labeled dataset without manual annotation.

A limitation of this method is that an individual sentence may express positive sentiment even if the overall review rating is negative.

4. Modeling Approach

The objective was to predict sentiment for each extracted attribute segment.
Text Representation

TF-IDF vectorization was used to convert text into numerical features.

Configuration:
Unigrams and Bigrams
Maximum Features: 5,000

TF-IDF was selected because it is simple, interpretable, and performs well on text classification tasks.

Model Selection

Logistic Regression was chosen as the classification model.

Reasons for selection:
1.	Fast training
2.	Easy interpretation
3.	Strong baseline performance
4.	Suitable for sparse TF-IDF features

To address class imbalance, class weights were set to "balanced".

Train-Test Split

Data was split into:
Training: 80%
Testing: 20%

Stratified sampling was used to preserve class distribution.

5. Evaluation Results

Separate models were trained for each attribute.

Cleanliness

Accuracy          68%   
Weighted F1 Score  0.71 

 Staff Service
Accuracy          65%   
Weighted F1 Score 0.68  

Wi-Fi
Accuracy          62%   
Weighted F1 Score  0.62  

Observations

The positive class achieved the best performance across all attributes because positive examples dominated the dataset.
The neutral class showed the weakest performance. This was expected because sentiment labels were generated from overall review ratings rather than sentence-level sentiment. As a result, some neutral examples contained positive or negative language, creating label noise.
Despite these limitations, the models successfully learned useful attribute-level sentiment patterns.

6. Tier Aggregation Framework

The next step was converting attribute sentiment predictions into quality tiers.

Sentiment Score Mapping

Predicted sentiments were converted into numeric scores:
Positive = +1
Neutral = 0
Negative = -1

Score Aggregation

For each review-attribute pair, sentiment scores were averaged.

Example:

Attribute     Score
Cleanliness    1.0   
Staff Service  0.5   
Wi-Fi          -1.0  

Evidence Count
The number of supporting attribute-related segments was calculated.
This was important because many review-attribute pairs contained only one evidence sentence.
Evidence Distribution:
Single Evidence Segment: 29,697
Multiple Evidence Segments: 13,862


Confidence Assignment

Confidence levels were introduced:
High Confidence - 2 or more evidence segments
Low Confidence - 1 evidence segment

Tier Assignment
The following tiering strategy was designed:

Tier    Condition                           
Elite    Score ≥ 0.75 and Evidence Count ≥ 2 
Superior  Score ≥ 0.25                        
Premium  Score between -0.25 and 0.25        
Fail     Score < -0.25                       

This approach prevents a single positive sentence from automatically producing an Elite rating.

Final Tier Distribution

Tier     Count  
Superior  22,651 
Premium  8,422  
Fail      7,354  
Elite     5,132  

The distribution appears balanced and more realistic than assigning Elite purely based on sentiment score.

7. Explainability and Evidence Extraction

One of the key requirements of the project was explainability.
For every tier assignment, the system stores the sentences that contributed to the final decision.

Example:
Attribute: Staff Service
Evidence:
"Staff were friendly and helpful."
"Reception team quickly resolved our issue."

Predicted Sentiment:
Positive
Assigned Tier:
Elite

This allows users to understand why a particular rating was assigned rather than relying on a black-box prediction.

8. Limitations and Future Improvements

Although the system produces useful results, several limitations remain.
Current Limitations

1. Attribute extraction relies on keyword matching and may miss implicit mentions.
2. Weak supervision introduces noisy sentiment labels.
3. No hotel identifier was available, so review-level aggregation was used as a proxy for hotel-level aggregation.
4. Numeric attributes such as room size were not extracted as exact measurements.

Future Improvements

Given additional development time, the following improvements would be implemented:

1. Manual annotation of a high-quality validation dataset.
2. Transformer-based models such as BERT or DistilBERT.
3. Named Entity Recognition for more accurate attribute extraction.
4. Probability-based scoring instead of discrete sentiment scores.
5. True hotel-level aggregation using hotel identifiers.
6. Extraction of factual attributes such as room dimensions and location proximity.

9. Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for hotel attribute extraction and tier assignment using unstructured review text.

The system successfully identifies hotel-related attributes, predicts attribute-level sentiment, aggregates signals into meaningful quality tiers, and provides evidence supporting each decision. While the approach uses simple models and weak supervision, it provides a scalable and explainable foundation for hotel quality assessment in a corporate travel booking platform.

The project highlights the importance of combining text mining, machine learning, aggregation logic, and explainability to transform unstructured customer feedback into actionable business insights.
