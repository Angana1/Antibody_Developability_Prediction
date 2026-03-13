# 🧬 Antibody Developability Prediction
### Protein Design Hackathon, EPFL 
### Team Members: Angana, Annabelle, Onur
You can view the project presentation here: [Project Presentation](https://docs.google.com/presentation/d/1mEzIXelF-CP5MxH5i_iDIPjPIG9TaWtvpu_pxjqVafc/edit?usp=sharing)


### Project Overview
This project focuses on predicting antibody developability factors, namely thermostability and expressibility, using language model embeddings and machine learning pipelines. The goal is to evaluate antibody candidates early in the discovery pipeline without time-consuming, expensive wet lab tests, and to design generalisable models applicable to sequences collected under different experimental conditions. By leveraging bioinformatics, machine learning, and antibody sequence representations, our model helps researchers prioritize antibody candidates across multiple studies (cross-domain). 

### Methodology
We collected cross-domain antibody sequence data for thermostability and expressability factors by combining multiple studies from the [FLAb dataset](https://github.com/Graylab/FLAb). We used pretrained large language models such as ESM-2,IgLM (antibody-specific model), and Prot-T5 to generate high-quality embeddings. Then we trained lightweight machine learning classifiers (Logistic regression, Ridge regression and XGBoost classifier) to predict developability scores from the embeddings.

### Key findings
By utilising cross-domain hyperparameter tuning and out-of-distribution testing using multiple studies, we were able to achieve generalisable results for antibody fitness prediction. More interestingly, our ML model showed better cross-study generalisability when predicting aggregate developability scores (combination of thermostability and expressability) across studies, as compared to individual developability scores. Our results showed promising future avenues for cross-study generalisation of machine learning models for antibody fitness prediction.

This project was developed during the above hackathon for a project at Gray Lab, Johns Hopkins University, where interdisciplinary teams collaborated to design innovative solutions for biotechnology and drug discovery challenges.
