# Discover, Explain, Improve: An Automatic Slice Detection Benchmark for Natural Language Processing

<img width="808" alt="Screen Shot 2023-09-10 at 1 03 29 AM" src="https://github.com/Wenyueh/DEIM/assets/28013619/314275fd-f77c-48c1-87ee-a707d8506e2f">

# Abstract

Pretrained natural language processing (NLP) models have achieved high overall performance, but they still make systematic errors. Instead of manual error analysis, research on slice detection models (SDM), which automatically identify underperforming groups of datapoints, has caught escalated attention in Computer Vision for both understanding model behaviors and providing insights for future model training and designing. However, little research on SDM and quantitative evaluation of their effectiveness have been conducted on NLP tasks. Our paper fills the gap by proposing a benchmark named ``Discover, Explain, Improve (DEIM)" for classification NLP tasks along with a new SDM Edisa. Edisa discovers coherent and underperforming groups of datapoints; DEIM then unites them under human-understandable concepts and provides comprehensive evaluation tasks and corresponding quantitative metrics. The evaluation in DEIM shows that Edisa can accurately select error-prone datapoints with informative semantic features that summarize error patterns. Detecting difficult datapoints directly boosts model performance without tuning any original model parameters, showing that discovered slices are actionable for users.

