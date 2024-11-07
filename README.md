# BagStacking: An Integrated Ensemble Learning Approach for Freezing of Gait Detection in Parkinson’s Disease

Abstract:
This paper introduces BagStacking, a novel ensemble learning method designed to enhance 1
the detection of Freezing of Gait (FOG) in Parkinson’s Disease (PD) by using a lower-back sensor to 2
track acceleration. Building on the principles of bagging and stacking, BagStacking aims to achieve 3
the variance reduction benefit of bagging’s bootstrap sampling while also learning sophisticated 4
blending through stacking. The method involves training a set of base models on bootstrap samples 5
from the training data, followed by a meta-learner trained on the base model outputs and true 6
labels to find an optimal aggregation scheme. The experimental evaluation demonstrates significant 7
improvements over other state-of-the-art machine learning methods on the validation set. Specifically, 8
BagStacking achieved a MAP score of 0.306, outperforming LightGBM (0.234) and classic Stacking 9
(0.286). Additionally, the run-time of BagStacking was measured at 3828 seconds, illustrating an 10
efficient approach compared to Regular Stacking’s 8350 seconds. BagStacking presents a promising 11
direction for handling the inherent variability in FOG detection data, offering a robust and scalable 12
solution to improve patient care in PD. 13
