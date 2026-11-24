---
title: "A Multimodal TinyML and Edge-AI Framework for Continuous Health Anomaly Detection Using Wearable IoT Sensor Networks"
---

**Author 1**^a,\*^, **Author 2**^b^, **Author 3**^a^ *(author names, affiliations and corresponding-author e-mail to be completed by the authors)*

^a^ Department, Institution, City, Country ^b^ Department, Institution, City, Country

\* Corresponding author.

**Abstract**

Continuous detection of health anomalies on wearable devices must cope with large differences between people, multiple sensors that may drop out, microcontroller-class memory and computation, and limited battery energy. This paper proposes TinyPAD, a tiny personalised multimodal anomaly detector for wearable IoT sensor networks. TinyPAD encodes each sensor modality with a depthwise-separable 1-D CNN, fuses the modalities with masked attention trained under random modality dropout, and personalises the detector to a new wearer without labels or fine-tuning by expressing every signal relative to the wearer's own five-minute resting baseline. A causal two-stage cascade keeps power-hungry sensors and the full model off unless an always-on, low-power stage flags a possible anomaly. TinyPAD was evaluated on WESAD (stress detection from seven chest and wrist modalities, leave-one-subject-out) and on the MIT-BIH Arrhythmia Database (abnormal-beat detection, inter-patient DS1→DS2 split) against a hand-crafted-feature random forest, a convolutional auto-encoder, early- and late-fusion CNNs, a CNN-LSTM and a tiny transformer under identical inputs and training budgets; all experiments were executed on RunPod. On WESAD, TinyPAD achieved the highest accuracy ({{wesad_t_acc}}), stress F1 ({{wesad_t_f1}}), sensitivity ({{wesad_t_sens}}) and balanced accuracy ({{wesad_t_bacc}}). On MIT-BIH it achieved the highest sensitivity ({{mitbih_t_sens}}), balanced accuracy ({{mitbih_t_bacc}}) and AUC ({{mitbih_t_auc}}), detected {{mitbih_t_seS}} of supraventricular beats (best baseline {{mitbih_max_seS_other}}), and matched the best baseline in F1 ({{mitbih_t_f1}} versus {{mitbih_b_f1}}). The models have {{mitbih_params}}–{{wesad_params}} parameters ({{mitbih_kb8}}–{{wesad_kb8}} kB with int8 weights, with less than one point of F1 change), personalisation saturates after about two minutes of rest data, and TinyPAD keeps an average F1 of {{wesad_miss_t_mean}} on WESAD when any one sensor is missing. The cascade reduces duty-cycled sensor on-time to {{wesad_c98_on}} of the monitored time on WESAD at an F1 cost, and reduces computation to {{mitbih_c90_comp}} on MIT-BIH while increasing F1. Baseline advantages that remain (specificity, subject-averaged AUC on WESAD) are reported alongside.

**Keywords:** wearable IoT; TinyML; edge AI; health anomaly detection; multimodal fusion; personalisation; stress detection; arrhythmia detection; energy-efficient monitoring
