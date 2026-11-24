# 5. Conclusion and Future Enhancements

This paper presented TinyPAD, a multimodal TinyML and edge-AI framework for continuous health-anomaly detection with wearable IoT sensor networks. TinyPAD encodes every sensor with a tiny depthwise-separable CNN, fuses the modalities with masked attention trained under random modality dropout, personalises the detector to each new wearer using only five minutes of unlabelled rest data, and wraps the detector in a causal two-stage cascade in which always-on, low-power signals decide when to wake the duty-cycled sensors and the full model.

With respect to the research objectives, the experiments on WESAD (leave-one-subject-out) and the MIT-BIH Arrhythmia Database (inter-patient) support the following conclusions. (Objective 1) {{CONCL_OBJ1}} (Objective 2) {{CONCL_OBJ2}} (Objective 3) {{CONCL_OBJ3}}

The study has limitations that define future work. First, energy is reported through its measurable drivers—sensor on-time and multiply–accumulate operations—rather than measured on a physical wearable; deploying the int8 model on a microcontroller board and measuring current draw is the next step. Second, WESAD stress episodes are induced in a laboratory protocol and the dataset has 15 subjects; evaluation on free-living data with longer recordings and more participants is needed before clinical claims can be made, and TinyPAD is a research prototype, not a medical device. Third, the personalisation requires a short rest period with the wearer's normal physiology; handling wearers whose resting state is itself abnormal (for example, frequent ectopic beats at rest) and updating the baseline over days and seasons are open problems. Fourth, label-free test-time adaptation beyond baseline normalisation (AdaBN and deviation scoring) reduced quality in our validation study, which suggests investigating safer adaptation schemes such as confidence-filtered or federated personalisation. Finally, future work will extend the framework to further anomaly types (falls, atrial fibrillation, hypoxaemia), to explainable alerts that indicate which sensor triggered a detection, and to privacy-preserving collaborative training across wearers.

# Code and Data Availability

All code required to reproduce this study—dataset download and pre-processing (`prepare_health.py`), the TinyPAD model and the six baselines (`healthlib.py`), the experiment runner including the cascade, missing-sensor, calibration-length and int8 evaluations (`run_health.py`, `run_all.sh`), the validation-only design selection (`dev_select.py`), the analysis that produces every figure and table (`analyze_health.py`, `fig1_tinypad.py`) and the manuscript build script—is publicly available at https://github.com/vengicse/TinyPAD. Raw per-run records (one JSON file per run with per-subject metrics and scores) and execution logs are included. Both datasets are publicly available: WESAD from its authors' distribution page (https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx) and the MIT-BIH Arrhythmia Database from PhysioNet (https://physionet.org/content/mitdb/1.0.0/). The pre-processing script regenerates the exact windows used here; SHA-256 hashes of the processed arrays are listed in the repository.

# Authorship Contribution Statement

*(To be completed by the authors using the CRediT taxonomy; the roles below are a template.)* **Author 1:** Conceptualization, Methodology, Software, Formal analysis, Investigation, Writing – original draft. **Author 2:** Data curation, Validation, Visualization, Writing – review & editing. **Author 3:** Supervision, Resources, Project administration, Writing – review & editing. All authors have read and agreed to the published version of the manuscript.

# Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Declaration of Generative AI and AI-Assisted Technologies

During the preparation of this work the authors used Claude (Anthropic), an AI assistant, to help implement the data-preparation, model and analysis code, orchestrate the experiments on RunPod, search and verify bibliographic metadata (via the Crossref and OpenAlex APIs), and draft the text. All reported results were produced by the executed code and are traceable to the released run records. The authors reviewed and edited the content and take full responsibility for the publication. *(Authors: adapt this statement to the target journal's policy.)*

# Ethics Statement

This study uses only publicly available, de-identified datasets (WESAD and the MIT-BIH Arrhythmia Database) under their published terms of use; no new data were collected from human participants.

# Funding

*(To be completed by the authors.)*
