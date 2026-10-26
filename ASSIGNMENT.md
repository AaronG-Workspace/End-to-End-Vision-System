# Week 07 Capstone: End-to-End Vision System with Anomaly Detection, Adversarial Robustness, and Governance Memo

Due Oct 26, 2026 11:59 PM. 150 points. CLO1 to CLO6.

## Scenario
You are the lead computer vision engineer at MedVision Systems, a healthcare AI company contracted to build an automated quality control system for a medical device manufacturer. Your CTO and compliance officer need a complete, professional-grade vision system that detects anomalies in device components, withstands adversarial perturbations, and ships with a governance memo connecting your quantitative failure evidence to specific regulatory obligations. Build your complete system using Python/PyTorch with Claude Code assistance.

Your system must include:
1. A dataset you curate from a publicly available source (MVTec AD, a Kaggle manufacturing inspection dataset, or synthetic data generated with Python) with documented curation decisions covering class balance, split rationale, and preprocessing choices.
2. An unsupervised anomaly detection pipeline using PatchCore, a reconstruction-based autoencoder, or a self-supervised approach.
3. A demonstrably leakage-free evaluation protocol with a held-out test set that was never touched during model selection, reporting AUROC, F1, and precision-recall.
4. An adversarial robustness assessment implementing at least FGSM in PyTorch across at least three epsilon values, with a clean-versus-attacked performance table.
5. INT8 or FP16 post-training quantization benchmarked against the full-precision model in an accuracy-latency-model-size tradeoff table.
6. A one-page governance memo in Google Docs or Word that maps your specific quantitative failure evidence to EU AI Act, FDA 510(k)/De Novo, or NIST AI RMF obligations, explicitly distinguishing what the regulation compels to be measured from what your current system demonstrates, with labeled source quality.

Use Claude Code to scaffold your pipeline, debug quantization errors, and generate your tradeoff tables. Record a camera-on walkthrough presenting all six components to your CTO and compliance officer as if this is a pre-production readiness review.

## Requirements
8 to 10 minute camera-on project walkthrough video + working Python/PyTorch system artifact + one-page governance memo with cited regulatory frameworks.

## Video
- Maximum duration: 10 minutes
- Format: MP4 (preferred), MOV, or WebM
- Face visible on camera for the entire video. Picture-in-picture during screen share is fine. Voice-over slides without a visible face is not permitted.
- Explain in your own words. No reading a verbatim script. A brief outline or speaker notes is fine.
- Clear narration, quiet space.
- Submit via the LMS dropbox.

## Rubric (30 points each, 150 total)
| Criterion | Exemplary (30) |
|---|---|
| Dataset Curation and Pipeline Architecture (CLO1, CLO6) | Student-sourced dataset is well curated and documented, and the pipeline architecture is justified against the problem rather than defaulted |
| Leakage-Free Evaluation and Anomaly Detection Performance (CLO3, CLO6) | Evaluation protocol is provably leakage-free and anomaly detection performance is reported with appropriate metrics and honest error analysis |
| Adversarial Robustness Assessment (CLO3) | Adversarial attack executed correctly, degradation quantified across attack strength, and results reported even where unflattering |
| Edge Deployment and Accuracy-Efficiency Tradeoff (CLO2) | Model deployed to an edge target with the accuracy-efficiency trade-off measured on realistic hardware assumptions |
| Governance Memo Quality and Regulatory Mapping (CLO4, CLO5) | Memo connects the student's own quantitative failure evidence to named framework provisions and reads as something a professional team could act on |

Lower bands: Proficient 26, Developing 21, Beginning 15 per criterion. Developing on adversarial = "attack attempted but results are thin or misinterpreted". Developing on edge = "trade-off asserted rather than measured". Developing on memo = "cites frameworks generically or is disconnected from the measured results".
