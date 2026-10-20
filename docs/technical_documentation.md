# P3 Technical Documentation: On-Device Cybersecurity and Digital Safety AI

## Status and evidence boundary

This document describes the code, artifacts, and reports present in this repository at the time of inspection. The Streamlit prototype, three scikit-learn detector artifacts, an ExecuTorch-format Llama model bundle, and training reports are present in the working tree. The small detector artifacts and reports, datasets, and Llama bundle are ignored by Git, so a fresh clone does not include them. No ARM device, mobile package, benchmark results, or device test logs were found. Treat detector metrics below as the saved split reports, not as independent validation or proof of production readiness.

## 1. Problem statement and objectives

The project aims to help users review potentially unsafe SMS messages, URLs, and network flow records locally. The prototype provides separate machine learning classifiers and a language model based explanation/advice step. Its objectives, as evidenced by the implementation, are:

- Classify SMS as `ham` or `spam`.
- Classify submitted URLs as `benign` or `phishing` without visiting them.
- Classify numeric network-flow records as `benign` or `suspicious`.
- Present the detector result and scores, and give concise, evidence-bounded safety advice.
- Allow inference without sending scanned text or URLs to a remote detection service.

This is a user-operated prototype. It does not monitor SMS or network traffic in the background, block threats, or establish that a benign result is safe.

## 2. System architecture

```text
User input in Streamlit
  ├─ SMS text ── TF-IDF word features ── Logistic Regression
  ├─ URL text ── TF-IDF character features ── Logistic Regression
  └─ CSV flows ── numeric feature selection/imputation ── Random Forest
                       │
                       ├─ label and detector scores/results in UI
                       └─ bounded result summary ── local ExecuTorch Llama 3.2 1B
                                                    └─ advice; fallback if unavailable
```

`app/main.py` loads joblib artifacts from `models/`, caches them for the Streamlit process, and calls `src/detection_results.py` and `src/advisor.py`. Single SMS and URL results include predicted labels and `predict_proba` scores. Network results show predicted rows and summarize class counts for the advisor. The network result display is capped at 100 rows, while the downloadable CSV includes the full analyzed input and predictions.

## 3. SMS dataset, preprocessing, and classifier

The training script expects the UCI SMS Spam Collection file (`SMSSpamCollection`) somewhere below `data/raw/`. A corpus file and accompanying readme are present locally. The readme identifies 5,574 English messages, 4,827 ham and 747 spam. Training drops missing and exact duplicate label/message records, normalizes labels to lowercase, retains only `ham` and `spam`, and uses an 80/20 stratified split with `random_state=42`.

The pipeline applies word TF-IDF with unigrams and bigrams, `min_df=2`, and sublinear term frequency, followed by class-weight-balanced Logistic Regression (`max_iter=1000`, seed 42). Model and report outputs are `models/sms_detector.joblib` and `reports/sms_report.txt`; these are present locally but ignored by Git.

## 4. Phishing URL dataset, preprocessing, and classifier

Training expects `data/raw/urls/urls.csv`, with `url` and `label` columns. Locally, the raw `openphish_raw.txt` list and generated CSV are present. `src/prepare_data.py` reads the OpenPhish lines and adds five hard-coded benign example URLs (Google, GitHub, Wikipedia, Python.org, and Stack Overflow); it labels the feed entries `phishing`. The training script strips strings, lowercases labels, removes empty URLs and labels outside `benign`/`phishing`, deduplicates by URL, and requires both classes with at least two examples each. It then uses an 80/20 stratified split with seed 42.

The model is a character-level TF-IDF vectorizer (3–5 character n-grams, `min_df=1`, at most 100,000 features, sublinear term frequency) followed by class-weight-balanced Logistic Regression (`max_iter=1000`, seed 42). The application accepts only URL strings starting with `http://` or `https://`; the separate CLI also checks for a hostname. Neither path fetches the submitted URL. Artifacts are `models/url_detector.joblib` and `reports/url_report.txt`.

The benign class contains only five hand-selected domains in the generation code. The saved evaluation test set contains one benign sample, so the reported benign performance is not a reliable estimate of real-world generalization.

## 5. Network intrusion dataset and classifier

`src/train_network.py` reads the first CSV found recursively in `data/raw/network/`. The checked-in working copy has a six-row example CSV with columns `duration`, `protocol_type`, `src_bytes`, `dst_bytes`, and `label`. Training requires a `Label` column (case-insensitive lookup), maps `benign` and `normal` to `benign`, and maps every other label to `suspicious`. It converts candidate features to numeric, turns infinities into missing values, drops all-missing and constant columns, removes duplicate feature/target rows, and performs a stratified 80/20 split with seed 42.

The model pipeline imputes missing values with the median and fits a 100-tree class-weight-balanced Random Forest (`n_jobs=2`, seed 42). The feature column list is stored alongside the model in `models/network_detector.joblib` and used by the UI to select columns in uploaded CSVs. Non-numeric fields such as `protocol_type` become missing during training and are consequently excluded if entirely missing. At inference, absent required feature columns cause an error; values that cannot be converted become missing and are imputed.

The tiny example dataset is not representative of operational intrusion detection data. Use a properly licensed, documented, sufficiently large labelled dataset and validate the feature schema before drawing conclusions.

## 6. Llama 3.2 1B Instruct advisory component

`src/advisor.py` loads a local ExecuTorch `NativeLlamaRunner` lazily when advice is requested. The expected files are `Llama-3.2-1B-Instruct-QLORA_INT4_EO8.pte`, `params.json`, and `tokenizer.model` under `models/Llama-3.2-1B-Instruct-INT4/`. A local bundle exists, including a roughly 1.18 GB `.pte` file. Parameters describe a 1B model configuration with 16 layers and grouped quantization; the bundle README identifies the base as Meta Llama 3.2 1B and the artifact as an ExecuTorch QLoRA INT4 export. The repository does not document a fine-tuning dataset or establish that project-specific fine-tuning occurred.

Only the component name, classification label, bounded class scores, and optional aggregate network counts are passed to the prompt builder. Scanned message text, URLs, and uploaded flow rows are not included. Prompt size and generated output are bounded, generation uses temperature zero, and generation errors/empty output fall back to fixed local advice. The UI labels advice as separate from detector evidence. The LLM is explanatory, not a detector; its text may still be inaccurate and should not override the detector output.

## 7. ExecuTorch runtime and deployment

The current Python adapter imports `executorch.examples.models.llama.runner.native.NativeLlamaRunner` from the installed ExecuTorch package at request time. It supplies the `.pte`, model parameters, tokenizer, a 512-token sequence limit, and disables the KV cache. The code caps newly generated tokens at 96. The checked-in `requirements.txt` pins `executorch==1.5.1`, `sentencepiece`, `pytorch_tokenizers`, and a large environment-wide set of packages. There is no dedicated minimal deployment lockfile, native mobile application, Android/iOS build configuration, or documented export/build script in the project source.

Model directories and binary artifacts are ignored by `.gitignore`; deployment requires provisioning them separately. Preserve the model's applicable license and notices when redistributing it. The local model README contains the Llama 3.2 Community License terms and points to Meta's acceptable use policy.

## 8. Offline operation and privacy

The application code performs predictions locally and makes no network request for SMS or URL analysis. The URL is parsed or classified as text; the app explicitly does not open it. The advisor loads local model files. Thus inference can be offline after software and artifacts have been installed. Initial dependency/model acquisition may require network access, but no downloader is implemented in the app.

No persistence or telemetry of submitted SMS/URLs is implemented in the inspected source. The advisor intentionally receives only structured detector output, and the network advisor receives aggregate counts rather than uploaded rows. Network CSV content is shown in the local UI and is included in a user-triggered downloadable output. This privacy description is limited to repository code; hosting Streamlit on a remote server would mean inputs are processed on that server, not on the user's device. Use the app on a trusted local machine for local-only processing.

## 9. Application interface

The Streamlit UI in `app/main.py` has three tabs:

- **SMS Scanner:** text area capped at 10,000 characters; displays the predicted class, class scores, and advice.
- **URL Scanner:** input capped at 4,096 characters; requires an HTTP(S) prefix; displays label, scores, and advice.
- **Network Scanner:** accepts CSV uploads, checks for the model's required columns, predicts rows, displays up to 100 rows, counts suspicious predictions, and offers a full-result CSV download.

Missing detector model artifacts produce instructions to run the corresponding training script. The UI includes warnings that model scores do not prove a threat or safety. It does not expose a mobile-native interface or automatic monitoring.

## 10. Evaluation metrics and measured results

The following values are transcribed from the saved report files. They come from the training scripts' single stratified random holdout splits (seed 42); there is no cross-validation, external test set, or device evaluation report in the repository.

| Detector | Holdout support | Accuracy | Class precision / recall / F1 | Confusion matrix |
|---|---:|---:|---|---|
| SMS | 1,034 (903 ham, 131 spam) | 0.98 | ham 0.99 / 0.99 / 0.99; spam 0.90 / 0.92 / 0.91 | `[[890, 13], [11, 120]]` (actual rows ham, spam; predicted columns ham, spam) |
| URL | 61 (1 benign, 60 phishing) | 0.98 | benign 0.50 / 1.00 / 0.67; phishing 1.00 / 0.98 / 0.99 | `[[1, 0], [1, 59]]` (actual rows benign, phishing; predicted columns benign, phishing) |
| Network | 2 (1 benign, 1 suspicious) | 1.00 | benign 1.00 / 1.00 / 1.00; suspicious 1.00 / 1.00 / 1.00 | `[[1, 0], [0, 1]]` (actual rows benign, suspicious; predicted columns benign, suspicious) |

The network result is two correctly classified test rows and should not be interpreted as evidence of useful accuracy. The URL test has only one benign record. Metrics describe these exact split samples only. The network UI does not compute or display probability scores; its advisor score fields represent fractions of predicted labels, not calibrated confidence. SMS and URL `predict_proba` values are model scores and are not necessarily calibrated probabilities.

## 11. False positives, false negatives, and limitations

For the saved SMS holdout, there were 13 ham messages predicted spam (false positives) and 11 spam messages predicted ham (false negatives). For the URL holdout, one phishing URL was predicted benign (false negative); the single benign URL was predicted phishing correctly. The network holdout had no errors in its two rows, which is too small to estimate either error rate. These counts follow the confusion matrices above.

Additional limitations supported by the implementation/data include narrow or dated datasets, possible source and distribution bias, random rather than time/domain/group-separated URL evaluation, very small URL benign class, tiny network sample, numeric-only flow features, collapse of all attack types into one class, no drift monitoring or threshold tuning, and no calibrated risk threshold. The UI requires manual input/upload and is not an inline blocker. The LLM output is generated text and may omit or misstate advice; fallback behavior is available when it cannot run. A benign prediction is not a safety guarantee.

## 12. ARM deployment and performance measurements

The intended direction is local inference on ARM hardware, and the project includes an ExecuTorch-exported model artifact. However, no ARM target, device-specific build, benchmark, latency, throughput, peak memory, battery use, or thermal measurement was found. No ARM performance claim can therefore be made. The existing Python runner path is not evidence of a completed mobile deployment. Record device model, OS, runtime build, model files/checksums, warm/cold latency, memory, and test conditions before reporting ARM performance.

## 13. Installation and execution

The commands below reflect the repository paths. Training requires the local datasets described above, and inference requires the generated joblib artifacts. The large Llama bundle is separately provisioned and is ignored by Git. The pinned `requirements.txt` is broad and includes packages unrelated to this app; a clean, minimal supported environment has not been established in this project.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Train models (datasets must already be placed under data/raw/)
python src/train_sms.py
python src/prepare_data.py       # expects data/raw/urls/openphish_raw.txt
python src/train_url.py
python src/train_network.py     # expects a labelled CSV under data/raw/network/

# Run the interface
streamlit run app/main.py
```

For SMS training, extract the UCI corpus so `SMSSpamCollection` is found below `data/raw/`. For URL training, the current preparation helper constructs the labelled file from the raw OpenPhish list. For network training, provide a CSV containing a `Label` column and numeric feature columns. If the local ExecuTorch runner or Llama files are unavailable, detector results remain usable and the advisor returns fallback text. The repository does not define a supported ARM installation procedure.

## 14. References and dataset attribution

- Almeida, T. A., Gómez Hidalgo, J. M., and Yamakami, A. “Contributions to the Study of SMS Spam Filtering: New Collection and Results.” ACM Symposium on Document Engineering (DocEng), 2011. The local corpus readme requests citation and attributes the corpus to Almeida and Gómez Hidalgo. [SMS Spam Collection project page](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/).
- OpenPhish community feed/list is the source named in the local URL preparation script. The repository does not record an acquisition date, feed snapshot provenance, or license/redistribution terms; confirm current terms before redistribution. [OpenPhish](https://openphish.com/).
- The local six-row network CSV has no source, license, or provenance metadata. It should be treated as an example fixture, not as a traceable benchmark dataset.
- Meta, [Llama 3.2](https://www.llama.com/llama-downloads/) and [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). The model bundle's README identifies its base model as `meta-llama/Llama-3.2-1B` and its artifact as an ExecuTorch QLoRA INT4 export. Retain applicable license notices and follow the license and acceptable-use terms for use or redistribution.
- ExecuTorch runtime project: [ExecuTorch documentation](https://pytorch.org/executorch/).
