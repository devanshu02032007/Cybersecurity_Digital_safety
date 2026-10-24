# Offline Ubuntu deployment

The app binds to `127.0.0.1:8501`, disables Streamlit usage statistics, and
does not call detection or AI services. Start it with:

```bash
streamlit run app/main.py
```

Install `requirements.txt` and provision the existing files in `models/` while
connected, then disconnect before starting the app. Runtime inference never
downloads models or datasets. Keep the three Joblib detector artifacts in
`models/`; training scripts overwrite them, so do not run training when you
intend to preserve the current trained models.

Llama advice is optional. Its artifact bundle is expected in
`models/Llama-3.2-1B-Instruct-INT4/`. The adapter uses the local `.pte`,
`params.json`, `config.json`, and `tokenizer.model`. Install a compatible
ExecuTorch/PyTorch runtime and native custom-op libraries separately before
expecting generation. The current project environment lacks `torch`, which
the bundled `NativeLlamaRunner` imports. If Llama cannot initialize, the app
identifies that condition and provides deterministic local advice, clearly
marked as fallback. No model is fetched automatically.

Scanned message text, URLs, and CSV rows are not written to application logs
or sent to external services. The app does not maintain scan history. Uploaded
CSV data remains in the local Streamlit process; output is only downloaded if
the user selects the download button. Do not expose the Streamlit port on a
public interface if local-only processing is required.
