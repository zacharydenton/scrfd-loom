//! Pinned model weights in the shared Hugging Face cache.
use anyhow::{Context, Result, bail};
use hf_hub::{HFClientSync, HFError};
use std::{io::IsTerminal, path::PathBuf, sync::OnceLock};

/// Hugging Face model repository.
pub const REPO: (&str, &str) = ("immich-app", "buffalo_l");
/// Revision whose weights are covered by this crate's numerical tests.
pub const REVISION: &str = "d09715916a0778919a770c343533641e250b8699";
/// Original model file within the repository.
pub const FILE: &str = "detection/model.onnx";

/// Resolve the pinned weights, downloading only on a cache miss.
///
/// `offline` or `HF_HUB_OFFLINE=1` forbids network access. The client respects
/// `HF_HOME`, `HF_HUB_CACHE`, `HF_ENDPOINT` and the normal Hugging Face token settings.
pub fn weights(offline: bool) -> Result<PathBuf> {
    // The blocking client owns a runtime thread; reuse it across model loads.
    static CLIENT: OnceLock<std::result::Result<HFClientSync, String>> = OnceLock::new();
    let client = CLIENT
        .get_or_init(|| HFClientSync::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("cannot initialize Hugging Face client: {e}"))?;
    let offline = offline
        || std::env::var_os("HF_HUB_OFFLINE")
            .is_some_and(|value| !value.is_empty() && value != "0");
    fetch(client, offline)
}

fn fetch(client: &HFClientSync, offline: bool) -> Result<PathBuf> {
    let repository = client.model(REPO.0, REPO.1);
    let request = || repository.download_file().filename(FILE).revision(REVISION);
    match request().local_files_only(true).send() {
        Ok(path) => return Ok(path),
        Err(HFError::LocalEntryNotFound { .. }) => {}
        Err(error) => return Err(error).context("cannot read the Hugging Face cache"),
    }
    if offline {
        bail!(
            "{}/{} at {REVISION}: {FILE} is not cached and model downloads are disabled",
            REPO.0,
            REPO.1
        );
    }
    if std::io::stderr().is_terminal() {
        eprintln!("Downloading {}/{}: {FILE}", REPO.0, REPO.1);
    }
    request().send().with_context(|| {
        format!(
            "cannot download {}/{} at {REVISION}: {FILE}",
            REPO.0, REPO.1
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_miss_and_cached_load_need_no_server() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let client = HFClientSync::from_inner(
            hf_hub::HFClient::builder()
                .cache_dir(dir.path())
                .endpoint("http://127.0.0.1:9")
                .build()?,
        )?;
        let error = fetch(&client, true).unwrap_err().to_string();
        assert!(error.contains("not cached") && error.contains(FILE));
        let path = dir.path().join(format!(
            "models--{}--{}/snapshots/{REVISION}/{FILE}",
            REPO.0, REPO.1
        ));
        std::fs::create_dir_all(path.parent().unwrap())?;
        std::fs::write(&path, b"cached model")?;
        // Even with downloads enabled, a cache hit does not contact the server.
        for offline in [true, false] {
            assert_eq!(fetch(&client, offline)?, path);
        }
        Ok(())
    }
}
