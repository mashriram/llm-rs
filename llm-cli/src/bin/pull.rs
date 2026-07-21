//! `llm pull <model>` — download a model from the Hugging Face Hub.
//!
//! Resolves a bare name (searches HF) or an explicit `owner/repo`, lists the
//! available quantization/format variants found in that repo, recommends one
//! based on this machine's detected `HardwareProfile` (free RAM/VRAM), and
//! downloads the chosen weight file plus whatever tokenizer/config sidecars
//! exist alongside it — mirroring what `llm serve`/`chat` expect to find next
//! to a model file.
//!
//! Supports GGUF-quantized community repos (any quant naming: Q4_K_M, Q8_0,
//! F16, IQ4_XS, ...) and native HF safetensors repos. Detects (but does not
//! yet implement dequantization for) bitsandbytes/AWQ/GPTQ-quantized
//! safetensors repos via `config.json`'s `quantization_config` field, and
//! fails with a clear, actionable error rather than silently mis-loading
//! them — see `llm-core/src/loader` for what actually loads today.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use futures_util::StreamExt;
use serde::Deserialize;

use llm_core::profile::{BackendChoice, HardwareProfile};

const HF_BASE: &str = "https://huggingface.co";

#[derive(Parser, Debug)]
#[command(author, version, about = "Download a model from the Hugging Face Hub")]
struct Args {
    /// Model to download: either "owner/repo" (e.g. Qwen/Qwen2.5-0.5B-Instruct-GGUF)
    /// or a bare search term (e.g. "qwen2.5 0.5b instruct gguf").
    model: String,

    /// Quant/file to download by exact filename (skip the interactive/
    /// auto-recommended choice). Use --list first to see exact names.
    #[arg(long)]
    quant: Option<String>,

    /// Only list available variants and the hardware-based recommendation;
    /// don't download anything.
    #[arg(long)]
    list: bool,

    /// Directory to download into. Defaults to `./models/<repo-name>`.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Re-download even if the target file already exists with the right size.
    #[arg(long)]
    force: bool,

    /// When `model` resolves to multiple search results, pick the Nth
    /// (0-indexed) instead of the top match.
    #[arg(long, default_value_t = 0)]
    pick: usize,
}

#[derive(Debug, Deserialize)]
struct SearchResult {
    id: String,
    #[serde(default)]
    downloads: i64,
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Clone)]
struct QuantVariant {
    filename: String,
    size_bytes: Option<u64>,
    quant_label: String,
    is_mmproj: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let client = reqwest::Client::builder()
        .user_agent("llm-rs/pull")
        .build()
        .context("failed to build HTTP client")?;

    let repo = resolve_repo(&client, &args.model, args.pick).await?;
    println!("Repository: {repo}");

    let entries = fetch_tree(&client, &repo).await?;
    if entries.is_empty() {
        bail!(
            "no files found in {repo} (repo may be private, gated, or nonexistent — \
             check https://huggingface.co/{repo} in a browser)"
        );
    }

    let gguf_variants = classify_gguf_variants(&entries);
    let safetensors_files: Vec<&TreeEntry> = entries
        .iter()
        .filter(|e| e.kind == "file" && e.path.ends_with(".safetensors"))
        .collect();

    if gguf_variants.is_empty() && safetensors_files.is_empty() {
        bail!(
            "no .gguf or .safetensors weight files found in {repo} — is this a \
             tokenizer-only or dataset repo? Check https://huggingface.co/{repo}/tree/main"
        );
    }

    let profile = HardwareProfile::get();
    print_hardware_summary(profile);

    if !gguf_variants.is_empty() {
        println!("\nGGUF quantization variants found in {repo}:");
        let recommended = recommend_variant(&gguf_variants, profile);
        for v in &gguf_variants {
            if v.is_mmproj {
                continue;
            }
            let size_str = v
                .size_bytes
                .map(|b| format!("{:.2} GB", b as f64 / 1e9))
                .unwrap_or_else(|| "size unknown".to_string());
            let marker = if Some(v.filename.as_str()) == recommended.map(|r| r.filename.as_str()) {
                " <- recommended for this machine"
            } else {
                ""
            };
            println!("  {:<12} {:>10}  {}{}", v.quant_label, size_str, v.filename, marker);
        }
        let mmprojs: Vec<&QuantVariant> = gguf_variants.iter().filter(|v| v.is_mmproj).collect();
        if !mmprojs.is_empty() {
            println!("\n  Multimodal projector file(s) available (vision/audio input support):");
            for m in &mmprojs {
                let size_str = m
                    .size_bytes
                    .map(|b| format!("{:.2} GB", b as f64 / 1e9))
                    .unwrap_or_else(|| "size unknown".to_string());
                println!("    {:>10}  {}", size_str, m.filename);
            }
        }

        if args.list {
            return Ok(());
        }

        let chosen = if let Some(ref want) = args.quant {
            gguf_variants
                .iter()
                .find(|v| v.filename == *want || v.quant_label.eq_ignore_ascii_case(want))
                .ok_or_else(|| {
                    anyhow!(
                        "'{want}' does not match any file/quant-label in {repo} — run with --list \
                         to see exact names"
                    )
                })?
        } else {
            recommended.ok_or_else(|| {
                anyhow!(
                    "could not determine a recommended quant automatically (no size metadata \
                     available for any variant) — pass --quant <filename> explicitly, see --list"
                )
            })?
        };

        let output_dir = args
            .output_dir
            .unwrap_or_else(|| default_output_dir(&repo));
        std::fs::create_dir_all(&output_dir)
            .with_context(|| format!("failed to create output directory {:?}", output_dir))?;

        println!("\nDownloading {} -> {:?}", chosen.filename, output_dir);
        download_file(&client, &repo, &chosen.filename, &output_dir, args.force).await?;

        // Also grab the matching mmproj file (if the recommended text quant
        // has one, download that too, since a model without its mmproj file
        // can't actually do vision/audio - matching what the user asked for
        // when they picked a multimodal-capable repo) plus tokenizer/config
        // sidecars, skipping anything already present unless --force.
        if let Some(mmproj) = find_matching_mmproj(&chosen.filename, &gguf_variants) {
            println!("Downloading matching multimodal projector: {}", mmproj.filename);
            download_file(&client, &repo, &mmproj.filename, &output_dir, args.force).await?;
        }
        download_sidecars(&client, &repo, &entries, &output_dir, args.force).await?;
    } else {
        // Native HF safetensors repo.
        println!("\nNative HF safetensors repo ({} weight shard(s)):", safetensors_files.len());
        let total: u64 = safetensors_files.iter().filter_map(|f| f.size).sum();
        println!("  Total weight size: {:.2} GB", total as f64 / 1e9);

        check_quantization_method(&client, &repo).await?;

        if args.list {
            return Ok(());
        }

        let required = (total as f64 * 1.15) as u64;
        let available = profile
            .gpu_vram_free_bytes
            .filter(|_| profile.backend != BackendChoice::Cpu)
            .unwrap_or(profile.system_ram_free_bytes);
        if total > 0 && required > available {
            println!(
                "\nWARNING: this repo's weights ({:.2} GB, +15% headroom = {:.2} GB) exceed this \
                 machine's detected available memory ({:.2} GB). Loading will likely fail with an \
                 out-of-memory error. Consider a GGUF-quantized version of this model instead, if \
                 one exists (search: `llm pull \"{repo}\" gguf` won't work directly - search \
                 huggingface.co for a community GGUF quantization of this model).",
                total as f64 / 1e9, required as f64 / 1e9, available as f64 / 1e9
            );
        }

        let output_dir = args
            .output_dir
            .unwrap_or_else(|| default_output_dir(&repo));
        std::fs::create_dir_all(&output_dir)
            .with_context(|| format!("failed to create output directory {:?}", output_dir))?;

        for f in &safetensors_files {
            download_file(&client, &repo, &f.path, &output_dir, args.force).await?;
        }
        download_sidecars(&client, &repo, &entries, &output_dir, args.force).await?;
    }

    println!("\nDone. Try:\n  ./chat --model-path {:?} --tokenizer-path <dir>/tokenizer.json",
        default_output_dir(&repo));
    Ok(())
}

/// Resolve `model` to an `owner/repo` string: pass through directly if it
/// already looks like one, otherwise search the HF Hub and pick a result.
async fn resolve_repo(client: &reqwest::Client, model: &str, pick: usize) -> Result<String> {
    if model.contains('/') && !model.contains(' ') {
        return Ok(model.to_string());
    }

    let url = format!("{HF_BASE}/api/models?search={}&limit=15", urlencode(model));
    let mut results: Vec<SearchResult> = client
        .get(&url)
        .send()
        .await
        .context("failed to reach huggingface.co (check your network connection)")?
        .error_for_status()
        .context("huggingface.co search API returned an error")?
        .json()
        .await
        .context("failed to parse huggingface.co search response")?;

    // HF's search API returns results in a relevance order that doesn't
    // reliably put the canonical/most-used repo first (e.g. searching for a
    // well-known model can surface an obscure fork with 30 downloads ahead
    // of the official quantization with 150k+). Sort by download count so
    // the default (--pick 0) pick is the one most users actually want.
    results.sort_by(|a, b| b.downloads.cmp(&a.downloads));

    if results.is_empty() {
        bail!(
            "no Hugging Face repos found matching \"{model}\" — try a more specific search, \
             or pass an exact \"owner/repo\" (e.g. \"Qwen/Qwen2.5-0.5B-Instruct-GGUF\")"
        );
    }

    println!("Search results for \"{model}\":");
    for (i, r) in results.iter().enumerate().take(10) {
        let marker = if i == pick { " <- selected" } else { "" };
        println!("  [{i}] {} ({} downloads){marker}", r.id, r.downloads);
    }
    if results.len() > 1 && pick == 0 {
        println!(
            "\n(Showing the top match. Pass --pick <N> to select a different one from the list above.)"
        );
    }

    results
        .get(pick)
        .map(|r| r.id.clone())
        .ok_or_else(|| anyhow!("--pick {pick} is out of range (only {} results)", results.len()))
}

async fn fetch_tree(client: &reqwest::Client, repo: &str) -> Result<Vec<TreeEntry>> {
    let url = format!("{HF_BASE}/api/models/{repo}/tree/main?recursive=true");
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to reach huggingface.co for {repo}"))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        bail!(
            "repository \"{repo}\" not found on Hugging Face (404) — check the spelling, or that \
             it isn't a gated/private repo requiring authentication (not yet supported by this \
             downloader)"
        );
    }
    let entries: Vec<TreeEntry> = resp
        .error_for_status()
        .with_context(|| format!("huggingface.co returned an error listing files for {repo}"))?
        .json()
        .await
        .context("failed to parse huggingface.co file-listing response")?;
    Ok(entries.into_iter().filter(|e| e.kind == "file").collect())
}

/// Extract a human-readable quant label (Q4_K_M, Q8_0, F16, IQ4_XS, ...)
/// from a GGUF filename. No fixed enum of "known" quant names - matches the
/// generic `[A-Z][A-Z0-9_]+` token immediately before `.gguf`, which covers
/// every naming scheme actually used by GGUF-quantizing tools in practice,
/// current or future, without hardcoding a list that inevitably goes stale.
fn extract_quant_label(filename: &str) -> String {
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    // Split only on '-'/'.' (the usual segment separators between the model
    // name and its quant suffix) - NOT '_', since quant labels themselves
    // routinely contain underscores (Q4_K_M, IQ4_XS, ...) and splitting on
    // it would shred a real label into meaningless fragments ("m", "k").
    for part in stem.rsplit(['-', '.']).take(3) {
        let looks_like_quant = part.len() >= 2
            && part.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
            && part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if looks_like_quant {
            return part.to_uppercase();
        }
    }
    "unknown".to_string()
}

fn classify_gguf_variants(entries: &[TreeEntry]) -> Vec<QuantVariant> {
    entries
        .iter()
        .filter(|e| e.path.ends_with(".gguf"))
        .map(|e| {
            let filename = e.path.clone();
            let is_mmproj = filename.to_lowercase().contains("mmproj");
            let quant_label = if is_mmproj {
                "mmproj".to_string()
            } else {
                extract_quant_label(&filename)
            };
            QuantVariant {
                filename,
                size_bytes: e.size,
                quant_label,
                is_mmproj,
            }
        })
        .collect()
}

/// Recommend the largest (best-quality) GGUF variant whose size fits this
/// machine's detected available memory with the same 15% headroom
/// `HardwareProfile::choose_device` uses elsewhere in this engine — so the
/// recommendation logic matches the load-time safety check the user will
/// actually hit, instead of an independent guess.
fn recommend_variant<'a>(variants: &'a [QuantVariant], profile: &HardwareProfile) -> Option<&'a QuantVariant> {
    let available = profile
        .gpu_vram_free_bytes
        .filter(|_| profile.backend != BackendChoice::Cpu)
        .unwrap_or(profile.system_ram_free_bytes);

    let mut candidates: Vec<&QuantVariant> = variants
        .iter()
        .filter(|v| !v.is_mmproj && v.size_bytes.is_some())
        .collect();
    candidates.sort_by_key(|v| v.size_bytes.unwrap());

    candidates
        .into_iter()
        .filter(|v| (v.size_bytes.unwrap() as f64 * 1.15) as u64 <= available)
        .next_back()
}

fn find_matching_mmproj<'a>(text_filename: &str, variants: &'a [QuantVariant]) -> Option<&'a QuantVariant> {
    variants.iter().find(|v| v.is_mmproj).map(|_| {
        // Prefer any mmproj file present; if several exist (different
        // precisions), take the smallest to keep the extra download light -
        // vision/audio quality is dominated by the text model's quant, not
        // the projector's.
        variants
            .iter()
            .filter(|v| v.is_mmproj)
            .min_by_key(|v| v.size_bytes.unwrap_or(u64::MAX))
            .unwrap()
    }).filter(|_| !text_filename.is_empty())
}

async fn check_quantization_method(client: &reqwest::Client, repo: &str) -> Result<()> {
    let url = format!("{HF_BASE}/{repo}/resolve/main/config.json");
    let Ok(resp) = client.get(&url).send().await else { return Ok(()) };
    let Ok(text) = resp.text().await else { return Ok(()) };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else { return Ok(()) };

    if let Some(qc) = json.get("quantization_config") {
        let method = qc
            .get("quant_method")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        match method.to_lowercase().as_str() {
            "awq" | "gptq" => {
                println!(
                    "\nNOTE: this repo is pre-quantized with '{method}'. llm-rs dequantizes \
                     {method} weights to dense F16/F32 at load time (see llm-core/src/loader/{method}.rs) \
                     - this is a correctness-first path, not yet the fast tensor-core kernel path \
                     (Marlin-class), so expect similar throughput to loading the unquantized model, \
                     not {method}'s usual speed advantage. This dequant path has been reviewed against \
                     real safetensors headers but not yet numerically verified against Python \
                     (transformers/autoawq/auto-gptq) output on real hardware - if generation looks \
                     wrong, that verification step is the first thing to check."
                );
            }
            "bitsandbytes" | "bnb" => {
                println!(
                    "\nNOTE: this repo is pre-quantized with '{method}'. llm-rs does not yet \
                     implement dequantization for bitsandbytes NF4/FP4's packed-weight layout. \
                     Loading this repo as-is will fail with a clear error at load time rather than \
                     silently misreading the packed weights as plain F16/BF16. Use a GGUF-quantized \
                     version of this model instead (search huggingface.co), or the unquantized/F16 \
                     version of this repo."
                );
            }
            other => {
                println!("\nNOTE: quantization_config present (method: '{other}') - support status unverified, may fail to load.");
            }
        }
    }
    Ok(())
}

fn default_output_dir(repo: &str) -> PathBuf {
    let name = repo.rsplit('/').next().unwrap_or(repo);
    PathBuf::from("models").join(sanitize_dirname(name))
}

fn sanitize_dirname(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

fn print_hardware_summary(profile: &HardwareProfile) {
    let (label, free) = match profile.backend {
        BackendChoice::Cpu => ("System RAM", profile.system_ram_free_bytes),
        BackendChoice::Cuda => ("Free VRAM", profile.gpu_vram_free_bytes.unwrap_or(profile.system_ram_free_bytes)),
        BackendChoice::Metal => ("Free Unified Memory", profile.gpu_vram_free_bytes.unwrap_or(profile.system_ram_free_bytes)),
    };
    println!(
        "Detected hardware: {:?} backend, {:.2} GB {} available",
        profile.backend, free as f64 / 1e9, label
    );
}

async fn download_file(
    client: &reqwest::Client,
    repo: &str,
    filename: &str,
    output_dir: &std::path::Path,
    force: bool,
) -> Result<()> {
    let dest = output_dir.join(filename);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let url = format!("{HF_BASE}/{repo}/resolve/main/{filename}");
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to start download of {filename}"))?
        .error_for_status()
        .with_context(|| format!("huggingface.co returned an error downloading {filename}"))?;

    let total = resp.content_length();

    if !force && dest.exists() {
        if let (Ok(meta), Some(total)) = (std::fs::metadata(&dest), total) {
            if meta.len() == total {
                println!("  {filename}: already present ({:.2} GB), skipping (--force to re-download)", total as f64 / 1e9);
                return Ok(());
            }
        }
    }

    let mut file = std::fs::File::create(&dest)
        .with_context(|| format!("failed to create {:?}", dest))?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_pct_printed = -1i64;

    use std::io::Write;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("download interrupted (network error mid-transfer)")?;
        file.write_all(&chunk).context("failed writing downloaded data to disk")?;
        downloaded += chunk.len() as u64;
        if let Some(total) = total {
            let pct = (downloaded as f64 / total as f64 * 100.0) as i64;
            if pct != last_pct_printed && pct % 5 == 0 {
                print!("\r  {filename}: {:.1}%  ({:.2}/{:.2} GB)   ", pct as f64, downloaded as f64 / 1e9, total as f64 / 1e9);
                use std::io::Write as _;
                std::io::stdout().flush().ok();
                last_pct_printed = pct;
            }
        }
    }
    // Verify completeness before declaring success: an interrupted transfer
    // (network drop, timeout, killed process) can leave a truncated file on
    // disk that still "exists" - loading it later fails with a confusing
    // low-level parser error (e.g. candle's GGUF reader reporting "failed to
    // read tensor X") that looks like a model-compatibility bug but is
    // actually just an incomplete download. Catch it here instead, where the
    // real cause is obvious, and remove the truncated file so a subsequent
    // run doesn't mistake it for a valid cached copy.
    if let Some(total) = total {
        if downloaded != total {
            let _ = std::fs::remove_file(&dest);
            bail!(
                "{filename}: download incomplete ({downloaded} of {total} bytes received) - the \
                 transfer was interrupted. Removed the partial file; re-run to retry."
            );
        }
    }
    println!("\r  {filename}: done ({:.2} GB)                    ", downloaded as f64 / 1e9);
    Ok(())
}

/// Download the small tokenizer/config sidecar files that live alongside
/// the weights in most HF repos, if present - `chat`/`llm serve` need
/// `tokenizer.json` next to the model, and `config.json` matters for the
/// HF-safetensors loading path.
async fn download_sidecars(
    client: &reqwest::Client,
    repo: &str,
    entries: &[TreeEntry],
    output_dir: &std::path::Path,
    force: bool,
) -> Result<()> {
    const SIDECARS: &[&str] = &[
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "chat_template.jinja",
    ];
    let present: HashMap<&str, &str> = entries
        .iter()
        .filter_map(|e| SIDECARS.iter().find(|&&s| e.path == s).map(|&s| (s, e.path.as_str())))
        .collect();

    for name in SIDECARS {
        if present.contains_key(name) {
            download_file(client, repo, name, output_dir, force).await.ok();
        }
    }

    // Many official GGUF-quantization repos (e.g. "Qwen/Qwen2.5-0.5B-
    // Instruct-GGUF") ship ONLY the .gguf files, with no tokenizer.json at
    // all - the tokenizer lives in the base (non-GGUF) repo instead. Without
    // it, `chat`/`llm serve` have nothing to load (this engine doesn't yet
    // extract a tokenizer from GGUF's embedded metadata - see goal.md's
    // Phase 1 "tokenizer auto-detection" fallback chain, which isn't
    // implemented here). Try the obvious sibling repo name as a fallback
    // rather than leaving the user stuck with an unusable download.
    if !present.contains_key("tokenizer.json") {
        if let Some(base_repo) = strip_gguf_suffix(repo) {
            println!(
                "\nNo tokenizer.json in {repo} - trying the likely base repo {base_repo} for it..."
            );
            let url = format!("{HF_BASE}/{base_repo}/resolve/main/tokenizer.json");
            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp.bytes().await.context("failed reading tokenizer.json response")?;
                    let dest = output_dir.join("tokenizer.json");
                    std::fs::write(&dest, &bytes)
                        .with_context(|| format!("failed to write {:?}", dest))?;
                    println!("  tokenizer.json: fetched from {base_repo} ({:.2} MB)", bytes.len() as f64 / 1e6);
                }
                _ => {
                    println!(
                        "  Could not find tokenizer.json in {base_repo} either. You'll need to \
                         supply --tokenizer-path pointing at a compatible tokenizer.json manually \
                         (this engine does not yet extract a tokenizer from GGUF's embedded \
                         metadata, unlike llama.cpp's own CLI)."
                    );
                }
            }
        } else {
            println!(
                "\nNo tokenizer.json found in {repo}, and no obvious base-repo name to try. \
                 You'll need to supply --tokenizer-path manually."
            );
        }
    }
    Ok(())
}

/// Given a GGUF-quantization repo name, guess its likely non-quantized base
/// repo by stripping a trailing "-GGUF"/"-gguf" (and common quant-suffix
/// variants like "-Q4_K_M-GGUF"). Best-effort - the caller verifies the
/// guess actually has a tokenizer.json before relying on it, never assumes.
fn strip_gguf_suffix(repo: &str) -> Option<String> {
    let lower = repo.to_lowercase();
    let idx = lower.rfind("-gguf")?;
    if idx + 5 != lower.len() {
        return None;
    }
    Some(repo[..idx].to_string())
}

fn urlencode(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~') {
                c.to_string()
            } else if c == ' ' {
                "+".to_string()
            } else {
                let mut buf = [0u8; 4];
                c.encode_utf8(&mut buf)
                    .bytes()
                    .map(|b| format!("%{:02X}", b))
                    .collect::<String>()
            }
        })
        .collect()
}
