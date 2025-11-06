//! Local inference module using OpenVINO
//!
//! This module is only available when the `local-inference` feature is enabled.

#![cfg(feature = "local-inference")]

use anyhow::{Context, Result, anyhow};
use openvino::Core;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Device type for OpenVINO inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceDevice {
    CPU,
    GPU,
    NPU,
}

impl InferenceDevice {
    pub fn as_str(&self) -> &'static str {
        match self {
            InferenceDevice::CPU => "CPU",
            InferenceDevice::GPU => "GPU",
            InferenceDevice::NPU => "NPU",
        }
    }

    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "CPU" => Ok(InferenceDevice::CPU),
            "GPU" => Ok(InferenceDevice::GPU),
            "NPU" => Ok(InferenceDevice::NPU),
            _ => Err(anyhow!("Invalid device type: {}. Use CPU, GPU, or NPU", s)),
        }
    }
}

/// Configuration for local model inference
#[derive(Debug, Clone)]
pub struct LocalModelConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub device: InferenceDevice,
    pub max_tokens: usize,
    pub temperature: f32,
}

impl LocalModelConfig {
    pub fn new(model_dir: impl AsRef<Path>, device: InferenceDevice) -> Self {
        let model_dir = model_dir.as_ref();
        Self {
            model_path: model_dir.join("openvino_model.xml"),
            tokenizer_path: model_dir.join("tokenizer.json"),
            device,
            max_tokens: 4000,
            temperature: 0.0,
        }
    }
}

/// Local LLM inference engine using OpenVINO
pub struct LocalInferenceEngine {
    model_path: PathBuf,
    tokenizer: Tokenizer,
    config: LocalModelConfig,
    eos_token_id: u32,
}

impl LocalInferenceEngine {
    /// Create a new inference engine with the given configuration
    pub fn new(config: LocalModelConfig) -> Result<Self> {
        // Check if model files exist
        if !config.model_path.exists() {
            return Err(anyhow!(
                "Model file not found: {}. Please download the model first.",
                config.model_path.display()
            ));
        }

        if !config.tokenizer_path.exists() {
            return Err(anyhow!(
                "Tokenizer file not found: {}",
                config.tokenizer_path.display()
            ));
        }

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Get EOS token ID from tokenizer
        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .unwrap_or(2); // Default to 2 if not found

        if cfg!(debug_assertions) {
            eprintln!("✓ Tokenizer loaded (EOS token: {})", eos_token_id);
            eprintln!("✓ Device: {}", config.device.as_str());
        }

        Ok(Self {
            model_path: config.model_path.clone(),
            tokenizer,
            config,
            eos_token_id,
        })
    }

    /// Generate text response for the given prompt
    pub fn generate(&self, prompt: &str) -> Result<String> {
        // Tokenize input
        let encoding = self.tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        if input_ids.is_empty() {
            return Err(anyhow!("Tokenization produced empty input"));
        }

        // Initialize OpenVINO Core
        let mut core = Core::new()?;

        // Read and compile model
        let model_xml = self.model_path.to_string_lossy();
        let model_bin = self.model_path.with_extension("bin").to_string_lossy().to_string();

        let _model = core.read_model_from_file(&model_xml, &model_bin)?;

        // TODO: Complete the implementation with proper OpenVINO API calls
        // The OpenVINO Rust API v0.7 has specific method signatures that need to be matched.
        // Key steps needed:
        // 1. Compile model: compiled_model = core.compile_model(&model, "CPU")?
        // 2. Create infer request: infer_request = compiled_model.create_infer_request()?
        // 3. Set input tensor with proper shape and data
        // 4. Run inference: infer_request.infer()?
        // 5. Get output tensor and extract logits
        // 6. Implement autoregressive loop with token sampling
        //
        // For now, return a helpful message showing the integration is partially complete.

        eprintln!("✓ OpenVINO Core initialized");
        eprintln!("✓ Model loaded: {}", model_xml);
        eprintln!("✓ Tokenized {} tokens", input_ids.len());
        eprintln!("✓ Device: {}", self.config.device.as_str());

        Ok(format!(
            "OpenVINO inference framework initialized successfully!\n\n\
            ✅ Model: {}\n\
            ✅ Device: {}\n\
            ✅ Tokenizer: {} tokens\n\
            ✅ EOS token: {}\n\n\
            🚧 Autoregressive generation loop implementation in progress.\n\n\
            The OpenVINO Rust bindings (v0.7) require specific API calls that are being finalized.\n\
            Key infrastructure is complete:\n\
            - Model loading ✓\n\
            - Tokenization ✓\n\
            - Device selection ✓\n\n\
            What's needed:\n\
            - Tensor creation and data copying with correct API\n\
            - Inference request execution\n\
            - Logits extraction and token sampling\n\
            - Autoregressive generation loop\n\n\
            Your prompt was: \"{}\"\n\n\
            For production use, please use remote providers:\n\
            • qq --profile groq \"your question\"\n\
            • qq --profile openai \"your question\"\n\n\
            See src/local_inference.rs for implementation details.",
            self.model_path.display(),
            self.config.device.as_str(),
            input_ids.len(),
            self.eos_token_id,
            prompt
        ))
    }

    /// Sample the next token from logits (greedy or temperature sampling)
    #[allow(dead_code)]
    fn sample_token(&self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(anyhow!("Empty logits"));
        }

        // Greedy decoding (argmax) - always use for now
        let mut max_idx = 0;
        let mut max_val = logits[0];

        for (i, &logit) in logits.iter().enumerate().skip(1) {
            if logit > max_val {
                max_val = logit;
                max_idx = i;
            }
        }

        Ok(max_idx as u32)

        // TODO: Implement temperature sampling when needed
        // if self.config.temperature >= 0.01 {
        //     Apply softmax with temperature
        //     Sample from the distribution
        // }
    }

    /// Generate text with streaming callback
    /// Currently calls the non-streaming generate and returns all text at once
    pub fn generate_stream<F>(&self, prompt: &str, mut callback: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        // TODO: Implement true token-by-token streaming once autoregressive loop is complete
        // For now, generate all text and call callback once
        let response = self.generate(prompt)?;
        callback(&response);
        Ok(())
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        format!("OpenVINO on {}", self.config.device.as_str())
    }
}

/// Model downloader for HuggingFace models
pub struct ModelDownloader {
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new() -> Result<Self> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow!("Could not determine home directory"))?;
        let cache_dir = home.join(".cache").join("qqqa").join("models");

        if !cache_dir.exists() {
            fs_err::create_dir_all(&cache_dir)
                .with_context(|| format!("Creating model cache dir: {}", cache_dir.display()))?;
        }

        Ok(Self { cache_dir })
    }

    /// Download a model from HuggingFace if not already cached
    pub async fn download_model(&self, repo_id: &str) -> Result<PathBuf> {
        let model_dir = self.cache_dir.join(repo_id.replace('/', "_"));

        // Check if model already exists
        let model_xml = model_dir.join("openvino_model.xml");
        let tokenizer_json = model_dir.join("tokenizer.json");

        if model_xml.exists() && tokenizer_json.exists() {
            println!("Model already cached at: {}", model_dir.display());
            return Ok(model_dir);
        }

        println!("Downloading model '{}' from HuggingFace...", repo_id);
        println!("This may take a while depending on model size and network speed.");

        // Create model directory
        fs_err::create_dir_all(&model_dir)?;

        // Download using hf-hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(repo_id.to_string());

        // Download required files
        println!("Downloading OpenVINO model files...");
        let model_xml_remote = repo.get("openvino_model.xml")
            .context("Failed to download openvino_model.xml")?;
        let model_bin_remote = repo.get("openvino_model.bin")
            .context("Failed to download openvino_model.bin")?;
        let tokenizer_remote = repo.get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;

        // Copy to cache directory
        fs_err::copy(&model_xml_remote, &model_xml)?;
        fs_err::copy(&model_bin_remote, model_dir.join("openvino_model.bin"))?;
        fs_err::copy(&tokenizer_remote, &tokenizer_json)?;

        println!("Model downloaded successfully to: {}", model_dir.display());
        Ok(model_dir)
    }

    /// Get the path to a cached model, or None if not downloaded
    pub fn get_cached_model(&self, repo_id: &str) -> Option<PathBuf> {
        let model_dir = self.cache_dir.join(repo_id.replace('/', "_"));
        let model_xml = model_dir.join("openvino_model.xml");
        let tokenizer_json = model_dir.join("tokenizer.json");

        if model_xml.exists() && tokenizer_json.exists() {
            Some(model_dir)
        } else {
            None
        }
    }

    /// List all cached models
    pub fn list_cached_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();

        if !self.cache_dir.exists() {
            return Ok(models);
        }

        for entry in fs_err::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let model_name = entry.file_name().to_string_lossy().to_string();
                let model_dir = entry.path();

                // Check if it has the required files
                if model_dir.join("openvino_model.xml").exists()
                    && model_dir.join("tokenizer.json").exists() {
                    models.push(model_name.replace('_', "/"));
                }
            }
        }

        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_from_str() {
        assert_eq!(InferenceDevice::from_str("CPU").unwrap(), InferenceDevice::CPU);
        assert_eq!(InferenceDevice::from_str("cpu").unwrap(), InferenceDevice::CPU);
        assert_eq!(InferenceDevice::from_str("GPU").unwrap(), InferenceDevice::GPU);
        assert_eq!(InferenceDevice::from_str("NPU").unwrap(), InferenceDevice::NPU);
        assert!(InferenceDevice::from_str("invalid").is_err());
    }

    #[test]
    fn test_model_config() {
        let config = LocalModelConfig::new("/path/to/model", InferenceDevice::NPU);
        assert_eq!(config.device, InferenceDevice::NPU);
        assert!(config.model_path.ends_with("openvino_model.xml"));
        assert!(config.tokenizer_path.ends_with("tokenizer.json"));
    }
}
