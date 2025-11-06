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
    _core: Core,
    tokenizer: Tokenizer,
    config: LocalModelConfig,
}

impl LocalInferenceEngine {
    /// Create a new inference engine with the given configuration
    pub fn new(config: LocalModelConfig) -> Result<Self> {
        // Initialize OpenVINO core
        let core = Core::new()?;

        eprintln!(
            "OpenVINO initialized. Device: {}",
            config.device.as_str()
        );

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

        eprintln!("Tokenizer loaded successfully");

        // Note: Actual model loading and compilation will be implemented
        // once the autoregressive generation loop is completed.
        // For now, this validates the setup and prepares the infrastructure.

        Ok(Self {
            _core: core,
            tokenizer,
            config,
        })
    }

    /// Generate text response for the given prompt
    pub fn generate(&self, prompt: &str) -> Result<String> {
        // Tokenize input to validate the tokenizer works
        let encoding = self.tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let input_ids = encoding.get_ids();

        if input_ids.is_empty() {
            return Err(anyhow!("Tokenization produced empty input"));
        }

        // TODO: Implement autoregressive generation loop
        // This requires:
        // 1. Load and compile the model with Core.read_model()
        // 2. Create inference request
        // 3. For each generation step:
        //    - Prepare input tensor from token IDs
        //    - Run inference to get logits
        //    - Sample or argmax to get next token
        //    - Append token and repeat until EOS
        // 4. Decode generated tokens to string
        //
        // See OpenVINO Rust examples for reference:
        // https://github.com/intel/openvino-rs

        eprintln!("OpenVINO inference: Tokenized {} tokens", input_ids.len());
        eprintln!("Device: {}", self.config.device.as_str());
        eprintln!("Model: {}", self.config.model_path.display());

        // Return informative message for now
        Ok(format!(
            "🔧 OpenVINO Integration Status: Framework Ready\n\n\
            ✅ OpenVINO Core initialized\n\
            ✅ Tokenizer loaded ({} tokens)\n\
            ✅ Device configured: {}\n\
            ✅ Model path validated: {}\n\n\
            ⚠️  Autoregressive generation loop not yet implemented.\n\n\
            Your prompt was: \"{}\"\n\n\
            To use LLM inference now, please use a remote provider:\n\
            • qq --profile groq \"your question\"\n\
            • qq --profile openai \"your question\"\n\n\
            See README.md for implementation details and contribution guidelines.",
            input_ids.len(),
            self.config.device.as_str(),
            self.config.model_path.display(),
            prompt
        ))
    }

    /// Generate text with streaming callback (for compatibility with existing API)
    pub fn generate_stream<F>(&self, prompt: &str, mut callback: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        // For now, implement as non-streaming and call callback once
        // TODO: Implement proper token-by-token streaming
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
