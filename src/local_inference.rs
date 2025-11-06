//! Local inference module using OpenVINO
//!
//! This module is only available when the `local-inference` feature is enabled.

#![cfg(feature = "local-inference")]

use anyhow::{Context, Result, anyhow};
use openvino::{Core, DeviceType, ElementType, Shape, Tensor};
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

        let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();

        if input_ids.is_empty() {
            return Err(anyhow!("Tokenization produced empty input"));
        }

        if cfg!(debug_assertions) {
            eprintln!("✓ Tokenized {} tokens", input_ids.len());
        }

        // Initialize OpenVINO Core
        let mut core = Core::new()?;

        if cfg!(debug_assertions) {
            eprintln!("✓ OpenVINO Core initialized");
        }

        // Read model
        let model_xml = self.model_path.to_string_lossy();
        let model_bin = self.model_path.with_extension("bin").to_string_lossy().to_string();
        let model = core.read_model_from_file(&model_xml, &model_bin)?;

        if cfg!(debug_assertions) {
            eprintln!("✓ Model loaded from {}", model_xml);
        }

        // Select device
        let device = match self.config.device {
            InferenceDevice::CPU => DeviceType::CPU,
            InferenceDevice::GPU => DeviceType::GPU,
            InferenceDevice::NPU => DeviceType::NPU,
        };

        if cfg!(debug_assertions) {
            eprintln!("✓ Compiling for {:?}", self.config.device);
        }

        // Compile model
        let mut compiled_model = core.compile_model(&model, device)?;

        if cfg!(debug_assertions) {
            eprintln!("✓ Model compiled successfully");
        }

        // Create inference request
        let mut infer_request = compiled_model.create_infer_request()?;

        // Autoregressive generation loop
        let mut generated_tokens = Vec::new();
        let max_new_tokens = self.config.max_tokens.min(512);

        for step in 0..max_new_tokens {
            // Prepare input tensor
            let seq_len = input_ids.len();
            let shape = Shape::new(&[1, seq_len as i64])?;
            let mut input_tensor = Tensor::new(ElementType::I64, &shape)?;

            // Copy token IDs to tensor
            {
                let data = input_tensor.get_data_mut::<i64>()?;
                for (i, &token_id) in input_ids.iter().enumerate() {
                    data[i] = token_id as i64;
                }
            }

            // Set input and run inference
            infer_request.set_input_tensor(&input_tensor)?;
            infer_request.infer()?;

            // Get output tensor
            let output_tensor = infer_request.get_output_tensor()?;
            let output_shape = output_tensor.get_shape()?;
            let dims = output_shape.get_dimensions();

            // Output shape is [batch_size, seq_len, vocab_size]
            if dims.len() != 3 {
                return Err(anyhow!("Unexpected output shape: expected 3 dimensions, got {}", dims.len()));
            }

            let _batch_size = dims[0] as usize;
            let output_seq_len = dims[1] as usize;
            let vocab_size = dims[2] as usize;

            // Get logits for the last token
            let output_data = output_tensor.get_data::<f32>()?;
            let last_token_offset = (output_seq_len - 1) * vocab_size;
            let logits = &output_data[last_token_offset..last_token_offset + vocab_size];

            // Sample next token
            let next_token = self.sample_token(logits)?;

            // Check for EOS
            if next_token == self.eos_token_id {
                if cfg!(debug_assertions) {
                    eprintln!("✓ Generated {} tokens (EOS reached)", step);
                }
                break;
            }

            // Add to generated sequence
            generated_tokens.push(next_token);
            input_ids.push(next_token);

            // Progress indicator
            if cfg!(debug_assertions) && step > 0 && step % 10 == 0 {
                eprintln!("  Generated {} tokens...", step);
            }
        }

        if cfg!(debug_assertions) {
            eprintln!("✓ Generation complete: {} tokens", generated_tokens.len());
        }

        // Decode generated tokens
        let response = self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;

        Ok(response)
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

    /// Generate text with streaming callback - calls callback for each generated token
    pub fn generate_stream<F>(&self, prompt: &str, mut callback: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        // Tokenize input
        let encoding = self.tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();

        if input_ids.is_empty() {
            return Err(anyhow!("Tokenization produced empty input"));
        }

        // Initialize OpenVINO Core
        let mut core = Core::new()?;
        let model_xml = self.model_path.to_string_lossy();
        let model_bin = self.model_path.with_extension("bin").to_string_lossy().to_string();
        let model = core.read_model_from_file(&model_xml, &model_bin)?;

        // Select device
        let device = match self.config.device {
            InferenceDevice::CPU => DeviceType::CPU,
            InferenceDevice::GPU => DeviceType::GPU,
            InferenceDevice::NPU => DeviceType::NPU,
        };

        // Compile model and create inference request
        let mut compiled_model = core.compile_model(&model, device)?;
        let mut infer_request = compiled_model.create_infer_request()?;

        // Streaming generation loop
        let max_new_tokens = self.config.max_tokens.min(512);

        for _step in 0..max_new_tokens {
            // Prepare input tensor
            let seq_len = input_ids.len();
            let shape = Shape::new(&[1, seq_len as i64])?;
            let mut input_tensor = Tensor::new(ElementType::I64, &shape)?;

            // Copy token IDs to tensor
            {
                let data = input_tensor.get_data_mut::<i64>()?;
                for (i, &token_id) in input_ids.iter().enumerate() {
                    data[i] = token_id as i64;
                }
            }

            // Set input and run inference
            infer_request.set_input_tensor(&input_tensor)?;
            infer_request.infer()?;

            // Get output tensor
            let output_tensor = infer_request.get_output_tensor()?;
            let output_shape = output_tensor.get_shape()?;
            let dims = output_shape.get_dimensions();

            if dims.len() != 3 {
                return Err(anyhow!("Unexpected output shape"));
            }

            let output_seq_len = dims[1] as usize;
            let vocab_size = dims[2] as usize;

            // Get logits for the last token
            let output_data = output_tensor.get_data::<f32>()?;
            let last_token_offset = (output_seq_len - 1) * vocab_size;
            let logits = &output_data[last_token_offset..last_token_offset + vocab_size];

            // Sample next token
            let next_token = self.sample_token(logits)?;

            // Check for EOS
            if next_token == self.eos_token_id {
                break;
            }

            // Decode and stream this single token
            let token_text = self.tokenizer
                .decode(&[next_token], false)
                .map_err(|e| anyhow!("Token decoding failed: {}", e))?;

            callback(&token_text);

            // Add to sequence for next iteration
            input_ids.push(next_token);
        }

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
