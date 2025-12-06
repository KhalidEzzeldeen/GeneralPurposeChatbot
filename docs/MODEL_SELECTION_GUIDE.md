# Model Selection Guide

## Overview

The application now supports multiple LLM models with easy switching in Settings. You can choose between different models based on your speed/quality preferences.

## Available Models

### 🚀 Recommended Fast Models

#### 1. **Qwen2.5 3B Instruct** (`qwen2.5:3b-instruct`)
- **Size:** 3B parameters
- **Speed:** Fast (2-3x faster than 7B)
- **Quality:** High (similar to 7B version)
- **Best for:** General use when you need faster responses
- **Install:** `ollama pull qwen2.5:3b-instruct`

#### 2. **Llama 3.2 3B** (`llama3.2:3b`)
- **Size:** 3B parameters
- **Speed:** Very Fast
- **Quality:** High
- **Best for:** Quick responses with excellent quality
- **Install:** `ollama pull llama3.2:3b`

#### 3. **Gemma 2 2B** (`gemma2:2b`)
- **Size:** 2B parameters
- **Speed:** Very Fast (fastest option)
- **Quality:** Good
- **Best for:** Maximum speed, acceptable quality
- **Install:** `ollama pull gemma2:2b`

### 🎯 Balanced Models

#### 4. **Qwen2.5 7B Instruct** (`qwen2.5:7b-instruct`) - Default
- **Size:** 7B parameters
- **Speed:** Medium
- **Quality:** High
- **Best for:** Balanced performance and quality
- **Install:** `ollama pull qwen2.5:7b-instruct`

#### 5. **Mistral 7B Instruct** (`mistral:7b-instruct`)
- **Size:** 7B parameters
- **Speed:** Medium
- **Quality:** High
- **Best for:** Alternative to Qwen with similar performance
- **Install:** `ollama pull mistral:7b-instruct`

### 🌟 High Quality Models

#### 6. **Llama 3.1 8B Instruct** (`llama3.1:8b-instruct`)
- **Size:** 8B parameters
- **Speed:** Medium
- **Quality:** Very High
- **Best for:** Complex tasks requiring highest quality
- **Install:** `ollama pull llama3.1:8b-instruct`

#### 7. **Phi-3 Mini** (`phi3:mini`)
- **Size:** 3.8B parameters
- **Speed:** Very Fast
- **Quality:** Good
- **Best for:** Microsoft's efficient model
- **Install:** `ollama pull phi3:mini`

## Speed Comparison

| Model | Relative Speed | Quality | Use Case |
|-------|---------------|---------|----------|
| gemma2:2b | ⚡⚡⚡⚡⚡ Fastest | Good | Maximum speed |
| llama3.2:3b | ⚡⚡⚡⚡ Very Fast | High | Fast + Quality |
| qwen2.5:3b-instruct | ⚡⚡⚡ Fast | High | Balanced fast |
| phi3:mini | ⚡⚡⚡ Very Fast | Good | Efficient |
| qwen2.5:7b-instruct | ⚡⚡ Medium | High | Balanced |
| mistral:7b-instruct | ⚡⚡ Medium | High | Alternative |
| llama3.1:8b-instruct | ⚡ Medium | Very High | Best quality |

## How to Use

### Method 1: Model Selector (Recommended)
1. Go to **Settings** → **LLM Settings**
2. Use the **"Choose Model"** dropdown
3. See model details (size, speed, quality)
4. Click **"Save LLM Settings"**

### Method 2: Quick Switch Buttons
1. Go to **Settings** → **LLM Settings**
2. Scroll to **"Quick Switch"** section
3. Click a button:
   - 🚀 **Fastest** → Gemma 2B
   - ⚡ **Very Fast** → Qwen 3B
   - 🎯 **Balanced** → Qwen 7B
   - 🌟 **High Quality** → Llama 8B

### Method 3: Custom Model
1. Check **"Use custom model name"**
2. Enter any Ollama model name
3. Save settings

## Model Installation

Before using a model, install it in Ollama:

```bash
# Fast models
ollama pull qwen2.5:3b-instruct
ollama pull llama3.2:3b
ollama pull gemma2:2b

# Balanced models
ollama pull qwen2.5:7b-instruct
ollama pull mistral:7b-instruct

# High quality
ollama pull llama3.1:8b-instruct
ollama pull phi3:mini
```

## Recommendations

### For Speed (2-3x Faster)
✅ **Use:** `qwen2.5:3b-instruct` or `llama3.2:3b`
- Similar quality to 7B models
- 2-3x faster responses
- Lower memory usage

### For Maximum Speed
✅ **Use:** `gemma2:2b`
- Fastest option
- Good quality for most tasks
- Lowest memory usage

### For Best Quality
✅ **Use:** `qwen2.5:7b-instruct` or `llama3.1:8b-instruct`
- Highest quality responses
- Better for complex queries
- More accurate

### For Balanced Performance
✅ **Use:** `qwen2.5:7b-instruct` (default)
- Good balance of speed and quality
- Reliable performance
- Well-tested

## Model Availability Check

The Settings page automatically checks if a model is available in Ollama:
- ✅ Green checkmark = Model is installed
- ⚠️ Warning = Model not found (install with `ollama pull`)

## Performance Impact

### Speed Improvements
- **3B models:** 2-3x faster than 7B
- **2B models:** 3-4x faster than 7B
- **Response time:** 1-2 seconds (3B) vs 3-5 seconds (7B)

### Quality Comparison
- **7B models:** Highest quality, best for complex tasks
- **3B models:** High quality, similar to 7B for most tasks
- **2B models:** Good quality, sufficient for simple queries

### Memory Usage
- **7B models:** ~4-5 GB RAM
- **3B models:** ~2-3 GB RAM
- **2B models:** ~1-2 GB RAM

## Tips

1. **Start with Qwen 3B:** Best balance of speed and quality
2. **Switch to 7B:** If you need higher quality for complex queries
3. **Use 2B:** For maximum speed on simple queries
4. **Test different models:** See which works best for your use case
5. **Cache is model-specific:** Switching models clears cache (expected)

## Technical Details

- **Context Window:** Smaller models (2B/3B) use 2K context, larger (7B+) use 4K
- **Caching:** Each model is cached separately
- **Embeddings:** Always uses BGE-M3 (consistent across all models)
- **Temperature:** Adjustable per model in Settings

## Troubleshooting

**Model not found?**
- Run `ollama pull <model_name>` to install
- Check Ollama is running: `ollama list`

**Model too slow?**
- Switch to a 3B or 2B model
- Check system resources (CPU/RAM)

**Model quality not good?**
- Switch to a 7B or 8B model
- Adjust temperature (lower = more focused)

