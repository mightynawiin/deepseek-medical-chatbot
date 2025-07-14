# 🧠 Reasoning-Enhanced Medical Assistant Using Fine-Tuned DeepSeek-R1
---



![DeepSeek](https://img.shields.io/badge/DeepSeek-R1-blue?style=flat-square)
![Unsloth](https://img.shields.io/badge/Unsloth-Optimized-orange?style=flat-square)
![LoRA](https://img.shields.io/badge/LoRA-LowRank-yellow?style=flat-square)
![SFT](https://img.shields.io/badge/SFT-Trainer-green?style=flat-square)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-ffcc00?logo=huggingface&style=flat-square)
![TRL](https://img.shields.io/badge/TRL-Finetune-9cf?style=flat-square)
![WandB](https://img.shields.io/badge/W&B-Logged-black?logo=wandb&style=flat-square)
![Colab](https://img.shields.io/badge/Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white&style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&style=flat-square)
![HF Hub](https://img.shields.io/badge/Model-HF%20Hub-red?logo=huggingface&style=flat-square)
![CoT](https://img.shields.io/badge/Reasoning-Chain_of_Thought-purple?style=flat-square)




A fine-tuned Medical Intelligence that answers complex clinical questions with **step-by-step reasoning**, built using **LoRA** and **SFTTrainer** on unsloth/DeepSeek-R1-Distill-Llama-8B.

>Unlike typical chatbots, this model **doesn't guess** — it **reasons**, analyzing medical questions through a **Chain-of-Thought** (CoT) format before producing accurate diagnostic responses.


---

## 🔍 Model Highlights

- **Model Base:** `unsloth/DeepSeek-R1-Distill-Llama-8B`
- **Tuning Method:** LoRA + SFT (Supervised Fine-Tuning)
- **Reasoning Style:** Chain-of-Thought (CoT) → Final Answer
- **Inference Engine:** FastLanguageModel (Unsloth)

---

## 🗃️ Dataset Used

- **Name:** `FreedomIntelligence/medical-o1-reasoning-SFT`
- **Core Fields:**
  - `Question` – Medical question
  - `Complex_CoT` – Step-by-step clinical reasoning
  - `Response` – Final diagnosis / answer

---

## 🏷️ Tech Stack & Tools

| Tool / Framework         | Role                                      | Badge                                                                 |
|--------------------------|-------------------------------------------|------------------------------------------------------------------------|
| 🧠 **DeepSeek R1**        | Base model for fine-tuning                | ![DeepSeek](https://img.shields.io/badge/DeepSeek-R1-blue?style=flat-square) |
| ⚡ **Unsloth**            | Memory-efficient training backend         | ![Unsloth](https://img.shields.io/badge/Unsloth-Optimized-orange?style=flat-square) |
| 🔧 **LoRA (Adaptation)** | Parameter-efficient fine-tuning method    | ![LoRA](https://img.shields.io/badge/LoRA-LowRank-yellow?style=flat-square) |
| 🎯 **SFT Trainer (TRL)** | Supervised fine-tuning framework          | ![SFT](https://img.shields.io/badge/SFT-Supervised%20Finetuning-green?style=flat-square) |
| 🤗 **Transformers**      | Model loading & tokenizer utilities       | ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-ffcc00?logo=huggingface&style=flat-square) |
| 🧪 **TRL (HF)**           | Trainer and logging utilities             | ![TRL](https://img.shields.io/badge/TRL-SFTTrainer-9cf?style=flat-square) |
| 📊 **Weights & Biases**  | Experiment tracking and visualization     | ![WandB](https://img.shields.io/badge/Weights_&_Biases-Logged-black?logo=wandb&style=flat-square) |
| 💻 **Google Colab**      | Training and testing environment          | ![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white&style=flat-square) |
| 🐍 **Python 3.10**        | Primary programming language              | ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&style=flat-square) |
| 📓 **Jupyter Notebook**  | Interactive coding and documentation      | ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&style=flat-square) |
| ☁️ **HuggingFace Hub**   | Model hosting and sharing platform        | ![HF Hub](https://img.shields.io/badge/Model-HF%20Hub-red?logo=huggingface&style=flat-square) |

---


## 🚦 Training Flow (Think → Reason → Answer)

1. Load and inspect dataset
2. Format prompts (`Instruction → Question → Reasoning → Answer`)
3. Tokenize using Unsloth’s tokenizer
4. Apply LoRA to selected model layers
5. Fine-tune using `SFTTrainer`
6. Save locally and upload to Hugging Face Hub
7. Perform inference with structured prompts

```plaintext
 ┌─────────────────────────────┐
 │ 1. Load medical dataset     │
 └────────────┬────────────────┘
              ↓
 ┌─────────────────────────────┐
 │ 2. Format prompts:          │
 │    Instruction + CoT        │
 └────────────┬────────────────┘
              ↓
 ┌─────────────────────────────┐
 │ 3. Tokenize using Unsloth   │
 └────────────┬────────────────┘
              ↓
 ┌─────────────────────────────┐
 │ 4. Apply LoRA to layers     │
 └────────────┬────────────────┘
              ↓
 ┌─────────────────────────────┐
 │ 5. Fine-tune using SFT      │
 └────────────┬────────────────┘
              ↓
 ┌─────────────────────────────┐
 │ 6. Save + push to HF Hub    │
 └────────────┬────────────────┘
              ↓
 ┌─────────────────────────────┐
 │ 7. Inference:               │
 │    Model reasons first,     │
 │    then answers.            │
 └─────────────────────────────┘
```

---

## 🔬 Key Concepts

- **Fine-Tuning:** Customizing a pre-trained model to a domain-specific task
- **LoRA (Low-Rank Adaptation):** Efficient adaptation of large models by tuning fewer parameters
- **SFT (Supervised Fine-Tuning):** Training with labeled input-output pairs
- **Chain-of-Thought (CoT):** Generating intermediate reasoning steps before the final output
- **Tokenizer:** Converts text into tokens that the model can understand

---

## ⚡ Challenges & Resolutions

| Challenge                   | Solution                               |
|----------------------------|----------------------------------------|
| GPU memory limitations     | Used 4-bit quantization with LoRA      |
| Prompt structure tuning    | Developed a CoT-style prompt template  |
| Token length truncation    | Increased `max_seq_length`             |
| Evaluation variance        | Combined metric-based and manual eval  |
| Device mismatch in inference | Ensured CUDA usage with proper config  |

---

## 🧠 Example Behavior

**Input:**  
Why does iron deficiency lead to anemia?

**_Model Thinking (Chain-of-Thought Reasoning):_**  
> Iron is essential for the production of hemoglobin in red blood cells.  
> Hemoglobin binds oxygen for transport through the bloodstream.  
> Without enough iron, fewer red blood cells are produced and oxygen delivery is reduced.

**Final Output:**  
➤ Thus, iron deficiency results in anemia due to reduced oxygen-carrying capacity.



---



❤️ **Thanks for visiting! Your support means a lot.**  
⭐ *If you find this repo helpful, please consider giving it a star!*

---
