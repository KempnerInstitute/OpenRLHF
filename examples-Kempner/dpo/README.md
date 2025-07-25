# Qwen Fine-Tuning with DPO on SLURM

This example demonstrates how to fine-tune the [Qwen](https://huggingface.co/Qwen) language model using **Direct Preference Optimization (DPO)** on the Kempner AI Cluster.

---

##  What is DPO?

**Direct Preference Optimization (DPO)** is a gradient-based method for aligning models to human preferences without reinforcement learning or KL penalties. It relies solely on pairwise comparison data.

Learn more in the [DPO paper](https://arxiv.org/abs/2305.18290) and the [OpenRLHF repository](https://github.com/OpenLMLab/OpenRLHF).

---

##  About `training_qwen_dpo.slrm`

This SLURM script launches a distributed DPO training job using `train_dpo.py` from the OpenRLHF repo. It supports:

- Qwen model fine-tuning
- Distributed multi-GPU training
- Customizable resource allocation

### Key Parameters to Customize

- `MODEL_NAME_OR_PATH`: Pretrained Qwen model (e.g., `Qwen/Qwen-1_8B-Chat`)
- `DATA_PATH`: Path to your dataset in DPO format
- `OUTPUT_DIR`: Output directory for fine-tuned checkpoints
- `GPUS_PER_NODE`, `NODES`, `CPUS_PER_TASK`, `MEM_PER_GPU`: SLURM resource specs

---

##  How to Run

1. **Prepare Your Dataset**  
   Format your data as preference pairs (chosen vs rejected completions).

2. **Clone OpenRLHF**

   ```bash
   git clone https://github.com/KempnerInstitute/OpenRLHF.git
   cd OpenRLHF/examples-Kempner/dpo
   ```

3. **Configure the SLURM Script**  
   Edit `training_qwen_dpo.slrm` to update model path, data path, and job settings.

4. **Submit the Job**

   ```bash
   sbatch training_qwen_dpo.slrm
   ```

---

##  Requirements

- Follow OpenRLHF installation instructions **OR**
- Use the provided **Singularity container** available on the Kempner AI Cluster

---

##  References

- [OpenRLHF (GitHub)](https://github.com/OpenLMLab/OpenRLHF)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Qwen Models on Hugging Face](https://huggingface.co/Qwen)
