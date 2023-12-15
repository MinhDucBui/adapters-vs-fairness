<div align="center">

# The Trade-off between Fairness and Efficiency in Adapter Modules

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## 📌&nbsp;&nbsp;Introduction
Our research addresses the trade-off between performance, efficiency, and fairness in NLP. We compare fully finetuned models to adapter modules in toxic text classification on the Jigsaw dataset. Adapters maintain accuracy and reduce training time, but fairness slightly decreases. Results vary across groups and settings, highlighting the need for case-specific evaluation.

## 🚀&nbsp;&nbsp;Quickstart

Configure your environment first.

```bash
# clone project, cd to project folder

# [OPTIONAL] create conda environment
conda create -n adapters_vs_fairness python=3.9
conda activate adapters_vs_fairness

# install requirements
pip install -r requirements.txt

# Please make sure to have the right Pytorch Version: Install pytorch according to instructions
# https://pytorch.org/get-started/
```

Download Jigaw dataset from Kaggle [here](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data). Put all csv files into data/jigsaw folder.

Execute the main script. The default setting uses BERT.
```bash
# Choose GPU Device
export CUDA_VISIBLE_DEVICES=0

# execute main script
python run.py

# To change your accelerator
python run.py trainer.accelerator=cpu

# To change the learning rate
python run.py module.optimizer.lr=0.000002

# To change the random seed
python run.py seed=0
```

### ⚡&nbsp;&nbsp;All Experiments

<details>
<summary><b>Change to Bert+Adapters</b></summary>


```bash
python run.py experiment=bert_adapters
```

  
  
</details>

<details>
<summary><b>Change to Bert+LoRA</b></summary>


```bash
python run.py experiment=bert_lora
```

  
  
</details>

<details>
<summary><b>Change to RoBERTa</b></summary>


```bash
python run.py experiment=roberta
```

</details>

<details>
<summary><b>Change to RoBERTa+Adapters</b></summary>


```bash
python run.py experiment=roberta_adapters
```

</details>

<details>
<summary><b>Change to RoBERTa+LoRA</b></summary>


```bash
python run.py experiment=roberta_lora
```

</details>

<details>
<summary><b>Change to GPT-2</b></summary>


```bash
python run.py experiment=gpt2
```

</details>

<details>
<summary><b>Change to GPT-2+Adapters</b></summary>


```bash
python run.py experiment=gpt2_adapters
```

</details>

<details>
<summary><b>Change to GPT-2+LoRA</b></summary>


```bash
python run.py experiment=gpt2_lora
```

</details>


<details>
<summary><b>Change to RoBERTa_Large</b></summary>


```bash
python run.py experiment=roberta_large
```

</details>

<details>
<summary><b>Change to RoBERTa_Large+Adapters</b></summary>


```bash
python run.py experiment=roberta_large_adapters
```

</details>

<details>
<summary><b>Change to RoBERTa_Large+LoRA</b></summary>


```bash
python run.py experiment=roberta_large_lora
```

</details>

<br>

<br>
