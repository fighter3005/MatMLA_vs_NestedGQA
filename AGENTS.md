# Agent Guidelines

## Rules
- 

# Information
## Environment
You are working either on a MacBook or on a server with NVidia GPUs.
If pyenv is available (usually MacBook), use that. Otherwise use uv with virtual environments and to set python version.

## Files
attention_moe.txt: covers some files from a repo that trains language models with Q, K or V experts (MoE style) on small datasets like wikitext.
hydralvlm.txt: covers hydra and MatFormer style sliced models (gemma, smolvlm). The pipeline includes importance based reordering, model implementation and training.
Both files only include a subset of the respecite repositories.
189_MatMLA_Matryoshka_Multi_He.pdf: The paper I want to replicate and evaluate against other solutions.
MatMLA_Compressed KV Cache Flow.md: some Q&A on the paper (189_MatMLA_Matryoshka_Multi_He)