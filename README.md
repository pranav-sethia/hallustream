# HalluStream

**Live hallucination highlighting for open LLMs.** A small probe reads Qwen3's hidden states and flags names, dates, numbers and citations the model probably can't back up, token by token, as the text streams.

> 🚧 **Work in progress.** The results, demo link and `pip install` instructions will appear here once they exist. Every number will come from a logged run.

## What this builds on
HalluStream applies the method from Obeso et al., [*Real-Time Detection of Hallucinated Entities in Long-Form Generation*](https://arxiv.org/abs/2509.03531) (2025), to Qwen3. It adds:
- training and testing on the model's **own** answers, using an automatic Wikipedia-based labeler
- a streaming Python library
- a live demo that doubles as an MCP tool for AI agents

The evaluation plan is fixed in advance in [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md). Data sources and licenses are listed in [docs/DATA.md](docs/DATA.md).

## License
Apache-2.0
