# Reading list (from search snippets only, not yet read in full; verify before citing in the report)

## Temperature / sampling
- Dubois, Yvon, Piantanida, *How Sampling Affects the Detectability of Machine-written Texts*, arXiv 2510.13681 (Oct 2025). Small changes to temperature or top-p can drop a detector from near-perfect to about 1% AUROC. Released a dataset with 37 decoding configurations. Idea: perplexity features alone are brittle, so test them on texts from different decoding settings.
- Kempton et al., *TempTest: Local Normalization Distortion and the Detection of Machine-generated Text*, arXiv 2503.20421 (Mar 2025). Zero-shot detector that picks up the distortion temperature and top-k leave in normalized token probabilities. Idea: add as a feature.
- Hsu & Yang, *Estimating Self-Scoring AI-Text Detectors' Failure Temperatures from Human Text*, SSRN 7553362. Fast-DetectGPT and Binoculars fall to chance as temperature rises.
- *Learn-to-Distance*, arXiv 2601.21895 (2026). Stays accurate when train and test temperatures differ.
- DetectLLM (log-rank), arXiv 2306.05540. Low temperature makes log-likelihood/log-rank methods work very well.

## Human + AI mixed text (our "fusion")
- *FAID*, arXiv 2505.14271 (2025). Three classes (human / AI / human-AI collaborative), the closest match to this task.
- *LLM-DetectAIve*, arXiv 2408.04284. Four classes, including human-written then machine-polished.
- Boundary / segment detection: *Mixture of Detectors* arXiv 2509.22147; *Sentence-level segmentation* arXiv 2509.17830; *DAMASHA* arXiv 2512.04838; *D2C-Routing* arXiv 2608.27380 (2026). Idea: per-chunk detector scores and their variance as fusion features.
- *TSM-Bench*, arXiv 2605.31113 (2026). LLM-edited Wikipedia text.

## Multilingual / zero-shot
- *DetectRL-X*, arXiv 2605.15518 (ACL 2026). Zero-shot detectors are weak outside English.
- Hans et al., *Binoculars*, arXiv 2401.12070. Cross-perplexity between two models.
- *LLM-specific fine-tuning for detecting AI-generated text*, arXiv 2601.20006 (2026).

## Experiments to try (each compared on the fixed split in `data/split.csv`)
1. Encoder (XLM-R / mDeBERTa) vs. the TF-IDF baseline; long-text handling (head+tail vs. chunk averaging).
2. Per-chunk perplexity / entropy / log-rank features from a small open LM (e.g. Qwen2.5-0.5B), fed to a GBM or stacked with the encoder (Binoculars / TempTest-style).
3. Class weights or thresholds tuned for macro-F1 (fusion is the hard class).

## Structure-based detection (chosen direction: ensemble of token-level group + structure group)
- Kim et al., *Threads of Subtlety: Detecting Machine-Generated Texts Through Discourse Motifs*, ACL 2024 (aclanthology 2024.acl-long.298). RST trees from the DMRST parser (Liu et al. 2021), counting recurring subgraph "motifs"; human text shows more structural variability; robust on paraphrased/OOD text. Limits: parser has a short input limit, code listed as "TBA".
- Tripto et al., *Beyond Checkmate: Exploring the Creative Choke Points for AI Generated Texts*, EMNLP 2025 (aclanthology 2025.emnlp-main.600). Splits a text into intro / body / conclusion; the body is the most informative segment, and human texts vary more in style across segments. Code: github.com/tripto03/chess_inspired_human_ai_text_distinction (licence not stated).
- Anthropic, *How Claude's text watermarking works* (Aug 2026): NOT usable. Needs the provider's key, closed API (rule 1), Claude-only.
