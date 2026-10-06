# References

Verification status: `verified` = checked against the live source this session; `unverified` = from the brief only.

- [S1] Nightingale Collective (Von Arx, Slade Byrd, Kitts, Larsen). *Discovery of a new OpenAI agent message board.* https://collusion.wiki/ (2026-09-04). verified 2026-09-09 from ada.
- [S2] Collusion Wiki export. https://collusion.wiki/explorer/download (redirect from `/download.html`). Draft banner present. verified 2026-09-09.
- [S3a] OpenAI. *The Hugging Face incident and the road ahead.* https://openai.com/index/hugging-face-incident-and-the-road-ahead/ (2026-08-26). verified (browser).
- [S3b] Hugging Face. *Agent intrusion: technical timeline.* https://huggingface.co/blog/agent-intrusion-technical-timeline (2026-07-27). verified.
- [S4] METR / Redwood Research. Investigation post, https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/ (2026-08-26). verified.
- [S5] Betley, Treutlein, Dumas. *Steering towards "automated grading" degrades alignment.* LessWrong (2026-09-03). verified.
- [S6] Qwen/Qwen3.6-27B model card and files, revision 6a9e13bd (2026-04-24). verified.
- [S8] Tice et al. *Alignment Pretraining.* arXiv:2601.10160. abstract verified.
- [S9] Betley et al. *Value Leakage.* arXiv:2607.14345 (2026-07-15). abstract + App. F summary verified.
- [S10] TruthfulAI-research/value_leakage @ f7e5480 (2026-07-17), no licence. verified.
- [S13] myprimals.com/for-researchers. use terms verified.
- [S14] Clifton & Yaden (2021). Brief measures of the four highest-order primal world beliefs. Psychological Assessment. unverified (cited by the guide).
- [S15] Clifton. *Primals Inventories Administration Instructions*, updated June 2022. local PDF sha256 b755f4d5... verified (page 6 PI-18 items and SAS code).
- [S17] anthropics/evals coordinate-itself.jsonl. verified (first rows).
- [S19] johny-b/public-steering-vectors @ cb715869 (2026-09-02), MIT. verified.
- [U1] harsh-kumar9/demand-worlds-corpus @ 1970b55. verified (submodule).
- [S20] Tan, Bostock, draganover, ma-rmartinez, sidbaines, Africa. *Your Model Organisms Might Be Fried.* LessWrong (2026-06-18), https://www.lesswrong.com/posts/WmEcgcstzYCcMpc7z/your-model-organisms-might-be-fried; code ArcadiaImpact/fried-model-organisms (Apache-2.0; `src/README.md`, `elicit.py`, `panel.py`, `thurstone.py`, `config/questions/main.jsonl` read 2026-10-02); items `arcadia-impact/question-consistency-datasets` config `items_500`. verified (post and the files listed).
- [S21] anthropics/jacobian-lens (Apache-2.0), companion to *Verbalizable Representations Form a Global Workspace in Language Models* (Transformer Circuits, 2026): lens_l(h) = unembed(E[dh_final/dh_l] h), fitted on about 1,000 web sequences of 128 tokens. README read 2026-10-02 via a summary; paper not read. A GPT-2 replication is reported to fail (greaterwrong.com/posts/tgn3pD2gLpZvepkWk). unverified.
- [S22] *Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models.* arXiv:2605.11887. Abstract only: 14 groups of SAEs across 7 Qwen3 / Qwen3.5 backbones. unverified.

## Literature ([L] codes, cited in `docs/STORY.md`)

Verification: `verified` = title, authors and id checked against the live source on 2026-10-06; `unverified` = cited from memory, check before the paper.

- [L1] Turner, A. *Self-fulfilling misalignment data might be poisoning our AI models.* LessWrong / turntrout.com (2025-03-02). https://www.lesswrong.com/posts/QkEyry3Mqo8umbhoK/self-fulfilling-misalignment-data-might-be-poisoning-our-ai. verified.
- [L2] Anthropic Alignment Science. *Modifying LLM Beliefs with Synthetic Document Finetuning* (2025). https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf. verified (follow-up "believe it or not" at alignment.anthropic.com/2025/believe-it-or-not, not read).
- [L3] Greenblatt, Denison, Wright, Roger, MacDiarmid, Marks, Treutlein, et al. *Alignment faking in large language models.* arXiv:2412.14093 (2024). SDF used to teach facts about the training setup. unverified.
- [L4] Marks, Treutlein, Bricken, et al. *Auditing language models for hidden objectives.* arXiv:2503.10965 (2025). SDF used to implant a hidden objective. unverified.
- [L5] MacDiarmid, Wright, Uesato, Benton, Kutasov, Price, et al. *Natural emergent misalignment from reward hacking in production RL.* arXiv:2511.18397 (2025). SDF seeds reward-hacking knowledge before RL; the hacking generalises to sabotage and deception. verified.
- [L6] Berglund, Stickland, Balesni, Kaufmann, Tong, Korbak, Kokotajlo, Evans. *Taken out of context: On measuring situational awareness in LLMs.* arXiv:2309.00667 (2023). Out-of-context reasoning: a description learned in finetuning, applied at test time without demonstrations. verified.
- [L7] Treutlein, Choi, Betley, Marks, Anil, Grosse, Evans. *Connecting the dots: LLMs can infer and verbalize latent structure from disparate training data.* arXiv:2406.14546 (2024). unverified.
- [L8] Betley, Tan, Warncke, Sztyber-Betley, Bao, Soto, Labenz, Evans. *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* arXiv:2502.17424 (2025; ICML 2025). verified.
- [L9] Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson. *Fine-tuning aligned language models compromises safety, even when users do not intend to!* arXiv:2310.03693 (2023). Benign finetuning erodes post-training. unverified.
- [L10] Lynch, Wright, Larson, Troy, Ritchie, Mindermann, Perez, Hubinger. *Agentic Misalignment: How LLMs could be insider threats.* Anthropic (2025-06); code anthropic-experimental/agentic-misalignment (MIT), vendored at `third_party/agentic_misalignment`. unverified (post), code verified.
- [L11] Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid. *Activation addition: Steering language models without optimization.* arXiv:2308.10248 (2023). unverified.
- [L12] Rimsky, Gabrieli, Schulz, Tong, Hubinger, Turner. *Steering Llama 2 via contrastive activation addition.* arXiv:2312.06681 (2023; ACL 2024). The mean-difference construction our vectors use. unverified.
- [L13] Needham, Edkins, Pimpale, Bartsch, Hobbhahn. *Large Language Models Often Know When They Are Being Evaluated.* arXiv:2505.23836 (2025). verified.
- [L14] Laine, Chughtai, Betley, Hariharan, Scheurer, Balesni, Hobbhahn, Meinke, Evans. *Me, Myself, and AI: The Situational Awareness Dataset.* arXiv:2407.04694 (2024). unverified.
- [L15] Chen, Arditi, Sleight, Evans, Lindsey. *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509 (2025). Finetuning shifts projected onto trait directions; the test our H3/H6 fails. verified.
- [L16] Soligo, Turner, Rajamanoharan, Nanda. *Convergent linear representations of emergent misalignment.* arXiv:2506.11618 (2025). unverified.
- [L17] Wang, Chen, Mossing, et al. (OpenAI). *Persona features control emergent misalignment.* arXiv:2506.19823 (2025). unverified.
- [L18] Dmitrii (BSc thesis, advised by Vili Kohonen). *Investigating Self-Fulfilling Misalignment and Collusion in AI Control.* LessWrong (2025). GPT-4.1 finetuned on 1.6M tokens of AI-control literature; blackmail in Agentic Misalignment rises from 7% to 64% (ethical goal). https://www.lesswrong.com/posts/vfxoCKW9T9wJFbCfv/investigating-self-fulfilling-misalignment-and-collusion-in. verified (summary only).
- [L19] Taylor, Chua, Betley, Treutlein, Evans. *School of Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs.* arXiv:2508.17511 (2025). The dataset behind the vendored SoRH task; its generalisation result is the "misalignment from doing" case. verified.
- [L20] Tan, Chua, Betley, Treutlein, Evans. *Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time.* arXiv:2510.04340 (2025); and Wichers et al., *Inoculation prompting* (2025), arXiv id unverified. verified (Tan et al. id).
- [L21] Marks, Rager, Michaud, Belinkov, Bau, Mueller. *Sparse feature circuits.* arXiv:2403.19647 (2024). Attribution through a dictionary, the construction `scripts/sae_attr.py` follows. unverified.
- [L22] Hubinger, Denison, Mu, et al. *Sleeper agents: Training deceptive LLMs that persist through safety training.* arXiv:2401.05566 (2024). Cue-conditional misbehaviour that survives post-training; the comparison point for the "expects cheating" tripwire. unverified.
