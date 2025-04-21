# Perception, <span style="color:purple">R</span>eason, <span style="color:purple">T</span>hink, and <span style="color:purple">P</span>lan: 
# A Survey on Large Multimodal Reasoning Models

## Table of Contents

1. **Introduction**
   - Background and Motivation
   - Definition of Large Multimodal Reasoning Models (LMRMs)
   - Survey Scope and Organization

2. **Background and Preliminaries**
   - Fundamentals of Multimodal Learning
   - Cognitive Theories of Reasoning (System 1 vs. System 2)
   - Metrics and Evaluation Methods

3. **Stage 1: Perception-Driven Reasoning**
   - [Neural Modular Reasoning Networks](#neural-modular-reasoning-networks)
   - [VLMs-based Modular Reasoning](#vlms-based-modular-reasoning)
   - Limitations of Perception-Driven Approaches

4. **Stage 2: Language-Centric Short Reasoning (System-1 Reasoning)**
   - Multimodal Chain-of-Thought (MCoT)
   - Structured Reasoning in MLLMs
      - [Multimodal Defined Reasoning](#multimodal-defined-reasoning)
      - [Multimodal Structural Reasoning](#multimodal-structural-reasoning)
   - External Enhanced Reasoning
      - [Tool-Augmented Reasoning](#approaches-enhancing-multimodal-reasoning-through-textual-mechanisms)
      - [Retrieval-Augmented Reasoning](#approaches-enhance-multimodal-reasoning-through-retrieval-mechanisms)
      - [Cross-Modal Reasoning](#approaches-enhancing-cross-modal-reasoning)
      - [Visual Experts](#multimodal-reasoning-with-visual-experts)
      - [External Enhanced (Overview)](#external-enhanced)

5. **Stage 3: Language-Centric Long Reasoning (System-2 Thinking and Planning)**
   - [Multimodal O1-style Long Thinking](#approach-of-mm-o1)
   - [Multimodal R1-style Long Planning](#approach-of-mm-r1)
   - Emerging Capabilities and Limitations

6. **Stage 4: Towards Native Large Multimodal Reasoning Models (N-LMRMs)**
   - Multimodal Agentic Reasoning
      - Long-horizon Planning
      - Dynamic Adaptation
      - Embodied Learning
   - Omni-Modal Understanding and Generative Reasoning
      - Heterogeneous Data Fusion
      - Contextual Multimodal Generation
      - Modality-agnostic Inference

7. **Benchmarks and Datasets**
   - [Multimodal Understanding Benchmarks](#multimodal-understanding)
     - [Visual Centric](#visual-centric)
     - [Audio Centric](#audio-centric)
   - [Multimodal Generation Benchmarks](#multimodal-generation)
     - [Cross-modal Generation](#cross-modal-generation)
     - [Joint Multimodal Generation](#joint-multimodal-generation)
   - [Multimodal Reasoning Benchmarks](#multimodal-reasoning)
     - [General Visual Reasoning](#general-visual-reasoning)
     - [Domain-specific Reasoning](#domain-specific-reasoning)
   - [Multimodal Planning Benchmarks](#multimodal-planning)
     - [GUI Navigation](#gui-navigation)
     - [Embodied and Simulated Environments](#embodied-and-simulated-environments)

8. **Open Challenges and Future Directions**
   - Scaling Laws for Multimodal Reasoning
   - Theoretical Foundations
   - Ethical Considerations
   - Cognitive Alignment

9. **Conclusion**

10. **References**


### Neural Modular Reasoning Networks

| **Model** | **Year** | **Architecture** | **Highlight** | **Training Method** |
|-----------|----------|------------------|---------------|---------------------|
| NMN (andreas2016neural: Neural module networks) | 2016 | Modular | Dynamically assembles task-specific modules for visual-textual reasoning. | Supervised learning |
| HieCoAtt (lu2016hierarchical: Hierarchical question-image co-attention for visual question answering) | 2016 | Attention-based | Aligns question semantics with image regions via hierarchical cross-modal attention. | Supervised learning |
| [MCB](https://arxiv.org/abs/1606.01847) | 2016 | Bilinear | Optimizes cross-modal feature interactions with efficient bilinear modules. | Supervised learning |
| [SANs](https://doi.org/10.1109/CVPR.2016.10) | 2016 | Attention-based | Iteratively refines reasoning through multiple attention hops over visual features. | Supervised learning |
| DMN (xiong2016dynamic: Dynamic memory networks for visual and textual question answering) | 2016 | Memory-based | Integrates memory modules for multi-episode reasoning over sequential inputs. | Supervised learning |
| [ReasonNet](https://proceedings.neurips.cc/paper/2017/hash/f61d6947467ccd3aa5af24db320235dd-Abstract.html) | 2017 | Modular | Decomposes reasoning into entity-relation modules for structured inference. | Supervised learning |
| UpDn (anderson2018bottom: Bottom-up and top-down attention for image captioning and visual question answering) | 2018 | Attention-based | Combines bottom-up and top-down attention for object-level reasoning. | Supervised learning |
| [MAC](https://arxiv.org/abs/1803.03067) | 2018 | Memory-based | Uses a memory-augmented control unit for iterative compositional reasoning. | Supervised learning |
| BAN (kim2018bilinear: Bilinear attention networks) | 2018 | Bilinear | Captures high-order interactions via bilinear attention across modalities. | Supervised learning |
| [HeteroMemory](http://openaccess.thecvf.com/content_CVPR_2019/html/Fan_Heterogeneous_Memory_Enhanced_Multimodal_Attention_Model_for_Video_Question_Answering_CVPR_2019_paper.html) | 2019 | Memory-based | Synchronizes appearance and motion modules for video-based temporal reasoning. | Supervised learning |
| MuRel (cadene2019murel: Murel: Multimodal relational reasoning for visual question answering) | 2019 | Relational | Models reasoning as a relational network over object pairs for fine-grained inference. | Supervised learning |
| MCAN (yu2019deep: Deep modular co-attention networks for visual question answering) | 2019 | Attention-based | Employs modular co-attention with self- and guided-attention for deep reasoning. | Supervised learning |

### VLMs-based Modular Reasoning

| **Model** | **Year** | **Architecture** | **Highlight** | **Training Method** |
|-----------|----------|------------------|---------------|---------------------|
| ViLBERT (lu2019vilbert: Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks) | 2019 | Dual-Encoder | Aligns visual-text features via dual-stream Transformers with cross-modal attention. | Pretraining + fine-tuning |
| [LXMERT](https://arxiv.org/abs/1908.07490) | 2019 | Dual-Encoder | Enhances cross-modal reasoning with dual-stream pretraining on diverse tasks. | Pretraining + fine-tuning |
| X-[LXMERT](https://arxiv.org/abs/1908.07490) | 2020 | Dual-Encoder | Extends dual-stream reasoning with generative cross-modal pretraining. | Pretraining + fine-tuning |
| ALBEF (li2021align: Align before fuse: Vision and language representation learning with momentum distillation) | 2021 | Dual-Encoder | Integrates contrastive learning with momentum distillation for robust reasoning. | Contrastive + generative pretraining |
| [SimVLM](https://arxiv.org/abs/2108.10904) | 2021 | Dual-Encoder | Uses prefix-based pretraining for flexible cross-modal reasoning. | Pretraining + fine-tuning |
| VLMo (bao2022vlmo: Vlmo: Unified vision-language pre-training with mixture-of-modality-experts) | 2022 | Dual-Encoder | Employs a mixture-of-modality-experts for dynamic cross-modal reasoning. | Pretraining + fine-tuning |
| METER (dou2022empirical: An empirical study of training end-to-end vision-and-language transformers) | 2022 | Dual-Encoder | Enhances reasoning with a modular encoder-decoder for robust alignment. | Pretraining + fine-tuning |
| BLIP (li2022blip: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation) | 2022 | Dual-Encoder | Bootstraps alignment with contrastive learning for efficient reasoning. | Contrastive + generative pretraining |
| [VisualBERT](https://arxiv.org/abs/1908.03557) | 2019 | Single-Transformer-Backbone | Fuses visual-text inputs in a single Transformer for joint contextual reasoning. | Pretraining + fine-tuning |
| [VL-BERT](https://arxiv.org/abs/1908.08530) | 2019 | Single-Transformer-Backbone | Enhances cross-modal reasoning with unified visual-language pretraining. | Pretraining + fine-tuning |
| [UNITER](https://doi.org/10.1007/978-3-030-58577-8_7) | 2020 | Single-Transformer-Backbone | Reasons via joint contextual encoding in a single Transformer backbone. | Pretraining + fine-tuning |
| [PixelBERT](https://arxiv.org/abs/2004.00849) | 2020 | Single-Transformer-Backbone | Processes pixels with CNN+Transformer for fine-grained cross-modal reasoning. | Pretraining + fine-tuning |
| [UniVL](https://arxiv.org/abs/2002.06353) | 2020 | Single-Transformer-Backbone | Unifies video-language reasoning with a single Transformer for temporal tasks. | Pretraining + fine-tuning |
| [Oscar](https://doi.org/10.1007/978-3-030-58577-8_8) | 2020 | Single-Transformer-Backbone | Anchors reasoning with object tags in a unified Transformer for semantic inference. | Pretraining + fine-tuning |
| VinVL (zhang2021vinvl: Vinvl: Revisiting visual representations in vision-language models) | 2021 | Single-Transformer-Backbone | Boosts reasoning with enhanced visual features in a single Transformer. | Pretraining + fine-tuning |
| ERNIE-ViL (yu2021ernie: Ernie-vil: Knowledge enhanced vision-language representations through scene graphs) | 2021 | Single-Transformer-Backbone | Integrates scene graph knowledge for structured visual-language reasoning. | Pretraining + fine-tuning |
| UniT (hu2021unit: Unit: Multimodal multitask learning with a unified transformer) | 2021 | Single-Transformer-Backbone | Streamlines multimodal tasks with a shared self-attention Transformer backbone. | Pretraining + fine-tuning |
| [Flamingo](https://arxiv.org/abs/2204.14198) | 2022 | Single-Transformer-Backbone | Prioritizes dynamic vision-text interactions via cross-attention. | Pretraining + fine-tuning |
| [CoCa](https://openreview.net/forum?id=Ee277P3AYC) | 2022 | Single-Transformer-Backbone | Combines contrastive and generative heads for versatile cross-modal reasoning. | Contrastive + generative pretraining |
| [BEiT-3](https://arxiv.org/abs/2208.10442) | 2022 | Single-Transformer-Backbone | Unifies vision-language learning with masked data modeling. | Pretraining + fine-tuning |
| OFA (wang2022ofa: Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework) | 2022 | Single-Transformer-Backbone | Provides a unified multimodal framework for efficient cross-modal reasoning. | Pretraining + fine-tuning |
| [PaLI](https://arxiv.org/abs/2209.06794) | 2022 | Single-Transformer-Backbone | Scales reasoning with a multilingual single-Transformer framework. | Pretraining + fine-tuning |
| [BLIP-2](https://proceedings.mlr.press/v202/li23q.html) | 2023 | Single-Transformer-Backbone | Uses a querying Transformer for improved cross-modal reasoning efficiency. | Pretraining + fine-tuning |
| [Kosmos-1](https://proceedings.mlr.press/v202/li23q.html) | 2023 | Single-Transformer-Backbone | Enables interleaved input processing for flexible multimodal understanding. | Pretraining + fine-tuning |
| [Kosmos-2](https://proceedings.mlr.press/v202/li23q.html) | 2023 | Single-Transformer-Backbone | Enhances grounding capability for precise object localization and reasoning. | Pretraining + fine-tuning |
| [CLIP-Cap](https://arxiv.org/abs/2111.09734) | 2021 | Vision-Encoder-LLM | Projects CLIP visual features into an LLM for reasoning and captioning. | Fine-tuning |
| [LLaVA](http://papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html) | 2023 | Vision-Encoder-LLM | Tunes ViT-LLM integration for conversational multimodal reasoning. | Instruction tuning |
| [MiniGPT-4](https://arxiv.org/abs/2304.10592) | 2023 | Vision-Encoder-LLM | Aligns ViT to a frozen LLM via projection for streamlined reasoning. | Fine-tuning |
| [InstructBLIP](https://arxiv.org/abs/2305.06500) | 2023 | Vision-Encoder-LLM | Uses instruction tuning to align ViT with LLM for multimodal reasoning. | Instruction tuning |
| [Qwen-VL](https://arxiv.org/abs/2309.16609) | 2023 | Vision-Encoder-LLM | Incorporates spatial-aware ViT for enhanced grounded reasoning. | Pretraining + fine-tuning |
| [mPLUG-Owl](https://arxiv.org/abs/2304.14178) | 2023 | Vision-Encoder-LLM | Integrates modular visual encoder with LLM for instruction-following reasoning. | Instruction tuning |
| [Otter](https://arxiv.org/abs/2305.03726) | 2023 | Vision-Encoder-LLM | Combines modular visual encoder with LLM for in-context multimodal reasoning. | Instruction tuning |

## The Structural Reasoning

| Name | Modality | Task | Reasoning Structure | Datasets | Highlight |
|------|----------|------|---------------------|----------|-----------|
| Cantor (gao2024cantor: Cantor: Inspiring multimodal chain-of-thought of mllm) | T,I | VQA | Perception, Decision | - | Decouples perception and reasoning via feature extraction and CoT-style integration. |
| [TextCoT](https://arxiv.org/abs/2404.09797) | T,I | VQA | Caption, Localization, Precise observation | - | First summarizes visual context, then generates CoT-based responses. |
| [Grounding-Prompter](https://arxiv.org/abs/2312.17117) | T,V,A | Temporal Sentence Grounding | Denoising | VidChapters-7M | Grounding-Prompter performs global parsing, denoising, partitioning before reasoning. |
| [Audio-CoT](https://arxiv.org/abs/2501.07246) | T,A | AQA | Manual-CoT, Zero-Shot-CoT, Desp-CoT | - | Enhances visual reasoning by utilizing three chain-of-thought paradigms. |
| [VIC](https://arxiv.org/abs/2411.12591) | I,T | VQA | Thinking before looking | - | Breaks tasks into text-based sub-steps before integrating visual inputs to form final rationales. |
| Visual [Sketchpad](https://arxiv.org/abs/2406.09403) | I,T | VQA, math QA | Sketch-based reasoning paradigm | - | Organizes rationales into "Thought, Action, Observation" phases. |
| Det-CoT (wu2024dettoolchain: Dettoolchain: A new prompting paradigm to unleash detection ability of mllm) | I,T | VQA | Subtask decomposition, Execution, and Verification | - | Formalizes VQA reasoning as a combination of subtasks and reviews. |
| BDoG (zheng2024picture: A Picture Is Worth a Graph: A Blueprint Debate Paradigm for Multimodal Reasoning) | I,T | VQA | Entity update, Relation update, Graph pruning | - | Utilizes a dedicated debate-summarization pipeline with specialized agents. |
| CoTDet (tang2023cotdet: Cotdet: Affordance knowledge prompting for task driven object detection) | I,T | object detection | Object listing, Affordance analysis, Visual feature summarization | COCO-Tasks | Achieves object detection via human-like procedure of listing, analyzing and summarizing. |
| [CoCoT](https://arxiv.org/abs/2401.02582) | I,T | VQA | Contrastive prompting strategy | - | Systematically contrasts input similarities and differences. |
| SegPref (wang2024avs_cot: Can Textual Semantics Mitigate Sounding Object Segmentation Preference?) | T,A,V | Temporal Sentence Grounding | Visual summary, Sound filtering, Denoising | Youtube-8M, Semantic-ADE20K | Robustly localizes sounding objects in the visual space through global understanding, sounding object filtering, and noise removal. |
| [EMMAX](https://arxiv.org/abs/2412.11974) | I,T | Robotic task | Grounded CoT reasoning, Look-ahead spatial reasoning | Dataset based on BridgeV2 | Integrates grounded planning and predictive. |
| DDCoT (zheng2023ddcot: Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models) | T,I | VQA | Question Deconstruct, Rationale | ScienceQA | Maintains a critical attitude by identifying reasoning and recognition responsibilities through the combined effect of negative-space design and visual deconstruction. |
| AVQA-CoT (li2024avqa_cot: AVQA-CoT: When CoT Meets Question Answering in Audio-Visual Scenarios) | T,A,V | AVQA | Question Deconstruct, Question Selection, Rationale | MUSIC-AVQA | Decomposes complex questions into multiple simpler sub-questions and leverages LLMs to select relevant sub-questions for audio-visual question answering. |
| [CoT-PT](https://arxiv.org/abs/2304.07919) | T,I | Image Classification, Image-Text Retrieval, VQA | Coarse-to-Fine Image Concept Representation | ImageNet | First to successfully adapt CoT for prompt tuning by combining visual and textual embeddings in the vision domain. |
| [IoT](https://arxiv.org/abs/2405.13872) | T,I | VQA | Visual Action Selection, Execution, Rationale, Summary, Self-Refine | - | Enhances visual reasoning by integrating visual and textual rationales through a model-driven multimodal reasoning chain. |
| [Shikra](https://arxiv.org/abs/2306.15195) | T,I | VQA, PointQA | Caption, Object Grounding | ScienceQA | Maintains a critical attitude by identifying reasoning and recognition responsibilities through the combined effect of negative-space design and visual deconstruction. |
| [E-CoT](https://arxiv.org/abs/2407.08693) | T,I,A | Policies' Generalization | Task Rephrase, Planning, Task Deconstruct, Object Grounding | Bidgedata v2 | Integrates semantic planning with low-level perceptual and motor reasoning, advancing task formulations in embodied intelligence. |
| [CoS](https://arxiv.org/abs/2403.12966) | T,I | VQA | Object Grounding, Rationale | Llava665K | Guides the model to identify and focus on key image regions relevant to a question, enabling multi-granularity understanding without compromising resolution. |
| [TextCoT](https://arxiv.org/abs/2404.09797) | T,I | VQA | Caption, Object Grounding, Image Zoom | Llava665K, SharedGPT4V | Enables accurate and interpretable multimodal question answering through staged processing: overview, coarse localization, and fine-grained observation. |
| DCoT (jia2024dcot: DCoT: Dual Chain-of-Thought Prompting for Large Multimodal Models) | T,I | VQA | Object Grounding, Fine-Grained Image Generation, Similar Example Retrieve, Rationale | - | Uses a dual-guidance mechanism by combining bounding box cues to focus attention on relevant image regions and retrieving the most suitable examples from a curated demonstration cluster as contextual support. |



## Multimodal Defined Reasoning

| Name | Modality | Task | Reasoning Structure | Datasets | Highlight |
|------|----------|------|---------------------|----------|-----------|
| Cantor (gao2024cantor: Cantor: Inspiring multimodal chain-of-thought of mllm) | I,T | VQA | perception, decision | - | Decouples perception and reasoning via feature extraction and CoT-style integration |
| [TextCoT](https://arxiv.org/abs/2404.09797) | I,T | VQA | caption, localization, precise observation | - | first summarizes visual context, then generates CoT-based responses |
| [Grounding-Prompter](https://arxiv.org/abs/2312.17117) | V,A,T | Temporal Sentence Grounding | Denoising | VidChapters-7M | Grounding-Prompter performs global parsing, denoising, partitioning before reasoning |


## Multimodal Structural Reasoning

| Name | Modality | Task | Reasoning Structure | Training Datasets | Highlight |
|------|----------|------|---------------------|-------------------|-----------|
| DDCoT (zheng2023ddcot: Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models) | T,I | VQA | Question Deconstruct,Rationale | ScienceQA | Maintains a critical attitude by identifying reasoning and recognition responsibilities through the combined effect of negative-space design and visual deconstruction. |
| AVQA-CoT (li2024avqa_cot: AVQA-CoT: When CoT Meets Question Answering in Audio-Visual Scenarios) | T,A,V | AVQA | Question Deconstruct, Question Selection, Rationale | MUSIC-AVQA | Decomposes complex questions into multiple simpler sub-questions and leverages LLMs to select relevant sub-questions for audio-visual question answering. |
| [CoT-PT](https://arxiv.org/abs/2304.07919) | T,I | Image Classification, Image-Text Retrieval, VQA | Coarse-to-Fine Image Concept Representation | ImageNet | First to successfully adapt CoT for prompt tuning by combining visual and textual embeddings in the vision domain. |
| [IoT](https://arxiv.org/abs/2405.13872) | T,I | VQA | Visual Action Selection, Execution, Rationale, Summary, Self-Refine | - | Enhances visual reasoning by integrating visual and textual rationales through a model-driven multimodal reasoning chain. |
| [Shikra](https://arxiv.org/abs/2306.15195) | T,I | VQA, PointQA | Caption, Object Grounding | ScienceQA | Maintains a critical attitude by identifying reasoning and recognition responsibilities through the combined effect of negative-space design and visual deconstruction. |
| [E-CoT](https://arxiv.org/abs/2407.08693) | T,I,A | Policies' Generalization | Task Rephrase, Planning, Task Deconstruct, Object Grounding | Bidgedata v2 | Integrates semantic planning with low-level perceptual and motor reasoning, advancing task formulations in embodied intelligence. |
| [CoS](https://arxiv.org/abs/2403.12966) | T,I | VQA | Object Grounding, Rationale | Llava665K | Guides the model to identify and focus on key image regions relevant to a question, enabling multi-granularity understanding without compromising resolution. |
| [TextCoT](https://arxiv.org/abs/2404.09797) | T,I | VQA | Caption, Object Grounding, Image Zoom | Llava665K, SharedGPT4V | Enables accurate and interpretable multimodal question answering through staged processing: overview, coarse localization, and fine-grained observation. |
| DCoT (jia2024dcot: DCoT: Dual Chain-of-Thought Prompting for Large Multimodal Models) | T,I | VQA | Object Grounding, Fine-Grained Image Generation, Similar Example Retrieve, Rationale | - | Uses a dual-guidance mechanism by combining bounding box cues to focus attention on relevant image regions and retrieving the most suitable examples from a curated demonstration cluster as contextual support. |



## External Enhanced

| Name | Modality | Task | Enhancement Type | External Source | Highlight |
|------|----------|------|------------------|-----------------|-----------|
| MM-ToT (gomez2023mmtot: MultiModal-ToT) | T,I | Image Generation | Search Algorithm | DFS,BFS | Applies DFS and BFS to select optimal outputs. |
| [HoT](https://arxiv.org/abs/2308.06207) | T,I | VQA | Search Algorithm | multi-hop random walks on graph | Generates linked thoughts from multimodal data in a hyperedge. |
| [AGoT](https://arxiv.org/abs/2404.04538) | T,I | Text-Image Retrieval, VQA | Search Algorithm | prompt aggregation and prompt flow operations | Builds a graph to aggregate multi-faceted reasoning with visuals. |
| BDoG (zheng2024picture: A Picture Is Worth a Graph: A Blueprint Debate Paradigm for Multimodal Reasoning) | T,I | VQA | Search Algorithm | Graph Condensation: Entity update, Relation update, Graph pruning | Effective three-agent debate forms thought graph for multimodal queries. |
| [L3GO](https://arxiv.org/abs/2402.09052) | T,I | 3D Object Generation & Composition | Tools | Blender, ControlNet | Iterative part-based 3D construction through LLM reasoning in a simulation environment. |
| HYDRA (ke2024hydra: HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning) | T,I | Knowledge-QA, Visual Grounding | Tools | RL agent controller, Visual Foundation Models | RL agent controls multi-stage visual reasoning through dynamic instruction selection. |
| Det-CoT (wu2024dettoolchain: Dettoolchain: A new prompting paradigm to unleash detection ability of mllm) | T,I | object detection | Tools | Visual Processing Prompts | Visual prompts guide MLLM attention for structured detection reasoning. |
| [Chain-of-Image](https://arxiv.org/abs/2311.09241) | T,I | Geometric, chess & commonsense reasoning | Tools | Chain of Images prompting | Generates intermediate images during reasoning for visual pattern recognition. |
| AnyMAL (moon2024anymal: Anymal: An efficient and scalable any-modality augmented language model) | T, I, A, V | Cross-modal reasoning, multimodal QA | Tools | Pre-trained alignment module | Efficient integration of diverse modalities; strong reasoning via LLaMA-2 backend. |
| SE-CMRN (zhang2021explicit: Explicit cross-modal representation learning for visual commonsense reasoning) | T,I | Visual Commonsense Reasoning | Tools | Syntactic Graph Convolutional Network | Enhances language-guided visual reasoning via syntactic GCN in a dual-branch network. |
| [RAGAR](https://arxiv.org/abs/2404.12065) | T,I | Political Fact-Checking | RAG | DuckDuckGo & SerpAPI | Integrates MLLMs with retrieval-augmented reasoning to verify facts using text and image evidence. |
| [Chain-of-action](https://arxiv.org/abs/2403.17359) | T,I | Info retrieval | RAG | Google Search, ChromaDB | Decomposes questions into reasoning chains with configurable retrieval actions to resolve conflicts between knowledge sources. |
| KAM-CoT (mondal2024kam: Kam-cot: Knowledge augmented multimodal chain-of-thoughts reasoning) | T,I, KG | Educational science reasoning | RAG | ConceptNet knowledge graph | Enhances reasoning by retrieving structured knowledge from graphs and integrating it through two-stage training. |
| [AR-MCTS](https://arxiv.org/abs/2412.14835) | T,I | Multi-step reasoning | RAG | Contriever, CLIP dual-stream | Step-wise retrieval with Monte Carlo Tree Search for verified reasoning. |
| [MR-MKG](https://arxiv.org/abs/2406.02030) | T, I | General multimodal reasoning | RAG | RGAT | Enhances multimodal reasoning by integrating information from multimodal knowledge graphs. |
| Reverse-HP (zhu2022multimodal: Multimodal reasoning based on knowledge graph embedding for specific diseases) | T, I | Disease-related reasoning | RAG | reverse hyperplane projection | Utilizes KG embeddings to enhance reasoning for specific diseases with multimodal data. |
| [MarT](https://arxiv.org/abs/2210.00312) | T, I | Analogical reasoning | RAG | Structure-guided relation transfer | Uses structure mapping theory and relation-oriented transfer for analogical reasoning with KG. |
| MCoT-Memory (liang2024memory_driven: Memory-Driven Multimodal Chain of Thought for Embodied Long-Horizon Task Planning) | T,I | VQA | Multimodal Information Enhancing | LLAVA | Memory framework and scene graph construction for effective long-horizon task planning |
| [MGCoT](https://arxiv.org/abs/2305.16582) | T,I | VQA | Multimodal Embedding Enhancing | ViT-large encoder | Precise visual feature extraction aiding multimodal reasoning |
| CCoT (mitra2024compositional: Compositional chain-of-thought prompting for large multimodal models) | T,I | VQA | Multimodal Perception Enhancing | Scene Graphs | Utilization of the generated scene graph as an intermediate reasoning step. |
| [CVR-LLM](https://arxiv.org/abs/2409.13980) | T,I | VQA | Multimodal Embedding Enhancing | BLIP2flant5 & BLIP2 multi-embedding | Precise context-aware image descriptions through iterative self-refinement and effective text-multimodal factors integrations |
| TeSO (wang2024avs_cot: Can Textual Semantics Mitigate Sounding Object Segmentation Preference?) | T,V,A | Temporal Sentence Grounding (TSG) | Multimodal Information Enhancing | VGGish | Integrates text semantics to mitigate segmentation preference for better audio-visual correlation boosting AVS performance. |
| [CAT](https://arxiv.org/abs/2305.02677) | T,I | Image Captioning | Multimodal Perception Enhancing | SAM | Promising pre-trained image caption generators, SAM, and instruction-tuned large language models integration |

## Approaches enhancing multimodal reasoning through textual mechanisms

| Name | Modality | Task | Tool | Purpose of Tool | Training Datasets | Highlight |
|------|----------|------|------|-----------------|-------------------|-----------|
| [L3GO](https://arxiv.org/abs/2402.09052) | T,I | 3D object generation & composition | Blender, ControlNet | Part-based 3D construction | - | Iterative part-based 3D construction through LLM reasoning in a simulation environment. |
| HYDRA (ke2024hydra: HYDRA: A Hyper Agent for Dynamic Compositional Visual Reasoning) | T,I | Knowledge-QA, visual grounding | RL agent controller, Visual Foundation Models | Agent scheduling | RL agent with specific rewards | RL agent controls multi-stage visual reasoning through dynamic instruction selection. |
| Det-CoT (wu2024dettoolchain: Dettoolchain: A new prompting paradigm to unleash detection ability of mllm) | T,I | object detection | Visual Processing Prompts | Visual attention guidance | - | Visual prompts guide MLLM attention for structured detection reasoning. |
| [Chain-of-Image](https://arxiv.org/abs/2311.09241) | T,I | Geometric, chess & commonsense reasoning | Chain of Images prompting | visual pattern recognition | Geometric & Chess datasets | Generates intermediate images during reasoning for visual pattern recognition. |
| AnyMAL (moon2024anymal: Anymal: An efficient and scalable any-modality augmented language model) | T, I, A, V | Cross-modal reasoning, multimodal QA | Pre-trained alignment module | diverse signals to text representations | Manual instruction set | Efficient integration of diverse modalities; strong reasoning via LLaMA-2 backend. |
| SE-CMRN (zhang2021explicit: Explicit cross-modal representation learning for visual commonsense reasoning) | T,I | Visual Commonsense Reasoning | Syntactic Graph Convolutional Network | Enhance visual reasoning | VCR dataset | Enhances language-guided visual reasoning via syntactic GCN in a dual-branch network. |

# Approaches enhance multimodal reasoning through retrieval mechanisms

| Name | Modality | Task | Search source | Search engine | Search query | Training Datasets | Highlight |
|------|----------|------|--------------|--------------|-------------|-------------------|-----------|
| [RAGAR](https://arxiv.org/abs/2404.12065) | T,I | Political Fact-Checking | Web, News sites | DuckDuckGo & SerpAPI | LLM-generated questions | MOCHEG dataset | Integrates MLLMs with retrieval-augmented reasoning to verify facts using text and image evidence. |
| [Chain-of-action](https://arxiv.org/abs/2403.17359) | T,I | Info retrieval | Web, domain databases, tabular data | Google Search, ChromaDB | Combined sub-questions with embeddings | Pre-trained LLMs, text-embedding-ada-002 | Decomposes questions into reasoning chains with configurable retrieval actions to resolve conflicts between knowledge sources. |
| KAM-CoT (mondal2024kam: Kam-cot: Knowledge augmented multimodal chain-of-thoughts reasoning) | T,I, KG | Educational science reasoning | ConceptNet knowledge graph | Custom graph extraction | Context-based text, image captions | ScienceQA | Enhances reasoning by retrieving structured knowledge from graphs and integrating it through two-stage training. |
| [AR-MCTS](https://arxiv.org/abs/2412.14835) | T,I | Multi-step reasoning | Math datasets, Wikipedia, COIG | Contriever, CLIP dual-stream | Dynamic state-based retrieval | - | Step-wise retrieval with Monte Carlo Tree Search for verified reasoning. |
| [MR-MKG](https://arxiv.org/abs/2406.02030) | T, I | General multimodal reasoning | MMKGs | RGAT | Top-N Triple Retrieval | ScienceQA, MARS | Enhances multimodal reasoning by integrating information from multimodal knowledge graphs. |
| Reverse-HP (zhu2022multimodal: Multimodal reasoning based on knowledge graph embedding for specific diseases) | T, I | Disease-related reasoning | SDKG-11 | reverse hyperplane projection | entity + relation to entity | Disease entity-set | Utilizes KG embeddings to enhance reasoning for specific diseases with multimodal data. |
| [MarT](https://arxiv.org/abs/2210.00312) | T, I | Analogical reasoning | MarKG | Structure-guided relation transfer | Analogical entity prediction | MARS | Uses structure mapping theory and relation-oriented transfer for analogical reasoning with KG. |


## Multimodal Reasoning with Visual Experts

| Name | Modality | Task | Tools | Propose of Tool | Training Datasets | Highlight |
|------|----------|------|-------|----------------|-------------------|-----------|
| MCoT-Memory (liang2024memory_driven: Memory-Driven Multimodal Chain of Thought for Embodied Long-Horizon Task Planning) | image to text | VQA reasoning | MLLM-based expert (e.g., LLAVA) | scene graph construction | training free | Memory framework and scene graph construction for effective long-horizon task planning |
| [MGCoT](https://arxiv.org/abs/2305.16582) | image to text | VQA reasoning | ViT-large encoder | extracts patch-level features of images enhancing the visual information | AQUA-RAT & ScienceQA | Precise visual feature extraction aiding multimodal reasoning |
| CCoT (mitra2024compositional: Compositional chain-of-thought prompting for large multimodal models) | image to text | VQA reasoning | Scene Graphs | represent visual scenes as structured graphs | training free | Utilization of the generated scene graph as an intermediate reasoning step. |
| [CVR-LLM](https://arxiv.org/abs/2409.13980) | image to text | VQA reasoning | BLIP2flant5xx & BLIP2 multi-embedding | basic captioner & encoding multi-modal information | training free | Precise context-aware image descriptions through iterative self-refinement and effective text-multimodal factors integrations |
| TeSO (wang2024avs_cot: Can Textual Semantics Mitigate Sounding Object Segmentation Preference?) | video, audio to text | Temporal Sentence Grounding (TSG) | Mask2Former & LLaVA-1.5 | provides visual information in the AVS task & generates dense scene descriptions | Youtube-8M & semantic-ADE20K | Effective visual tools for better audio-visual correlation boosting AVS performance. |
| [CAT](https://arxiv.org/abs/2305.02677) | image to text | image captioning | SAM | generate pixel - level masks corresponding to user-selected regions, facilitating object-centered perception | training free | Promising pre-trained image caption generators, SAM, and instruction-tuned large language models integration |

## Approaches enhancing Cross-Modal Reasoning

| Name | Modality | Cross-Modal Reasoning | Task | Highlight |
|------|----------|------------------------|------|-----------|
| [IdealGPT](https://doi.org/10.18653/v1/2023.findings-emnlp.755) | T, I | Answer sub-questions about image via gpt | VQA, Text Entailment | Using gpt to iteratively decompose and solve visual reasoning tasks |
| [AssistGPT](https://arxiv.org/abs/2301.12597) | T, I, V | Plan, Execute, Inspect via External Tools(gpt4, OCR, Grounding, et al.) | VQA, Causal Reasoning | Using an interleaved code and language reasoning approach to handle complex multimodal tasks |
| ProViQ (choudhury2023zero: Zero-Shot Video Question Answering with Procedural Programs) | T, V | Generate and execute Python programs for the video | Video VQA | Using procedural programs to solve visual subtasks in videos |
| [MM-REACT](https://arxiv.org/abs/2303.11381) | T, I, V | Use CV tools for sub-taskss about image | VQA, Video VQA | Vision experts combined with GPT for multimodal reasoning and action |
| [VisualReasoner](https://arxiv.org/abs/2406.19934) | T, I | Synthesize multi-step reasoning(Using exteral CV tools) data | GQA, VQA | Proposing a least-to-most visual reasoning paradigm and a data synthesis approach for training |
| [Multi-model-thought](https://arxiv.org/abs/2502.11514) | T, I | External Tools(Visual Sketchpad) | Geometry, Math, VQA | Investigating inference-time scaling for multi-modal thought across diverse tasks |
| [FaST](https://arxiv.org/abs/2408.08862) | T, I | System switch adapter for visual reasoning | VQA | Integrating fast and slow thinking mechanisms into visual agents |
| [ICoT](https://arxiv.org/abs/2411.19488) | T, I | Generate interleaved visual-textual reasoning via ADS | VQA | Using visual patches as reasoning carriers to improve LMMs' fine-grained reasoning |
| [Image-of-Thought](https://arxiv.org/abs/2405.13872) | T, I | Extract visual rationales step-by-step via IoT prompting | VQA | Using visual rationales to enhance LLMs' reasoning accuracy and interpretability |
| CoTDiffusion (ni2024generate: Generate Subgoal Images before Act: Unlocking the Chain-of-Thought Reasoning in Diffusion Model for Robot Manipulation with Multimodal Prompts) | T, I | External Algorithms | Robotics | Generating subgoal images before action to enhance reasoning in long-horizon robot manipulation tasks |
| [T-SciQ](https://arxiv.org/abs/2305.03453) | T, I | Model-Intrinsic Capabilities | ScienceQA | Using LLM-generated reasoning signals to teach multimodal reasoning for complex science QA |
| [Visual-CoT](https://arxiv.org/abs/2305.02317) | T, I | Model-Intrinsic Capabilities | VQA, DocQA, ChartQA | Using visual-text pairs as reasoning carriers to bridge logical gaps in sequential data |
| [VoCoT](https://arxiv.org/abs/2405.16919) | T, I | Model-Intrinsic Capabilities | VQA | Using visually-grounded object-centric reasoning paths for multi-step reasoning |
| [MVoT](https://arxiv.org/abs/2501.07542) | T, I | Model-Intrinsic Capabilities | Spatial Reasoning | Using multimodal reasoning with image visualizations to enhance complex spatial reasoning in LMMs |

## Approach of MM-o1

| **Name** | **Backbone** | **Dataset** | **Modality** | **Reasoning Paradigm** | **Task Type** | **Highlight** |
|----------|--------------|-------------|--------------|------------------------|---------------|---------------|
| [Macro-O1](https://doi.org/10.48550/arXiv.2411.14405) | Qwen2-7B-Instruct | Open-O1 CoT + Marco-o1 CoT + Marco-o1 Instruction | T | MCTS-guided Thinking | Math, Translate | MCTS for solution expansion and reasoning action strategy |
| [llamaberry](https://arxiv.org/abs/2410.02884) | LLaMA-3.1-8B | PRM800K + OpenMathInstruct-1 | T | MCTS-guided Thinking | Math | SR-MCTS for search and PPRM for evaluation |
| [LLaVA-CoT](https://arxiv.org/abs/2411.10440) | Llama-3.2V-11B-cot | LLaVA-CoT-100k | T, I | Summary, Caption, Thinking | Science, General | Introduce LLaVA-CoT-100k and scalable beam search |
| [LlamaV-o1](https://arxiv.org/abs/2501.06186) | Llama-3.2V-11B-cot | LLaVA-CoT-100k + PixMo | T, I | Summary, Caption, Thinking | Science, General | Introduce VCR-Bench and outperforms |
| [Mulberry](https://arxiv.org/abs/2412.18319) | Llama-3.2V-11B-cot, LLaVA-Next-8B, Qwen2-VL-7B | Mulberry-260K | T, I | Caption, Rationales, Thinking | Math, General | Introduce Mulberry-260k and CoMCTS for collective learning |
| [RedStar-Geo](https://arxiv.org/abs/2501.11284) | InternVL2-8B | GeoQA | T, I | Long-Thinking | Math | Competitive with minimal Long-CoT data |

## Approach of MM-R1

| **Approach** | **Backbone** | **Dataset** | **RL Algorithm** | **Modality** | **Task Type** | **RL Framework** | **Cold Start** | **Rule-base/RM** |
|--------------|--------------|-------------|------------------|--------------|---------------|------------------|----------------|------------------|
| RLHF-V (yu2024rlhf: Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback) | LLaVA-13B | RLHF-V-Dataset(1.4k) | DPO | T, I | VQA | Muffin | - | (unknown) |
| InternVL2.[5](https://doi.org/10.48550/arXiv.2411.10442) | InternVL | MMPR(3m) | MPO(DPO) | T, I | VQA | - | - | (unknown) |
| [Insight-V](https://arxiv.org/abs/2411.14432) | LLaMA3-LLaVA-Next | - | DPO | T, I | VQA | trl | - | (unknown) |
| [LLaVA-Reasoner-DPO](https://doi.org/10.48550/arXiv.2410.16198) | LLaMA3-LLaVA-Next | ShareGPT4o-reasoning-dpo(6.6k) | DPO | T, I | VQA | trl | - | (unknown) |
| [VLM-R1](https://arxiv.org/abs/2504.07615) | Qwen2.5-VL | coco , LISA , Refcoco | GRPO | T, I | Grounding ,Math , Open-Vocabulary Detection | trl | No | Rule-base |
| R1-V (chen2025r1v: R1-V: Reinforcing Super Generalization Ability in Vision-Language Models with Less Than $3) | Qwen2-VL | CLEVR  , GEOQA | GRPO | T, I | Counting , Math | trl | No | Rule-base |
| [MM-EUREKA](https://github.com/ModalMinds/MM-EUREKA) | InternVL2.5 | K12 , MMPR | RLOO | T, I | Math | OpenRLHF | Yes | Rule-base |
| [MM-EUREKA-Qwen](https://github.com/ModalMinds/MM-EUREKA) | Qwen2.5-VL | K12 , MMPR | GRPO | T, I | Math | OpenRLHF | No | Rule-base |
| [Video-R1](https://arxiv.org/abs/2503.21776) | Qwen2.5-VL | Video-R1(260K) | GRPO | T, I, V | Video VQA | trl | Yes | Rule-base |
| LMM-R1 (peng2025lmmr1: LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL) | Qwen2.5-VL | VerMulti | PPO | T, I | Math | OpenRLHF | No | RM |
| [Vision-R1](https://arxiv.org/abs/2503.06749) | Qwen2.5-VL | LLaVA-CoT , Mulberry | GRPO | T, I | Math | - | Yes | Rule-base |
| [Visual-RFT](https://arxiv.org/abs/2503.01785) | Qwen2-VL | coco , LISA , ... | GRPO | T, I | Detection , Classification | trl | No | Rule-base |
| [R1-OneVision](https://arxiv.org/abs/2503.10615) | Qwen2.5-VL | R1-Onevision-Dataset | GRPO | T, I | Math , Science , General , Doc | - | Yes | Rule-base |
| [Seg-Zero](https://arxiv.org/abs/2503.06520) | Qwen2.5-VL , SAM2 | RefCOCOg , ReasonSeg | GRPO | T, I | Grounding | verl | No | Rule-base |
| [VisualThinker-R1-Zero](https://arxiv.org/abs/2503.05132) | Qwen2-VL | SAT dataset | GRPO | T, I | Spatial Reasoning | trl | No | Rule-base |
| R1-Omni (zhao2025r1omni: R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning) | HumanOmni | MAFW , DFEW | GRPO | T, I, A, V | emotion recognition | trl | Yes | Rule-base |
| [OThink-MR1](https://arxiv.org/abs/2503.16081) | Qwen2.5-VL | CLEVR , GEOQA | GRPO | T, I | Counting , Math | - | No | Rule-base |
| [Multimodal-Open-R1](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) | Qwen2-VL | multimodal-open-r1-8k-verified(based on Math360K and Geo170K) | GRPO | T,I | Math | trl | No | Rule-base |
| [Curr-ReFT](https://arxiv.org/abs/2503.07065) | Qwen2.5-VL | RefCOCOg , Math360K , Geo170K | GRPO | T,I | Detection , Classification , Math | Curr-RL | No | RM |
| Open-R1-Video (wang-2025-open-r1-video: Open-R1-Video) | Qwen2-VL | open-r1-video-4k | GRPO | T, I, V | Video VQA | trl | No | Rule-base |
| [VisRL](https://arxiv.org/abs/2503.07523) | Qwen2.5-VL | VisCoT | DPO | T,I | VQA | trl | Yes | RM |
| [R1-VL](https://arxiv.org/abs/2503.12937) | Qwen2-VL | Mulberry-260k | StepGRPO | T,I | Math , ChartQA | not release | No | Rule-base |

## Multimodal Understanding

### Visual Centric

| **Benchmark** | **Dataset** |
|---------------|-------------|
| [VQA](http://arxiv.org/abs/1610.01465), [GQA](http://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html) | [ALIGN](https://arxiv.org/abs/2102.05918), [LTIP](https://arxiv.org/abs/2410.05249) |
| [DocVQA](https://doi.org/10.1109/WACV48630.2021.00225), [TextVQA](http://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html) | [YFCC100M](http://dx.doi.org/10.1145/2812802), [DocVQA](https://doi.org/10.1109/WACV48630.2021.00225) |
| [OCR-VQA](https://doi.org/10.1109/ICDAR.2019.00156), [CMMLU](https://doi.org/10.18653/v1/2024.findings-acl.671) | Visual [Genome](https://arxiv.org/abs/1602.07332), [Wukong](https://arxiv.org/abs/2202.06767) |
| [C-Eval](http://papers.nips.cc/paper_files/paper/2023/hash/c6ec1844bec96d6d32ae95ae694e23d8-Abstract-Datasets_and_Benchmarks.html), [MTVQA](https://doi.org/10.48550/arXiv.2405.11985) | [CC3M](https://arxiv.org/abs/2102.08981), [ActivityNet-QA](https://doi.org/10.1609/aaai.v33i01.33019127) |
| [Perception-Test](https://doi.org/10.48550/arXiv.2411.19941), [Video-MMMU](https://doi.org/10.48550/arXiv.2501.13826) | [SBU](https://proceedings.neurips.cc/paper/2011/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html), [AI2D](https://doi.org/10.1007/s10579-020-09517-1) |
| [Video-MME](https://doi.org/10.48550/arXiv.2405.21075), [MMBench](https://doi.org/10.1007/978-3-031-72658-3_13) | [LAION-5B](https://arxiv.org/abs/2210.08402), [LAION-400M](https://arxiv.org/abs/2111.02114) |
| [Seed-Bench](https://doi.org/10.48550/arXiv.2307.16125), [MME-RealWorld](https://doi.org/10.48550/arXiv.2408.13257) | [MS-COCO](https://doi.org/10.1007/978-3-319-10602-1_48), [Virpt](https://arxiv.org/abs/2406.06040) |
| [MMMU](https://doi.org/10.1109/CVPR52733.2024.00913), [MM-Vet](https://openreview.net/forum?id=KOTutrSR2y) | [OpenVid-1M](https://doi.org/10.48550/arXiv.2407.02371), [VidGen-1M](https://arxiv.org/abs/2408.02629) |
| [MMT-Bench](https://openreview.net/forum?id=R4Ng8zYaiz), [Hallu-PI](https://doi.org/10.1145/3664647.3681251) | [Flickr30k](https://doi.org/10.1007/s11263-016-0965-7), [COYO-700M](https://arxiv.org/abs/2305.15248) |
| [ColorBench](https://arxiv.org/abs/2504.10514), [DVQA](http://openaccess.thecvf.com/content_cvpr_2018/html/Kafle_DVQA_Understanding_Data_CVPR_2018_paper.html) | [WebVid](https://arxiv.org/abs/2104.00650), [Youku-mPLUG](https://arxiv.org/abs/2306.04362) |
| [MMStar](http://papers.nips.cc/paper_files/paper/2024/hash/2f8ee6a3d766b426d2618e555b5aeb39-Abstract-Conference.html) | [VideoCC3M](https://arxiv.org/abs/2204.00679), [FILIP](https://arxiv.org/abs/2111.07783) |
| | [CLIP](http://proceedings.mlr.press/v139/radford21a.html), [YouTube8M](https://arxiv.org/abs/1609.08675) |

### Audio Centric

| **Benchmark** | **Dataset** |
|---------------|-------------|
| [AudioBench](https://doi.org/10.48550/arXiv.2406.16020), [VoiceBench](https://doi.org/10.48550/arXiv.2410.17196) | [Librispeech](https://doi.org/10.1109/ICASSP.2015.7178964), Common [Voice](https://aclanthology.org/2020.lrec-1.520/) |
| [Fleurs](https://doi.org/10.1109/SLT54892.2023.10023141), [MusicBench](https://doi.org/10.18653/v1/2024.naacl-long.459) | [Aishell](https://doi.org/10.1109/ICSDA.2017.8384449), [Fleurs](https://doi.org/10.1109/SLT54892.2023.10023141), [MELD](https://doi.org/10.18653/v1/p19-1050) |
| [Air-Bench](https://doi.org/10.18653/v1/2024.acl-long.109), [MMAU](https://doi.org/10.48550/arXiv.2410.19168) | [CoVoST2](https://arxiv.org/abs/2007.10310), [SIFT-50M](https://arxiv.org/abs/2504.09081) |
| [SD-eval](http://papers.nips.cc/paper_files/paper/2024/hash/681fe4ec554beabdc9c84a1780cd5a8a-Abstract-Datasets_and_Benchmarks_Track.html), [CoVoST2](https://arxiv.org/abs/2007.10310) | [Clotho](https://doi.org/10.1109/ICASSP40776.2020.9052990), [AudioCaps](https://doi.org/10.18653/v1/n19-1011) |
| [MusicNet](https://openreview.net/forum?id=rkFBJv9gg) | [ClothoAQA](https://ieeexplore.ieee.org/document/9909680), [MusicNet](https://openreview.net/forum?id=rkFBJv9gg) |
| | [NSynth](http://proceedings.mlr.press/v70/engel17a.html), [MusicCaps](https://doi.org/10.48550/arXiv.2301.11325) |

## Multimodal Generation

### Cross-modal Generation

| **Benchmark** | **Dataset** |
|---------------|-------------|
| [GenEval](http://papers.nips.cc/paper_files/paper/2023/hash/a3bf71c7c63f0c3bcb7ff67c67b1e7b1-Abstract-Datasets_and_Benchmarks.html), T2I-CompBench++ \citep{huang2025t2icompbenchenhancedcomprehensivebenchmark} | [MS-COCO](https://doi.org/10.1007/978-3-319-10602-1_48), [Flickr30k](https://doi.org/10.1007/s11263-016-0965-7) |
| [DPG-Bench](https://doi.org/10.48550/arXiv.2403.05135), [GenAI-Bench](https://doi.org/10.48550/arXiv.2406.13743) | Conceptual [Captions](https://aclanthology.org/P18-1238/), [RedCaps](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/e00da03b685a0dd18fb6a08af0923de0-Abstract-round1.html) |
| [VBench](https://doi.org/10.1109/CVPR52733.2024.02060), [VideoScore](https://aclanthology.org/2024.emnlp-main.127) | [CommonPool](http://papers.nips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html), [LLaVA-Pretrain](http://papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html) |
| [WorldSimBench](https://doi.org/10.48550/arXiv.2410.18072), [WorldModelBench](https://doi.org/10.48550/arXiv.2502.20694) | [Aishell1](https://doi.org/10.1109/ICSDA.2017.8384449), [ThreeDWorld](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/735b90b4568125ed6c3f678819b6e058-Abstract-round1.html) |
| [MagicBrush](http://papers.nips.cc/paper_files/paper/2023/hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_Benchmarks.html), VBench++~\citep{DBLP:journals/corr/abs-2411-13503} | [X2I](https://arxiv.org/abs/2409.11340), [GAIA-1](https://doi.org/10.48550/arXiv.2309.17080) |
| MJHQ-30K (li2024playground: Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation), VBench 2.[0](https://arxiv.org/abs/2503.21755) | [UniSim](https://arxiv.org/abs/2310.06114), [VidProM](http://papers.nips.cc/paper_files/paper/2024/hash/78b3e7836e3b7dea79d809b0c99cb097-Abstract-Datasets_and_Benchmarks_Track.html) |
| [AIGCBench](http://openaccess.thecvf.com/content_CVPR_2019/html/Fan_Heterogeneous_Memory_Enhanced_Multimodal_Attention_Model_for_Video_Question_Answering_CVPR_2019_paper.html), [EvalCrafter](https://doi.org/10.1109/CVPR52733.2024.02090) | [LWM](https://doi.org/10.48550/arXiv.2402.08268), Genesis (authors2024genesis: Genesis: A universal and generative physics engine for robotics and beyond) |
| | [HQ-Edit](https://doi.org/10.48550/arXiv.2404.09990), [InstructPix2Pix](https://doi.org/10.1109/CVPR52729.2023.01764) |
| | [MagicBrush](http://papers.nips.cc/paper_files/paper/2023/hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_Benchmarks.html) |

### Joint Multimodal Generation

| **Benchmark** | **Dataset** |
|---------------|-------------|
| [MM-Interleaved](https://arxiv.org/abs/2401.10208), [ANOLE](https://arxiv.org/abs/2407.06135) | [DreamLLM](https://arxiv.org/abs/2309.11499), [SEED-Story](https://arxiv.org/abs/2407.08683) |
| InterleavedEval (liu2024holistic: Holistic Evaluation for Interleaved Text-and-Image Generation), [OpenLEAF](https://doi.org/10.1145/3664647.3685511) | NextGPT (wu24next: NExT-GPT: Any-to-Any Multimodal LLM), [DreamFactory](https://arxiv.org/abs/2408.11788) |
| OpenING (zhou2024GATE: GATE OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation), [M2RAG](https://arxiv.org/abs/2411.16365) | [DreamRunner](https://arxiv.org/abs/2411.16657), [EVA](https://arxiv.org/abs/2410.15461) |
| [SEED-Bench](https://doi.org/10.48550/arXiv.2307.16125), [SEED-Bench-2](https://arxiv.org/abs/2311.17092) | |

## Multimodal Reasoning

### General Visual Reasoning

| **Benchmark** | **Dataset** |
|---------------|-------------|
| [NaturalBench](http://papers.nips.cc/paper_files/paper/2024/hash/1e69ff56d0ebff0752ff29caaddc25dd-Abstract-Datasets_and_Benchmarks_Track.html), [VCR](http://openaccess.thecvf.com/content_CVPR_2019/html/Zellers_From_Recognition_to_Cognition_Visual_Commonsense_Reasoning_CVPR_2019_paper.html) | [VCR](http://openaccess.thecvf.com/content_CVPR_2019/html/Zellers_From_Recognition_to_Cognition_Visual_Commonsense_Reasoning_CVPR_2019_paper.html), [TDIUC](https://arxiv.org/abs/1703.09684) |
| [PhysBench](https://doi.org/10.48550/arXiv.2501.16411), [MMBench](https://doi.org/10.1007/978-3-031-72658-3_13) | [MMPR](https://doi.org/10.48550/arXiv.2411.10442), [ChartQA](https://doi.org/10.18653/v1/2022.findings-acl.177) |
| [MMMU](https://doi.org/10.1109/CVPR52733.2024.00913), [AGIEval](https://doi.org/10.18653/v1/2024.findings-naacl.149) | [SWAG](https://arxiv.org/abs/1808.05326), [LLaVA-CoT](https://doi.org/10.48550/arXiv.2411.10440) |
| [MMStar](http://papers.nips.cc/paper_files/paper/2024/hash/2f8ee6a3d766b426d2618e555b5aeb39-Abstract-Conference.html), [InfographicVQA](https://doi.org/10.1109/WACV51458.2022.00264) | [CLEVR](https://arxiv.org/abs/1612.06890), [Mulberry-260K](https://arxiv.org/abs/2412.18319) |
| [VCRBench](https://arxiv.org/abs/2504.07956), [VisualPuzzles](https://arxiv.org/abs/2504.10342) | [ShareGPT4oReasoning](https://doi.org/10.48550/arXiv.2410.16198), [R1-Onevision](https://arxiv.org/abs/2503.10615) |
| | [Video-R1-data](https://arxiv.org/abs/2503.21776), [Visual-CoT](https://arxiv.org/abs/2403.16999) |

### Domain-specific Reasoning

| **Benchmark** | **Dataset** |
|---------------|-------------|
| [MathVista](https://openreview.net/forum?id=KUNzEQMWU7), [MATH-Vision](http://papers.nips.cc/paper_files/paper/2024/hash/ad0edc7d5fa1a783f063646968b7315b-Abstract-Datasets_and_Benchmarks_Track.html) | [Habitat](https://doi.org/10.1109/ICCV.2019.00943), [AI2-THOR](http://arxiv.org/abs/1712.05474) |
| [VLM-Bench](http://papers.nips.cc/paper_files/paper/2022/hash/04543a88eae2683133c1acbef5a6bf77-Abstract-Datasets_and_Benchmarks.html), [GemBench](https://doi.org/10.48550/arXiv.2410.01345) | [Gibson](http://openaccess.thecvf.com/content_cvpr_2018/html/Xia_Gibson_Env_Real-World_CVPR_2018_paper.html), [GeoQA](https://arxiv.org/abs/2105.14517) |
| [GeoQA](https://arxiv.org/abs/2105.14517), [VIMA-Bench](https://doi.org/10.48550/arXiv.2210.03094) | Isaac [Lab](https://doi.org/10.1109/LRA.2023.3270034), [ProcTHOR](http://papers.nips.cc/paper_files/paper/2022/hash/27c546ab1e4f1d7d638e6a8dfbad9a07-Abstract-Conference.html) |
| [WorldSimBench](https://doi.org/10.48550/arXiv.2410.18072), [WorldModelBench](https://doi.org/10.48550/arXiv.2502.20694) | [CALVIN](https://doi.org/10.1109/LRA.2022.3180108) |
| [ScienceQA](http://papers.nips.cc/paper_files/paper/2022/hash/11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html), ChartQA (\citep{DBLP:conf/acl/MasryLTJH22}) | |
| [MathQA](https://doi.org/10.18653/v1/n19-1245), [Habitat](https://doi.org/10.1109/ICCV.2019.00943) | |
| [AI2-THOR](http://arxiv.org/abs/1712.05474), [Gibson](http://openaccess.thecvf.com/content_cvpr_2018/html/Xia_Gibson_Env_Real-World_CVPR_2018_paper.html) | |
| [iGibson](https://arxiv.org/abs/2108.03272), Isaac [Lab](https://doi.org/10.1109/LRA.2023.3270034) | |

## Multimodal Planning

### GUI Navigation

| **Benchmark** | **Dataset** |
|---------------|-------------|
| [WebArena](https://openreview.net/forum?id=oKn9c6ytLx), [Mind2Web](http://papers.nips.cc/paper_files/paper/2023/hash/5950bf290a1570ea401bf98882128160-Abstract-Datasets_and_Benchmarks.html) | [AMEX](https://doi.org/10.48550/arXiv.2407.17490), RiCo (Deka:2017:Rico: Rico: A Mobile App Dataset for Building Data-Driven Design Applications) |
| [VisualWebBench](https://doi.org/10.48550/arXiv.2404.05955), [OSWorld](http://papers.nips.cc/paper_files/paper/2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets_and_Benchmarks_Track.html) | [WebSRC](https://arxiv.org/abs/2101.09465), [E-ANT](https://doi.org/10.48550/arXiv.2406.14250) |
| [OmniACT](https://doi.org/10.1007/978-3-031-73113-6_10), [VisualAgentBench](https://doi.org/10.48550/arXiv.2408.06327) | [AndroidEnv](https://arxiv.org/abs/2105.13231), [GUI-World](https://doi.org/10.48550/arXiv.2406.10819) |
| [LlamaTouch](https://doi.org/10.1145/3654777.3676382), Windows Agent [Arena](https://doi.org/10.48550/arXiv.2409.08264) | |
| [Ferret-UI](https://doi.org/10.1007/978-3-031-73039-9_14), [WebShop](http://papers.nips.cc/paper_files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference.html) | |
| SWE-BENCH [M](https://arxiv.org/abs/2410.03859), [MineDojo](http://papers.nips.cc/paper_files/paper/2022/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html) | |
| [TeamCraft](https://arxiv.org/abs/2412.05255), [V-MAGE](https://arxiv.org/abs/2504.06148) | |

### Embodied and Simulated Environments

| **Benchmark** | **Dataset** |
|---------------|-------------|
| [MineDojo](http://papers.nips.cc/paper_files/paper/2022/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html), [MuEP](https://www.ijcai.org/proceedings/2024/15) | [MineDojo](http://papers.nips.cc/paper_files/paper/2022/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html), Habitat 3.[0](https://openreview.net/forum?id=4znwzG92CE) |
| [GVCCI](https://doi.org/10.1109/IROS55552.2023.10342021), [BEHAVIOR-1K](https://doi.org/10.48550/arXiv.2403.09227) | [SAPIEN](https://openaccess.thecvf.com/content_CVPR_2020/html/Xiang_SAPIEN_A_SimulAted_Part-Based_Interactive_ENvironment_CVPR_2020_paper.html), [HomeRobot](https://proceedings.mlr.press/v229/yenamandra23a.html) |
| Habitat 3.[0](https://openreview.net/forum?id=4znwzG92CE), [SAPIEN](https://openaccess.thecvf.com/content_CVPR_2020/html/Xiang_SAPIEN_A_SimulAted_Part-Based_Interactive_ENvironment_CVPR_2020_paper.html) | [HoloAssist](https://doi.org/10.1109/ICCV51070.2023.01854), [DrivingDojo](https://doi.org/10.48550/arXiv.2207.11432) |
| [HomeRobot](https://proceedings.mlr.press/v229/yenamandra23a.html), [HoloAssist](https://doi.org/10.1109/ICCV51070.2023.01854) | [OmmiHD-Scenes](https://arxiv.org/abs/2412.10734) |
| [DrivingDojo](https://doi.org/10.48550/arXiv.2207.11432), [WolfBench](https://arxiv.org/abs/2410.07869) | |
