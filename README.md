# Perception, <span style="color:purple">R</span>eason, <span style="color:purple">T</span>hink, and <span style="color:purple">P</span>lan:
# A Survey on Large Multimodal Reasoning Models


## The classific works of the initial stage of perception-driven multimodal reasoning, where VLMs and MLLMs play a significant role to advance the performance of multimodal reasoning tasks.

### Neural Modular Reasoning Networks

| **Model** | **Year** | **Architecture** | **Highlight** | **Training Method** |
|-----------|----------|------------------|---------------|---------------------|
| NMN~\citep{andreas2016neural} | 2016 | Modular | Dynamically assembles task-specific modules for visual-textual reasoning. | Supervised learning |
| HieCoAtt~\citep{lu2016hierarchical} | 2016 | Attention-based | Aligns question semantics with image regions via hierarchical cross-modal attention. | Supervised learning |
| MCB~\citep{fukui2016multimodal} | 2016 | Bilinear | Optimizes cross-modal feature interactions with efficient bilinear modules. | Supervised learning |
| SANs~\citep{DBLP:conf/cvpr/YangHGDS16} | 2016 | Attention-based | Iteratively refines reasoning through multiple attention hops over visual features. | Supervised learning |
| DMN~\citep{xiong2016dynamic} | 2016 | Memory-based | Integrates memory modules for multi-episode reasoning over sequential inputs. | Supervised learning |
| ReasonNet~\citep{DBLP:conf/nips/IlievskiF17} | 2017 | Modular | Decomposes reasoning into entity-relation modules for structured inference. | Supervised learning |
| UpDn~\citep{anderson2018bottom} | 2018 | Attention-based | Combines bottom-up and top-down attention for object-level reasoning. | Supervised learning |
| MAC~\citep{hudson2018compositional} | 2018 | Memory-based | Uses a memory-augmented control unit for iterative compositional reasoning. | Supervised learning |
| BAN~\citep{kim2018bilinear} | 2018 | Bilinear | Captures high-order interactions via bilinear attention across modalities. | Supervised learning |
| HeteroMemory~\citep{DBLP:conf/cvpr/FanZZW0H19} | 2019 | Memory-based | Synchronizes appearance and motion modules for video-based temporal reasoning. | Supervised learning |
| MuRel~\citep{cadene2019murel} | 2019 | Relational | Models reasoning as a relational network over object pairs for fine-grained inference. | Supervised learning |
| MCAN~\citep{yu2019deep} | 2019 | Attention-based | Employs modular co-attention with self- and guided-attention for deep reasoning. | Supervised learning |

### VLMs-based Modular Reasoning

| **Model** | **Year** | **Architecture** | **Highlight** | **Training Method** |
|-----------|----------|------------------|---------------|---------------------|
| ViLBERT~\citep{lu2019vilbert} | 2019 | Dual-Encoder | Aligns visual-text features via dual-stream Transformers with cross-modal attention. | Pretraining + fine-tuning |
| LXMERT~\citep{tan2019lxmert} | 2019 | Dual-Encoder | Enhances cross-modal reasoning with dual-stream pretraining on diverse tasks. | Pretraining + fine-tuning |
| X-LXMERT~\citep{tan2019lxmert} | 2020 | Dual-Encoder | Extends dual-stream reasoning with generative cross-modal pretraining. | Pretraining + fine-tuning |
| ALBEF~\citep{li2021align} | 2021 | Dual-Encoder | Integrates contrastive learning with momentum distillation for robust reasoning. | Contrastive + generative pretraining |
| SimVLM~\citep{wang2021simvlm} | 2021 | Dual-Encoder | Uses prefix-based pretraining for flexible cross-modal reasoning. | Pretraining + fine-tuning |
| VLMo~\citep{bao2022vlmo} | 2022 | Dual-Encoder | Employs a mixture-of-modality-experts for dynamic cross-modal reasoning. | Pretraining + fine-tuning |
| METER~\citep{dou2022empirical} | 2022 | Dual-Encoder | Enhances reasoning with a modular encoder-decoder for robust alignment. | Pretraining + fine-tuning |
| BLIP~\citep{li2022blip} | 2022 | Dual-Encoder | Bootstraps alignment with contrastive learning for efficient reasoning. | Contrastive + generative pretraining |
| VisualBERT~\citep{li2019visualbert} | 2019 | Single-Transformer-Backbone | Fuses visual-text inputs in a single Transformer for joint contextual reasoning. | Pretraining + fine-tuning |
| VL-BERT~\citep{su2019vl} | 2019 | Single-Transformer-Backbone | Enhances cross-modal reasoning with unified visual-language pretraining. | Pretraining + fine-tuning |
| UNITER~\citep{DBLP:conf/eccv/ChenLYK0G0020} | 2020 | Single-Transformer-Backbone | Reasons via joint contextual encoding in a single Transformer backbone. | Pretraining + fine-tuning |
| PixelBERT~\citep{huang2020pixel} | 2020 | Single-Transformer-Backbone | Processes pixels with CNN+Transformer for fine-grained cross-modal reasoning. | Pretraining + fine-tuning |
| UniVL~\citep{luo2020univl} | 2020 | Single-Transformer-Backbone | Unifies video-language reasoning with a single Transformer for temporal tasks. | Pretraining + fine-tuning |
| Oscar~\citep{DBLP:conf/eccv/Li0LZHZWH0WCG20} | 2020 | Single-Transformer-Backbone | Anchors reasoning with object tags in a unified Transformer for semantic inference. | Pretraining + fine-tuning |
| VinVL~\citep{zhang2021vinvl} | 2021 | Single-Transformer-Backbone | Boosts reasoning with enhanced visual features in a single Transformer. | Pretraining + fine-tuning |
| ERNIE-ViL~\citep{yu2021ernie} | 2021 | Single-Transformer-Backbone | Integrates scene graph knowledge for structured visual-language reasoning. | Pretraining + fine-tuning |
| UniT~\citep{hu2021unit} | 2021 | Single-Transformer-Backbone | Streamlines multimodal tasks with a shared self-attention Transformer backbone. | Pretraining + fine-tuning |
| Flamingo~\citep{alayrac2022flamingo} | 2022 | Single-Transformer-Backbone | Prioritizes dynamic vision-text interactions via cross-attention. | Pretraining + fine-tuning |
| CoCa~\citep{DBLP:journals/tmlr/YuWVYSW22} | 2022 | Single-Transformer-Backbone | Combines contrastive and generative heads for versatile cross-modal reasoning. | Contrastive + generative pretraining |
| BEiT-3~\citep{wang2022image} | 2022 | Single-Transformer-Backbone | Unifies vision-language learning with masked data modeling. | Pretraining + fine-tuning |
| OFA~\citep{wang2022ofa} | 2022 | Single-Transformer-Backbone | Provides a unified multimodal framework for efficient cross-modal reasoning. | Pretraining + fine-tuning |
| PaLI~\citep{chen2022pali} | 2022 | Single-Transformer-Backbone | Scales reasoning with a multilingual single-Transformer framework. | Pretraining + fine-tuning |
| BLIP-2~\citep{DBLP:conf/icml/0008LSH23} | 2023 | Single-Transformer-Backbone | Uses a querying Transformer for improved cross-modal reasoning efficiency. | Pretraining + fine-tuning |
| Kosmos-1~\citep{DBLP:conf/icml/0008LSH23} | 2023 | Single-Transformer-Backbone | Enables interleaved input processing for flexible multimodal understanding. | Pretraining + fine-tuning |
| Kosmos-2~\citep{DBLP:conf/icml/0008LSH23} | 2023 | Single-Transformer-Backbone | Enhances grounding capability for precise object localization and reasoning. | Pretraining + fine-tuning |
| CLIP-Cap~\citep{mokady2021clipcap} | 2021 | Vision-Encoder-LLM | Projects CLIP visual features into an LLM for reasoning and captioning. | Fine-tuning |
| LLaVA~\citep{DBLP:conf/nips/LiuLWL23a} | 2023 | Vision-Encoder-LLM | Tunes ViT-LLM integration for conversational multimodal reasoning. | Instruction tuning |
| MiniGPT-4~\citep{zhu2023minigpt} | 2023 | Vision-Encoder-LLM | Aligns ViT to a frozen LLM via projection for streamlined reasoning. | Fine-tuning |
| InstructBLIP~\citep{dai2023instructblip} | 2023 | Vision-Encoder-LLM | Uses instruction tuning to align ViT with LLM for multimodal reasoning. | Instruction tuning |
| Qwen-VL~\citep{bai2023qwen} | 2023 | Vision-Encoder-LLM | Incorporates spatial-aware ViT for enhanced grounded reasoning. | Pretraining + fine-tuning |
| mPLUG-Owl~\citep{ye2023mplug} | 2023 | Vision-Encoder-LLM | Integrates modular visual encoder with LLM for instruction-following reasoning. | Instruction tuning |
| Otter~\citep{li2023otter} | 2023 | Vision-Encoder-LLM | Combines modular visual encoder with LLM for in-context multimodal reasoning. | Instruction tuning |

## The Structural Reasoning

| Name | Modality | Task | Reasoning Structure | Datasets | Highlight |
|------|----------|------|---------------------|----------|-----------|
| Cantor~\citeyearpar{gao2024cantor} | T,I | VQA | Perception, Decision | - | Decouples perception and reasoning via feature extraction and CoT-style integration. |
| TextCoT\citeyearpar{luan2024textcot} | T,I | VQA | Caption, Localization, Precise observation | - | First summarizes visual context, then generates CoT-based responses. |
| Grounding-Prompter\citeyearpar{chen2023grounding} | T,V,A | Temporal Sentence Grounding | Denoising | VidChapters-7M | Grounding-Prompter performs global parsing, denoising, partitioning before reasoning. |
| Audio-CoT\citeyearpar{ma2025audio_cot} | T,A | AQA | Manual-CoT, Zero-Shot-CoT, Desp-CoT | - | Enhances visual reasoning by utilizing three chain-of-thought paradigms. |
| VIC\citeyearpar{zheng2024thinking} | I,T | VQA | Thinking before looking | - | Breaks tasks into text-based sub-steps before integrating visual inputs to form final rationales. |
| Visual Sketchpad\citeyearpar{hu2024visual} | I,T | VQA, math QA | Sketch-based reasoning paradigm | - | Organizes rationales into "Thought, Action, Observation" phases. |
| Det-CoT\citeyearpar{wu2024dettoolchain} | I,T | VQA | Subtask decomposition, Execution, and Verification | - | Formalizes VQA reasoning as a combination of subtasks and reviews. |
| BDoG\citeyearpar{zheng2024picture} | I,T | VQA | Entity update, Relation update, Graph pruning | - | Utilizes a dedicated debate-summarization pipeline with specialized agents. |
| CoTDet\citeyearpar{tang2023cotdet} | I,T | object detection | Object listing, Affordance analysis, Visual feature summarization | COCO-Tasks | Achieves object detection via human-like procedure of listing, analyzing and summarizing. |
| CoCoT\citeyearpar{zhang2024cocot} | I,T | VQA | Contrastive prompting strategy | - | Systematically contrasts input similarities and differences. |
| SegPref\citeyearpar{wang2024avs_cot} | T,A,V | Temporal Sentence Grounding | Visual summary, Sound filtering, Denoising | Youtube-8M, Semantic-ADE20K | Robustly localizes sounding objects in the visual space through global understanding, sounding object filtering, and noise removal. |
| EMMAX\citeyearpar{sun2024emmax} | I,T | Robotic task | Grounded CoT reasoning, Look-ahead spatial reasoning | Dataset based on BridgeV2 | Integrates grounded planning and predictive. |
| DDCoT \citeyearpar{zheng2023ddcot} | T,I | VQA | Question Deconstruct, Rationale | ScienceQA | Maintains a critical attitude by identifying reasoning and recognition responsibilities through the combined effect of negative-space design and visual deconstruction. |
| AVQA-CoT \citeyearpar{li2024avqa_cot} | T,A,V | AVQA | Question Deconstruct, Question Selection, Rationale | MUSIC-AVQA | Decomposes complex questions into multiple simpler sub-questions and leverages LLMs to select relevant sub-questions for audio-visual question answering. |
| CoT-PT \citeyearpar{ge2023chain} | T,I | Image Classification, Image-Text Retrieval, VQA | Coarse-to-Fine Image Concept Representation | ImageNet | First to successfully adapt CoT for prompt tuning by combining visual and textual embeddings in the vision domain. |
| IoT~\citeyearpar{zhou2024image} | T,I | VQA | Visual Action Selection, Execution, Rationale, Summary, Self-Refine | - | Enhances visual reasoning by integrating visual and textual rationales through a model-driven multimodal reasoning chain. |
| Shikra ~\citeyearpar{chen2023shikra} | T,I | VQA, PointQA | Caption, Object Grounding | ScienceQA | Maintains a critical attitude by identifying reasoning and recognition responsibilities through the combined effect of negative-space design and visual deconstruction. |
| E-CoT ~\citeyearpar{zawalski2024robotic} | T,I,A | Policies' Generalization | Task Rephrase, Planning, Task Deconstruct, Object Grounding | Bidgedata v2 | Integrates semantic planning with low-level perceptual and motor reasoning, advancing task formulations in embodied intelligence. |
| CoS ~\citeyearpar{liu2024chain} | T,I | VQA | Object Grounding, Rationale | Llava665K | Guides the model to identify and focus on key image regions relevant to a question, enabling multi-granularity understanding without compromising resolution. |
| TextCoT ~\citeyearpar{luan2024textcot} | T,I | VQA | Caption, Object Grounding, Image Zoom | Llava665K, SharedGPT4V | Enables accurate and interpretable multimodal question answering through staged processing: overview, coarse localization, and fine-grained observation. |
| DCoT ~\citeyearpar{jia2024dcot} | T,I | VQA | Object Grounding, Fine-Grained Image Generation, Similar Example Retrieve, Rationale | - | Uses a dual-guidance mechanism by combining bounding box cues to focus attention on relevant image regions and retrieving the most suitable examples from a curated demonstration cluster as contextual support. |



## Multimodal Defined Reasoning

| Name | Modality | Task | Reasoning Structure | Datasets | Highlight |
|------|----------|------|---------------------|----------|-----------|
| Cantor~\citeyearpar{gao2024cantor} | I,T | VQA | perception, decision | - | Decouples perception and reasoning via feature extraction and CoT-style integration |
| TextCoT\citeyearpar{luan2024textcot} | I,T | VQA | caption, localization, precise observation | - | first summarizes visual context, then generates CoT-based responses |
| Grounding-Prompter\citeyearpar{chen2023grounding} | V,A,T | Temporal Sentence Grounding | Denoising | VidChapters-7M | Grounding-Prompter performs global parsing, denoising, partitioning before reasoning |


## Multimodal Structural Reasoning

| Name | Modality | Task | Reasoning Structure | Training Datasets | Highlight |
|------|----------|------|---------------------|-------------------|-----------|
| DDCoT \citeyearpar{zheng2023ddcot} | T,I | VQA | Question Deconstruct,Rationale | ScienceQA | Maintains a critical attitude by identifying reasoning and recognition responsibilities through the combined effect of negative-space design and visual deconstruction. |
| AVQA-CoT \citeyearpar{li2024avqa_cot} | T,A,V | AVQA | Question Deconstruct, Question Selection, Rationale | MUSIC-AVQA | Decomposes complex questions into multiple simpler sub-questions and leverages LLMs to select relevant sub-questions for audio-visual question answering. |
| CoT-PT \citeyearpar{ge2023chain} | T,I | Image Classification, Image-Text Retrieval, VQA | Coarse-to-Fine Image Concept Representation | ImageNet | First to successfully adapt CoT for prompt tuning by combining visual and textual embeddings in the vision domain. |
| IoT \citeyearpar{zhou2024image} | T,I | VQA | Visual Action Selection, Execution, Rationale, Summary, Self-Refine | - | Enhances visual reasoning by integrating visual and textual rationales through a model-driven multimodal reasoning chain. |
| Shikra \citeyearpar{chen2023shikra} | T,I | VQA, PointQA | Caption, Object Grounding | ScienceQA | Maintains a critical attitude by identifying reasoning and recognition responsibilities through the combined effect of negative-space design and visual deconstruction. |
| E-CoT \citeyearpar{zawalski2024robotic} | T,I,A | Policies' Generalization | Task Rephrase, Planning, Task Deconstruct, Object Grounding | Bidgedata v2 | Integrates semantic planning with low-level perceptual and motor reasoning, advancing task formulations in embodied intelligence. |
| CoS \citeyearpar{liu2024chain} | T,I | VQA | Object Grounding, Rationale | Llava665K | Guides the model to identify and focus on key image regions relevant to a question, enabling multi-granularity understanding without compromising resolution. |
| TextCoT \citeyearpar{luan2024textcot} | T,I | VQA | Caption, Object Grounding, Image Zoom | Llava665K, SharedGPT4V | Enables accurate and interpretable multimodal question answering through staged processing: overview, coarse localization, and fine-grained observation. |
| DCoT \citeyearpar{jia2024dcot} | T,I | VQA | Object Grounding, Fine-Grained Image Generation, Similar Example Retrieve, Rationale | - | Uses a dual-guidance mechanism by combining bounding box cues to focus attention on relevant image regions and retrieving the most suitable examples from a curated demonstration cluster as contextual support. |



## External Enhanced

| Name | Modality | Task | Enhancement Type | External Source | Highlight |
|------|----------|------|------------------|-----------------|-----------|
| MM-ToT ~\citeyearpar{gomez2023mmtot} | T,I | Image Generation | Search Algorithm | DFS,BFS | Applies DFS and BFS to select optimal outputs. |
| HoT ~\citeyearpar{yao2023HoT} | T,I | VQA | Search Algorithm | multi-hop random walks on graph | Generates linked thoughts from multimodal data in a hyperedge. |
| AGoT ~\citeyearpar{yang2024AGoT} | T,I | Text-Image Retrieval, VQA | Search Algorithm | prompt aggregation and prompt flow operations | Builds a graph to aggregate multi-faceted reasoning with visuals. |
| BDoG ~\citeyearpar{zheng2024picture} | T,I | VQA | Search Algorithm | Graph Condensation: Entity update, Relation update, Graph pruning | Effective three-agent debate forms thought graph for multimodal queries. |
| L3GO ~\citeyearpar{yamada2024Co3D_Thoughts} | T,I | 3D Object Generation & Composition | Tools | Blender, ControlNet | Iterative part-based 3D construction through LLM reasoning in a simulation environment. |
| HYDRA ~\citeyearpar{ke2024hydra} | T,I | Knowledge-QA, Visual Grounding | Tools | RL agent controller, Visual Foundation Models | RL agent controls multi-stage visual reasoning through dynamic instruction selection. |
| Det-CoT ~\citeyearpar{wu2024dettoolchain} | T,I | object detection | Tools | Visual Processing Prompts | Visual prompts guide MLLM attention for structured detection reasoning. |
| Chain-of-Image ~\citeyearpar{meng2023chain} | T,I | Geometric, chess & commonsense reasoning | Tools | Chain of Images prompting | Generates intermediate images during reasoning for visual pattern recognition. |
| AnyMAL ~\citeyearpar{moon2024anymal} | T, I, A, V | Cross-modal reasoning, multimodal QA | Tools | Pre-trained alignment module | Efficient integration of diverse modalities; strong reasoning via LLaMA-2 backend. |
| SE-CMRN ~\citeyearpar{zhang2021explicit} | T,I | Visual Commonsense Reasoning | Tools | Syntactic Graph Convolutional Network | Enhances language-guided visual reasoning via syntactic GCN in a dual-branch network. |
| RAGAR ~\citeyearpar{khaliq2024ragar} | T,I | Political Fact-Checking | RAG | DuckDuckGo & SerpAPI | Integrates MLLMs with retrieval-augmented reasoning to verify facts using text and image evidence. |
| Chain-of-action ~\citeyearpar{pan2024chain} | T,I | Info retrieval | RAG | Google Search, ChromaDB | Decomposes questions into reasoning chains with configurable retrieval actions to resolve conflicts between knowledge sources. |
| KAM-CoT ~\citeyearpar{mondal2024kam} | T,I, KG | Educational science reasoning | RAG | ConceptNet knowledge graph | Enhances reasoning by retrieving structured knowledge from graphs and integrating it through two-stage training. |
| AR-MCTS ~\citeyearpar{dong2024progressive_retrieval} | T,I | Multi-step reasoning | RAG | Contriever, CLIP dual-stream | Step-wise retrieval with Monte Carlo Tree Search for verified reasoning. |
| MR-MKG ~\citeyearpar{lee2024multimodal} | T, I | General multimodal reasoning | RAG | RGAT | Enhances multimodal reasoning by integrating information from multimodal knowledge graphs. |
| Reverse-HP ~\citeyearpar{zhu2022multimodal} | T, I | Disease-related reasoning | RAG | reverse hyperplane projection | Utilizes KG embeddings to enhance reasoning for specific diseases with multimodal data. |
| MarT ~\citeyearpar{zhang2022multimodal} | T, I | Analogical reasoning | RAG | Structure-guided relation transfer | Uses structure mapping theory and relation-oriented transfer for analogical reasoning with KG. |
| MCoT-Memory~\citeyearpar{liang2024memory_driven} | T,I | VQA | Multimodal Information Enhancing | LLAVA | Memory framework and scene graph construction for effective long-horizon task planning |
| MGCoT~\citeyearpar{yao2023beyond} | T,I | VQA | Multimodal Embedding Enhancing | ViT-large encoder | Precise visual feature extraction aiding multimodal reasoning |
| CCoT~\citeyearpar{mitra2024compositional} | T,I | VQA | Multimodal Perception Enhancing | Scene Graphs | Utilization of the generated scene graph as an intermediate reasoning step. |
| CVR-LLM~\citeyearpar{li2024enhancing} | T,I | VQA | Multimodal Embedding Enhancing | BLIP2flant5 & BLIP2 multi-embedding | Precise context-aware image descriptions through iterative self-refinement and effective text-multimodal factors integrations |
| TeSO~\citeyearpar{wang2024avs_cot} | T,V,A | Temporal Sentence Grounding (TSG) | Multimodal Information Enhancing | VGGish | Integrates text semantics to mitigate segmentation preference for better audio-visual correlation boosting AVS performance. |
| CAT~\citeyearpar{wang2023caption} | T,I | Image Captioning | Multimodal Perception Enhancing | SAM | Promising pre-trained image caption generators, SAM, and instruction-tuned large language models integration |

## Approaches enhancing multimodal reasoning through textual mechanisms

| Name | Modality | Task | Tool | Purpose of Tool | Training Datasets | Highlight |
|------|----------|------|------|-----------------|-------------------|-----------|
| L3GO ~\citep{yamada2024Co3D_Thoughts} | T,I | 3D object generation & composition | Blender, ControlNet | Part-based 3D construction | - | Iterative part-based 3D construction through LLM reasoning in a simulation environment. |
| HYDRA ~\citep{ke2024hydra} | T,I | Knowledge-QA, visual grounding | RL agent controller, Visual Foundation Models | Agent scheduling | RL agent with specific rewards | RL agent controls multi-stage visual reasoning through dynamic instruction selection. |
| Det-CoT ~\citep{wu2024dettoolchain} | T,I | object detection | Visual Processing Prompts | Visual attention guidance | - | Visual prompts guide MLLM attention for structured detection reasoning. |
| Chain-of-Image ~\citep{meng2023chain} | T,I | Geometric, chess & commonsense reasoning | Chain of Images prompting | visual pattern recognition | Geometric & Chess datasets | Generates intermediate images during reasoning for visual pattern recognition. |
| AnyMAL ~\citep{moon2024anymal} | T, I, A, V | Cross-modal reasoning, multimodal QA | Pre-trained alignment module | diverse signals to text representations | Manual instruction set | Efficient integration of diverse modalities; strong reasoning via LLaMA-2 backend. |
| SE-CMRN ~\citep{zhang2021explicit} | T,I | Visual Commonsense Reasoning | Syntactic Graph Convolutional Network | Enhance visual reasoning | VCR dataset | Enhances language-guided visual reasoning via syntactic GCN in a dual-branch network. |

# Approaches enhance multimodal reasoning through retrieval mechanisms

| Name | Modality | Task | Search source | Search engine | Search query | Training Datasets | Highlight |
|------|----------|------|--------------|--------------|-------------|-------------------|-----------|
| RAGAR ~\citep{khaliq2024ragar} | T,I | Political Fact-Checking | Web, News sites | DuckDuckGo & SerpAPI | LLM-generated questions | MOCHEG dataset | Integrates MLLMs with retrieval-augmented reasoning to verify facts using text and image evidence. |
| Chain-of-action ~\citep{pan2024chain} | T,I | Info retrieval | Web, domain databases, tabular data | Google Search, ChromaDB | Combined sub-questions with embeddings | Pre-trained LLMs, text-embedding-ada-002 | Decomposes questions into reasoning chains with configurable retrieval actions to resolve conflicts between knowledge sources. |
| KAM-CoT ~\citep{mondal2024kam} | T,I, KG | Educational science reasoning | ConceptNet knowledge graph | Custom graph extraction | Context-based text, image captions | ScienceQA | Enhances reasoning by retrieving structured knowledge from graphs and integrating it through two-stage training. |
| AR-MCTS ~\citep{dong2024progressive_retrieval} | T,I | Multi-step reasoning | Math datasets, Wikipedia, COIG | Contriever, CLIP dual-stream | Dynamic state-based retrieval | - | Step-wise retrieval with Monte Carlo Tree Search for verified reasoning. |
| MR-MKG ~\citep{lee2024multimodal} | T, I | General multimodal reasoning | MMKGs | RGAT | Top-N Triple Retrieval | ScienceQA, MARS | Enhances multimodal reasoning by integrating information from multimodal knowledge graphs. |
| Reverse-HP ~\citep{zhu2022multimodal} | T, I | Disease-related reasoning | SDKG-11 | reverse hyperplane projection | entity + relation to entity | Disease entity-set | Utilizes KG embeddings to enhance reasoning for specific diseases with multimodal data. |
| MarT ~\citep{zhang2022multimodal} | T, I | Analogical reasoning | MarKG | Structure-guided relation transfer | Analogical entity prediction | MARS | Uses structure mapping theory and relation-oriented transfer for analogical reasoning with KG. |


## Multimodal Reasoning with Visual Experts

| Name | Modality | Task | Tools | Propose of Tool | Training Datasets | Highlight |
|------|----------|------|-------|----------------|-------------------|-----------|
| MCoT-Memory\cite{liang2024memory_driven} | image to text | VQA reasoning | MLLM-based expert (e.g., LLAVA) | scene graph construction | training free | Memory framework and scene graph construction for effective long-horizon task planning |
| MGCoT\cite{yao2023beyond} | image to text | VQA reasoning | ViT-large encoder | extracts patch-level features of images enhancing the visual information | AQUA-RAT & ScienceQA | Precise visual feature extraction aiding multimodal reasoning |
| CCoT\cite{mitra2024compositional} | image to text | VQA reasoning | Scene Graphs | represent visual scenes as structured graphs | training free | Utilization of the generated scene graph as an intermediate reasoning step. |
| CVR-LLM\cite{li2024enhancing} | image to text | VQA reasoning | BLIP2flant5xx & BLIP2 multi-embedding | basic captioner & encoding multi-modal information | training free | Precise context-aware image descriptions through iterative self-refinement and effective text-multimodal factors integrations |
| TeSO\cite{wang2024avs_cot} | video, audio to text | Temporal Sentence Grounding (TSG) | Mask2Former & LLaVA-1.5 | provides visual information in the AVS task & generates dense scene descriptions | Youtube-8M & semantic-ADE20K | Effective visual tools for better audio-visual correlation boosting AVS performance. |
| CAT\cite{wang2023caption} | image to text | image captioning | SAM | generate pixel - level masks corresponding to user-selected regions, facilitating object-centered perception | training free | Promising pre-trained image caption generators, SAM, and instruction-tuned large language models integration |

## Approaches enhancing Cross-Modal Reasoning

| Name | Modality | Cross-Modal Reasoning | Task | Highlight |
|------|----------|------------------------|------|-----------|
| IdealGPT~\citeyearpar{DBLP:conf/emnlp/YouSW0WACC23} | T, I | Answer sub-questions about image via gpt | VQA, Text Entailment | Using gpt to iteratively decompose and solve visual reasoning tasks |
| AssistGPT~\citeyearpar{gao2023assistgpt} | T, I, V | Plan, Execute, Inspect via External Tools(gpt4, OCR, Grounding, et al.) | VQA, Causal Reasoning | Using an interleaved code and language reasoning approach to handle complex multimodal tasks |
| ProViQ~\citeyearpar{choudhury2023zero} | T, V | Generate and execute Python programs for the video | Video VQA | Using procedural programs to solve visual subtasks in videos |
| MM-REACT~\citeyearpar{yang2023mm} | T, I, V | Use CV tools for sub-taskss about image | VQA, Video VQA | Vision experts combined with GPT for multimodal reasoning and action |
| VisualReasoner~\citeyearpar{cheng2024least} | T, I | Synthesize multi-step reasoning(Using exteral CV tools) data | GQA, VQA | Proposing a least-to-most visual reasoning paradigm and a data synthesis approach for training |
| Multi-model-thought~\citeyearpar{lin2025investigating} | T, I | External Tools(Visual Sketchpad) | Geometry, Math, VQA | Investigating inference-time scaling for multi-modal thought across diverse tasks |
| FaST~\citeyearpar{sun2024visual} | T, I | System switch adapter for visual reasoning | VQA | Integrating fast and slow thinking mechanisms into visual agents |
| ICoT~\citeyearpar{gao2024interleaved} | T, I | Generate interleaved visual-textual reasoning via ADS | VQA | Using visual patches as reasoning carriers to improve LMMs' fine-grained reasoning |
| Image-of-Thought~\citeyearpar{zhou2024image} | T, I | Extract visual rationales step-by-step via IoT prompting | VQA | Using visual rationales to enhance LLMs' reasoning accuracy and interpretability |
| CoTDiffusion~\citeyearpar{ni2024generate} | T, I | External Algorithms | Robotics | Generating subgoal images before action to enhance reasoning in long-horizon robot manipulation tasks |
| T-SciQ~\citeyearpar{wang2023t} | T, I | Model-Intrinsic Capabilities | ScienceQA | Using LLM-generated reasoning signals to teach multimodal reasoning for complex science QA |
| Visual-CoT~\citeyearpar{rose2023visual} | T, I | Model-Intrinsic Capabilities | VQA, DocQA, ChartQA | Using visual-text pairs as reasoning carriers to bridge logical gaps in sequential data |
| VoCoT~\citeyearpar{li2024vocot} | T, I | Model-Intrinsic Capabilities | VQA | Using visually-grounded object-centric reasoning paths for multi-step reasoning |
| MVoT~\citeyearpar{li2025imagine} | T, I | Model-Intrinsic Capabilities | Spatial Reasoning | Using multimodal reasoning with image visualizations to enhance complex spatial reasoning in LMMs |

## Approach of MM-o1

| **Name** | **Backbone** | **Dataset** | **Modality** | **Reasoning Paradigm** | **Task Type** | **Highlight** |
|----------|--------------|-------------|--------------|------------------------|---------------|---------------|
| Macro-O1~\citeyearpar{DBLP:journals/corr/abs-2411-14405} | Qwen2-7B-Instruct | Open-O1 CoT + Marco-o1 CoT + Marco-o1 Instruction | T | MCTS-guided Thinking | Math, Translate | MCTS for solution expansion and reasoning action strategy |
| llamaberry~\citeyearpar{zhang2024llamaberry} | LLaMA-3.1-8B | PRM800K + OpenMathInstruct-1 | T | MCTS-guided Thinking | Math | SR-MCTS for search and PPRM for evaluation |
| LLaVA-CoT~\citeyearpar{xu2024llava_cot} | Llama-3.2V-11B-cot | LLaVA-CoT-100k | T, I | Summary, Caption, Thinking | Science, General | Introduce LLaVA-CoT-100k and scalable beam search |
| LlamaV-o1~\citeyearpar{thawakar2025llamav_o1} | Llama-3.2V-11B-cot | LLaVA-CoT-100k + PixMo | T, I | Summary, Caption, Thinking | Science, General | Introduce VCR-Bench and outperforms |
| Mulberry~\citeyearpar{yao2024mulberry} | Llama-3.2V-11B-cot, LLaVA-Next-8B, Qwen2-VL-7B | Mulberry-260K | T, I | Caption, Rationales, Thinking | Math, General | Introduce Mulberry-260k and CoMCTS for collective learning |
| RedStar-Geo~\citeyearpar{xu2025redstar} | InternVL2-8B | GeoQA | T, I | Long-Thinking | Math | Competitive with minimal Long-CoT data |

## Approach of MM-R1

| **Approach** | **Backbone** | **Dataset** | **RL Algorithm** | **Modality** | **Task Type** | **RL Framework** | **Cold Start** | **Rule-base/RM** |
|--------------|--------------|-------------|------------------|--------------|---------------|------------------|----------------|------------------|
| RLHF-V~\citeyearpar{yu2024rlhf} | LLaVA-13B | RLHF-V-Dataset(1.4k) | DPO | T, I | VQA | Muffin | - | (unknown) |
| InternVL2.5~\citeyearpar{DBLP:journals/corr/abs-2411-10442} | InternVL | MMPR(3m) | MPO(DPO) | T, I | VQA | - | - | (unknown) |
| Insight-V~\citeyearpar{dong2024insight} | LLaMA3-LLaVA-Next | - | DPO | T, I | VQA | trl | - | (unknown) |
| LLaVA-Reasoner-DPO~\citeyearpar{DBLP:journals/corr/abs-2410-16198} | LLaMA3-LLaVA-Next | ShareGPT4o-reasoning-dpo(6.6k) | DPO | T, I | VQA | trl | - | (unknown) |
| VLM-R1~\citeyearpar{shen2025vlmr1} | Qwen2.5-VL | coco , LISA , Refcoco | GRPO | T, I | Grounding ,Math , Open-Vocabulary Detection | trl | No | Rule-base |
| R1-V~\citeyearpar{chen2025r1v} | Qwen2-VL | CLEVR  , GEOQA | GRPO | T, I | Counting , Math | trl | No | Rule-base |
| MM-EUREKA~\citeyearpar{meng2025mmeureka} | InternVL2.5 | K12 , MMPR | RLOO | T, I | Math | OpenRLHF | Yes | Rule-base |
| MM-EUREKA-Qwen~\citeyearpar{meng2025mmeureka} | Qwen2.5-VL | K12 , MMPR | GRPO | T, I | Math | OpenRLHF | No | Rule-base |
| Video-R1~\citeyearpar{feng2025video} | Qwen2.5-VL | Video-R1(260K) | GRPO | T, I, V | Video VQA | trl | Yes | Rule-base |
| LMM-R1~\citeyearpar{peng2025lmmr1} | Qwen2.5-VL | VerMulti | PPO | T, I | Math | OpenRLHF | No | RM |
| Vision-R1~\citeyearpar{huang2025vision} | Qwen2.5-VL | LLaVA-CoT , Mulberry | GRPO | T, I | Math | - | Yes | Rule-base |
| Visual-RFT~\citeyearpar{liu2025visual} | Qwen2-VL | coco , LISA , ... | GRPO | T, I | Detection , Classification | trl | No | Rule-base |
| R1-OneVision~\citeyearpar{yang2025r1onevisionadvancinggeneralizedmultimodal} | Qwen2.5-VL | R1-Onevision-Dataset | GRPO | T, I | Math , Science , General , Doc | - | Yes | Rule-base |
| Seg-Zero~\citeyearpar{liu2025segzero} | Qwen2.5-VL , SAM2 | RefCOCOg , ReasonSeg | GRPO | T, I | Grounding | verl | No | Rule-base |
| VisualThinker-R1-Zero~\citeyearpar{zhou2025VisualThinker-R1-Zero} | Qwen2-VL | SAT dataset | GRPO | T, I | Spatial Reasoning | trl | No | Rule-base |
| R1-Omni~\citeyearpar{zhao2025r1omni} | HumanOmni | MAFW , DFEW | GRPO | T, I, A, V | emotion recognition | trl | Yes | Rule-base |
| OThink-MR1~\citeyearpar{liu2025othink} | Qwen2.5-VL | CLEVR , GEOQA | GRPO | T, I | Counting , Math | - | No | Rule-base |
| Multimodal-Open-R1~\citeyearpar{multimodal-open-r1} | Qwen2-VL | multimodal-open-r1-8k-verified(based on Math360K and Geo170K) | GRPO | T,I | Math | trl | No | Rule-base |
| Curr-ReFT~\citeyearpar{deng2025boostinggeneralizationreasoningvision} | Qwen2.5-VL | RefCOCOg , Math360K , Geo170K | GRPO | T,I | Detection , Classification , Math | Curr-RL | No | RM |
| Open-R1-Video~\citeyearpar{wang-2025-open-r1-video} | Qwen2-VL | open-r1-video-4k | GRPO | T, I, V | Video VQA | trl | No | Rule-base |
| VisRL~\citeyearpar{chen2025visrl} | Qwen2.5-VL | VisCoT | DPO | T,I | VQA | trl | Yes | RM |
| R1-VL~\citeyearpar{zhang2025r1vl} | Qwen2-VL | Mulberry-260k | StepGRPO | T,I | Math , ChartQA | not release | No | Rule-base |




