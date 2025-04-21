# The overview of Multimodal Benchmarks and Datasets

## 目录

- [Multimodal Understanding](#multimodal-understanding)
  - [Visual Centric](#visual-centric)
  - [Audio Centric](#audio-centric)
- [Multimodal Generation](#multimodal-generation)
  - [Cross-modal Generation](#cross-modal-generation)
  - [Joint Multimodal Generation](#joint-multimodal-generation)
- [Multimodal Reasoning](#multimodal-reasoning)
  - [General Visual Reasoning](#general-visual-reasoning)
  - [Domain-specific Reasoning](#domain-specific-reasoning)
- [Multimodal Planning](#multimodal-planning)
  - [GUI Navigation](#gui-navigation)
  - [Embodied and Simulated Environments](#embodied-and-simulated-environments)

## Multimodal Understanding

### Visual Centric

| **Benchmark** | **Dataset** |
|---------------|-------------|
| VQA [1], GQA [2] | ALIGN [3], LTIP [4] |
| DocVQA [5], TextVQA [6] | YFCC100M [7], DocVQA [5] |
| OCR-VQA [8], CMMLU [9] | Visual Genome [10], Wukong [11] |
| C-Eval [12], MTVQA [13] | CC3M [14], ActivityNet-QA [15] |
| Perception-Test [16], Video-MMMU [17] | SBU [18], AI2D [19] |
| Video-MME [20], MMBench [21] | LAION-5B [22], LAION-400M [23] |
| Seed-Bench [24], MME-RealWorld [25] | MS-COCO [26], Virpt [27] |
| MMMU [28], MM-Vet [29] | OpenVid-1M [30], VidGen-1M [31] |
| MMT-Bench [32], Hallu-PI [33] | Flickr30k [34], COYO-700M [35] |
| ColorBench [36], DVQA [37] | WebVid [38], Youku-mPLUG [39] |
| MMStar [40] | VideoCC3M [41], FILIP [42] |
| | CLIP [43], YouTube8M [44] |

### Audio Centric

| **Benchmark** | **Dataset** |
|---------------|-------------|
| AudioBench [45], VoiceBench [46] | Librispeech [47], Common Voice [48] |
| Fleurs [49], MusicBench [50] | Aishell [51], Fleurs [49], MELD [52] |
| Air-Bench [53], MMAU [54] | CoVoST2 [55], SIFT-50M [56] |
| SD-eval [57], CoVoST2 [55] | Clotho [58], AudioCaps [59] |
| MusicNet [60] | ClothoAQA [61], MusicNet [60] |
| | NSynth [62], MusicCaps [63] |

## Multimodal Generation

### Cross-modal Generation

| **Benchmark** | **Dataset** |
|---------------|-------------|
| GenEval [64], T2I-CompBench++ [65] | MS-COCO [26], Flickr30k [34] |
| DPG-Bench [66], GenAI-Bench [67] | Conceptual Captions [68], RedCaps [69] |
| VBench [70], VideoScore [71] | CommonPool [72], LLaVA-Pretrain [73] |
| WorldSimBench [74], WorldModelBench [75] | Aishell1 [51], ThreeDWorld [76] |
| MagicBrush [77], VBench++ [78] | X2I [79], GAIA-1 [80] |
| MJHQ-30K [81], VBench 2.0 [82] | UniSim [83], VidProM [84] |
| AIGCBench [85], EvalCrafter [86] | LWM [87], Genesis [88] |
| | HQ-Edit [89], InstructPix2Pix [90] |
| | MagicBrush [77] |

### Joint Multimodal Generation

| **Benchmark** | **Dataset** |
|---------------|-------------|
| MM-Interleaved [91], ANOLE [92] | DreamLLM [93], SEED-Story [94] |
| InterleavedEval [95], OpenLEAF [96] | NextGPT [97], DreamFactory [98] |
| OpenING [99], M2RAG [100] | DreamRunner [101], EVA [102] |
| SEED-Bench [24], SEED-Bench-2 [103] | |

## Multimodal Reasoning

### General Visual Reasoning

| **Benchmark** | **Dataset** |
|---------------|-------------|
| NaturalBench [104], VCR [105] | VCR [105], TDIUC [106] |
| PhysBench [107], MMBench [21] | MMPR [108], ChartQA [109] |
| MMMU [28], AGIEval [110] | SWAG [111], LLaVA-CoT [112] |
| MMStar [40], InfographicVQA [113] | CLEVR [114], Mulberry-260K [115] |
| VCRBench [116], VisualPuzzles [117] | ShareGPT4oReasoning [118], R1-Onevision [119] |
| | Video-R1-data [120], Visual-CoT [121] |

### Domain-specific Reasoning

| **Benchmark** | **Dataset** |
|---------------|-------------|
| MathVista [122], MATH-Vision [123] | Habitat [124], AI2-THOR [125] |
| VLM-Bench [126], GemBench [127] | Gibson [128], GeoQA [129] |
| GeoQA [129], VIMA-Bench [130] | Isaac Lab [131], ProcTHOR [132] |
| WorldSimBench [74], WorldModelBench [75] | CALVIN [133] |
| ScienceQA [134], ChartQA [109] | |
| MathQA [135], Habitat [124] | |
| AI2-THOR [125], Gibson [128] | |
| iGibson [136], Isaac Lab [131] | |

## Multimodal Planning

### GUI Navigation

| **Benchmark** | **Dataset** |
|---------------|-------------|
| WebArena [137], Mind2Web [138] | AMEX [139], RiCo [140] |
| VisualWebBench [141], OSWorld [142] | WebSRC [143], E-ANT [144] |
| OmniACT [145], VisualAgentBench [146] | AndroidEnv [147], GUI-World [148] |
| LlamaTouch [149], Windows Agent Arena [150] | |
| Ferret-UI [151], WebShop [152] | |
| SWE-BENCH M [153], MineDojo [154] | |
| TeamCraft [155], V-MAGE [156] | |

### Embodied and Simulated Environments

| **Benchmark** | **Dataset** |
|---------------|-------------|
| MineDojo [154], MuEP [157] | MineDojo [154], Habitat 3.0 [158] |
| GVCCI [159], BEHAVIOR-1K [160] | SAPIEN [161], HomeRobot [162] |
| Habitat 3.0 [158], SAPIEN [161] | HoloAssist [163], DrivingDojo [164] |
| HomeRobot [162], HoloAssist [163] | OmmiHD-Scenes [165] |
| DrivingDojo [164], WolfBench [166] | |

## 参考文献

[1] DBLP:journals/corr/KafleK16  
[2] DBLP:conf/cvpr/HudsonM19  
[3] jia2021scalingvisualvisionlanguagerepresentation  
[4] wu2024lotlipimprovinglanguageimagepretraining  
[5] DBLP:conf/wacv/MathewKJ21  
[6] DBLP:conf/cvpr/SinghNSJCBPR19  
[7] Thomee_2016  
[8] DBLP:conf/icdar/0001SSC19  
[9] DBLP:conf/acl/0002ZKY0GDB24  
[10] krishna2016visualgenomeconnectinglanguage  
[11] gu2022wukong100millionlargescale  
[12] DBLP:conf/nips/HuangBZZZSLLZLF23  
[13] DBLP:journals/corr/abs-2405-11985  
[14] changpinyo2021conceptual12mpushingwebscale  
[15] DBLP:conf/aaai/YuXYYZZT19  
[16] DBLP:journals/corr/abs-2411-19941  
[17] DBLP:journals/corr/abs-2501-13826  
[18] DBLP:conf/nips/OrdonezKB11  
[19] DBLP:journals/lre/HiippalaAHKLOTS21  
[20] DBLP:journals/corr/abs-2405-21075  
[21] DBLP:conf/eccv/LiuDZLZZYWHLCL24  
[22] schuhmann2022laion5bopenlargescaledataset  
[23] schuhmann2021laion400mopendatasetclipfiltered  
[24] DBLP:journals/corr/abs-2307-16125  
[25] DBLP:journals/corr/abs-2408-13257  
[26] DBLP:conf/eccv/LinMBHPRDZ14  
[27] yang2024vriptvideoworththousands  
[28] DBLP:conf/cvpr/YueNZ0LZSJRSWYY24  
[29] DBLP:conf/icml/YuYLWL0WW24  
[30] DBLP:journals/corr/abs-2407-02371  
[31] tan2024vidgen1mlargescaledatasettexttovideo  
[32] DBLP:conf/icml/YingMWLLYZZLLLL24  
[33] DBLP:conf/mm/DingWKMCCCCH24  
[34] DBLP:journals/ijcv/PlummerWCCHL17  
[35] lu2023delvingdeeperdatascaling  
[36] liang2025colorbench  
[37] DBLP:conf/cvpr/KaflePCK18  
[38] bain2022frozentimejointvideo  
[39] xu2023youkumplug10millionlargescale  
[40] DBLP:conf/nips/ChenLDZZCDWQLZ24  
[41] nagrani2022learningaudiovideomodalitiesimage  
[42] yao2021filipfinegrainedinteractivelanguageimage  
[43] DBLP:conf/icml/RadfordKHRGASAM21  
[44] abuelhaija2016youtube8mlargescalevideoclassification  
[45] DBLP:journals/corr/abs-2406-16020  
[46] DBLP:journals/corr/abs-2410-17196  
[47] DBLP:conf/icassp/PanayotovCPK15  
[48] DBLP:conf/lrec/ArdilaBDKMHMSTW20  
[49] DBLP:conf/slt/ConneauMKZADRRB22  
[50] DBLP:conf/naacl/MelechovskyGGMH24  
[51] DBLP:conf/ococosda/BuDNWZ17  
[52] DBLP:conf/acl/PoriaHMNCM19  
[53] DBLP:conf/acl/YangXLC0ZLLZZZ24  
[54] DBLP:journals/corr/abs-2410-19168  
[55] DBLP:journals/corr/abs-2007-10310  
[56] pandey2025sift50m  
[57] DBLP:conf/nips/AoWTCZ0W0024  
[58] DBLP:conf/icassp/DrossosLV20  
[59] DBLP:conf/naacl/KimKLK19  
[60] DBLP:conf/iclr/ThickstunHK17  
[61] DBLP:conf/eusipco/LippingSDV22  
[62] DBLP:conf/icml/EngelRRDNES17  
[63] DBLP:journals/corr/abs-2301-11325  
[64] DBLP:conf/nips/GhoshHS23  
[65] huang2025t2icompbenchenhancedcomprehensivebenchmark  
[66] DBLP:journals/corr/abs-2403-05135  
[67] DBLP:journals/corr/abs-2406-13743  
[68] DBLP:conf/acl/SoricutDSG18  
[69] DBLP:conf/nips/DesaiKA021  
[70] DBLP:conf/cvpr/HuangHYZS0Z0JCW24  
[71] DBLP:conf/emnlp/HeJZKSSCCJAWDNL24  
[72] DBLP:conf/nips/GadreIFHSNMWGZO23  
[73] DBLP:conf/nips/LiuLWL23a  
[74] DBLP:journals/corr/abs-2410-18072  
[75] DBLP:journals/corr/abs-2502-20694  
[76] DBLP:conf/nips/GanSAMSTFKBHSKW21  
[77] DBLP:conf/nips/ZhangMCSS23  
[78] DBLP:journals/corr/abs-2411-13503  
[79] xiao2024omnigenunifiedimagegeneration  
[80] DBLP:journals/corr/abs-2309-17080  
[81] li2024playground  
[82] zheng2025vbench  
[83] yang2024learninginteractiverealworldsimulators  
[84] DBLP:conf/nips/WangY24  
[85] DBLP:conf/cvpr/FanZZW0H19  
[86] DBLP:conf/cvpr/LiuC0WZCLZCS24  
[87] DBLP:journals/corr/abs-2402-08268  
[88] authors2024genesis  
[89] DBLP:journals/corr/abs-2404-09990  
[90] DBLP:conf/cvpr/BrooksHE23  
[91] tian2024mm  
[92] chern2024anole  
[93] dong2023dreamllm  
[94] yang2024seedstory  
[95] liu2024holistic  
[96] an2024openleaf  
[97] wu24next  
[98] xie2024dreamfactory  
[99] zhou2024GATE  
[100] ma2025multimodalretrievalaugmentedmultimodal  
[101] zun2024dreamrunner  
[102] chi2024eva  
[103] li2023seedbench2benchmarkingmultimodallarge  
[104] DBLP:conf/nips/LiLPNJMKKNR24  
[105] DBLP:conf/cvpr/ZellersBFC19  
[106] kafle2017analysisvisualquestionanswering  
[107] DBLP:journals/corr/abs-2501-16411  
[108] DBLP:journals/corr/abs-2411-10442  
[109] DBLP:conf/acl/MasryLTJH22  
[110] DBLP:conf/naacl/ZhongCGLLWSCD24  
[111] zellers2018swaglargescaleadversarialdataset  
[112] DBLP:journals/corr/abs-2411-10440  
[113] DBLP:conf/wacv/MathewBTKVJ22  
[114] johnson2016clevrdiagnosticdatasetcompositional  
[115] yao2024mulberry  
[116] qi2025vcrbench  
[117] song2025visualpuzzlesdecouplingmultimodalreasoning  
[118] DBLP:journals/corr/abs-2410-16198  
[119] yang2025r1onevisionadvancinggeneralizedmultimodal  
[120] feng2025video  
[121] shao2024visualcotadvancingmultimodal  
[122] DBLP:conf/iclr/LuBX0LH0CG024  
[123] DBLP:conf/nips/WangPSLRZZL24  
[124] DBLP:conf/iccv/SavvaMPBKMZWJSL19  
[125] DBLP:journals/corr/abs-1712-05474  
[126] DBLP:conf/nips/ZhengCJW22  
[127] DBLP:journals/corr/abs-2410-01345  
[128] DBLP:conf/cvpr/XiaZHSMS18  
[129] chen2022geoqageometricquestionanswering  
[130] DBLP:journals/corr/abs-2210-03094  
[131] DBLP:journals/ral/MittalYYLRHYSGMMBSHG23  
[132] DBLP:conf/nips/DeitkeVHWESHKKM22  
[133] DBLP:journals/ral/MeesHRB22  
[134] DBLP:conf/nips/LuMX0CZTCK22  
[135] DBLP:conf/naacl/AminiGLKCH19  
[136] DBLP:journals/corr/abs-2108-03272  
[137] DBLP:conf/iclr/ZhouX0ZLSCOBF0N24  
[138] DBLP:conf/nips/DengGZCSWSS23  
[139] DBLP:journals/corr/abs-2407-17490  
[140] Deka:2017:Rico  
[141] DBLP:journals/corr/abs-2404-05955  
[142] DBLP:conf/nips/XieZCLZCHCSLLXZ24  
[143] DBLP:journals/corr/abs-2101-09465  
[144] DBLP:journals/corr/abs-2406-14250  
[145] DBLP:conf/eccv/KapoorBRKKAS24  
[146] DBLP:journals/corr/abs-2408-06327  
[147] DBLP:journals/corr/abs-2105-13231  
[148] DBLP:journals/corr/abs-2406-10819  
[149] DBLP:conf/uist/ZhangWJZYGLX24  
[150] DBLP:journals/corr/abs-2409-08264  
[151] DBLP:conf/eccv/YouZSWSNYG24  
[152] DBLP:conf/nips/Yao0YN22  
[153] yang2024swe  
[154] DBLP:conf/nips/FanWJMYZTHZA22  
[155] long2024teamcraftbenchmarkmultimodalmultiagent  
[156] zheng2025vmagegameevaluationframework  
[157] DBLP:conf/ijcai/LiYZZZZYCSC0LT024  
[158] DBLP:conf/iclr/PuigUSCYPDCHMVG24  
[159] DBLP:conf/iros/KimKKSZ23  
[160] DBLP:journals/corr/abs-2403-09227  
[161] DBLP:conf/cvpr/XiangQMXZLLJYWY20  
[162] DBLP:conf/corl/YenamandraRYWKG23  
[163] DBLP:conf/iccv/WangKRPCABFTFJP23  
[164] DBLP:journals/corr/abs-2207-11432  
[165] zheng2025omnihdscenesnextgenerationmultimodaldataset  
[166] qiao2024benchmarking
