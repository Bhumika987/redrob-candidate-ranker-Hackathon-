"""
jd_requirements.py
Structured encoding of the Senior AI Engineer JD from data/job_description.docx.

Source: Redrob AI — Series A, Pune/Noida, Founding Team hire.
This is a pure data module — no scorer logic, no LLM calls.
Import what you need:
    from jd_requirements import (
        HARD_REQUIREMENTS, DISQUALIFIERS, NICE_TO_HAVE,
        PREFERRED_LOCATIONS, TARGET_EXPERIENCE_YEARS,
        PRODUCTION_SIGNAL_KEYWORDS, SERVICES_COMPANIES, IRRELEVANT_TITLES,
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


# =============================================================================
# HARD REQUIREMENTS
# =============================================================================
# JD section: "Things you absolutely need"
# These are ELIMINATORS if absent — not bonuses. The JD frames each one as
# something that will make the role "very painful" if missing, or that the
# company screens on directly. A candidate missing any of these should be
# ranked near the bottom regardless of other signals.
# =============================================================================

@dataclass(frozen=True)
class HardRequirement:
    id: str
    name: str
    # Verbatim or close-paraphrase of the JD rationale for this requirement.
    why: str
    # Keywords to search for across profile summary, career descriptions, skills.
    signal_terms: List[str]
    # Specific technologies named in the JD as examples.
    example_tech: List[str]


HARD_REQUIREMENTS: List[HardRequirement] = [
    HardRequirement(
        id="embeddings_retrieval_production",
        name="Production embeddings-based retrieval, deployed to real users",
        why=(
            "JD: 'Production experience with embeddings-based retrieval systems "
            "(sentence-transformers, OpenAI embeddings, BGE, E5, or similar) "
            "deployed to real users. We don't care which model — we care that "
            "you've handled embedding drift, index refresh, retrieval-quality "
            "regression in production.' The first 90-day plan assumes this "
            "knowledge exists on day 1 to audit BM25 + rules and ship v2."
        ),
        signal_terms=[
            "embedding", "embeddings", "dense retrieval", "semantic search",
            "sentence-transformers", "embedding drift", "index refresh",
            "retrieval quality", "vector search", "ANN",
            "approximate nearest neighbor", "bi-encoder", "cross-encoder",
            "deployed", "production", "real users",
        ],
        example_tech=[
            "sentence-transformers", "OpenAI embeddings", "BGE", "E5",
            "bge-large", "instructor-xl", "text-embedding-ada-002",
        ],
    ),
    HardRequirement(
        id="vector_db_hybrid_search",
        name="Vector database / hybrid search infrastructure in production",
        why=(
            "JD: 'Production experience with vector databases or hybrid search "
            "infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
            "Elasticsearch, FAISS, or something similar. The specific tech doesn't "
            "matter; the operational experience does.' The role owns the intelligence "
            "layer — someone without this cannot hit the Week 4-8 delivery target "
            "of shipping a v2 ranker with hybrid retrieval."
        ),
        signal_terms=[
            "vector database", "vector db", "vector store", "hybrid search",
            "BM25", "inverted index", "sparse-dense", "ANN index",
            "Elasticsearch", "OpenSearch", "Pinecone", "Weaviate", "Qdrant",
            "Milvus", "FAISS", "pgvector", "Chroma", "Vespa", "Annoy",
            "HNSW", "IVF",
        ],
        example_tech=[
            "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
            "Elasticsearch", "FAISS", "pgvector", "Vespa", "Chroma",
        ],
    ),
    HardRequirement(
        id="strong_python",
        name="Strong Python — code quality matters",
        why=(
            "JD: 'Strong Python. Yes really, we care about code quality.' "
            "The role writes production code from day 1 on a 4-person team "
            "scaling to 12. Bad code compounds fast at this stage."
        ),
        signal_terms=[
            "Python", "python", "PySpark", "FastAPI", "pydantic", "pytest",
            "type hints", "mypy", "asyncio", "NumPy", "pandas",
        ],
        example_tech=["Python", "PySpark", "FastAPI", "NumPy", "pandas"],
    ),
    HardRequirement(
        id="ranking_evaluation_frameworks",
        name="Designed evaluation frameworks for ranking systems (NDCG/MRR/MAP)",
        why=(
            "JD: 'Hands-on experience designing evaluation frameworks for ranking "
            "systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test "
            "interpretation. If you've never thought about how to evaluate a "
            "ranking system rigorously, this role will be very painful.' "
            "Weeks 9-12 of the 90-day plan are explicitly about eval infra."
        ),
        signal_terms=[
            "NDCG", "MRR", "MAP", "precision@k", "recall@k", "DCG",
            "offline evaluation", "online evaluation", "A/B test", "A/B testing",
            "interleaving", "ranking evaluation", "evaluation framework",
            "relevance judgement", "offline-to-online", "click-through rate",
            "engagement metric", "benchmark", "holdout", "nDCG",
        ],
        example_tech=["NDCG", "MRR", "MAP", "A/B testing", "interleaving"],
    ),
]


# =============================================================================
# DISQUALIFIERS
# =============================================================================
# Sources: JD section "Things we explicitly do NOT want" + hard eliminators
# stated in the body ("we will not move forward").
# A single confirmed disqualifier should heavily penalise a candidate.
# Note: some are HARD stops ("we will not move forward"); others are softer
# ("we will probably not move forward") — flagged in severity.
# =============================================================================

@dataclass(frozen=True)
class Disqualifier:
    id: str
    name: str
    severity: str         # "hard" = will not move forward; "soft" = probably not
    why: str              # JD rationale verbatim or close paraphrase
    # Plain-English hint for the scorer on how to detect this from the schema.
    check_hint: str


DISQUALIFIERS: List[Disqualifier] = [
    Disqualifier(
        id="pure_research",
        name="Entire career in pure research (no production deployment ever)",
        severity="hard",
        why=(
            "JD: 'If you've spent your career in pure research environments "
            "(academic labs, research-only roles) without any production deployment "
            "— we will not move forward. We are explicit about this. "
            "We've tried it twice and it didn't work for either side.'"
        ),
        check_hint=(
            "Scan all career_history descriptions for PRODUCTION_SIGNAL_KEYWORDS. "
            "If none appear anywhere in the career, and/or all titles are "
            "'Research Scientist', 'PhD Student', 'Postdoc', 'Research Intern' "
            "or similar, treat as hard disqualifier."
        ),
    ),
    Disqualifier(
        id="recent_llm_wrapper_only",
        name="AI experience is <12 months of LangChain/OpenAI calls with no pre-LLM production ML",
        severity="soft",
        why=(
            "JD: 'If your AI experience consists primarily of recent (under 12 months) "
            "projects using LangChain to call OpenAI — we will probably not move forward, "
            "unless you can demonstrate substantial pre-LLM-era ML production experience. "
            "We're looking for people who understood retrieval and ranking before it "
            "became fashionable.'"
        ),
        check_hint=(
            "Flag if: (a) skills/career contain LANGCHAIN_WRAPPER_TERMS as the dominant "
            "AI signal AND (b) no PRE_LLM_ML_TERMS appear in career descriptions or skills. "
            "A candidate with XGBoost ranking work from 2019 who now also uses LangChain "
            "is NOT disqualified. Only disqualify if LangChain/OpenAI is the ENTIRE "
            "AI story and no pre-2022 production ML evidence exists."
        ),
    ),
    Disqualifier(
        id="services_only_career",
        name="Entire career at pure IT-services / consulting firms",
        severity="soft",
        why=(
            "JD: 'People who have only worked at consulting firms (TCS, Infosys, "
            "Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career. "
            "We've had bad fit experiences in both directions.' CRITICAL CARVE-OUT: "
            "'If you're currently at one of these companies but have prior product-company "
            "experience, that's fine.' Do NOT disqualify if any prior role was at a "
            "product company — only disqualify if every role maps to SERVICES_COMPANIES."
        ),
        check_hint=(
            "Cross-reference every career_history[*].company against SERVICES_COMPANIES "
            "(case-insensitive substring match). If ALL match → apply disqualifier. "
            "If even one non-services company exists in the history → skip this disqualifier."
        ),
    ),
    Disqualifier(
        id="cv_speech_robotics_without_nlp_ir",
        name="Primary expertise CV / speech / robotics with no NLP/IR exposure",
        severity="soft",
        why=(
            "JD: 'People whose primary expertise is computer vision, speech, or "
            "robotics without significant NLP/IR exposure. We respect your work "
            "but you'd be re-learning fundamentals here.'"
        ),
        check_hint=(
            "Count CV_SPEECH_ROBOTICS_TERMS hits vs NLP_IR_TERMS hits across "
            "skills + career descriptions. If CV/speech/robotics hits are >3x the "
            "NLP/IR hits AND total NLP/IR hit count is < 3, flag as disqualifier. "
            "Having some CV skills is fine — the issue is absence of NLP/IR entirely."
        ),
    ),
    Disqualifier(
        id="code_inactive_18m",
        name="Has not written production code in last 18 months (moved to architecture/leadership)",
        severity="soft",
        why=(
            "JD: 'If you are a senior engineer who hasn't written production code in "
            "the last 18 months because you've moved into architecture or tech lead "
            "roles — we will probably not move forward. This role writes code.'"
        ),
        check_hint=(
            "Check career_history entries in the last 18 months for titles containing "
            "'Architect', 'VP Engineering', 'Director of Engineering', 'Head of', 'CTO', "
            "'Tech Lead' WITHOUT coding signals in the description. "
            "Also check github_activity_score: -1 (no GitHub linked) combined with "
            "such a title is a strong signal of code inactivity."
        ),
    ),
    Disqualifier(
        id="title_chaser",
        name="Title-chaser: average job tenure < 18 months across career",
        severity="soft",
        why=(
            "JD: 'If your career trajectory shows you optimizing for Senior -> Staff -> "
            "Principal titles by switching companies every 1.5 years, we're not a fit. "
            "We need someone who plans to be here for 3+ years.'"
        ),
        check_hint=(
            "Compute mean(career_history[*].duration_months) excluding the current "
            "open-ended role (is_current=True). If mean < TITLE_CHASER_THRESHOLD_MONTHS "
            "flag as disqualifier. Additionally check if consecutive titles across "
            "different companies show rapid seniority escalation (e.g. "
            "Engineer -> Senior -> Staff -> Principal across 4 companies in 5 years)."
        ),
    ),
    Disqualifier(
        id="closed_source_no_external_validation",
        name="5+ years exclusively on closed-source proprietary systems, no external validation",
        severity="soft",
        why=(
            "JD: 'People whose work has been entirely on closed-source proprietary "
            "systems for 5+ years without external validation (papers, talks, "
            "open-source). We need to see how you think, not just trust that "
            "you can think.'"
        ),
        check_hint=(
            "Check: github_activity_score == -1 (no GitHub linked) AND "
            "certifications list is empty AND no skill endorsements > 0 AND "
            "all career_history companies are large enterprises or services firms. "
            "If years_of_experience > 5 and all these hold, flag this disqualifier."
        ),
    ),
]

# Threshold used by the title_chaser disqualifier check.
TITLE_CHASER_THRESHOLD_MONTHS: int = 18  # JD: "switching every 1.5 years"


# =============================================================================
# NICE TO HAVE
# =============================================================================
# JD section: "Things we'd like you to have but won't reject you for."
# Use as additive score bonuses. None of these is an eliminator.
# =============================================================================

@dataclass(frozen=True)
class NiceToHave:
    id: str
    name: str
    why: str
    signal_terms: List[str]


NICE_TO_HAVE: List[NiceToHave] = [
    NiceToHave(
        id="llm_finetuning",
        name="LLM fine-tuning experience (LoRA / QLoRA / PEFT)",
        why=(
            "JD explicitly lists 'LLM fine-tuning experience (LoRA, QLoRA, PEFT)' "
            "as a nice-to-have. Relevant for the long-term architecture of the "
            "candidate-JD matching system."
        ),
        signal_terms=[
            "LoRA", "QLoRA", "PEFT", "fine-tuning", "fine-tune", "finetuning",
            "instruction tuning", "RLHF", "DPO", "SFT", "adapter",
            "parameter-efficient", "low-rank adaptation",
        ],
    ),
    NiceToHave(
        id="learning_to_rank",
        name="Learning-to-rank models (XGBoost-based or neural)",
        why=(
            "JD lists 'Experience with learning-to-rank models (XGBoost-based or "
            "neural)'. Directly relevant to the ranker the role will build."
        ),
        signal_terms=[
            "learning to rank", "LambdaMART", "LambdaRank", "RankNet",
            "LTR", "pointwise", "pairwise", "listwise",
            "XGBoost ranking", "GBDT ranking", "gradient boosted trees",
            "neural ranking", "ColBERT", "MonoT5", "RankLLM",
        ],
    ),
    NiceToHave(
        id="hrtech_marketplace",
        name="HR-tech / recruiting / marketplace product background",
        why=(
            "JD lists 'Prior exposure to HR-tech, recruiting tech, or marketplace "
            "products'. Domain familiarity reduces onboarding cost on Redrob's "
            "specific product nuances (two-sided talent marketplace)."
        ),
        signal_terms=[
            "HR tech", "recruiting", "recruitment", "talent acquisition",
            "ATS", "applicant tracking", "marketplace", "two-sided market",
            "job board", "candidate matching", "HR platform", "talent intelligence",
        ],
    ),
    NiceToHave(
        id="distributed_inference",
        name="Distributed systems / large-scale inference optimization",
        why=(
            "JD lists 'Background in distributed systems or large-scale inference "
            "optimization'. Needed as the platform scales from Series A to growth."
        ),
        signal_terms=[
            "distributed", "Spark", "Kafka", "Flink", "Ray", "Dask",
            "TensorRT", "ONNX", "quantization", "model serving",
            "inference optimization", "latency optimization", "throughput",
            "horizontal scaling", "sharding", "replication", "Triton",
        ],
    ),
    NiceToHave(
        id="open_source_contributions",
        name="Open-source AI/ML contributions",
        why=(
            "JD lists 'Open-source contributions in the AI/ML space'. Also connects "
            "to the closed-source disqualifier — the JD needs to 'see how you think'. "
            "github_activity_score > 0 is a direct Redrob platform signal for this."
        ),
        signal_terms=[
            "open-source", "open source", "GitHub", "contributor", "maintainer",
            "OSS", "pull request", "PyPI", "Hugging Face",
        ],
    ),
]


# =============================================================================
# PREFERRED LOCATIONS
# =============================================================================
# JD: "Pune/Noida-preferred but flexible... Candidates in Hyderabad, Pune,
# Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, but we
# don't sponsor work visas."
# Scoring guide: primary > acceptable > willing_to_relocate > outside India.
# =============================================================================

PREFERRED_LOCATIONS: Dict = {
    # Highest location score — no relocation friction.
    "primary": ["pune", "noida"],

    # JD explicitly names these as acceptable without relocation sponsorship.
    "acceptable": [
        "hyderabad",
        "mumbai",
        "delhi",
        "delhi ncr",
        "gurgaon",
        "gurugram",   # same metro as Delhi NCR
        "bengaluru",  # JD says "Tier-1 Indian cities" — counts if relocating
        "bangalore",
    ],

    # Candidates outside the above but with willing_to_relocate=True:
    # apply a moderate penalty rather than exclude.
    "relocation_note": (
        "JD: 'Open to relocation candidates from Tier-1 Indian cities.' "
        "If redrob_signals.willing_to_relocate is True and location is a Tier-1 "
        "Indian city not in primary/acceptable, apply small penalty only."
    ),

    # Outside India: case-by-case; no visa sponsorship means high friction.
    "outside_india_note": (
        "JD: 'Outside India: case-by-case, but we don't sponsor work visas.' "
        "Down-weight but do not hard-exclude. willing_to_relocate must be True."
    ),

    # Notice period preference — affects time-to-hire urgency at a Series A.
    "notice_period": {
        "preferred_max_days": 30,   # JD: "We'd love sub-30-day notice"
        "buyout_max_days": 30,      # JD: "We can buy out up to 30 days"
        # 30-90 days: still in scope per JD but "bar gets higher" — penalise
        "penalty_threshold_days": 30,
        # Beyond 90 days: heavy penalty (not in JD but implied by urgency)
        "heavy_penalty_days": 90,
    },
}


# =============================================================================
# TARGET EXPERIENCE YEARS
# =============================================================================
# JD: "Experience Required: 5–9 years ... some people hit senior judgment at
# 4 years; some never hit it after 15."
# Ideal candidate section: "6-8 years total, of which 4-5 are in applied
# ML/AI roles at product companies."
# Use as a scoring curve, not a hard cutoff.
# =============================================================================

TARGET_EXPERIENCE_YEARS: Dict = {
    "min": 4,             # JD: "some people hit senior judgment at 4 years"
    "ideal_low": 5,       # JD stated range start
    "ideal_high": 9,      # JD stated range end
    "sweet_spot_low": 6,  # JD ideal candidate: "6-8 years total"
    "sweet_spot_high": 8, # JD ideal candidate: "6-8 years total"
    "max_considered": 12, # implied; beyond 12 the "never hits senior judgment" comment applies
    # Subset of total experience that must be in applied ML/AI at product companies.
    "min_applied_ml_years": 4,  # JD ideal: "4-5 years in applied ML/AI at product companies"
}


# =============================================================================
# PRODUCTION SIGNAL KEYWORDS
# =============================================================================
# The sharpest distinction in the JD is production ML vs research/demo ML.
# The JD explicitly penalises candidates who "understood retrieval and ranking
# before it became fashionable" — i.e., it rewards demonstrated production history.
# These terms, found in career descriptions or the summary, are evidence of
# shipped production systems. Used for:
#   1. Satisfying the pure_research disqualifier (absence of these → disqualify)
#   2. Satisfying hard requirement embeddings_retrieval_production
#   3. Boosting score for candidates whose title is irrelevant but career shows depth
# =============================================================================

PRODUCTION_SIGNAL_KEYWORDS: List[str] = [
    # Deployment evidence
    "deployed", "deployment", "in production", "prod", "production",
    "real users", "live", "shipped", "launched", "released", "at scale",
    "serving", "served", "millions of", "billions of",
    # Operational / reliability signals
    "QPS", "requests per second", "latency", "throughput", "SLA", "SLO",
    "uptime", "reliability", "on-call", "incident", "monitoring", "alerting",
    # ML infra / pipeline
    "pipeline", "feature store", "feature pipeline", "model serving",
    "inference", "batch inference", "online inference", "real-time",
    "streaming", "data pipeline", "ETL", "model registry",
    # Retrieval / search / ranking specific
    "retrieval", "ranking", "recommendation", "search", "re-ranking",
    "reranking", "recall", "precision", "relevance", "vector search",
    "embedding", "embeddings", "ANN", "approximate nearest neighbor",
    "hybrid search", "BM25", "inverted index", "dense retrieval",
    "sparse retrieval", "passage retrieval",
    # Evaluation & experimentation — production teams run these; researchers don't
    "A/B test", "A/B testing", "experiment", "online evaluation",
    "offline evaluation", "NDCG", "MRR", "MAP", "click-through",
    "engagement metric", "conversion", "interleaving",
    # Team/product context implying real users
    "users", "customers", "product", "platform", "API",
    "microservice", "service", "endpoint", "traffic",
]


# =============================================================================
# SERVICES COMPANIES
# =============================================================================
# JD explicitly names the firms whose sole-career presence is a disqualifier.
# User prompt also adds Mindtree and HCL.
# IMPORTANT: The JD carve-out — "If you're currently at one of these companies
# but have prior product-company experience, that's fine." Match case-insensitively.
# =============================================================================

SERVICES_COMPANIES: Set[str] = {
    # Named explicitly in the JD
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant", "cognizant technology solutions", "cts",
    "capgemini",
    # Added by user request (common IT-services firms, same disqualifier logic)
    "mindtree", "ltimindtree", "lti mindtree",
    "hcl", "hcl technologies", "hcltech",
    # Close variants / commonly seen spellings
    "lti", "larsen & toubro infotech", "l&t infotech",
    "tech mahindra", "tech-mahindra",
    "mphasis",
    "hexaware",
    "niit technologies",
    "dxc technology", "dxc",
    "unisys",
    "ibm global services",  # IBM has product divisions; this targets the services arm only
    "kforce",
}


# =============================================================================
# IRRELEVANT TITLES
# =============================================================================
# From the data exploration: ~68% of the 100K pool are non-technical roles
# injected as noise. The JD is explicit:
#   "A candidate whose title is 'Marketing Manager' is not a fit, no matter
#    how perfect their skill list looks."
# BUT the JD also warns: "A Tier 5 candidate may not use the words 'RAG' or
# 'Pinecone' but if their career history shows they built a recommendation
# system at a product company, they're a fit."
# So: irrelevant current title → STRONG prior against, but career_history can
# override. Do not hard-zero these candidates — penalise title, then let the
# career description scoring recover them if warranted.
# =============================================================================

IRRELEVANT_TITLES: Set[str] = {
    # Business / ops
    "hr manager", "hr executive", "human resources manager",
    "operations manager", "operations executive",
    "accountant", "senior accountant", "accounts manager", "finance manager",
    "business development manager", "business development executive",
    # Non-technical engineering
    "mechanical engineer", "civil engineer", "electrical engineer",
    "manufacturing engineer", "structural engineer",
    # Creative / marketing
    "content writer", "technical writer", "copywriter", "content strategist",
    "graphic designer", "ui designer", "visual designer", "ux designer",
    "marketing manager", "digital marketing manager", "marketing executive",
    "brand manager",
    # Sales / support
    "sales executive", "sales manager", "account executive", "account manager",
    "customer support", "customer success manager", "support engineer",
    "help desk", "customer service",
}


# =============================================================================
# CV / SPEECH / ROBOTICS TERMS  (used by cv_speech_robotics_without_nlp_ir)
# =============================================================================
# These terms alone don't disqualify — their DOMINANCE over NLP_IR_TERMS does.
# =============================================================================

CV_SPEECH_ROBOTICS_TERMS: List[str] = [
    # Computer Vision
    "computer vision", "image classification", "object detection",
    "image segmentation", "semantic segmentation", "instance segmentation",
    "OCR", "YOLO", "ResNet", "VGG", "EfficientNet", "convolutional",
    "CNN", "image recognition", "video analysis", "pose estimation",
    "depth estimation", "3D reconstruction", "OpenCV", "stereo vision",
    # Speech / Audio
    "speech recognition", "ASR", "automatic speech recognition",
    "TTS", "text-to-speech", "speech synthesis", "audio processing",
    "speaker diarization", "Whisper", "wav2vec", "DeepSpeech",
    "voice", "acoustic model", "speaker recognition", "keyword spotting",
    # Robotics / Physical AI
    "robotics", "ROS", "robot operating system", "SLAM",
    "autonomous vehicle", "self-driving", "path planning", "kinematics",
    "drone", "actuator", "control system", "embedded systems",
    "reinforcement learning for robotics",
]

# NLP / IR terms whose presence offsets the cv_speech_robotics disqualifier.
NLP_IR_TERMS: List[str] = [
    "NLP", "natural language processing", "text classification",
    "language model", "LLM", "large language model",
    "BERT", "RoBERTa", "transformer", "tokenizer", "tokenization",
    "embedding", "embeddings", "retrieval", "information retrieval",
    "IR", "search", "ranking", "question answering", "QA",
    "named entity recognition", "NER", "sentiment analysis",
    "text summarization", "summarization", "machine translation",
    "RAG", "retrieval augmented generation",
    "dense retrieval", "BM25", "passage retrieval", "document retrieval",
]


# =============================================================================
# LANGCHAIN / WRAPPER TERMS  (used by recent_llm_wrapper_only disqualifier)
# =============================================================================
# These are the "hot framework" tools the JD is cautious about when they are
# the candidate's ONLY AI signal. Not a disqualifier on their own.
# =============================================================================

LANGCHAIN_WRAPPER_TERMS: List[str] = [
    "LangChain", "LlamaIndex", "Haystack", "LangGraph",
    "AutoGen", "CrewAI", "Flowise", "Dify",
    "ChatGPT API", "OpenAI API", "GPT-4 API",
]

# Pre-LLM ML production terms that CANCEL the recent_llm_wrapper_only disqualifier.
PRE_LLM_ML_TERMS: List[str] = [
    "scikit-learn", "sklearn", "XGBoost", "LightGBM", "CatBoost",
    "gradient boosting", "random forest", "logistic regression",
    "recommendation system", "collaborative filtering", "matrix factorization",
    "BM25", "TF-IDF", "Lucene", "Solr", "Elasticsearch",
    "information retrieval", "learning to rank", "click model",
    "feature engineering", "feature store",
    "Spark MLlib", "Kafka", "Flink", "streaming ML", "online learning",
    "word2vec", "GloVe", "fastText",    # pre-transformer NLP
    "BERT", "RoBERTa",                  # pre-GPT-4 era transformers
    "PyTorch", "TensorFlow",            # only meaningful if pre-2022 context
]


# =============================================================================
# BEHAVIORAL THRESHOLDS
# =============================================================================
# JD's key behavioural insight (final note for hackathon participants):
#   "Your ranking system should also weigh behavioral signals — a perfect-on-paper
#    candidate who hasn't logged in for 6 months and has a 5% recruiter response
#    rate is, for hiring purposes, not actually available. Down-weight them
#    appropriately."
#
# These thresholds encode how each availability signal maps to a per-signal
# multiplier component.  The scorer combines them into a single availability
# multiplier constrained to [0.5, 1.0] that scales the base technical score.
#
# Structure of each signal block:
#   "jd_citation" — verbatim or paraphrase of the JD rationale
#   "tiers"       — ordered list of bands; evaluate top-to-bottom, use first match
#                   Each tier has: label, threshold condition, multiplier component,
#                   and a note explaining the reasoning.
#
# Data calibration note (from explore.py on the 100K dataset, date = 2026-07-01):
#   recruiter_response_rate : sparse above 0.8 (2.2% of pool); drops sharply above 0.75
#   last_active_date        : 0% <30d, 26% 30-90d, 41% 90-180d, 32% 180-365d, 0% >365d
#   open_to_work_flag       : 35% True, 65% False
#   interview_completion_rate: 0% in [0-0.2], clusters in [0.4-0.8]
# =============================================================================

BEHAVIORAL_THRESHOLDS: Dict = {

    # -------------------------------------------------------------------------
    # recruiter_response_rate  (float 0.0–1.0)
    # -------------------------------------------------------------------------
    # JD: "a 5% recruiter response rate is, for hiring purposes, not actually
    # available."  A candidate may be technically perfect but unreachable.
    # This is the strongest individual behavioural signal because it measures
    # willingness to engage with exactly the kind of outreach Redrob will send.
    # Weight: 0.35 — highest weight of the four signals.
    # -------------------------------------------------------------------------
    "recruiter_response_rate": {
        "jd_citation": (
            "JD: 'a perfect-on-paper candidate who ... has a 5% recruiter response rate "
            "is, for hiring purposes, not actually available. Down-weight them appropriately.'"
        ),
        "weight": 0.35,
        # Tiers are evaluated top-to-bottom; use the first band the value falls into.
        # multiplier: the component value this signal contributes before weighted sum.
        "tiers": [
            {
                "label": "ghost",
                "condition": "value < 0.10",
                "multiplier": 0.50,
                "note": (
                    "JD explicitly calls out 5% as the archetypal unavailable candidate. "
                    "Under 10% means the recruiter has to message ~10 times to get one reply "
                    "— not viable for a time-sensitive Series A hire. Hard floor at 0.50."
                ),
            },
            {
                "label": "low",
                "condition": "0.10 <= value < 0.25",
                "multiplier": 0.70,
                "note": (
                    "Replies to roughly 1 in 5–10 messages. Reachable but unreliable. "
                    "Meaningful penalty: a strong technical fit still makes sense to contact, "
                    "but expect slow / missed conversations."
                ),
            },
            {
                "label": "mid",
                "condition": "0.25 <= value < 0.50",
                "multiplier": 0.85,
                "note": (
                    "Replies to roughly 1 in 2–4 messages. Acceptable but not ideal. "
                    "Slight penalty to prefer more responsive candidates at equal technical score."
                ),
            },
            {
                "label": "high",
                "condition": "0.50 <= value < 0.75",
                "multiplier": 1.00,
                "note": (
                    "Responds to more than half of recruiter messages. No penalty. "
                    "Aligns with the JD's implicit expectation of an actively engaged candidate."
                ),
            },
            {
                "label": "very_high",
                "condition": "value >= 0.75",
                "multiplier": 1.00,
                "note": (
                    "Only 2.2% of the dataset reaches this band (data calibration). "
                    "No additional bonus over 'high' — the signal saturates at reliable engagement; "
                    "the JD offers no extra credit for near-perfect responsiveness."
                ),
            },
        ],
    },

    # -------------------------------------------------------------------------
    # last_active_date  (expressed as days_since_active = today - last_active_date)
    # -------------------------------------------------------------------------
    # JD: "hasn't logged in for 6 months" is the concrete example of an
    # unavailable candidate.  Recency of platform activity is a proxy for
    # whether the candidate is currently in job-search mode.
    # Data note: no candidate in this dataset was active <30d or >365d ago —
    # the full live range is 30–365d.  Tiers are calibrated accordingly,
    # with the penalty curve steepening around the JD's explicit 6-month mark.
    # Weight: 0.30 — second-highest; recency affects reachability directly.
    # -------------------------------------------------------------------------
    "last_active_days": {
        "jd_citation": (
            "JD: 'a perfect-on-paper candidate who hasn't logged in for 6 months ... "
            "is, for hiring purposes, not actually available.' "
            "Also: 'Active on Redrob platform (or has clear signal of being in the job "
            "market) so we can actually talk to them.'"
        ),
        "weight": 0.30,
        # days_since_active = (TODAY - last_active_date).days
        "tiers": [
            {
                "label": "very_fresh",
                "condition": "days_since_active < 30",
                "multiplier": 1.00,
                "note": (
                    "Active in the last month — clearly in the market right now. "
                    "No candidate in the current dataset falls here; this tier exists "
                    "for completeness and live-platform use."
                ),
            },
            {
                "label": "fresh",
                "condition": "30 <= days_since_active < 60",
                "multiplier": 1.00,
                "note": (
                    "Active within 30–60 days. Still clearly engaged. "
                    "No penalty — this is the best band actually present in the dataset."
                ),
            },
            {
                "label": "recent",
                "condition": "60 <= days_since_active < 120",
                "multiplier": 0.90,
                "note": (
                    "2–4 months since last activity. Likely still browsable but may "
                    "have reduced urgency. Small penalty to prioritise fresher candidates."
                ),
            },
            {
                "label": "cooling",
                "condition": "120 <= days_since_active < 180",
                "multiplier": 0.75,
                "note": (
                    "4–6 months. Approaching the JD's cited 6-month threshold. "
                    "Candidate may have found a role or lost interest. Meaningful penalty."
                ),
            },
            {
                "label": "stale",
                "condition": "180 <= days_since_active < 270",
                "multiplier": 0.60,
                "note": (
                    "6–9 months. This is the regime the JD explicitly flags: "
                    "'hasn't logged in for 6 months ... not actually available.' "
                    "Strong penalty. Worth contacting only if technical score is exceptional."
                ),
            },
            {
                "label": "very_stale",
                "condition": "days_since_active >= 270",
                "multiplier": 0.50,
                "note": (
                    "9+ months inactive. Hard floor at 0.50 — matched to the ghost tier "
                    "for response rate. For a Series A urgency hire, pursuing this "
                    "candidate carries high outreach cost for low expected conversion."
                ),
            },
        ],
    },

    # -------------------------------------------------------------------------
    # open_to_work_flag  (boolean)
    # -------------------------------------------------------------------------
    # JD: the ideal candidate is "Active on Redrob platform (or has clear signal
    # of being in the job market)". open_to_work_flag=True is the explicit
    # platform signal of that intent.  False does not mean unavailable — many
    # passive candidates respond well — but it reduces expected conversion.
    # Weight: 0.15 — smallest weight; it's a binary signal with less information
    # than the continuous signals.
    # -------------------------------------------------------------------------
    "open_to_work_flag": {
        "jd_citation": (
            "JD ideal candidate: 'Active on Redrob platform (or has clear signal of "
            "being in the job market) so we can actually talk to them.' "
            "open_to_work_flag=True is the direct Redrob-platform encoding of this."
        ),
        "weight": 0.15,
        "tiers": [
            {
                "label": "actively_open",
                "condition": "value is True",
                "multiplier": 1.00,
                "note": (
                    "Candidate has explicitly signalled openness to new roles. "
                    "No penalty — this is the ideal state from the JD's perspective."
                ),
            },
            {
                "label": "passive",
                "condition": "value is False",
                "multiplier": 0.85,
                "note": (
                    "Passive candidate. 65% of the dataset. Modest penalty: "
                    "conversion rates for passive outreach are lower, and Redrob "
                    "as a platform has less confidence in this candidate's receptivity. "
                    "Not a hard penalty — many excellent candidates are passively open."
                ),
            },
        ],
    },

    # -------------------------------------------------------------------------
    # interview_completion_rate  (float 0.0–1.0)
    # -------------------------------------------------------------------------
    # Not cited verbatim in the JD, but directly implied by the availability
    # theme: a candidate who schedules interviews but doesn't show up creates
    # operational waste for Redrob's recruiter customers.  Low completion rate
    # predicts the same ghost behaviour as low recruiter_response_rate.
    # Data note: 0% of the dataset is in [0-0.2], so the lowest band is
    # effectively theoretical; real penalty activates from [0.2-0.4] upward.
    # Weight: 0.20 — meaningful but secondary to response rate and recency.
    # -------------------------------------------------------------------------
    "interview_completion_rate": {
        "jd_citation": (
            "Implied by JD's availability theme: 'a candidate who ... has a 5% recruiter "
            "response rate is, for hiring purposes, not actually available.' "
            "Low interview completion rate is a lagged version of the same signal — "
            "a candidate who accepts interviews but doesn't attend is operationally "
            "equivalent to an unresponsive one."
        ),
        "weight": 0.20,
        "tiers": [
            {
                "label": "unreliable",
                "condition": "value < 0.40",
                "multiplier": 0.60,
                "note": (
                    "Misses more than 60% of scheduled interviews. Strong ghost signal. "
                    "No candidate in the dataset falls below 0.2 — this tier captures "
                    "the 11% of candidates in [0.2-0.4]. Penalty matches the 'cooling' "
                    "tier for last_active to avoid over-penalising a single signal."
                ),
            },
            {
                "label": "average",
                "condition": "0.40 <= value < 0.65",
                "multiplier": 0.85,
                "note": (
                    "Completes roughly 2 in 3 interviews. Acceptable but not ideal. "
                    "Covers ~45% of the dataset (the 0.4-0.6 and lower half of 0.6-0.8 bands). "
                    "Small penalty to prefer candidates who follow through."
                ),
            },
            {
                "label": "reliable",
                "condition": "0.65 <= value < 0.85",
                "multiplier": 1.00,
                "note": (
                    "Completes most scheduled interviews. No penalty. "
                    "Covers the upper half of the 0.6-0.8 band (~18% of dataset)."
                ),
            },
            {
                "label": "highly_reliable",
                "condition": "value >= 0.85",
                "multiplier": 1.00,
                "note": (
                    "Near-perfect follow-through (~20% of dataset in 0.8-1.0). "
                    "No additional bonus over 'reliable' — the signal saturates. "
                    "Avoids over-rewarding a metric that may reflect low interview volume."
                ),
            },
        ],
    },

    # -------------------------------------------------------------------------
    # Combination guidance (for the scorer — not executable logic)
    # -------------------------------------------------------------------------
    # The four per-signal multiplier components should be combined as a
    # weighted average:
    #   availability_mult = sum(weight_i * mult_i for each signal)
    # The weights above (0.35 + 0.30 + 0.15 + 0.20 = 1.0) are calibrated so
    # that a worst-case candidate (ghost RRR + 9-month inactive + passive + unreliable
    # interview) produces approximately:
    #   0.35*0.50 + 0.30*0.50 + 0.15*0.85 + 0.20*0.60 = 0.175+0.15+0.1275+0.12 = 0.5725
    # and a best-case candidate produces 1.0.
    # This keeps the combined multiplier in the [0.50, 1.00] range the scorer expects.
    # -------------------------------------------------------------------------
    "_combination": {
        "formula": "weighted_average of per-signal multiplier components",
        "weights_sum_check": 0.35 + 0.30 + 0.15 + 0.20,   # must equal 1.0
        "output_range": (0.50, 1.00),
        "worst_case_example": {
            "rrr": 0.50,       "rrr_mult": 0.50,
            "lad": 0.50,       "lad_mult": 0.50,
            "otw": 0.85,       "otw_mult": 0.85,   # passive=0.85 is the floor here
            "icr": 0.60,       "icr_mult": 0.60,
            "combined": round(0.35*0.50 + 0.30*0.50 + 0.15*0.85 + 0.20*0.60, 4),
        },
    },
}


# =============================================================================
# IDEAL CANDIDATE PROFILE SUMMARY
# =============================================================================
# JD: "The 'ideal candidate' we're imagining is roughly..."
# Use as a north-star for calibrating scorer weights, not as a hard filter.
# Key insight from the JD's final note: "We'd rather see 10 great matches
# than 1000 maybes." Score aggressively — precision over recall.
# =============================================================================

IDEAL_CANDIDATE_PROFILE: Dict = {
    "total_experience_years": (6, 8),
    "applied_ml_at_product_company_years": (4, 5),
    "shipped_ranking_or_search_end_to_end": True,
    "has_strong_opinions_on": [
        "hybrid vs dense retrieval",
        "offline vs online evaluation",
        "when to fine-tune vs prompt LLMs",
    ],
    "location_preference": "Noida or Pune (or willing to relocate from Tier-1 city)",
    "platform_activity": (
        "Active on Redrob: open_to_work_flag=True, recent last_active_date, "
        "recruiter_response_rate > 0.5. "
        "JD: Active on Redrob platform (or has clear signal of being in the job market) "
        "so we can actually talk to them."
    ),
    "scoring_philosophy": (
        "JD: We are not expecting to find many matches in a 100K candidate pool. "
        "We are explicitly OK with that -- we would rather see 10 great matches than "
        "1000 maybes. Score aggressively. Let precision beat recall."
    ),
}
