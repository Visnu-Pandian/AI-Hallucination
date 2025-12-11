import re
import os
import json
import joblib
from collections import Counter

# Try importing sklearn, but handle cases where it might be missing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =========================================
# SECTION 0: IMPORTING DATA
# =========================================

with open("hallucinations.json", encoding="utf-8") as f:
    hallucinations = [json.loads(line) for line in f if line.strip()]

final_ordered_complexities = []

# =========================================
# SECTION 1: MACHINE LEARNING COMPONENT
# =========================================

# Seed data to bootstrap the model if no external model exists
SEED_PROMPTS = [
    "hi", "hello world", "how are you", "simple text", "the cat sat on the mat",
    "write a story about a dog", "explain 2+2", "what is the weather",
    "analyze the quantum fluctuations of the particle",
    "explain the ontological paradox of existentialism",
    "derive the asymptotic complexity of the recursive algorithm",
    "discuss the geopolitical implications of the industrial revolution",
    "evaluate the thermodynamic entropy in a closed system",
    "critique the chiaroscuro techniques in baroque art",
    "calculate the eigenvector of the matrix using linear algebra"
]

SEED_SCORES = [
    1.0, 1.0, 1.5, 2.0, 2.0,
    3.0, 2.0, 2.5,
    9.0,
    9.5,
    9.0,
    8.5,
    9.0,
    8.5,
    9.0
]

def train_ml_model(training_prompts, training_scores):
    """
    Trains a Linear Regression model using TF-IDF.
    Note: Caller must ensure SKLEARN_AVAILABLE is True.
    """
    print("Training local ML model on seed data...")
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(training_prompts)
    model = LinearRegression()
    model.fit(X, training_scores)
    joblib.dump((vectorizer, model), 'complexity_model.pkl')
    print("Model trained and saved to complexity_model.pkl")


def ml_predict(prompt: str) -> float:
    """
    Predicts complexity using the pre-trained model.
    If the model file is missing, it auto-trains one using SEED_DATA.
    """
    if not SKLEARN_AVAILABLE:
        return 0.0

    model_path = 'complexity_model.pkl'

    # Auto-train if model doesn't exist
    if not os.path.exists(model_path):
        train_ml_model(SEED_PROMPTS, SEED_SCORES)

    try:
        vectorizer, model = joblib.load(model_path)
        X = vectorizer.transform([prompt])
        ml_score = float(model.predict(X)[0])
        return max(0.0, min(10.0, ml_score))
    except Exception as e:
        print(f"ML Prediction Error: {e}")
        return 0.0


# =========================================
# SECTION 2: LEXICAL DOMAINS & KEYWORDS
# =========================================

SCIENTIFIC_TERMS = [
    "quantum", "molecule", "algorithm", "thermodynamics", "relativity", "genome", 
    "spectroscopy", "entropy", "polymer", "photosynthesis", "isotope", "particle", 
    "catalyst", "trajectory", "hypothesis", "theoretical", "simulation"
]

MATHEMATICAL_TERMS = [
    "integral", "derivative", "tensor", "matrix", "eigenvalue", "complexity", 
    "bitwise", "asymptotic", "combinatorics", "probability", "logarithm", 
    "polynomial", "theorem", "stochastic", "vector", "calculus"
]

HISTORICAL_TERMS = [
    "ancient", "empire", "dynasty", "revolution", "medieval", "bronze age", 
    "renaissance", "ottoman", "industrial", "colonial", "monarchy", "feudalism", 
    "crusade", "antiquity", "heritage", "civilization", "archaeology"
]

GEOGRAPHIC_TERMS = [
    "continent", "longitude", "geopolitics", "terrain", "tectonic", "climate", 
    "biome", "archipelago", "ecosystem", "topography", "demographics", "urbanization", 
    "latitude", "migration", "hemisphere", "glacier"
]

COMPUTING_TERMS = [
    "compiler", "architecture", "protocol", "distributed", "encryption", 
    "virtualization", "containerization", "firmware", "kernel", "latency", 
    "bandwidth", "recursion", "neural network", "cybersecurity", "blockchain"
]

PHILOSOPHY_TERMS = [
    "epistemology", "ontology", "metaphysics", "phenomenology", "ethics", 
    "utilitarianism", "existentialism", "dialectics", "paradox", "cognition", 
    "consciousness", "morality", "aesthetic", "rationalism"
]

ART_TERMS = [
    "surrealism", "composition", "chiaroscuro", "perspective", "narrative", 
    "symbolism", "motif", "genre", "protagonist", "allegory", "minimalism", 
    "baroque", "impressionism", "typography", "palette"
]

MEDICAL_TERMS = [
    "pathology", "neurology", "chronic", "diagnosis", "cardiovascular", 
    "anatomy", "physiology", "immunology", "syndrome", "clinical", 
    "therapeutic", "biopsy", "genetic", "metabolism"
]

BUSINESS_TERMS = [
    "equity", "fiscal", "liquidity", "inflation", "stakeholder", 
    "dividend", "macroeconomics", "supply chain", "market cap", "liability", 
    "acquisition", "monetary", "recession", "portfolio"
]

LEGAL_TERMS = [
    "litigation", "jurisdiction", "precedent", "statute", "contract", "tort", 
    "liability", "arbitration", "regulation", "compliance", "constitutional", 
    "amendment", "prosecution", "defendant", "intellectual property"
]

EMOTIONAL_TERMS = [
    "anxiety", "joy", "sadness", "fear", "anger", "empathy", "affection", 
    "grief", "elation", "nostalgia", "melancholy", "euphoria", "resentment", 
    "compassion", "ambivalence"
]

DOMAIN_GROUPS = {
    "Science": SCIENTIFIC_TERMS,
    "Math": MATHEMATICAL_TERMS,
    "History": HISTORICAL_TERMS,
    "Geography": GEOGRAPHIC_TERMS,
    "Computing": COMPUTING_TERMS,
    "Philosophy": PHILOSOPHY_TERMS,
    "Art": ART_TERMS,
    "Medical": MEDICAL_TERMS,
    "Business": BUSINESS_TERMS,
    "Legal": LEGAL_TERMS,
    "Emotional": EMOTIONAL_TERMS
}

TIME_PERIODS = {
    "ancient": 5, "bronze age": 5, "iron age": 4, "medieval": 4,
    "renaissance": 3, "industrial revolution": 3, "enlightenment": 4,
    "17th century": 3, "18th century": 3, "19th century": 2,
    "20th century": 2, "present day": 1, "modern era": 1,
    "future": 3, "far future": 5, "post-apocalyptic": 4,
    "world war": 3, "prehistoric": 4, "victorian": 3
}

# Words that imply heavy "Thought Power" (Multi-step logic)
REASONING_TRIGGERS = [
    "step-by-step", "derive", "prove", "justify", "critique", "evaluate", 
    "chain of thought", "break down", "analyze", "compare", "contrast", 
    "design", "architect", "debug", "solve", "optimize", "find", "optimize",
    "what", "who", "when", "why", "theorize", "solve", "reflect", "calculate",
    "if", "then", "where", "which", "decrypt", "encrypt"
]

# Words that imply heavy "Memory/Resource" (Large Token Output)
GENERATION_TRIGGERS = [
    "essay", "story", "novel", "script", "code", "program", "html", "css", 
    "comprehensive", "detailed", "list all", "table", "curriculum", "chapter", 
    "article", "report", "summary", "extract", "report", "tokenize", "enumerate",
    "json", "pdf", "word", "pptx", "generate", "create", "write", "lab", "png", "jpg"
]

COMMON_WORDS = set("the a an is are was were be to of in that it and on for with as by from this these those which what when where how why who they he she we you i me my mine your yours his him her hers its our ours their theirs".split())

# =========================================
# SECTION 3: COMPLEXITY ALGORITHMS
# =========================================

# Question words that require different reasoning depths
SIMPLE_QUESTION_WORDS = ["what", "who", "where"]  # Often single-hop
COMPLEX_QUESTION_WORDS = ["why", "how", "which"]  # Often multi-hop

# Patterns that indicate multi-hop or comparative reasoning
COMPARISON_PATTERNS = [
    r'\b(more|less|greater|fewer|larger|smaller|longer|shorter|taller|higher|lower)\b',
    r'\b(first|last|earliest|latest|oldest|youngest|newest)\b',
    r'\b(both|either|neither|same|different|similar)\b',
    r'\b(before|after|during|between|since|until)\b',
    r'\b(compare|versus|vs\.?|or)\b',
]

# Patterns suggesting need for specific/precise answers
SPECIFICITY_PATTERNS = [
    r'\b(exactly|precisely|specifically)\b',
    r'\b(how many|how much|how long|how old|how far)\b',
    r'\b(what year|what date|what month|what day)\b',
    r'\b(what number|what percentage|what amount)\b',
    r'\b(name|named|called|titled|known as)\b',
]

# Patterns indicating reasoning chains
REASONING_PATTERNS = [
    r'\b(because|therefore|thus|hence|since|as a result)\b',
    r'\b(if|then|when|while|although|despite)\b',
    r'\b(based on|according to|due to|caused by)\b',
    r'\b(led to|resulted in|contributed to)\b',
]

# Negation and exception patterns (tricky for AI)
NEGATION_PATTERNS = [
    r'\b(not|never|no longer|neither|nor)\b',
    r'\b(except|besides|other than|apart from)\b',
    r'\b(without|lacking|missing)\b',
    r"\b(isn't|aren't|wasn't|weren't|didn't|doesn't|don't|won't|wouldn't|couldn't|shouldn't)\b",
]


def count_named_entities(prompt: str) -> int:
    """
    Estimates named entities by counting capitalized multi-word sequences
    and quoted phrases (titles, names, etc.)
    """
    # Capitalized words not at sentence start
    words = prompt.split()
    entities = 0
    
    for i, word in enumerate(words):
        # Skip first word of sentences
        if i == 0 or (i > 0 and words[i-1][-1] in '.!?'):
            continue
        # Count capitalized words (likely proper nouns)
        if word and word[0].isupper() and word.lower() not in COMMON_WORDS:
            entities += 1
    
    # Count quoted phrases as entities
    quotes = re.findall(r'"[^"]+"|\'[^\']+\'|"[^"]+"|\'[^\']+\'', prompt)
    entities += len(quotes)
    
    return entities


def calculate_reasoning_depth(prompt: str) -> float:
    """
    Measures how many reasoning steps are likely needed.
    Score 0-10 based on question structure.
    """
    lowered = prompt.lower()
    score = 0.0
    
    # 1. Question type analysis
    if any(w in lowered for w in ["why", "how"]):
        score += 2.0  # Explanatory questions need reasoning
    elif any(w in lowered for w in ["which", "what"]):
        score += 1.0
    
    # 2. Comparison/ranking (requires evaluating multiple items)
    for pattern in COMPARISON_PATTERNS:
        matches = len(re.findall(pattern, lowered))
        score += matches * 1.5
    
    # 3. Temporal reasoning
    temporal_words = len(re.findall(r'\b(before|after|first|last|earlier|later|when|during|while|until|since)\b', lowered))
    score += temporal_words * 1.2
    
    # 4. Multi-entity questions (need to track multiple things)
    entity_count = count_named_entities(prompt)
    if entity_count >= 2:
        score += min(entity_count * 0.8, 3.0)
    
    # 5. Possessive chains ("X's Y's Z") indicate relationship tracking
    possessive_chains = len(re.findall(r"'s\s+\w+(?:'s)?", prompt))
    score += possessive_chains * 1.0
    
    return min(score, 10.0)


def calculate_specificity_demand(prompt: str) -> float:
    """
    Measures how specific/precise the expected answer needs to be.
    Vague questions = easier, precise questions = harder.
    """
    lowered = prompt.lower()
    score = 0.0
    
    # 1. Explicit specificity markers
    for pattern in SPECIFICITY_PATTERNS:
        matches = len(re.findall(pattern, lowered))
        score += matches * 1.5
    
    # 2. Questions asking for numbers, dates, measurements
    numeric_asks = len(re.findall(r'\b(how many|how much|how long|how old|how far|what year|what date|in \d{4})\b', lowered))
    score += numeric_asks * 2.0
    
    # 3. Questions about specific attributes
    attribute_asks = len(re.findall(r'\b(what (is|was) the (name|title|location|city|country|year|date|number))\b', lowered))
    score += attribute_asks * 1.5
    
    # 4. "The" + specific noun (looking for THE answer, not A answer)
    the_specific = len(re.findall(r'\bthe\s+(name|title|year|date|city|country|author|director|founder|capital)\b', lowered))
    score += the_specific * 1.0
    
    return min(score, 10.0)


def calculate_ambiguity_traps(prompt: str) -> float:
    """
    Measures elements that can trip up an AI:
    - Negations, exceptions, conditionals
    - Similar-sounding entities
    - Tricky phrasing
    """
    lowered = prompt.lower()
    score = 0.0
    
    # 1. Negation patterns (AI often misses negations)
    for pattern in NEGATION_PATTERNS:
        matches = len(re.findall(pattern, lowered))
        score += matches * 2.0
    
    # 2. Conditional phrasing
    conditionals = len(re.findall(r'\b(if|when|while|although|despite|unless|except)\b', lowered))
    score += conditionals * 1.0
    
    # 3. Questions with "or" (binary choice - easy to get wrong)
    or_choices = len(re.findall(r'\b\w+\s+or\s+\w+', lowered))
    score += or_choices * 1.5
    
    # 4. Double negatives or complex logic
    if re.search(r'\b(not|never|no)\b.*\b(not|never|no|without)\b', lowered):
        score += 2.0
    
    # 5. Embedded clauses (commas within question suggest complexity)
    comma_count = prompt.count(',')
    score += min(comma_count * 0.5, 2.0)
    
    return min(score, 10.0)


def calculate_knowledge_obscurity(prompt: str) -> float:
    """
    Estimates how obscure/niche the required knowledge is.
    Common knowledge = easy, specialized = hard.
    """
    lowered = prompt.lower()
    score = 0.0
    
    # 1. Named entities suggest specific knowledge needed
    entity_count = count_named_entities(prompt)
    score += min(entity_count * 0.6, 3.0)
    
    # 2. Long proper nouns (full names, titles) = more specific
    long_caps = len(re.findall(r'\b[A-Z][a-z]{8,}\b', prompt))
    score += min(long_caps * 0.5, 2.0)
    
    # 3. Quoted titles suggest specific works
    quotes = len(re.findall(r'"[^"]+"|\'[^\']+\'', prompt))
    score += quotes * 1.0
    
    # 4. Questions about relationships between entities
    relationship_words = len(re.findall(r'\b(married|spouse|wife|husband|father|mother|son|daughter|sibling|brother|sister|founded|created|wrote|directed|starring|played|born|died)\b', lowered))
    score += min(relationship_words * 0.8, 2.5)
    
    # 5. Domain-specific terminology (from existing lists)
    domain_score, _ = calculate_domain_score(lowered)
    score += domain_score * 0.3
    
    return min(score, 10.0)


def calculate_question_length_complexity(prompt: str) -> float:
    """
    Longer questions often embed more constraints and context.
    But normalize to avoid over-weighting.
    """
    word_count = len(prompt.split())
    
    # Short questions (< 8 words) can still be hard, so floor at 1
    # Long questions (> 30 words) likely have many constraints
    if word_count <= 5:
        return 1.0
    elif word_count <= 10:
        return 2.0 + (word_count - 5) * 0.3
    elif word_count <= 20:
        return 3.5 + (word_count - 10) * 0.25
    elif word_count <= 35:
        return 6.0 + (word_count - 20) * 0.15
    else:
        return min(8.25 + (word_count - 35) * 0.05, 10.0)


def calculate_domain_score(lowered_prompt: str) -> tuple[float, list]:
    """Domain-specific vocabulary detection."""
    domain_hits = 0
    detected_domains = []
    for domain_name, terms in DOMAIN_GROUPS.items():
        hits = sum(1 for t in terms if t in lowered_prompt)
        if hits > 0:
            detected_domains.append(domain_name)
            domain_hits += min(hits, 3)
    diversity_bonus = len(detected_domains) * 0.5
    final_domain_score = min(domain_hits + diversity_bonus, 10.0)
    return final_domain_score, detected_domains


def assess_complexity(prompt: str) -> dict:
    """
    Main complexity assessment function.
    Focuses on cognitive load for AI: reasoning depth, specificity, traps.
    """
    if not prompt.strip():
        return {
            "word_count": 0,
            "detected_domains": [],
            "scores": {
                "reasoning_depth": 0.0, 
                "knowledge_obscurity": 0.0, 
                "question_length": 0.0,
                "ml_model": 0.0
            },
            "final_complexity": 0.0
        }

    words = prompt.strip().split()
    word_count = len(words)
    lowered = prompt.lower()

    # Calculate each complexity dimension
    reasoning_score = calculate_reasoning_depth(prompt)
    obscurity_score = calculate_knowledge_obscurity(prompt)
    length_score = calculate_question_length_complexity(prompt)
    ml_score = ml_predict(prompt)
    
    # Get domains for reference
    _, detected_domains = calculate_domain_score(lowered)
    
    # Weights focused on AI difficulty
    weights = {
        "reasoning": 0.30,      # Multi-hop reasoning is hard
        "obscurity": 0.15,      # Niche knowledge is harder
        "length": 0.15,         # More constraints = harder
        "ml": 0.40              # ML model's assessment (highest weight)
    }

    if ml_score == 0.0:
        weights["ml"] = 0
        weights["reasoning"] += 0.25
        weights["length"] += 0.15

    final_score = (
        weights["reasoning"] * reasoning_score +
        weights["obscurity"] * obscurity_score +
        weights["length"] * length_score +
        weights["ml"] * ml_score
    )

    return {
        "word_count": word_count,
        "detected_domains": detected_domains,
        "scores": {
            "reasoning_depth": round(reasoning_score, 2),
            "knowledge_obscurity": round(obscurity_score, 2),
            "question_length": round(length_score, 2),
            "ml_model": round(ml_score, 2)
        },
        "final_complexity": round(ml_score, 2) # temp change
    }


# =========================================
# SECTION 4: PROCESS HALLUCINATIONS
# =========================================

def format_justification(scores: dict) -> str:
    """
    Formats all sub-scores into a justification string.
    """
    parts = [
        f"reasoning_depth={scores['reasoning_depth']}",
        f"knowledge_obscurity={scores['knowledge_obscurity']}",
        f"question_length={scores['question_length']}",
        f"ml_model={scores['ml_model']}"
    ]
    return "Scores: " + ", ".join(parts)


def process_hallucinations():
    """
    Processes each hallucination from the loaded data.
    Extracts the 'question' field as the prompt, computes complexity,
    and outputs complexity + justification (list of all sub-scores).
    """
    print(f"Processing {len(hallucinations)} hallucinations...")
    
    for index, entry in enumerate(hallucinations, start=1):
        prompt = entry.get("question", "")
        
        if not prompt.strip():
            final_ordered_complexities.append({
                "complexity": 0,
                "justification": "Empty prompt"
            })
            continue
        
        result = assess_complexity(prompt)
        
        complexity_entry = {
            "complexity": result["final_complexity"],
            "justification": format_justification(result["scores"])
        }
        
        final_ordered_complexities.append(complexity_entry)
        
        if index % 50 == 0 or index == len(hallucinations):
            print(f"   [OK] Processed {index}/{len(hallucinations)}")
    
    # Save results to JSON
    with open("prompt_complexities.json", "w", encoding="utf-8") as f:
        json.dump({"complexities": final_ordered_complexities}, f, indent=2, ensure_ascii=False)
    
    print(f"\nDONE - Complexities saved to prompt_complexities.json")


if __name__ == '__main__':
    process_hallucinations()
