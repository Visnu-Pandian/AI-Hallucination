import re
import sys
import os
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

def linguistic_complexity(prompt: str) -> float:
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    if not words:
        return 0.0
    word_count = len(words)
    avg_len = sum(len(w) for w in words) / word_count
    lexical_density = len([w for w in words if w not in COMMON_WORDS]) / word_count
    rarity_count = sum(1 for w in words if len(w) > 10)
    rarity_score = min((rarity_count / word_count) * 100, 10)
    total = (avg_len / 8) * 3 + (lexical_density * 4) + (rarity_score * 0.3)
    return min(total, 10.0)


def calculate_coherence_score(prompt: str) -> float:
    """
    Analyzes 'Logical Coherence' (Does it make sense?).
    """
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    if not words:
        return 0.0
    
    word_count = len(words)
    
    # 1. Repetition (Spam Check)
    unique_ratio = len(set(words)) / word_count
    repetition_score = min(unique_ratio * 1.2, 1.0) * 10
    
    # 2. Natural Language "Glue" Check (Stopword Ratio)
    stopword_count = sum(1 for w in words if w in COMMON_WORDS)
    stopword_ratio = stopword_count / word_count
    
    structure_score = 10.0
    if stopword_ratio < 0.15:
        structure_score = 4.0 
    elif stopword_ratio > 0.60:
        structure_score = 5.0
        
    # 3. Formatting & Gibberish Check
    format_score = 0
    if prompt[0].isupper(): 
        format_score += 5
    if prompt.strip()[-1] in ".!?": 
        format_score += 5
        
    avg_len = sum(len(w) for w in words) / word_count
    if avg_len > 14:
        repetition_score = 0 
        structure_score = 0

    final_coherence = (repetition_score * 0.4) + (structure_score * 0.4) + (format_score * 0.2)
    return min(final_coherence, 10.0)


def calculate_inference_cost(prompt: str) -> float:
    """
    Estimates 'Inference Cost' or 'Thought Power'.
    Detects intent for Deep Reasoning or Large Generation.
    """
    lowered = prompt.lower()
    score = 0.0
    
    # 1. Thought Power (Reasoning Depth)
    # These are expensive because they require logic chains.
    reasoning_hits = sum(1 for t in REASONING_TRIGGERS if t in lowered)
    score += reasoning_hits * 1.5  # High weight for reasoning
    
    # 2. Resource Load (Output Volume)
    # These are expensive because they require generating many tokens.
    generation_hits = sum(1 for t in GENERATION_TRIGGERS if t in lowered)
    score += generation_hits * 0.8  # Moderate weight for volume
    
    # 3. Code Penalty
    # Generating code often burns more "thought" resources than text.
    if "code" in lowered or "function" in lowered or "script" in lowered:
        score += 1.0
        
    # Cap at 10
    return min(score, 10.0)


def calculate_domain_score(lowered_prompt: str) -> tuple[float, list]:
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
    if not prompt.strip():
        return {
            "word_count": 0,
            "detected_domains": [],
            "scores": {
                "length": 0.0, "domain": 0.0, "time": 0.0, 
                "linguistic": 0.0, "coherence": 0.0, "inference": 0.0, "ml_model": 0.0
            },
            "final_complexity": 0.0
        }

    words = prompt.strip().split()
    word_count = len(words)
    lowered = prompt.lower()

    length_score = min(word_count / 12, 10)
    domain_score, detected_domains = calculate_domain_score(lowered)

    time_score = 0
    for phrase, val in TIME_PERIODS.items():
        if phrase in lowered:
            time_score = max(time_score, val)
    time_score = min(time_score * 2, 10)

    ling_score = linguistic_complexity(prompt)
    coherence_score = calculate_coherence_score(prompt)
    inference_score = calculate_inference_cost(prompt)
    ml_score = ml_predict(prompt)
    
    weights = {
        "length": 0.10,      
        "domain": 0.20,
        "time": 0.05,
        "linguistic": 0.15,
        "coherence": 0.15,
        "inference": 0.20,
        "ml": 0.15
    }

    if ml_score == 0.0:
        weights["ml"] = 0
        weights["domain"] += 0.05
        weights["inference"] += 0.05  # Boost inference if ML is missing
        weights["linguistic"] += 0.05

    final_score = (
        weights["length"] * length_score +
        weights["domain"] * domain_score +
        weights["time"] * time_score +
        weights["linguistic"] * ling_score +
        weights["coherence"] * coherence_score +
        weights["inference"] * inference_score +
        weights["ml"] * ml_score
    )

    return {
        "word_count": word_count,
        "detected_domains": detected_domains,
        "scores": {
            "length": round(length_score, 2),
            "domain": round(domain_score, 2),
            "time": round(time_score, 2),
            "linguistic": round(ling_score, 2),
            "coherence": round(coherence_score, 2),
            "inference": round(inference_score, 2),
            "ml_model": round(ml_score, 2)
        },
        "final_complexity": round(final_score, 2)
    }


# =========================================
# SECTION 4: INTERFACE (GUI & CLI)
# =========================================

def run_gui():
    """Launches a modern, dark-themed windowed interface using Tkinter."""
    import tkinter as tk
    from tkinter import scrolledtext, messagebox, font

    # --- Theme Configuration ---
    COLOR_BG_MAIN = "#1e1e1e"       # Dark Charcoal
    COLOR_BG_INPUT = "#2d2d2d"      # Slightly lighter for inputs
    COLOR_FG_TEXT = "#d4d4d4"       # Light Grey (easy on eyes)
    COLOR_ACCENT = "#007acc"        # VS Code Blue
    COLOR_ACCENT_HOVER = "#0062a3"  # Darker Blue
    COLOR_BORDER = "#3e3e42"        # Subtle border color

    def on_analyze():
        prompt = txt_input.get("1.0", tk.END)
        if not prompt.strip():
            messagebox.showwarning("Input Required", "Please enter a prompt to analyze.")
            return
        
        result = assess_complexity(prompt)
        
        # Format Output
        out = f"COMPLEXITY SCORE: {result['final_complexity']} / 10\n"
        out += f"Word Count: {result['word_count']}\n"
        out += f"Domains: {', '.join(result['detected_domains']) if result.get('detected_domains') else 'General'}\n"
        out += "_" * 35 + "\n\n"
        out += "DETAILED BREAKDOWN:\n"
        if 'scores' in result:
            for k, v in result['scores'].items():
                # Custom labels for clarity
                if k == "coherence":
                    label = "LOGIC FLOW"
                elif k == "inference":
                    label = "THOUGHT POWER"
                else:
                    label = k.replace('_', ' ').upper()
                out += f"  • {label:<15} : {v}\n"
        
        lbl_result.config(state=tk.NORMAL)
        lbl_result.delete("1.0", tk.END)
        lbl_result.insert(tk.END, out)
        lbl_result.config(state=tk.DISABLED)

    def on_button_hover(e):
        btn_analyze['bg'] = COLOR_ACCENT_HOVER

    def on_button_leave(e):
        btn_analyze['bg'] = COLOR_ACCENT

    # --- Main Window Setup ---
    root = tk.Tk()
    root.title("Prompt Complexity Analyzer")
    root.geometry("600x650")
    root.configure(bg=COLOR_BG_MAIN)
    
    # Define custom fonts
    font_header = font.Font(family="Segoe UI", size=12, weight="bold")
    font_body = font.Font(family="Segoe UI", size=10)
    font_mono = font.Font(family="Consolas", size=10)

    # 1. Header / Instruction
    header_frame = tk.Frame(root, bg=COLOR_BG_MAIN)
    header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
    
    lbl_instr = tk.Label(
        header_frame, 
        text="Input Prompt:", 
        font=font_header, 
        bg=COLOR_BG_MAIN, 
        fg=COLOR_FG_TEXT,
        anchor="w"
    )
    lbl_instr.pack(fill=tk.X)

    # 2. Input Area
    txt_input = scrolledtext.ScrolledText(
        root, 
        height=8, 
        bg=COLOR_BG_INPUT, 
        fg=COLOR_FG_TEXT,
        insertbackground="white", # Cursor color
        font=font_body,
        relief=tk.FLAT,
        padx=10, pady=10,
        highlightthickness=1,
        highlightbackground=COLOR_BORDER,
        highlightcolor=COLOR_ACCENT
    )
    txt_input.pack(padx=20, pady=(0, 20), fill=tk.BOTH, expand=True)

    # 3. Action Button
    btn_analyze = tk.Button(
        root, 
        text="ANALYZE COMPLEXITY", 
        command=on_analyze, 
        bg=COLOR_ACCENT, 
        fg="white", 
        font=("Segoe UI", 10, "bold"),
        relief=tk.FLAT,
        activebackground=COLOR_ACCENT_HOVER,
        activeforeground="white",
        cursor="hand2",
        pady=8
    )
    btn_analyze.pack(padx=20, fill=tk.X)
    
    # Add hover effects
    btn_analyze.bind("<Enter>", on_button_hover)
    btn_analyze.bind("<Leave>", on_button_leave)

    # 4. Results Area
    lbl_res_title = tk.Label(
        root, 
        text="Analysis Results:", 
        font=font_header, 
        bg=COLOR_BG_MAIN, 
        fg=COLOR_FG_TEXT,
        anchor="w"
    )
    lbl_res_title.pack(padx=20, pady=(20, 5), fill=tk.X)
    
    lbl_result = scrolledtext.ScrolledText(
        root, 
        height=10, 
        state=tk.DISABLED, 
        bg=COLOR_BG_INPUT, 
        fg=COLOR_FG_TEXT,
        font=font_mono, # Monospace for aligned output
        relief=tk.FLAT,
        padx=10, pady=10,
        highlightthickness=1,
        highlightbackground=COLOR_BORDER
    )
    lbl_result.pack(padx=20, pady=(0, 20), fill=tk.BOTH, expand=True)
    
    print("Dark Mode GUI launched.")
    root.mainloop()


def run_cli():
    """Runs the command line interface version."""
    print("--- Prompt Complexity Analyzer (CLI Mode) ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            prompt = input("Enter your prompt: ")
            if prompt.lower() in ['exit', 'quit']:
                break
                
            result = assess_complexity(prompt)
            
            print("\n" + "="*30)
            print(f"FINAL SCORE: {result['final_complexity']} / 10")
            print("="*30)
            print(f"Word Count: {result['word_count']}")
            print(f"Domains: {', '.join(result['detected_domains']) if result.get('detected_domains') else 'General'}")
            print("-" * 20)
            print("Breakdown:")
            if 'scores' in result:
                for k, v in result['scores'].items():
                    print(f"  - {k.capitalize()}: {v}")
            print("\n")
        except EOFError:
            print("\nError: Input stream closed. Try running in GUI mode.")
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    # Check if we are running in a TTY (interactive terminal)
    # VS Code's 'Output' tab usually returns False for isatty()
    if sys.stdout.isatty() and sys.stdin.isatty():
        print("Interactive terminal detected. Starting CLI...")
        run_cli()
    else:
        print("Non-interactive environment detected (e.g., VS Code Output tab).")
        print("Launching GUI mode...")
        try:
            run_gui()
        except Exception as e:
            print(f"Failed to launch GUI: {e}")
            print("Please configure VS Code to 'Run in Terminal' to use CLI mode.")

if __name__ == '__main__':
    main()
    
# Example of high-complexity prompt
# In the far future, the macroeconomics of quantum encryption will likely dictate the geopolitics of interstellar civilization. We must rigorously analyze the stochastic thermodynamics affecting neural network architecture within a post-apocalyptic archipelago. Considering the epistemology of existentialism, the litigation regarding intellectual property in biotechnology and neurology creates profound ambivalence. The renaissance of surrealism in digital typography contrasts sharply with the feudalism of corporate jurisdiction. Inflation of market cap driven by blockchain protocols requires strict compliance with constitutional amendments. The pathology of chronic anxiety in rapid urbanization necessitates therapeutic intervention via genetic modification. Calculus and combinatorics prove the hypothesis that entropy in a closed simulation mirrors the melancholy of ancient monarchies. Furthermore, the asymptotic complexity of recursive algorithms utilized in fiscal policy modeling demonstrates a paradox within utilitarianism. We observe phenomenology overlapping with artistic composition and chiaroscuro techniques, creating a narrative of industrial revolution heritage. Arbitration in international tort law regarding liability for autonomous agents remains a controversial precedent. Spectroscopy analysis of molecular isotopes reveals photosynthesis variations, impacting agricultural dividends and stakeholder equity.
