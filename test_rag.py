import argparse
from query_data import query_rag
from langchain_ollama import ChatOllama
from typing import Optional

EVAL_PROMPT = """
Expected key information: {expected_response}
Actual response: {actual_response}
---
Task: Determine if the actual response contains the expected key information.

Answer 'true' if the actual response contains the expected information (even if phrased differently).
Answer 'false' if:
- The actual response contradicts the expected information
- The actual response is missing key information from the expected response

When answering 'false', explain what the actual response says instead of the expected information.

Answer (true/false):
"""

EVAL_MODEL = "mistral:7b"

def query_and_validate(
    question: str,
    expected_response: str,
    model: str = "llama3:8b",
    hybrid: bool = False,
    hybrid_weight: float = 0.7,
    mode_name: str = "vector",
    eval_model: Optional[str] = None
):
    """
    Query the RAG system and validate the response using LLM evaluation.
    
    Args:
        question: The question to ask
        expected_response: Expected answer or key information
        model: Ollama model to use for answering (default: llama3:8b)
        hybrid: Enable hybrid search
        hybrid_weight: Weight for hybrid search (0.0 = BM25-only, 1.0 = vector-only)
        mode_name: Name of the retrieval mode for display
        eval_model: Ollama model to use for evaluation (default: uses global EVAL_MODEL)
    
    Returns:
        bool: True if response matches expected, False otherwise
    """
    if eval_model is None:
        eval_model = EVAL_MODEL
    
    # Type narrowing: eval_model is guaranteed to be str at this point
    assert eval_model is not None
    
    print(f"\n[{mode_name.upper()}] Question: {question}")
    print(f"[{mode_name.upper()}] Expected: {expected_response}")
    
    response_text = query_rag(
        question=question,
        model=model,
        hybrid=hybrid,
        hybrid_weight=hybrid_weight
    )
    
    print(f"[{mode_name.upper()}] Actual: {response_text}")
    
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text
    )

    evaluator = ChatOllama(model=eval_model, temperature=0)
    evaluation_result = evaluator.invoke(prompt)
    # Extract content as string (handle both string and list responses)
    content = evaluation_result.content
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    else:
        content = str(content)
    evaluation_cleaned = content.strip().lower()

    if "true" in evaluation_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"[{mode_name.upper()}] ✓ PASS: {evaluation_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"[{mode_name.upper()}] ✗ FAIL: {evaluation_cleaned}" + "\033[0m")
        return False
    else:
        print("\033[93m" + f"[{mode_name.upper()}] ? UNCERTAIN: {evaluation_cleaned}" + "\033[0m")
        return False


# Anime Domain Tests

def test_naruto_character_vector():
    """Test: Who is Naruto? (Vector-only)"""
    assert query_and_validate(
        question="Who is Naruto?",
        expected_response="Naruto is a ninja and the main character of the Naruto series",
        mode_name="vector"
    )


def test_naruto_character_bm25():
    """Test: Who is Naruto? (BM25-only)"""
    assert query_and_validate(
        question="Who is Naruto?",
        expected_response="Naruto is a ninja and the main character of the Naruto series",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25"
    )


def test_naruto_character_hybrid():
    """Test: Who is Naruto? (Hybrid)"""
    assert query_and_validate(
        question="Who is Naruto?",
        expected_response="Naruto is a ninja and the main character of the Naruto series",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_one_piece_break_vector():
    """Test: When does One Piece take breaks? (Vector-only)"""
    assert query_and_validate(
        question="When is One Piece (the TV series) doing a break?",
        expected_response="One Piece takes breaks or hiatus information",
        mode_name="vector"
    )


def test_anime_definition_hybrid():
    """Test: What is anime? (Hybrid)"""
    assert query_and_validate(
        question="What is anime?",
        expected_response="Anime is a style of animation originating from Japan",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


# Technical Domain Tests

def test_python_language_vector():
    """Test: What is Python? (Vector-only)"""
    assert query_and_validate(
        question="What is Python?",
        expected_response="Python is a programming language",
        mode_name="vector"
    )


def test_python_language_bm25():
    """Test: What is Python? (BM25-only)"""
    assert query_and_validate(
        question="What is Python?",
        expected_response="Python is a programming language",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25"
    )


def test_python_language_hybrid():
    """Test: What is Python? (Hybrid)"""
    assert query_and_validate(
        question="What is Python?",
        expected_response="Python is a programming language",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_machine_learning_definition_hybrid():
    """Test: What is machine learning? (Hybrid)"""
    assert query_and_validate(
        question="What is machine learning?",
        expected_response="Machine learning is a subset of artificial intelligence",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_sql_purpose_bm25():
    """Test: What is SQL used for? (BM25-only - keyword heavy)"""
    assert query_and_validate(
        question="What is SQL used for?",
        expected_response="SQL is used for managing and querying databases",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25"
    )


def test_rest_api_definition_hybrid():
    """Test: What is a REST API? (Hybrid)"""
    assert query_and_validate(
        question="What is a REST API?",
        expected_response="REST API is an architectural style for web services",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


# Science Domain Tests

def test_quantum_computing_definition_vector():
    """Test: What is quantum computing? (Vector-only)"""
    assert query_and_validate(
        question="What is quantum computing?",
        expected_response="Quantum computing uses quantum mechanical phenomena",
        mode_name="vector"
    )


def test_photosynthesis_process_hybrid():
    """Test: How does photosynthesis work? (Hybrid)"""
    assert query_and_validate(
        question="How does photosynthesis work?",
        expected_response="Photosynthesis is the process by which plants convert light into energy",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_dna_structure_bm25():
    """Test: What is DNA? (BM25-only)"""
    assert query_and_validate(
        question="What is DNA?",
        expected_response="DNA is deoxyribonucleic acid, the genetic material",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25"
    )


def test_evolution_theory_hybrid():
    """Test: What is evolution? (Hybrid)"""
    assert query_and_validate(
        question="What is evolution?",
        expected_response="Evolution is the process of change in species over time",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


# History & Geography Domain Tests

def test_ww2_when_vector():
    """Test: When did World War II happen? (Vector-only)"""
    assert query_and_validate(
        question="When did World War II happen?",
        expected_response="World War II occurred in the 1940s or 1939-1945",
        mode_name="vector"
    )


def test_ancient_egypt_hybrid():
    """Test: What was Ancient Egypt? (Hybrid)"""
    assert query_and_validate(
        question="What was Ancient Egypt?",
        expected_response="Ancient Egypt was an ancient civilization in North Africa",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_mount_everest_height_bm25():
    """Test: How tall is Mount Everest? (BM25-only - specific fact)"""
    assert query_and_validate(
        question="How tall is Mount Everest?",
        expected_response="Mount Everest is approximately 8848 meters or 29029 feet",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25"
    )


# Technology Domain Tests

def test_blockchain_definition_hybrid():
    """Test: What is blockchain? (Hybrid)"""
    assert query_and_validate(
        question="What is blockchain?",
        expected_response="Blockchain is a distributed ledger technology",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_linux_operating_system_bm25():
    """Test: What is Linux? (BM25-only)"""
    assert query_and_validate(
        question="What is Linux?",
        expected_response="Linux is an open-source operating system",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25"
    )


def test_neural_network_definition_vector():
    """Test: What is a neural network? (Vector-only)"""
    assert query_and_validate(
        question="What is a neural network?",
        expected_response="Neural network is a computing system inspired by biological neural networks",
        mode_name="vector"
    )


# Additional Tests with Detailed Expected Responses

def test_python_features_detailed_hybrid():
    """Test: What are the key features of Python? (Hybrid - detailed)"""
    assert query_and_validate(
        question="What are the key features and characteristics of Python programming language?",
        expected_response="Python is a high-level, interpreted programming language known for its simple syntax, readability, and versatility. Key features include dynamic typing, automatic memory management, extensive standard library, support for multiple programming paradigms (object-oriented, functional, procedural), and strong community support. Python is widely used in web development, data science, artificial intelligence, automation, and scientific computing.",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_machine_learning_applications_detailed_vector():
    """Test: What are applications of machine learning? (Vector-only - detailed)"""
    assert query_and_validate(
        question="What are some real-world applications of machine learning?",
        expected_response="Machine learning has numerous applications across various industries including recommendation systems used by Netflix and Amazon, image and speech recognition in smartphones, natural language processing for chatbots and translation services, fraud detection in banking, medical diagnosis and drug discovery in healthcare, autonomous vehicles, predictive maintenance in manufacturing, and personalized marketing. These applications leverage algorithms that learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
        mode_name="vector"
    )


def test_quantum_computing_advantages_detailed_hybrid():
    """Test: What are advantages of quantum computing? (Hybrid - detailed)"""
    assert query_and_validate(
        question="What are the main advantages and potential applications of quantum computing?",
        expected_response="Quantum computing offers significant advantages over classical computing for certain problems, including the ability to perform parallel computations through quantum superposition, exponential speedup for specific algorithms like factorization and database search, and potential breakthroughs in cryptography, drug discovery, financial modeling, and optimization problems. Quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously, enabling them to process vast amounts of information more efficiently than classical computers for particular computational tasks.",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_photosynthesis_steps_detailed_bm25():
    """Test: What are the steps of photosynthesis? (BM25-only - detailed)"""
    assert query_and_validate(
        question="What are the main steps and processes involved in photosynthesis?",
        expected_response="Photosynthesis is a complex biochemical process where plants, algae, and some bacteria convert light energy into chemical energy. The process occurs in two main stages: the light-dependent reactions (light reactions) that take place in the thylakoid membranes and capture light energy to produce ATP and NADPH, and the light-independent reactions (Calvin cycle) that occur in the stroma and use ATP and NADPH to convert carbon dioxide into glucose. Chlorophyll, the green pigment in chloroplasts, plays a crucial role in absorbing light energy. The overall equation involves carbon dioxide, water, and light producing glucose and oxygen.",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25"
    )


def test_evolution_mechanisms_detailed_hybrid():
    """Test: What are the mechanisms of evolution? (Hybrid - detailed)"""
    assert query_and_validate(
        question="What are the main mechanisms that drive evolution?",
        expected_response="Evolution is driven by several key mechanisms including natural selection, where organisms with advantageous traits are more likely to survive and reproduce; genetic mutation, which introduces new genetic variations; genetic drift, the random change in allele frequencies in small populations; gene flow, the transfer of genetic material between populations through migration; and sexual selection, where certain traits increase mating success. These mechanisms work together over generations to cause changes in the characteristics of populations, leading to the diversity of life forms we observe today and the adaptation of species to their environments.",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_ww2_causes_detailed_vector():
    """Test: What caused World War II? (Vector-only - detailed)"""
    assert query_and_validate(
        question="What were the main causes and events that led to World War II?",
        expected_response="World War II was caused by multiple interconnected factors including the harsh terms of the Treaty of Versailles that ended World War I, which created economic hardship and resentment in Germany; the rise of fascist and militaristic regimes in Germany, Italy, and Japan; aggressive expansionist policies and territorial conquests; the failure of the League of Nations to prevent aggression; economic depression and instability in the 1930s; and the policy of appeasement by Western powers. Key events leading to the war included Germany's remilitarization of the Rhineland, the Anschluss with Austria, the Munich Agreement, and the invasion of Poland in 1939, which triggered the war in Europe.",
        mode_name="vector"
    )


def test_ancient_egypt_civilization_detailed_hybrid():
    """Test: What made Ancient Egypt significant? (Hybrid - detailed)"""
    assert query_and_validate(
        question="What were the key achievements and characteristics of Ancient Egyptian civilization?",
        expected_response="Ancient Egypt was one of the world's earliest and most influential civilizations, lasting for over 3000 years along the Nile River in North Africa. Key achievements include the construction of massive pyramids and temples, the development of hieroglyphic writing, advances in mathematics and astronomy, sophisticated irrigation systems for agriculture, mummification and medical knowledge, and a complex religious and political system centered around pharaohs. The civilization was known for its stability, architectural marvels like the Great Pyramid of Giza, the Sphinx, and extensive trade networks. Egyptian culture had a profound influence on later civilizations in the Mediterranean region.",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_blockchain_technology_detailed_bm25():
    """Test: How does blockchain work? (BM25-only - detailed)"""
    assert query_and_validate(
        question="How does blockchain technology work and what are its key components?",
        expected_response="Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) that are linked and secured using cryptography. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. Key features include decentralization (no central authority), immutability (records cannot be altered retroactively), transparency (all participants can view the ledger), and consensus mechanisms (like proof-of-work or proof-of-stake) that validate transactions. Blockchain enables secure, peer-to-peer transactions without intermediaries, making it the foundation for cryptocurrencies like Bitcoin and Ethereum, as well as applications in supply chain management, smart contracts, and digital identity verification.",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25"
    )


def test_linux_history_detailed_hybrid():
    """Test: What is the history of Linux? (Hybrid - detailed)"""
    assert query_and_validate(
        question="What is the history and development of the Linux operating system?",
        expected_response="Linux is an open-source, Unix-like operating system kernel originally created by Linus Torvalds in 1991 as a free alternative to proprietary Unix systems. It was developed as a hobby project and released under the GNU General Public License, allowing anyone to use, modify, and distribute it freely. Linux has since grown into a major operating system powering servers, supercomputers, embedded systems, and Android devices. The system is characterized by its modularity, security, stability, and the collaborative development model involving thousands of developers worldwide. Major Linux distributions include Ubuntu, Debian, Red Hat, and Fedora, each with different package management systems and user interfaces.",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_dna_function_detailed_vector():
    """Test: What is the function of DNA? (Vector-only - detailed)"""
    assert query_and_validate(
        question="What is the structure and function of DNA in living organisms?",
        expected_response="DNA (deoxyribonucleic acid) is the molecule that carries genetic information in all living organisms. It has a double-helix structure discovered by Watson and Crick, composed of two strands of nucleotides containing four bases: adenine (A), thymine (T), cytosine (C), and guanine (G). DNA's primary functions include storing genetic information that determines an organism's traits, replicating itself during cell division to pass genetic information to offspring, and providing instructions for protein synthesis through the processes of transcription and translation. The sequence of bases in DNA encodes the information needed to build and maintain an organism, and mutations in DNA can lead to genetic variations and evolution.",
        mode_name="vector"
    )


def test_rest_api_principles_detailed_hybrid():
    """Test: What are REST API principles? (Hybrid - detailed)"""
    assert query_and_validate(
        question="What are the core principles and characteristics of REST API design?",
        expected_response="REST (Representational State Transfer) is an architectural style for designing web services with several key principles: stateless communication where each request contains all information needed to process it, resource-based URLs that identify resources rather than actions, standard HTTP methods (GET, POST, PUT, DELETE) for operations, representation of resources in formats like JSON or XML, and a uniform interface for interaction. REST APIs are designed to be scalable, simple, and work well with HTTP. They enable communication between different systems over the internet, allowing applications to access and manipulate web resources using standard protocols. RESTful APIs are widely used in web development, mobile applications, and microservices architectures.",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


def test_climate_change_effects_detailed_bm25():
    """Test: What are effects of climate change? (BM25-only - detailed)"""
    assert query_and_validate(
        question="What are the main effects and impacts of climate change on the environment?",
        expected_response="Climate change has widespread and severe effects on the environment including rising global temperatures leading to melting ice caps and glaciers, sea level rise threatening coastal communities, increased frequency and intensity of extreme weather events like hurricanes and droughts, changes in precipitation patterns affecting agriculture and water availability, ocean acidification harming marine ecosystems, loss of biodiversity as species struggle to adapt, shifts in ecosystems and habitats, and increased risk of wildfires. These changes are primarily driven by human activities that increase greenhouse gas concentrations in the atmosphere, particularly carbon dioxide from burning fossil fuels. The impacts are already being observed worldwide and are projected to intensify without significant mitigation efforts.",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25"
    )


def test_anime_culture_detailed_hybrid():
    """Test: What is anime's cultural impact? (Hybrid - detailed)"""
    assert query_and_validate(
        question="What is the cultural significance and global impact of anime?",
        expected_response="Anime, Japanese animation, has had a profound cultural impact globally, becoming a major export of Japanese culture and influencing entertainment, art, and fashion worldwide. Anime has introduced audiences to Japanese storytelling, aesthetics, and values, creating a bridge between cultures. It has influenced Western animation styles, video games, and filmmaking. The medium covers diverse genres from action and fantasy to slice-of-life and educational content, appealing to audiences of all ages. Anime conventions, cosplay culture, and fan communities have grown internationally, contributing to cultural exchange and tourism. The industry has also driven technological innovation in animation techniques and has become a significant economic force, with merchandise, streaming services, and international licensing generating substantial revenue.",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid"
    )


# Negative Test Cases: These tests should FAIL

def test_naruto_wrong_character_negative():
    """Negative Test: Wrong character description (should fail)"""
    assert not query_and_validate(
        question="Who is Naruto?",
        expected_response="Naruto is a pirate and the main character of the One Piece series",
        mode_name="vector_negative"
    )


def test_python_wrong_type_negative():
    """Negative Test: Wrong description of Python (should fail)"""
    assert not query_and_validate(
        question="What is Python?",
        expected_response="Python is a compiled programming language that requires explicit memory management and uses curly braces for code blocks",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid_negative"
    )


def test_ww2_wrong_dates_negative():
    """Negative Test: Wrong dates for World War II (should fail)"""
    assert not query_and_validate(
        question="When did World War II happen?",
        expected_response="World War II occurred in the 1960s from 1965 to 1970",
        mode_name="vector_negative"
    )


def test_quantum_computing_wrong_concept_negative():
    """Negative Test: Wrong description of quantum computing (should fail)"""
    assert not query_and_validate(
        question="What is quantum computing?",
        expected_response="Quantum computing uses classical bits that can only exist in one state at a time, making it slower than traditional computers",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid_negative"
    )


def test_dna_wrong_structure_negative():
    """Negative Test: Wrong DNA structure description (should fail)"""
    assert not query_and_validate(
        question="What is DNA?",
        expected_response="DNA is a single-stranded molecule with a linear structure that contains only two bases: adenine and guanine",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25_negative"
    )


def test_linux_wrong_creator_negative():
    """Negative Test: Wrong creator of Linux (should fail)"""
    assert not query_and_validate(
        question="What is the history and development of the Linux operating system?",
        expected_response="Linux was created by Bill Gates in 1985 as a proprietary operating system for Microsoft",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid_negative"
    )


def test_photosynthesis_wrong_process_negative():
    """Negative Test: Wrong description of photosynthesis (should fail)"""
    assert not query_and_validate(
        question="How does photosynthesis work?",
        expected_response="Photosynthesis is a process where animals convert oxygen into carbon dioxide using heat energy from the sun",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid_negative"
    )


def test_blockchain_wrong_feature_negative():
    """Negative Test: Wrong blockchain characteristics (should fail)"""
    assert not query_and_validate(
        question="How does blockchain technology work and what are its key components?",
        expected_response="Blockchain is a centralized database technology controlled by a single authority that allows easy modification of historical records",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25_negative"
    )


def test_evolution_wrong_mechanism_negative():
    """Negative Test: Wrong evolution mechanism (should fail)"""
    assert not query_and_validate(
        question="What are the main mechanisms that drive evolution?",
        expected_response="Evolution is driven solely by intelligent design, where a creator directly modifies species to adapt to their environment",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid_negative"
    )


def test_machine_learning_wrong_definition_negative():
    """Negative Test: Wrong machine learning definition (should fail)"""
    assert not query_and_validate(
        question="What is machine learning?",
        expected_response="Machine learning is a subset of database management that requires explicit programming for every decision",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid_negative"
    )


def test_ancient_egypt_wrong_location_negative():
    """Negative Test: Wrong location for Ancient Egypt (should fail)"""
    assert not query_and_validate(
        question="What were the key achievements and characteristics of Ancient Egyptian civilization?",
        expected_response="Ancient Egypt was a civilization located in South America along the Amazon River that lasted for only 100 years",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid_negative"
    )


def test_mount_everest_wrong_height_negative():
    """Negative Test: Wrong height for Mount Everest (should fail)"""
    assert not query_and_validate(
        question="How tall is Mount Everest?",
        expected_response="Mount Everest is approximately 5000 meters or 16404 feet tall",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25_negative"
    )


def test_rest_api_wrong_principles_negative():
    """Negative Test: Wrong REST API principles (should fail)"""
    assert not query_and_validate(
        question="What are the core principles and characteristics of REST API design?",
        expected_response="REST APIs require stateful communication where each request depends on previous requests, use action-based URLs like /getUser or /deleteItem, and only support the POST method for all operations",
        hybrid=True,
        hybrid_weight=0.7,
        mode_name="hybrid_negative"
    )


def test_climate_change_wrong_cause_negative():
    """Negative Test: Wrong cause of climate change (should fail)"""
    assert not query_and_validate(
        question="What are the main effects and impacts of climate change on the environment?",
        expected_response="Climate change is caused by natural processes only, with no human contribution, and results in global cooling and shrinking ice caps",
        hybrid=True,
        hybrid_weight=0.0,
        mode_name="bm25_negative"
    )


# Runs all tests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG system with various retrieval modes")
    parser.add_argument(
        "--eval-model",
        type=str,
        default="mistral:7b",
        help="Ollama model to use for evaluation (default: mistral:7b)"
    )
    args = parser.parse_args()
    
    def set_eval_model(value: str) -> None:
        global EVAL_MODEL
        EVAL_MODEL = value
    
    set_eval_model(args.eval_model)
    
    print(f"\nAnswering model: llama3:8b")
    print(f"Evaluation model: {EVAL_MODEL}")
    print("Modes: Vector-only, BM25-only (hybrid_weight=0.0), Hybrid (hybrid_weight=0.7)")
    print("=" * 80)
    
    # Get all test functions
    test_functions = [name for name in dir() if name.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for test_name in test_functions:
        test_func = globals()[test_name]
        try:
            result = test_func()
            # If function returns None (assert passed), count as passed
            # If function returns False explicitly, count as failed
            if result is False:
                failed += 1
            else:
                passed += 1
        except AssertionError:
            # Assert failed means test failed
            failed += 1
        except Exception as e:
            print(f"\n\033[91mERROR in {test_name}: {str(e)}\033[0m")
            failed += 1
    
    print("TEST SUMMARY:")
    print(f"Total tests: {passed + failed}")
    print(f"\033[92mPassed: {passed}\033[0m")
    print(f"\033[91mFailed: {failed}\033[0m")