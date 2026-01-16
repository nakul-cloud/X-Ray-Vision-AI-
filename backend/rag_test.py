from rag_answer import answer_with_rag

print("=== RAG TEST START ===")

# Test queries
queries = [
    "Explain contrast improvement in X-ray enhancement",
    "What does high entropy in an X-ray image indicate?",
    "How does noise affect gradient variance?",
    "What does symmetry difference measure in an X-ray?",
    "Explain Otsu segmentation thresholding",
]

for q in queries:
    print("\n----------------------------------------")
    print(f"QUERY: {q}")
    print("ANSWER:")
    
    try:
        response = answer_with_rag(q, debug=True)
        print(response)
    except Exception as e:
        print("❌ ERROR:", e)

print("\n=== RAG TEST COMPLETE ===")
