from src.routing.llm_router import LLMRouter

router = LLMRouter()

tests = [
    "What does HTTP 404 mean?",
    "Check the current database health right now",
    "My GPU runs out of memory during training",
    "My API keeps returning HTTP 500 and I need help debugging it",
    "Our production credentials may have been compromised",
]

for text in tests:
    print("=" * 60)
    print("USER QUERY:")
    print(text)

    result = router.route(text)

    print("\nPARSED RESULT:")
    print(result)

    print("=" * 60)
    print()