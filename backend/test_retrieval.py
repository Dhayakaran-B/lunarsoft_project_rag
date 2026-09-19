from retrieve import retrieve


question = input("\nAsk a question about the PDF: ")

results = retrieve(question)

print("\n" + "=" * 70)
print("RETRIEVED INFORMATION")
print("=" * 70)

for i, result in enumerate(results, start=1):

    print(f"\nRESULT {i}")
    print(f"Page: {result['page']}")
    print(f"Distance: {result['distance']:.4f}")

    print("\nText:")
    print(result["text"])

    print("-" * 70)