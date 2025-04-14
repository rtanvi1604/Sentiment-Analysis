from transformers import pipeline
classifier = pipeline("sentiment-analysis")

while True:
    text = input("Enter text (type 'quit' to exit): ")
    if text.lower() == "quit":
        break
    result = classifier(text)[0]
    print(f"Label: {result['label']}, Confidence: {result['score']:.2f}")


