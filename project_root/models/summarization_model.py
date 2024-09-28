from transformers import pipeline

def load_summarization_model():
    # Load a pre-trained summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def summarize_text(summarizer, text):
    # Generate a summary of the input text
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']