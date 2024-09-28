from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

class SummarizationModel:
    def __init__(self):
        self.summarizer = LsaSummarizer()

    def summarize_text(self, text, sentences_count=3):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.summarizer(parser.document, sentences_count)
        return " ".join([str(sentence) for sentence in summary])
