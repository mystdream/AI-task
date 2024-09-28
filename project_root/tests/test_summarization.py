import pytest
from models.summarization_model import SummarizationModel

@pytest.fixture
def sample_text():
    return "This is a long text. It contains multiple sentences. Some of them are important. Others are not so important. The summarization model should extract the key information."

def test_summarization():
    model = SummarizationModel()
    summary = model.summarize_text(sample_text, sentences_count=2)
    assert isinstance(summary, str)
    assert len(summary.split('.')) <= 2
