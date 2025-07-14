# summarizer.py
import nltk
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

nltk.download("punkt")

def clean_text(text: str) -> str:
    """Clean extracted text to remove noise and formatting issues."""
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = re.sub(r"[^\w\s.,!?]", "", text)  # remove special chars
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4}", "", text)  # remove dates
    text = re.sub(r"\bPage\s*\d+\b", "", text, flags=re.IGNORECASE)  # remove page numbers
    return text.strip()

def summarize_text(text: str, min_words=120, max_words=150) -> str:
    try:
        # Clean text before processing
        cleaned_text = clean_text(text)
        parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
        summarizer = TextRankSummarizer()

        # Generate 40 ranked sentences
        summary_sentences = summarizer(parser.document, 40)

        result = []
        total_words = 0

        for sentence in summary_sentences:
            sentence_text = str(sentence).strip()

            # Skip too-short or low-quality lines
            if len(sentence_text.split()) < 5:
                continue
            if sentence_text.lower().count("the") > 5:
                continue

            word_count = len(sentence_text.split())
            if total_words + word_count > max_words:
                break

            result.append(sentence_text)
            total_words += word_count

        # Final summary
        final_summary = " ".join(result)

        # If too short, pad with more content (safely)
        if total_words < min_words:
            extra = cleaned_text.split()
            needed = min_words - total_words
            pad = " ".join(extra[:needed])
            final_summary += " " + pad

        return final_summary.strip()

    except Exception as e:
        return f"Summary failed: {e}"