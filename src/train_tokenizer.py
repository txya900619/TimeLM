import sentencepiece as spm

if __name__ == "__main__":
    spm.SentencePieceTrainer.train(
        input="data/ldc_tokenizer.txt",
        model_prefix="ldc_cna_char_8000",
        vocab_size=8000,
        character_coverage=1,
        model_type="char",
    )
