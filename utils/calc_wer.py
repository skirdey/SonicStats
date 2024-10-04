def compute_wer(reference, hypothesis):
    # Split sentences into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Compute Levenshtein distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    # Compute WER
    wer = d[-1][-1] / len(ref_words)
    
    return wer

# Example usage
reference = "The cat danced gracefully on the windowsill as the sun set behind the hills."
hypothesis = "I lived the night the only thing I could do. Then the star is dead?"
wer = compute_wer(reference, hypothesis)
print(f"Word Error Rate: {wer:.2f}")