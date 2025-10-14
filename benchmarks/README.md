# Benchmarks

This directory contains math problem benchmarks for evaluation.

## Structure

Each benchmark is a JSON file with the following format:

```json
{
  "name": "Benchmark Name",
  "version": "1.0",
  "problems": [
    {
      "id": "unique_id",
      "question": "Problem text",
      "answer": "Expected answer",
      "answer_type": "integer|float|symbolic|string",
      "tolerance": null,
      "difficulty": "easy|medium|hard",
      "tags": ["tag1", "tag2"]
    }
  ]
}
```

## Included Benchmarks

- **gsm8k_subset.json**: 25 problems from GSM8K (grade school math)
- **math_subset.json**: 25 problems from MATH dataset (competition math)

## Adding Custom Benchmarks

1. Create JSON file following the schema above
2. Place in this directory
3. Run: `mathagent run benchmarks/your_benchmark.json`

## Answer Types

- **integer**: Exact integer match
- **float**: Numeric comparison with tolerance (default 1e-6)
- **symbolic**: Sympy equivalence check (e.g., `x^2 - 1` ≡ `(x-1)(x+1)`)
- **string**: Normalized string comparison
