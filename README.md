# LLM in context bias
Evaluating LLM performance for mis-matched prompts

# Tasks
- [ ] Verify that model performance noticeably improves as the number of in-context examples increase
- [ ] Verify that the model performance is degraded by the number of out-of-context examples

---
## Results

### Results for incontext examples
- Note that the 

| Ex  | rt    | gigaword |
| --- | ----- | -------- |
| 0   |       | 91.28    |
| 2   |       | 91.74    |
| 4   |       | 92.12    |
| 6   |       | 91.93    |
| 8   |       | 92.21    |
| 10  | 91.84 | 92.31    |


### Results for incontext examples that are *not* shuffles

| Ex  | rt    | gigaword | snli | boolq |
| --- | ----- | -------- | ---- | ----- |
| 0   | 92.11 |
| 1   | 91.65 |
| 2   | 92.03 |


### Results for in context examples that *are* shuffled each iteration
| Ex  | rt    | gigaword |
| --- | ----- | -------- |
| 0   | 91.81 | 91.92    |
| 1   | 91.77 | 92.48    |
| 2   |       |

---
## Datasets

[Gigaword](https://huggingface.co/datasets/gigaword)
- ROUGE metric (see the huggingface website for the metric)

---
## Code Setup

To run GPT3.5, an openAI API key is required. Specify this in the `.env` file such as:

```.env
# llm-in-context/.env
OPENAI_API_KEY=mykey
```



