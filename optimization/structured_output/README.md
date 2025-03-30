# Structured outputs

There are three main techniques to get LLMs to return json objects (also known as structured outputs):

1. Built-in support: Some large language models like OpenAI or Gemini models support structured outputs
   out of the box - No extra steps necessary
2. Using tools: Many models support calling tools, this can be leveraged as tools require arguments to be
   specified using a JSON schema
3. Prompt engineering: By specifying the expected output schema in the system or developer prompt, we can
   push the model to return data in the right format

In this guide, we will focus on how to best utilize prompt engineering techniques to reliably get structured
outputs from models that don't support it out of the box or don't support tool calls.

## Current performance

We are going to test 10 different models on a Entity Extraction task, the goal will not be to assess the quality
of the responses but rather ensure that the LLM returns a valid JSON response.

We will be evaluating the following models available on Open-Router:

1. Gemma 3 - gemma-3-4b-it
2. DeepSeek R1 - deepseek/deepseek-r1
3. Llama 3.3 - 70B - meta-llama/llama-3.3-70b-instruct
4. Llama 3.2 - 3B - meta-llama/llama-3.2-3b-instruct
5. Llama 3.1 8B - perplexity/llama-3.1-sonar-small-128k-online
6. phi4 - microsoft/phi-4
8. Mistal - 7B - mistralai/mistral-7b-instruct
9. Qwen 2.5 - qwen/qwen2.5-32b-instruct
10. Command-a - cohere/command-a

We are going to run the evaluation based on the first 50 items in the [Synthetic PII dataset](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual/viewer/default/test). The goal is to extract the PII in the dataset that follows the following format:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "start": {
        "type": "integer",
        "minimum": 0
      },
      "end": {
        "type": "integer",
        "minimum": 0
      },
      "label": {
        "type": "string",
        "enum": [
          "account_pin", "api_key", "bank_routing_number", "bban", "company", 
          "credit_card_number", "credit_card_security_code", "customer_id", "date", 
          "date_of_birth", "date_time", "driver_license_number", "email", "employee_id", 
          "first_name", "iban", "ipv4", "ipv6", "last_name", "local_latlng", "name", 
          "passport_number", "password", "phone_number", "ssn", "street_address", 
          "swift_bic_code", "time", "user_name"
        ]
      }
    },
    "required": ["start", "end", "label"]
  }
}
```

*Notes:* To facilitate the evaluation, we will be using the ai-sdk.

## Results

Results for 50 optimization steps:

```
--- Running evaluation for google/gemma-3-4b-it ---
Best performing instruction for google/gemma-3-4b-it:
Score: 1.0
Optimization step: 0
Instruction: You MUST answer with a JSON object that matches the JSON schema above. Do not include any other text, only the JSON object and DO NOT return the data in markdown format.


--- Running evaluation for meta-llama/llama-3.3-70b-instruct ---
Best performing instruction for meta-llama/llama-3.3-70b-instruct:
Score: 1.0
Optimization step: 1
Instruction: Ensure that your response is a correctly formatted JSON object, validating it against the provided JSON schema. Avoid adding any explanatory text or formatting such as markdown. Strictly output only the JSON object.


--- Running evaluation for meta-llama/llama-3.2-3b-instruct ---
Best performing instruction for meta-llama/llama-3.2-3b-instruct:
Score: 0.4
Optimization step: 0
Instruction: You MUST answer with a JSON object that matches the JSON schema above. Do not include any other text, only the JSON object and DO NOT return the data in markdown format.

--- Running evaluation for perplexity/llama-3.1-sonar-small-128k-online ---
Best performing instruction for perplexity/llama-3.1-sonar-small-128k-online:
Score: 0.4
Optimization step: 3
Instruction: Provide your response as a JSON object by directly starting and ending with the JSON structure according to the given schema. Avoid adding any extra text, including explanations or confirmations, outside the braces that encapsulate the JSON content.


--- Running evaluation for microsoft/phi-4 ---
Best performing instruction for microsoft/phi-4:
Score: 1.0
Optimization step: 0
Instruction: You MUST answer with a JSON object that matches the JSON schema above. Do not include any other text, only the JSON object and DO NOT return the data in markdown format.


--- Running evaluation for mistralai/mistral-7b-instruct ---
Best performing instruction for mistralai/mistral-7b-instruct:
Score: 1.0
Optimization step: 28
Instruction: Respond solely with a valid JSON object that adheres strictly to the provided schema. Begin your output with the opening curly brace '{' and end with the closing curly brace '}', ensuring nothing precedes or follows these braces. Follow these precise JSON guidelines: use double quotes for all keys and string values, separate keys and values with colons, and separate key-value pairs with commas, taking special care to avoid trailing commas. To mitigate common errors, closely examine your JSON for misplaced punctuation or unexpected data. Utilize a trusted JSON validator to confirm your JSON is flawless and formed correctly, adjusting errors as necessary to achieve an error-free result before submission.


--- Running evaluation for cohere/command-a ---
Best performing instruction for cohere/command-a:
Score: 1.0
Optimization step: 0
Instruction: You MUST answer with a JSON object that matches the JSON schema above. Do not include any other text, only the JSON object and DO NOT return the data in markdown format.


--- Running evaluation for qwen/qwen2.5-32b-instruct ---
Best performing instruction for qwen/qwen2.5-32b-instruct:
Score: 1.0
Optimization step: 0
Instruction: You MUST answer with a JSON object that matches the JSON schema above. Do not include any other text, only the JSON object and DO NOT return the data in markdown format.
```