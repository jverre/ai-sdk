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

1. Gemma 3
2. DeepSeek R1
3. Llama 3.3 - 70B
4. Llama 3.2 - 3B
5. Llama 3.1 8B Instruct
6. phi4
7. phi4-mini
8. Mistal - 7B
9. Qwen 2.5
10. Command-a

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