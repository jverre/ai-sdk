from ai_sdk.openai import openai, create_openai_provider
from ai_sdk.openai.provider import OpenAIProviderSettings
from ai_sdk import generate_text
import os
import dotenv
from utils import create_system_prompt, PIISchema
from metric import valid_json
from typing import List, Tuple, Literal, Dict, Optional
import pandas as pd
import re
import json
from concurrent.futures import ThreadPoolExecutor
import time

dotenv.load_dotenv()

def get_dataset(nb_samples: int = 1, few_shot_samples: int = 2):
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/gretelai/synthetic_pii_finance_multilingual/" + splits["test"])
    df = df.sort_values(by='index', ascending=True).reset_index(drop=True)
    
    # Get few-shot examples
    few_shot_df = df.head(few_shot_samples)
    few_shot_examples = [
        {
            "input": row['generated_text'],
            "output": json.dumps(row['pii_spans'])
        }
        for _, row in few_shot_df.iterrows()
    ]
    
    # Get evaluation samples
    eval_df = df.iloc[few_shot_samples:few_shot_samples + nb_samples]
    eval_df = eval_df[['index', 'generated_text', 'pii_spans']]
    
    return eval_df, few_shot_examples


def eval_single_example(args: Tuple[str, str, str]) -> Tuple[bool, Optional[str], str]:
    """Evaluate a single example with retries."""
    generated_text, system_prompt, model = args
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            res = generate_text(
                model=model,
                system=system_prompt,
                prompt=f"Extract the PII from the following text: {generated_text}"
            )
            valid, error = valid_json(res.text)
            return valid, error, res.text
        except Exception as e:
            if attempt == max_retries - 1:
                return False, str(e), ""
            time.sleep(retry_delay * (2 ** attempt))
    
    return False, "Max retries exceeded", ""

def eval_dataset(df: pd.DataFrame, model, system_prompt: str, parallel: bool = True):
    """Evaluate dataset with optional parallel processing."""
    valid_count = 0
    total_count = 0
    errors = []
    error_examples = {}  # Track first example for each error
    
    if parallel:
        with ThreadPoolExecutor() as executor:
            # Prepare arguments for parallel processing
            args = [(row['generated_text'], system_prompt, model) for _, row in df.iterrows()]
            # Run evaluations in parallel
            results = list(executor.map(eval_single_example, args))
            
            for valid, error, response_text in results:
                total_count += 1
                if valid:
                    valid_count += 1
                else:
                    errors.append(error)
                    # Store first example for this error
                    if error not in error_examples:
                        error_examples[error] = response_text
    else:
        for _, row in df.iterrows():
            valid, error, response_text = eval_single_example((row['generated_text'], system_prompt, model))
            total_count += 1
            if valid:
                valid_count += 1
            else:
                errors.append(error)
                if error not in error_examples:
                    error_examples[error] = response_text

    return {
        "accuracy": valid_count / total_count, 
        "errors": errors,
        "error_examples": error_examples
    }


def generate_meta_prompt(
    old_instructions_and_scores: List[Dict[str, any]],
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    directions: Literal["maximize", "minimize"] = "maximize"
):
    """Generate meta-prompt with optional few-shot examples and error information."""
    # Format instructions with their scores and errors
    formatted_instructions = []

    sorted_instructions = sorted(old_instructions_and_scores, key=lambda x: x['score'], reverse=True)

    for x in sorted_instructions[0:5]:
        instruction_text = f"<Instruction>\n<Score> {x['score']}<\SCORE>\n<INS>{x['instruction']}</INS>"
        if 'errors' in x and x['errors']:
            # Get the most common error and its example
            error = max(set(x['errors']), key=x['errors'].count)
            example = x.get('error_examples', {}).get(error, '')
            instruction_text += f"\n<Error>{error}</Error>\n"
        instruction_text += "</Instruction>"
        formatted_instructions.append(instruction_text)
    
    formated_old_instructions = "\n\n".join(formatted_instructions)
    
    # Format few-shot examples if provided
    few_shot_text = ""
    if few_shot_examples:
        few_shot_text = "\n\nExample problems and their solutions:\n"
        for example in few_shot_examples:
            few_shot_text += f"\nInput: {example['input']}\nOutput: {example['output']}\n"
    
    res = f"""
Your task is to generate the instruction <INS> that {"maximizes" if directions == "maximize" else "minimizes"} the scores below.
The score ranges from 0 to 1.

Preview instructions and their performance:
{formated_old_instructions}

Generate an instruction that is different from all the instructions <INS> above, and has a higher score than all the instructions <INS> above.
When creating the instructions, review the errors and try to understand the root cause. Then, try to create an instruction that resolves these root causes.

The instruction should be effective, and generally applicable to all problems above. The instruction should begin with <INS> and end with </INS>.
"""
    return res

def generate_instructions(model, instructions: List[Dict[str, any]], few_shot_examples: Optional[List[Dict[str, str]]] = None):
    """Generate new instructions with optional few-shot examples."""
    meta_prompt = generate_meta_prompt(instructions, few_shot_examples)

    res = generate_text(
        model=openai("gpt-4o"),
        prompt=meta_prompt
    )

    # Parse text between <INS> tags
    ins_match = re.search(r'<INS>(.*?)</INS>', res.text, re.DOTALL)
    if ins_match:
        return ins_match.group(1).strip()
    else:
        raise ValueError("No instruction found in the response")

def run_evaluation(
    model_id="google/gemma-3-4b-it",
    base_instructions="You MUST answer with a JSON object that matches the JSON schema above. Make sure to only return the JSON and nothing else, DO NOT START the response with ```json.",
    nb_evaluation_steps: int = 10,
    nb_samples: int = 10,
    few_shot_samples: int = 2
):
    # Get the dataset and few-shot examples
    df, few_shot_examples = get_dataset(nb_samples=nb_samples, few_shot_samples=few_shot_samples)

    # Create the model
    provider = create_openai_provider(
        settings=OpenAIProviderSettings(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api",
        )
    )
    model = provider(model_id)

    # Run first eval
    system_prompt = create_system_prompt(
        PIISchema.model_json_schema(),
        prompt_suffix=base_instructions
    )

    eval_result = eval_dataset(df, model, system_prompt)
    instructions = [{
        "instruction": base_instructions, 
        "score": eval_result["accuracy"],
        "errors": eval_result["errors"],
        "error_examples": eval_result["error_examples"]
    }]
    print(f"Step 0 accuracy: {eval_result['accuracy']}, instruction: {base_instructions}\n")

    if eval_result['accuracy'] == 1:
        return instructions

    for i in range(nb_evaluation_steps):
        instruction = generate_instructions(model, instructions, few_shot_examples)

        system_prompt = create_system_prompt(
            PIISchema.model_json_schema(),
            prompt_suffix=instruction
        )

        eval_result = eval_dataset(df, model, system_prompt)
        instructions += [{
            "instruction": instruction, 
            "score": eval_result["accuracy"],
            "errors": eval_result["errors"],
            "error_examples": eval_result["error_examples"]
        }]
        print(f"Step {i+1} accuracy: {eval_result['accuracy']}, instruction: {instruction}\n")

        if eval_result['accuracy'] == 1:
            break

    return instructions

if __name__ == "__main__":
    for model_id in [
        "google/gemma-3-4b-it",
        #"deepseek/deepseek-r1", 
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "perplexity/llama-3.1-sonar-small-128k-online",
        "microsoft/phi-4",
        "mistralai/mistral-7b-instruct",
        "cohere/command-a",
        "qwen/qwen2.5-32b-instruct"
    ]:
        print(f"--- Running evaluation for {model_id} ---")
        instructions = run_evaluation(
            model_id=model_id,
            nb_samples=10,
            few_shot_samples=0,
            nb_evaluation_steps=50,
            #base_instructions="Provide your response strictly as a JSON object in accordance with the given schema. Ensure there is no introductory text, no supplementary explanations, and avoid formatting indicators like ```json."
            #base_instructions="Directly output the JSON object as specified by the schema without any introductory or surrounding text. Ensure there are no empty lines or formatting marks like ``` preceding or following the JSON, and confirm the JSON object begins immediately with '{' and ends immediately with '}'. Verify the JSON syntax and all required fields are present to ensure successful parsing."
            base_instructions="You MUST answer with a JSON object that matches the JSON schema above. Do not include any other text, only the JSON object and DO NOT return the data in markdown format."
        )

        best_instruction = max(instructions, key=lambda x: x['score'])
        print(f"\nBest performing instruction for {model_id}:")
        print(f"Score: {best_instruction['score']}")
        print(f"Instruction: {best_instruction['instruction']}\n\n")