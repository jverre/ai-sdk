import pandas as pd
from pydantic import BaseModel
from typing import List, Literal

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/gretelai/synthetic_pii_finance_multilingual/" + splits["test"])
df = df.sort_values(by='index', ascending=True).reset_index(drop=True).head(50)
df = df[['index', 'generated_text', 'pii_spans']]

from ai_sdk.openai import openai, create_openai_provider
from ai_sdk.openai.provider import OpenAIProviderSettings
from ai_sdk import generate_object
import os
import dotenv

dotenv.load_dotenv()

provider = create_openai_provider(
    settings=OpenAIProviderSettings(
        api_key=os.getenv("OPENAI_API_KEY"),
        #api_key=os.getenv("OPENROUTER_API_KEY"),
        #base_url="https://openrouter.ai/api/v1",
    )
)

model = provider.create_chat_model("deepseek/deepseek-v3-base:free")
#model = provider.create_chat_model("gpt-4o")

class PII_SPAN(BaseModel):
    start: int
    end: int
    label: Literal["account_pin", "api_key", "bank_routing_number", "bban", "company", "credit_card_number", "credit_card_security_code", "customer_id", "date", "date_of_birth", "date_time", "driver_license_number", "email", "employee_id", "first_name", "iban", "ipv4", "ipv6", "last_name", "local_latlng", "name", "passport_number", "password", "phone_number", "ssn", "street_address", "swift_bic_code", "time", "user_name"]

class PIISchema(BaseModel):
    pii_spans: List[PII_SPAN]

generated_text = df.iloc[0]['generated_text']
res = generate_object(
    model=model,
    schema=PIISchema,
    prompt=f"Extract the PII from the following text: {generated_text}"
)

print(generated_text)
print(res.object)