import pandas as pd
from pydantic import BaseModel
from typing import List, Literal
from ai_sdk.core.language_model import LanguageModel

def get_eval_samples(nb_samples: int = 50) -> pd.DataFrame:
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/gretelai/synthetic_pii_finance_multilingual/" + splits["test"])
    df = df.sort_values(by='index', ascending=True).reset_index(drop=True).head(nb_samples)
    df = df[['index', 'generated_text', 'pii_spans']]

    return df

class PII_SPAN(BaseModel):
    start: int
    end: int
    label: Literal["account_pin", "api_key", "bank_routing_number", "bban", "company", "credit_card_number", "credit_card_security_code", "customer_id", "date", "date_of_birth", "date_time", "driver_license_number", "email", "employee_id", "first_name", "iban", "ipv4", "ipv6", "last_name", "local_latlng", "name", "passport_number", "password", "phone_number", "ssn", "street_address", "swift_bic_code", "time", "user_name"]

class PIISchema(BaseModel):
    pii_spans: List[PII_SPAN]

def create_system_prompt(json_schema: str, prompt_suffix: str = "") -> str:
    return f"""JSON schema:
{json_schema}

{prompt_suffix}
"""