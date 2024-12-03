""" 
Author: Saharsh Barve
Description: Pydantic model for the GSM8k dataset
"""

from pydantic import BaseModel
from typing import List, Literal, Union

# ========== Pydantic Model for the dataset ============
class GSM8KDataRow(BaseModel):
    question: str
    correct_answer: str
    generated_answer: str
    strategy: Union[Literal["zero-shot", "few-shot", "cot", "sc-cot"], Literal[""]]
    priority: Union[int, None]

class GSM8KDataset(BaseModel):
    data: List[GSM8KDataRow]
# ======================================================