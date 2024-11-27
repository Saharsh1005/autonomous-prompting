""" 
Author: Saharsh Barve
Description: Pydantic model for the GSM8k dataset
"""

from pydantic import BaseModel
from typing import List, Literal, Union

# ========== Pydantic Model for the dataset ============
class GSM8KDataRow(BaseModel):
    question: str
    prompt: str
    answer: str
    response: str
    strategy: Union[Literal["zero-shot", "few-shot", "cot", "sc"], Literal[""]]
    # FIXME: Add Priority field if needed

class GSM8KDataset(BaseModel):
    data: List[GSM8KDataRow]
# ======================================================