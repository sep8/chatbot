from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator

def get_system_prompt_text():
    with open('public/system.txt', 'r') as f:
        system_text = f.read()
    return system_text

class PaimonPromptTemplate(StringPromptTemplate, BaseModel):
    @validator("input_variables", allow_reuse=True)
    def validate_input_variables(cls, v):
        if "input" not in v:
            raise ValueError("input must be the only input_variable.")
        return v

    def format(self, **kwargs) -> str:
        # Get the source code of the function
        system = get_system_prompt_text()
        input = kwargs["input"]
        input = f"旅行者：{input}"
        history = kwargs["history"]
        history = "\n".join(history)
        # Generate the prompt to be sent to the language model
        prompt = f"""{system}\n{history}\n{input}"""
        return prompt
    
    def _prompt_type(self):
        return "paimon"