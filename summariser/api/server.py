from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from summariser.core.summariser import Summariser

app = FastAPI()
summariser = Summariser()

first_example = \
"Microsoft Corporation intends to officially end free support for the Windows 7 operating system "
second_example = \
"根据该组织的官方门户网站，微软公司打算在2020年1月14日之后正式终止对Windows "

class Payload(BaseModel):
    sentences: List[str] = Field([first_example, 
                                  second_example],
                                 title="Input sentences")

class Summaries(BaseModel):
    summaries: List[str] = Field(None, title="Summaries")

@app.post("/summarise", response_model=Summaries, status_code=200)
def summarise(sentences: Payload):
    summaries = summariser.summarise(sentences.sentences)
    summaries = Summaries(summaries=summaries)
    return summaries