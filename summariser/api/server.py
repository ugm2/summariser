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
    max_length: int = Field(None,
                            title="Maximum summary length")
    min_length: int = Field(None,
                            title="Minimum summary length")
    num_beams: int = Field(4,
                           title="Number of exploratory beams")

class Summaries(BaseModel):
    summaries: List[str] = Field(None, title="Summaries")

async def summarise_async(sentences, max_length, min_length, num_beams):
    return summariser.summarise(sentences, max_length, min_length, num_beams)

@app.post("/summarise", response_model=Summaries, status_code=200, name="summarise")
async def summarise(payload: Payload):
    summaries = await summarise_async(payload.sentences,
                                      payload.max_length,
                                      payload.min_length,
                                      payload.num_beams)
    summaries = Summaries(summaries=summaries)
    return summaries