import pytest
from src import sample

assert sample.add_punct("Jump", "exclaimation")=="Jump!"

assert sample.add_punct("Did he go yet", "question")=="Did he go yet?"

assert sample.add_punct("I saw the cat walk away", "period")=="I saw the cat walk away."