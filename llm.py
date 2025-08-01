from ibm_watsonx_ai.foundation_models import Model
import os
from dotenv import load_dotenv
load_dotenv()

model = Model(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    credentials={"url": os.getenv("IBM_URL"), "apikey": os.getenv("IBM_API_KEY")},
    project_id=os.getenv("IBM_PROJECT_ID")
)

def generate_answer(question, context_chunks):
    prompt = f"""Answer based strictly on the following context:\n{chr(10).join(context_chunks)}\n\nQuestion: {question}\nAvoid hallucinations. Respond factually."""
    response = model.generate(prompt=prompt, max_new_tokens=300, temperature=0.5)
    return response['results'][0]['generated_text']
