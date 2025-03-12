import pandas as pd
import time
import torch
from openai import OpenAI
from db_utils import create_table, log_token_usage

# Initialize DeepSeek client
client = OpenAI(api_key="sk-1a87e1e386e4478395b227b17a8ebdc9", base_url="https://api.deepseek.com")

# Create the SQLite table if it doesn't exist
create_table()

# Load your training data
df = pd.read_csv("C:\\CS Tech\\Agents\\Daily\\NEW\\quebec_insurance_questions.csv")  # Only first 5 rows
df = df[['question', 'context']]  # We'll generate answers from teacher

teacher_answers = []

for idx, row in df.iterrows():
    question = row['question']
    context = row['context']
    
    # Construct prompt
    prompt = f"""
You are an AI trained to answer insurance-related questions based on Quebec insurance regulations.
Check for the answers from the context provided. Use only the information provided in the context to answer.
Always give professional answers.

Context:
{context}

Question:
{question}
"""

    # Call DeepSeek API
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained on Quebec insurance regulations."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        end_time = time.time()
        answer = response.choices[0].message.content.strip()
        total_tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        response_time = (end_time - start_time) * 1000  # ms

        print(f"\n🧠 Question {idx + 1}: {question}")
        print(f"📄 Context: {context[:100]}...")
        print(f"🎯 Teacher Answer: {answer}\n")

        # Save answer
        teacher_answers.append(answer)

        # Log API usage
        log_token_usage(
            api_name="deepseek-chat",
            prompt=prompt,
            response=answer,
            response_time=response_time,
            token_count=total_tokens
        )

    except Exception as e:
        print(f"❌ Error at row {idx}: {e}")
        teacher_answers.append("")

# Add generated answers to DataFrame
df['teacher_answer'] = teacher_answers

# Save for student training
torch.save(df, "C:\\CS Tech\\Agents\\Daily\\NEW\\new_teacher_predictions.pth")
print("✅ Saved teacher answers to 'new_teacher_predictions.pth'")
