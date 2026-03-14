from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("🔄 Loading GPT model...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

tokenizer.pad_token = tokenizer.eos_token


def generate_text(prompt):
    formatted_prompt = (
        "Write a detailed, structured, and educational explanation.\n"
        f"Topic: {prompt}\n"
        "Answer:\n"
    )

    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=450,        # 🔥 longer output
        min_length=200,        # 🔥 forces detail
        do_sample=True,
        temperature=0.75,      # balanced creativity
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.4,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean prompt from output
    if "Answer:" in text:
        text = text.split("Answer:")[1].strip()

    return text


if __name__ == "__main__":
    print("\n📝 GENERATIVE TEXT MODEL (GPT)\n")

    user_prompt = input("Enter a topic or prompt: ")

    print("\n⚡ Generating detailed text...\n")
    result = generate_text(user_prompt)

    print("✅ Generated Text:\n")
    print(result)
