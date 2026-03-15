from transformers import MarianTokenizer, MarianMTModel
model_name = "Helsinki-NLP/opus-mt-en-id"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Two women shaking hands in front of a wall of flags. Sri Mulyani I on the left is wearing a blue traditional Indian outfit with a patterned shawl draped over her shoulders. She has dark hair and is wearing glasses."

sentences = [s.strip() for s in text.split(".") if s.strip()]
to_translate = [s + "." for s in sentences]
tokens2 = tokenizer(to_translate, return_tensors="pt", padding=True)
out2 = model.generate(**tokens2)
print("Sentences padded:", " ".join(tokenizer.batch_decode(out2, skip_special_tokens=True)))
