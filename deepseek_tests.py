import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datetime
import re # For parsing the output

# --- Configuration ---
# 1. SELECT DEEPSEEK MODEL (REPLACE THIS WITH YOUR CHOSEN DEEPSEEK MODEL)
# Examples:
# DEEPSEEK_MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
# DEEPSEEK_MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base" (might need more careful prompting)
# DEEPSEEK_MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct" (less likely for translation, but an option)
DEEPSEEK_MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat" # <-- IMPORTANT: CHOOSE AND VERIFY!

# 2. Device Selection
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 3. Quantization (Optional, to save VRAM, might impact performance slightly)
# Set to None if you don't want quantization or have ample VRAM
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
# quantization_config = None # Uncomment this if not using quantization

# 4. Target Language (for prompting and etalon)
TARGET_LANGUAGE_NAME = "Russian"

# 5. Texts for Translation (Same as your NLLB script)
ALL_CANONICAL_TEXTS_DATA = [
    {
        "id": "Тибетский Текст 1: Драгоценная человеческая жизнь",
        "source_language_name": "Tibetan",
        "source_text": """དལ་འབྱོར་གྱི་མི་ལུས་རིན་པོ་ཆེ་འདི་ནི་ཐོབ་པར་ཤིན་ཏུ་དཀའ་བ་ཡིན། རྒྱུ་མཚན་ནི་འགྲོ་བ་རིགས་དྲུག་གི་ནང་ནས་མི་ལུས་ཐོབ་པ་ཉུང་ཞིང་། དེ་ལས་ཀྱང་ཆོས་བྱེད་པའི་དལ་ཁོམ་དང་འབྱོར་པ་ཚང་བ་ནི་དེ་བས་ཀྱང་དཀོན་པའི་ཕྱིར་རོ། །ལུས་འདི་ལན་ཅིག་ཐོབ་ཀྱང་རྟག་པ་མེད་པས་སྐད་ཅིག་ཀྱང་མི་སྡོད་པར་འཆི་བ་ལ་ཐུག་ཡོད། དེས་ན་ཚེ་འདི་ལ་དོན་མེད་པའི་བྱ་བ་ཁོ་ནས་དུས་འདའ་བར་མི་བྱ་བར། རང་གཞན་གཉིས་ཀྱི་དོན་ཆེན་པོ་སྒྲུབ་པའི་ཐབས་ལ་བརྩོན་པར་བྱའོ། །གལ་ཏེ་ད་ལྟ་ཆོས་ལ་མ་འབད་ན་ཕྱིས་ནམ་ཞིག་མི་ལུས་འདི་ལྟ་བུ་འཐོབ་པའི་ངེས་པ་མེད་དོ། །དེའི་ཕྱིར་དུ་བརྩོན་འགྲུས་ཆེན་པོས་དམ་པའི་ཆོས་ལ་སློབ་སྦྱོང་དང་ཉམས་ལེན་བྱེད་པ་ནི་ཤིན་ཏུ་གལ་ཆེའོ། །""",
        "etalon_russian": "Эту драгоценную человеческую жизнь, наделённую свободами и благами, очень трудно обрести. Причина в том, что среди шести уделов существ человеческое рождение встречается редко, а ещё реже – наличие досуга и благоприятных условий для практики Дхармы. Даже обретя это тело однажды, оно непостоянно и ни на мгновение не задерживаясь, движется к смерти. Поэтому, вместо того чтобы тратить эту жизнь на одни лишь бессмысленные дела, следует усердствовать в методах осуществления великого блага для себя и других. Если сейчас не усердствовать в Дхарме, нет уверенности, что в будущем когда-либо обретёшь подобное человеческое тело. Посему, с великим усердием изучать и практиковать святую Дхарму чрезвычайно важно."
    },
    {
        "id": "Санскрит Текст 1: Сутра Сердца (начало)",
        "source_language_name": "Sanskrit (Devanagari script)",
        "source_text": """इह शारिपुत्र रूपं शून्यता शून्यतैव रूपम्। रूपान्न पृथक् शून्यता शून्यया न पृथग् रूपम्। यद् रूपं सा शून्यता या शून्यता तद् रूपम्। एवमेव वेदना संज्ञा संस्कारा विज्ञानानि।""",
        "etalon_russian": "Здесь, о Шарипутра, форма есть пустота, и сама пустота есть форма; нет пустоты помимо формы, нет формы помимо пустоты. Что есть форма, то есть пустота, что есть пустота, то есть форма. Точно так же [обстоит дело с] ощущением, представлением, формирующими факторами и сознанием."
    },
    {
        "id": "Китайский Канон: Алмазная Сутра (фрагмент, начало)",
        "source_language_name": "Classical Chinese (Simplified script)",
        "source_text": """如是我聞。一時佛在舍衛國。祇樹給孤獨園。與大比丘眾。千二百五十人俱。""",
        "etalon_russian": """Так я слышал. Однажды Будда пребывал в Шравасти, в роще Джетавана, саду Анатхапиндики, с великим собранием бхикшу из тысячи двухсот пятидесяти человек."""
    },
    # Add more texts here...
]

# 6. Generation Parameters to Test
TEST_CONFIGURATIONS = [
    {
        "name": "Default (Greedy)",
        "params": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 1024, # Max *new* tokens to generate
            # "temperature": 0.6, # Not used if do_sample=False
            # "top_p": 0.9,       # Not used if do_sample=False
        }
    },
    {
        "name": "Beam Search: 3",
        "params": {
            "do_sample": False,
            "num_beams": 3,
            "early_stopping": True,
            "max_new_tokens": 1024,
        }
    },
    {
        "name": "Sampling (Temp: 0.7)",
        "params": {
            "do_sample": True,
            "num_beams": 1, # Typically 1 for sampling
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "max_new_tokens": 1024,
        }
    },
]

# --- Prompting Function ---
# This might need adjustment based on the specific DeepSeek model's preferred instruction format.
# For chat models, we'll use the tokenizer's chat template.
def create_translation_prompt_for_chat_model(tokenizer, source_language_name, text_to_translate, target_language_name="Russian"):
    messages = [
        {"role": "user", "content": f"Please translate the following {source_language_name} text into {target_language_name}. Only provide the {target_language_name} translation, without any introductory phrases, explanations, or the original text. \n\n{source_language_name} text:\n```\n{text_to_translate}\n```\n\n{target_language_name} translation:"}
    ]
    # `add_generation_prompt=True` is important for the model to know it should generate a response.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def create_translation_prompt_for_base_model(source_language_name, text_to_translate, target_language_name="Russian"):
    # For base models, a simpler instruction format might work.
    # You might need to experiment with what works best.
    # Adding "### Instruction:" and "### Response:" style can sometimes help.
    prompt = f"Translate the following text from {source_language_name} to {target_language_name}.\n"
    prompt += f"Input ({source_language_name}):\n{text_to_translate}\n"
    prompt += f"Output ({target_language_name}):\n"
    return prompt


# --- Output Parsing Function ---
def parse_translation_output(full_output, prompt_text):
    """
    Attempts to extract only the generated translation from the model's full output.
    This is a common challenge with Causal LMs.
    """
    # Simple approach: remove the prompt from the start of the output.
    # This assumes the model directly appends the translation.
    if full_output.startswith(prompt_text):
        translation = full_output[len(prompt_text):].strip()
    else:
        # If prompt not found at the start (e.g., due to chat templates adding their own BOS tokens),
        # this might be more complex. For chat models, the output is usually cleaner.
        # For now, let's assume the model's output *is* the translation if the prompt isn't directly prefixed.
        # This part might need refinement based on observed model behavior.
        translation = full_output.strip()

    # Further cleanup: Sometimes models add "Here is the translation:" or similar.
    # This is highly model-dependent and might require regex or more sophisticated parsing.
    # For now, we'll keep it simple.
    # Example: if model says "Russian translation:\n <actual_translation>", try to get <actual_translation>
    cleaned_translation = re.sub(rf"^(.*{TARGET_LANGUAGE_NAME} translation:[\s\n]*)", "", translation, flags=re.IGNORECASE | re.DOTALL)
    if cleaned_translation and len(cleaned_translation) < len(translation): # check if regex did something
        translation = cleaned_translation.strip()

    # Another common pattern: if the model repeats the "Source Text:" or similar from the prompt
    source_text_marker_variants = [
        f"{text_data['source_language_name']} text:",
        "```",
        "Input ("
    ]
    for marker in source_text_marker_variants:
        if marker in translation:
            translation = translation.split(marker)[0].strip()

    return translation

# --- Main Script ---
print(f"--- Initializing Translation Test with DeepSeek ---")
print(f"Timestamp: {datetime.datetime.now()}")
print(f"Using device: {device}")
print(f"Using DeepSeek model: {DEEPSEEK_MODEL_NAME}")

# Load Model and Tokenizer
print(f"\nLoading tokenizer for {DEEPSEEK_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_NAME, trust_remote_code=True)

# Set pad_token if not already set (common for Causal LMs)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        print(f"  Tokenizer `pad_token` is None. Setting `pad_token` to `eos_token` ('{tokenizer.eos_token}').")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Add a new pad token if EOS is also missing (less common for good models)
        new_pad_token = "[PAD]"
        tokenizer.add_special_tokens({'pad_token': new_pad_token})
        print(f"  Tokenizer `pad_token` and `eos_token` are None. Added new `pad_token`: '{new_pad_token}'.")
        # Important: if we add tokens, model embeddings might need resizing if not done by from_pretrained
        # model.resize_token_embeddings(len(tokenizer)) # May not be needed if from_pretrained handles it

print(f"Loading model {DEEPSEEK_MODEL_NAME} to {device}...")
if quantization_config and device == "cuda": # BNB quantization is typically for CUDA
    print(f"  Applying quantization: {quantization_config}")
    model = AutoModelForCausalLM.from_pretrained(
        DEEPSEEK_MODEL_NAME,
        torch_dtype=torch.bfloat16 if quantization_config.bnb_4bit_compute_dtype == torch.bfloat16 else torch.float16, # or torch.float16
        quantization_config=quantization_config,
        device_map="auto", # Automatically distribute layers if >1 GPU and model is large
        trust_remote_code=True
    )
else:
    print("  Loading model without bitsandbytes quantization.")
    model = AutoModelForCausalLM.from_pretrained(
        DEEPSEEK_MODEL_NAME,
        torch_dtype=torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16, # Use bfloat16 if available
        trust_remote_code=True
    ).to(device) # Move to device if not using device_map

model.eval()
print("DeepSeek Model and tokenizer loaded.")

# Determine if it's a chat model (simple heuristic based on name)
is_chat_model = "chat" in DEEPSEEK_MODEL_NAME.lower()
print(f"Model identified as {'Chat Model' if is_chat_model else 'Base Model'}.")


# --- Results Log ---
results_log = []

# --- Translation Loop ---
for text_data in ALL_CANONICAL_TEXTS_DATA:
    print(f"\n\n--- Processing Text ID: {text_data['id']} ---")
    print(f"Source Language: {text_data['source_language_name']}")
    print(f"Original Text:\n{text_data['source_text']}")
    print(f"Etalon {TARGET_LANGUAGE_NAME} Translation:\n{text_data['etalon_russian']}\n")

    for config_idx, config in enumerate(TEST_CONFIGURATIONS):
        print(f"  Translating with configuration: {config['name']} (Config {config_idx+1}/{len(TEST_CONFIGURATIONS)})...")
        # print(f"  Parameters: {config['params']}") # Redundant if name is descriptive

        if is_chat_model:
            prompt_text = create_translation_prompt_for_chat_model(
                tokenizer,
                text_data['source_language_name'],
                text_data['source_text'],
                TARGET_LANGUAGE_NAME
            )
        else:
            prompt_text = create_translation_prompt_for_base_model(
                text_data['source_language_name'],
                text_data['source_text'],
                TARGET_LANGUAGE_NAME
            )

        # print(f"  Generated Prompt:\n{prompt_text[:500]}...\n") # For debugging prompt

        try:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(device) # max_length for prompt tokenization

            generation_params = {
                **config['params'],
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            # Remove None params to avoid issues with generate function
            generation_params = {k: v for k, v in generation_params.items() if v is not None}


            with torch.no_grad():
                # Ensure all inputs to generate are on the correct device
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)

                output_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_params
                )

            # Decode the full output (prompt + generation)
            full_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

            # Parse the translation from the full output
            # For chat models, the output is often just the assistant's response.
            # For base models, we need to strip the prompt.
            if is_chat_model:
                # The `apply_chat_template` with `add_generation_prompt=True` means the prompt itself isn't in `output_tokens[0]`
                # if we decode `output_tokens[0, inputs.input_ids.shape[-1]:]` (i.e. only new tokens)
                # However, decoding the whole `output_tokens[0]` and then parsing is often more robust.
                # The `create_translation_prompt_for_chat_model` already formats it so the model should just give the answer.
                # Let's try to get only the generated part if possible.
                generated_tokens_only = output_tokens[0][inputs.input_ids.shape[-1]:]
                machine_translation = tokenizer.decode(generated_tokens_only, skip_special_tokens=True).strip()
                # If the above is empty or weird, fallback to parsing the full output.
                if not machine_translation:
                     machine_translation = parse_translation_output(full_output_text, prompt_text)
            else: # Base model
                 machine_translation = parse_translation_output(full_output_text, prompt_text)


            print(f"  Machine Translation ({config['name']}):\n  {machine_translation}\n")
            results_log.append({
                "text_id": text_data['id'],
                "source_language_name": text_data['source_language_name'],
                "source_text": text_data['source_text'],
                "etalon_russian": text_data['etalon_russian'],
                "config_name": config['name'],
                "generation_params": generation_params,
                "prompt_text_preview": prompt_text[:200] + "...", # Store a preview of the prompt
                "full_model_output_preview": full_output_text[:500] + "...", # Store a preview of the raw output
                "machine_translation_russian": machine_translation,
                "error": None
            })
        except Exception as e:
            print(f"  ERROR during generation for {text_data['id']} with config {config['name']}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            results_log.append({
                "text_id": text_data['id'],
                "source_language_name": text_data['source_language_name'],
                "source_text": text_data['source_text'],
                "etalon_russian": text_data['etalon_russian'],
                "config_name": config['name'],
                "generation_params": generation_params,
                "prompt_text_preview": prompt_text[:200] + "...",
                "full_model_output_preview": "ERROR OCCURRED",
                "machine_translation_russian": f"GENERATION ERROR: {e}",
                "error": str(e)
            })

print("\n\n--- All DeepSeek tests завершены. ---")

# --- Формирование отчета ---
report_filename = f"deepseek_translation_report_{DEEPSEEK_MODEL_NAME.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_filename, "w", encoding="utf-8") as f:
    f.write(f"--- ОТЧЕТ ПО РАБОТЕ МОДЕЛИ DEEPSEEK ({DEEPSEEK_MODEL_NAME}) ---\n")
    f.write(f"Дата и время: {datetime.datetime.now()}\n")
    f.write(f"Устройство: {device}\n\n")

    for log_entry in results_log:
        f.write(f"\n===================================================\n")
        f.write(f"Текст ID: {log_entry['text_id']}\n")
        f.write(f"Исходный язык: {log_entry['source_language_name']}\n")
        f.write(f"Настройки генерации: {log_entry['config_name']}\n")
        # f.write(f"Параметры: {log_entry['generation_params']}\n") # Can be verbose
        f.write(f"---------------------------------------------------\n")
        f.write(f"ОРИГИНАЛ:\n{log_entry['source_text']}\n")
        f.write(f"---------------------------------------------------\n")
        f.write(f"ЭТАЛОННЫЙ ПЕРЕВОД ({TARGET_LANGUAGE_NAME}):\n{log_entry['etalon_russian']}\n")
        f.write(f"---------------------------------------------------\n")
        f.write(f"МАШИННЫЙ ПЕРЕВОД ({TARGET_LANGUAGE_NAME}) (DeepSeek):\n{log_entry['machine_translation_russian']}\n")
        if log_entry['error']:
            f.write(f"---------------------------------------------------\n")
            f.write(f"ОШИБКА: {log_entry['error']}\n")
        # f.write(f"---------------------------------------------------\n")
        # f.write(f"Prompt Preview: {log_entry['prompt_text_preview']}\n")
        # f.write(f"Full Output Preview: {log_entry['full_model_output_preview']}\n")
        f.write(f"===================================================\n")

print(f"\n--- Отчет сохранен в файл: {report_filename} ---")
