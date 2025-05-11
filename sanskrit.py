from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Проверяем, доступен ли GPU (CUDA) или MPS (для Apple Silicon), иначе используем CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Выберите модель NLLB.
model_name = 'facebook/nllb-200-3.3B'
# model_name = 'facebook/nllb-200-1.3B'
# model_name = 'facebook/nllb-200-distilled-600M'

print(f"Загрузка токенизатора для {model_name}...")
# Код для санскрита (деванагари): san_Deva
# Код для санскрита (латиница, IAST и др.): san_Latn
source_lang_code = "san_Deva"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_lang_code)

print(f"Загрузка модели {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval() # Перевод модели в режим оценки

# --- Тексты для перевода (санскрит) и их эталонные переводы (русский) ---
sanskrit_texts_data = [
    {
        "id": "Текст 1 (Сутра сердца, санскрит)",
        "sanskrit": """इह शारिपुत्र रूपं शून्यता शून्यतैव रूपम्। रूपान्न पृथक् शून्यता शून्यया न पृथग् रूपम्। यद् रूपं सा शून्यता या शून्यता तद् रूपम्। एवमेव वेदना संज्ञा संस्कारा विज्ञानानि।""",
        "etalon_russian": "Здесь, о Шарипутра, форма есть пустота, и сама пустота есть форма; нет пустоты помимо формы, нет формы помимо пустоты. Что есть форма, то есть пустота, что есть пустота, то есть форма. Точно так же [обстоит дело с] ощущением, представлением, формирующими факторами и сознанием."
    },
    {
        "id": "Текст 2 (Муламадхьямакакарика, санскрит)",
        "sanskrit": """अनिरोधम् अनुत्पादम् अनुच्छेदम् अशाश्वतम्। अनेकार्थम् अनानार्थम् अनागमम् अनिर्गमम्। यः प्रतीत्यसमुत्पादं प्रपञ्चोपशमं शिवम्। देशयामास संबुद्धस्तं वन्दे वदतां वरम्॥""",
        "etalon_russian": "Неуничтожимое, невозникающее, непрерываемое, непостоянное, не имеющее одного смысла, не имеющее различных смыслов, не приходящее, не уходящее – эту взаимозависимым образом возникающую [реальность], умиротворение концептуальных построений, благую, возвестил Совершенный Будда. Ему, лучшему из учителей, я поклоняюсь."
    },
    {
        "id": "Текст 3 (Уданаварга, санскрит)",
        "sanskrit": """मनः पूर्वङ्गमा धर्मा मनः श्रेष्ठा मनोजवाः। मनसा चेत् प्रसन्नेन भाषते वा करोति वा। ततस्तं सुखमन्वेति च्छायेवानपायिनी॥""",
        "etalon_russian": "Ум предшествует всем состояниям (дхармам), ум – их наилучшая [часть], они созданы умом. Если кто-либо говорит или действует с ясным (чистым) умом, счастье следует за ним, как неотступная тень."
    }
]

# Код для русского языка: rus_Cyrl
target_lang_code = "rus_Cyrl"
target_lang_id = None

try:
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang_code)
    if target_lang_id is None or target_lang_id == tokenizer.unk_token_id:
        print(f"Предупреждение: Код языка '{target_lang_code}' не был распознан как известный токен напрямую "
              f"(ID: {target_lang_id}, UNK ID: {tokenizer.unk_token_id}).")
        if target_lang_code in tokenizer.get_vocab(): # Маловероятно для NLLB, но для полноты
            target_lang_id = tokenizer.get_vocab()[target_lang_code]
            print(f"Найден ID через get_vocab(): {target_lang_id}")
        else:
            print(f"Код языка '{target_lang_code}' не найден и в get_vocab(). "
                  "Проверьте правильность кода или версию библиотеки transformers.")
            if target_lang_id is None or target_lang_id == tokenizer.unk_token_id:
                 raise ValueError(f"Не удалось корректно определить ID для языка {target_lang_code}")
    print(f"ID для целевого языка ({target_lang_code}): {target_lang_id}")
except Exception as e:
    print(f"Критическая ошибка при получении ID для языка '{target_lang_code}': {e}")
    exit()

# --- Параметры для тестирования ---
test_configurations = []

# 1. Тесты с num_beams
beams_settings = [1, 5, 10]
for beams in beams_settings:
    test_configurations.append({
        "name": f"Beams: {beams}",
        "params": {
            "num_beams": beams,
            "do_sample": False,
            "early_stopping": True if beams > 1 else False
        }
    })

# 2. Тесты с temperature
temperature_settings = [0.6, 0.8, 1.0]
for temp in temperature_settings:
    test_configurations.append({
        "name": f"Temp: {temp}",
        "params": {
            "num_beams": 1,
            "do_sample": True,
            "temperature": temp,
            "top_k": 50,
            "top_p": 0.95
        }
    })

# --- Цикл тестирования ---
results_log = []

for text_data in sanskrit_texts_data:
    print(f"\n\n--- Тестирование для: {text_data['id']} ---")
    print(f"Исходный текст (санскрит):\n{text_data['sanskrit']}")
    print(f"Эталонный перевод (русский):\n{text_data['etalon_russian']}\n")

    inputs = tokenizer(text_data['sanskrit'], return_tensors="pt", truncation=True, max_length=1024).to(device)

    for config in test_configurations:
        print(f"  Перевод с настройками: {config['name']}...")

        generation_params = {
            "forced_bos_token_id": target_lang_id,
            "max_length": 1024, # Увеличим для философских текстов
            **config['params']
        }

        try:
            with torch.no_grad():
                 translated_tokens = model.generate(
                    **inputs,
                    **generation_params
                )
            machine_translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(f"  Машинный перевод ({config['name']}):\n  {machine_translation}\n")
            results_log.append({
                "text_id": text_data['id'],
                "source_text_sanskrit": text_data['sanskrit'],
                "etalon_russian": text_data['etalon_russian'],
                "config_name": config['name'],
                "generation_params": generation_params,
                "machine_translation_russian": machine_translation
            })
        except Exception as e:
            print(f"  Ошибка при генерации с параметрами {config['name']}: {e}")
            results_log.append({
                "text_id": text_data['id'],
                "source_text_sanskrit": text_data['sanskrit'],
                "etalon_russian": text_data['etalon_russian'],
                "config_name": config['name'],
                "generation_params": generation_params,
                "machine_translation_russian": f"ОШИБКА: {e}"
            })

print("\n\n--- Все тесты завершены. ---")

# --- Формирование отчета ---
print("\n\n--- ОТЧЕТ ПО РАБОТЕ МОДЕЛИ (САНСКРИТ -> РУССКИЙ) ---")
for log_entry in results_log:
    print(f"\n===================================================")
    print(f"Текст: {log_entry['text_id']}")
    print(f"Настройки: {log_entry['config_name']}")
    print(f"---------------------------------------------------")
    print(f"ОРИГИНАЛ (санскрит):\n{log_entry['source_text_sanskrit']}")
    print(f"---------------------------------------------------")
    print(f"ЭТАЛОННЫЙ ПЕРЕВОД (русский):\n{log_entry['etalon_russian']}")
    print(f"---------------------------------------------------")
    print(f"МАШИННЫЙ ПЕРЕВОД (русский):\n{log_entry['machine_translation_russian']}")
    print(f"===================================================")
