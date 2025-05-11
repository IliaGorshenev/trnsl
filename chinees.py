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
# Код для традиционного китайского: zho_Hant
# Код для упрощенного китайского: zho_Hans
source_lang_code = "zho_Hant"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_lang_code)

print(f"Загрузка модели {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval() # Перевод модели в режим оценки

# --- Тексты для перевода (китайский) и их эталонные переводы (русский) ---
chinese_texts_data = [
    {
        "id": "Текст 1 (Сутра сердца, китайский)",
        "chinese": """舍利子，色不異空，空不異色；色即是空，空即是色。受想行識，亦復如是。""",
        "etalon_russian": "Шарипутра, форма не отлична от пустоты, пустота не отлична от формы; форма – это и есть пустота, пустота – это и есть форма. Чувства, представления, побуждения и сознание – также таковы."
    },
    {
        "id": "Текст 2 (Алмазная сутра, китайский)",
        "chinese": """凡所有相，皆是虛妄。若見諸相非相，即見如來。""",
        "etalon_russian": "Все, что обладает признаками – иллюзорно. Если увидишь, что все признаки – не признаки, то есть узришь Татхагату."
    },
    {
        "id": "Текст 3 (Дхаммапада/Фацзю Цзин, китайский)",
        "chinese": """心為法本，心尊心使。中心念善，即言即行。福慶自隨，如影隨形。""",
        "etalon_russian": "Ум – основа всех состояний, ум – их властитель, ум их создает. Кто говорит иль поступает с мыслью доброй, за тем, как тень его, и счастье следует."
    }
]

# Код для русского языка: rus_Cyrl
target_lang_code = "rus_Cyrl"
target_lang_id = None

try:
    # Для NLLB ID языка для декодера устанавливается через forced_bos_token_id
    # Этот ID должен соответствовать коду языка в словаре токенизатора
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang_code)

    if target_lang_id is None or target_lang_id == tokenizer.unk_token_id:
        print(f"Предупреждение: Код языка '{target_lang_code}' не был распознан как известный токен напрямую "
              f"(ID: {target_lang_id}, UNK ID: {tokenizer.unk_token_id}).")
        # Дополнительная проверка, если язык есть в словаре как 'rus_Cyrl' (маловероятно для NLLB, но для полноты)
        if target_lang_code in tokenizer.get_vocab():
            target_lang_id = tokenizer.get_vocab()[target_lang_code]
            print(f"Найден ID через get_vocab(): {target_lang_id}")
        else:
            print(f"Код языка '{target_lang_code}' не найден и в get_vocab(). "
                  "Проверьте правильность кода или версию библиотеки transformers.")
            # Если все еще None или UNK, это проблема.
            if target_lang_id is None or target_lang_id == tokenizer.unk_token_id:
                 raise ValueError(f"Не удалось корректно определить ID для языка {target_lang_code}")

    print(f"ID для целевого языка ({target_lang_code}): {target_lang_id}")

except Exception as e:
    print(f"Критическая ошибка при получении ID для языка '{target_lang_code}': {e}")
    print("Возможные причины: неправильный код языка, несовместимая версия библиотеки transformers, "
          "или модель не поддерживает указанный язык в текущей конфигурации токенизатора.")
    print("Пожалуйста, убедитесь, что код языка корректен и токенизатор поддерживает его.")
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

for text_data in chinese_texts_data:
    print(f"\n\n--- Тестирование для: {text_data['id']} ---")
    print(f"Исходный текст (китайский):\n{text_data['chinese']}")
    print(f"Эталонный перевод (русский):\n{text_data['etalon_russian']}\n")

    # Токенизация текста. src_lang уже установлен при загрузке токенизатора
    inputs = tokenizer(text_data['chinese'], return_tensors="pt", truncation=True, max_length=1024).to(device)

    for config in test_configurations:
        print(f"  Перевод с настройками: {config['name']}...")

        generation_params = {
            "forced_bos_token_id": target_lang_id,
            "max_length": 1024,
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
                "source_text_chinese": text_data['chinese'],
                "etalon_russian": text_data['etalon_russian'],
                "config_name": config['name'],
                "generation_params": generation_params,
                "machine_translation_russian": machine_translation
            })
        except Exception as e:
            print(f"  Ошибка при генерации с параметрами {config['name']}: {e}")
            results_log.append({
                "text_id": text_data['id'],
                "source_text_chinese": text_data['chinese'],
                "etalon_russian": text_data['etalon_russian'],
                "config_name": config['name'],
                "generation_params": generation_params,
                "machine_translation_russian": f"ОШИБКА: {e}"
            })

print("\n\n--- Все тесты завершены. ---")

# --- Формирование отчета ---
print("\n\n--- ОТЧЕТ ПО РАБОТЕ МОДЕЛИ (КИТАЙСКИЙ -> РУССКИЙ) ---")
for log_entry in results_log:
    print(f"\n===================================================")
    print(f"Текст: {log_entry['text_id']}")
    print(f"Настройки: {log_entry['config_name']}")
    print(f"---------------------------------------------------")
    print(f"ОРИГИНАЛ (китайский):\n{log_entry['source_text_chinese']}")
    print(f"---------------------------------------------------")
    print(f"ЭТАЛОННЫЙ ПЕРЕВОД (русский):\n{log_entry['etalon_russian']}")
    print(f"---------------------------------------------------")
    print(f"МАШИННЫЙ ПЕРЕВОД (русский):\n{log_entry['machine_translation_russian']}")
    print(f"===================================================")
