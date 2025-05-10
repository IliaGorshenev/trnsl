from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Проверяем, доступен ли GPU (CUDA) или MPS (для Apple Silicon), иначе используем CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Выберите модель NLLB.
model_name = 'facebook/nllb-200-3.3B'
# model_name = 'facebook/nllb-200-1.3B'
# model_name = 'facebook/nllb-200-distilled-600M' # Еще меньшая модель для более быстрых тестов, если нужно

print(f"Загрузка токенизатора для {model_name}...")
# Код для стандартного тибетского: bod_Tibt
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="bod_Tibt")

print(f"Загрузка модели {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval() # Перевод модели в режим оценки

# --- Тексты для перевода и их эталонные переводы ---
tibetan_texts_data = [
    {
        "id": "Текст 1 (о драгоценной человеческой жизни)",
        "tibetan": """དལ་འབྱོར་གྱི་མི་ལུས་རིན་པོ་ཆེ་འདི་ནི་ཐོབ་པར་ཤིན་ཏུ་དཀའ་བ་ཡིན། རྒྱུ་མཚན་ནི་འགྲོ་བ་རིགས་དྲུག་གི་ནང་ནས་མི་ལུས་ཐོབ་པ་ཉུང་ཞིང་། དེ་ལས་ཀྱང་ཆོས་བྱེད་པའི་དལ་ཁོམ་དང་འབྱོར་པ་ཚང་བ་ནི་དེ་བས་ཀྱང་དཀོན་པའི་ཕྱིར་རོ། །ལུས་འདི་ལན་ཅིག་ཐོབ་ཀྱང་རྟག་པ་མེད་པས་སྐད་ཅིག་ཀྱང་མི་སྡོད་པར་འཆི་བ་ལ་ཐུག་ཡོད། དེས་ན་ཚེ་འདི་ལ་དོན་མེད་པའི་བྱ་བ་ཁོ་ནས་དུས་འདའ་བར་མི་བྱ་བར། རང་གཞན་གཉིས་ཀྱི་དོན་ཆེན་པོ་སྒྲུབ་པའི་ཐབས་ལ་བརྩོན་པར་བྱའོ། །གལ་ཏེ་ད་ལྟ་ཆོས་ལ་མ་འབད་ན་ཕྱིས་ནམ་ཞིག་མི་ལུས་འདི་ལྟ་བུ་འཐོབ་པའི་ངེས་པ་མེད་དོ། །དེའི་ཕྱིར་དུ་བརྩོན་འགྲུས་ཆེན་པོས་དམ་པའི་ཆོས་ལ་སློབ་སྦྱོང་དང་ཉམས་ལེན་བྱེད་པ་ནི་ཤིན་ཏུ་གལ་ཆེའོ། །""",
        "etalon_russian": "Эту драгоценную человеческую жизнь, наделённую свободами и благами, очень трудно обрести. Причина в том, что среди шести уделов существ человеческое рождение встречается редко, а ещё реже – наличие досуга и благоприятных условий для практики Дхармы. Даже обретя это тело однажды, оно непостоянно и ни на мгновение не задерживаясь, движется к смерти. Поэтому, вместо того чтобы тратить эту жизнь на одни лишь бессмысленные дела, следует усердствовать в методах осуществления великого блага для себя и других. Если сейчас не усердствовать в Дхарме, нет уверенности, что в будущем когда-либо обретёшь подобное человеческое тело. Посему, с великим усердием изучать и практиковать святую Дхарму чрезвычайно важно."
    },
    {
        "id": "Текст 2 (о пустотности)",
        "tibetan": """གང་ཞིག་རྟེན་ཅིང་འབྲེལ་བར་འབྱུང་བ་དེ་ཉིད་སྟོང་པ་ཉིད་དུ་བཤད། དེ་ནི་དོན་དམ་པའི་གནས་ལུགས་ཡིན་ནོ། །སྟོང་པ་ཉིད་མ་རྟོགས་ན་འཁོར་བ་ལས་མི་གྲོལ། དེ་བས་ན་སྟོང་པ་ཉིད་ལ་བསམ་པ་དང་སྒོམ་པ་གལ་ཆེའོ། །""",
        "etalon_russian": "То, что возникает зависимо и взаимосвязано, то и описывается как пустотность. Такова подлинная природа явлений. Не постигнув пустотность, невозможно освободиться от сансары. Поэтому размышлять и медитировать о пустотности очень важно."
    },
    {
        "id": "Текст 3 (о сострадании)",
        "tibetan": """སེམས་ཅན་ཐམས་ཅད་བདེ་བ་འདོད་ཅིང་སྡུག་བསྔལ་མི་འདོད་པ་མཚུངས་ཀྱང་། བདེ་བའི་རྒྱུ་དང་སྡུག་བསྔལ་གྱི་རྒྱུ་མི་ཤེས་པས་སྡུག་བསྔལ་མྱོང་། དེས་ན་སེམས་ཅན་ཐམས་ཅད་ལ་སྙིང་རྗེ་ཆེན་པོ་བསྐྱེད་དེ་དེ་དག་སྡུག་བསྔལ་ལས་སྒྲོལ་བར་བྱའོ། །""",
        "etalon_russian": "Хотя все живые существа одинаково желают счастья и не желают страданий, они испытывают страдания, так как не знают причин счастья и причин страданий. Поэтому, породив великое сострадание ко всем живым существам, следует освобождать их от страданий."
    }
]

# Код для русского языка: rus_Cyrl
target_lang_code = "rus_Cyrl"
target_lang_id = None

try:
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang_code)
    if target_lang_id is None or target_lang_id == tokenizer.unk_token_id:
        print(f"Предупреждение: Код языка '{target_lang_code}' не был распознан как известный токен напрямую (ID: {target_lang_id}, UNK ID: {tokenizer.unk_token_id}).")
        if target_lang_code in tokenizer.get_vocab():
            target_lang_id = tokenizer.get_vocab()[target_lang_code]
            print(f"Найден ID через get_vocab(): {target_lang_id}")
        else:
            print(f"Код языка '{target_lang_code}' не найден и в get_vocab(). Проверьте правильность кода или версию библиотеки transformers.")
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
            "early_stopping": True if beams > 1 else False # Early stopping полезен для beam search
        }
    })

# 2. Тесты с temperature
temperature_settings = [0.6, 0.8, 1.0]
for temp in temperature_settings:
    test_configurations.append({
        "name": f"Temp: {temp}",
        "params": {
            "num_beams": 1, # Temperature обычно используется с num_beams=1
            "do_sample": True,
            "temperature": temp,
            "top_k": 50, # Можно добавить top_k и top_p для контроля сэмплирования
            "top_p": 0.95
        }
    })


# --- Цикл тестирования ---
results_log = []

for text_data in tibetan_texts_data:
    print(f"\n\n--- Тестирование для: {text_data['id']} ---")
    print(f"Исходный текст (тибетский):\n{text_data['tibetan']}")
    print(f"Эталонный перевод (русский):\n{text_data['etalon_russian']}\n")

    inputs = tokenizer(text_data['tibetan'], return_tensors="pt", truncation=True, max_length=1024).to(device)

    for config in test_configurations:
        print(f"  Перевод с настройками: {config['name']}...")
        
        generation_params = {
            "forced_bos_token_id": target_lang_id,
            "max_length": 1024, # Увеличим max_length для более полных переводов философских текстов
            **config['params']
        }

        try:
            with torch.no_grad(): # Важно для экономии памяти и ускорения в режиме инференса
                 translated_tokens = model.generate(
                    **inputs,
                    **generation_params
                )
            
            machine_translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            print(f"  Машинный перевод ({config['name']}):\n  {machine_translation}\n")
            results_log.append({
                "text_id": text_data['id'],
                "tibetan_text": text_data['tibetan'],
                "etalon_russian": text_data['etalon_russian'],
                "config_name": config['name'],
                "generation_params": generation_params,
                "machine_translation_russian": machine_translation
            })
        except Exception as e:
            print(f"  Ошибка при генерации с параметрами {config['name']}: {e}")
            results_log.append({
                "text_id": text_data['id'],
                "tibetan_text": text_data['tibetan'],
                "etalon_russian": text_data['etalon_russian'],
                "config_name": config['name'],
                "generation_params": generation_params,
                "machine_translation_russian": f"ОШИБКА: {e}"
            })

print("\n\n--- Все тесты завершены. ---")

# --- Формирование отчета ---
print("\n\n--- ОТЧЕТ ПО РАБОТЕ МОДЕЛИ ---")
for log_entry in results_log:
    print(f"\n===================================================")
    print(f"Текст: {log_entry['text_id']}")
    print(f"Настройки: {log_entry['config_name']}")
    # print(f"Параметры генерации: {log_entry['generation_params']}") # Можно раскомментировать для детального лога
    print(f"---------------------------------------------------")
    print(f"ОРИГИНАЛ (тибетский):\n{log_entry['tibetan_text']}")
    print(f"---------------------------------------------------")
    print(f"ЭТАЛОННЫЙ ПЕРЕВОД (русский):\n{log_entry['etalon_russian']}")
    print(f"---------------------------------------------------")
    print(f"МАШИННЫЙ ПЕРЕВОД (русский):\n{log_entry['machine_translation_russian']}")
    print(f"===================================================")
